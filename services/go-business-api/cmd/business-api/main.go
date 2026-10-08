// 文件用途：启动公开 Business API，装配数据库、Control Plane/Python 客户端、HTTP 路由和优雅停机流程。
package main

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"knowledge-agent/go-business-api/internal/businessapi"
	"knowledge-agent/go-business-api/internal/controlplane"
	"knowledge-agent/go-business-api/internal/experimentqueue"
	"knowledge-agent/go-business-api/internal/python"

	_ "github.com/go-sql-driver/mysql"
)

// main 根据命令行参数选择健康检查模式或正常启动服务。
func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	if len(os.Args) == 2 && os.Args[1] == "healthcheck" {
		if err := healthcheck(); err != nil {
			log.Error("healthcheck failed", "error", err)
			os.Exit(1)
		}
		return
	}
	if err := run(log); err != nil {
		log.Error("business API stopped", "error", err)
		os.Exit(1)
	}
}

// run 初始化全部依赖、启动 HTTP Server，并在系统信号到达时优雅关闭。
func run(log *slog.Logger) error {
	dsn := strings.TrimSpace(os.Getenv("BUSINESS_DATABASE_DSN"))
	if dsn == "" {
		return errors.New("BUSINESS_DATABASE_DSN is required")
	}
	token := strings.TrimSpace(os.Getenv("CONTROL_PLANE_INTERNAL_TOKEN"))
	if len(token) < 16 {
		return errors.New("CONTROL_PLANE_INTERNAL_TOKEN must contain at least 16 characters")
	}
	db, err := sql.Open("mysql", dsn)
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}
	defer db.Close()
	db.SetMaxOpenConns(envInt("BUSINESS_DB_MAX_OPEN", 30))
	db.SetMaxIdleConns(envInt("BUSINESS_DB_MAX_IDLE", 10))
	db.SetConnMaxLifetime(4 * time.Minute)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	err = db.PingContext(ctx)
	cancel()
	if err != nil {
		return fmt.Errorf("connect database: %w", err)
	}

	grpcTimeout := envDuration("BUSINESS_PYTHON_TIMEOUT", 2*time.Minute)
	compute, err := python.New(env("PYTHON_AGENT_GRPC_TARGET", "localhost:50051"), token, grpcTimeout)
	if err != nil {
		return err
	}
	defer compute.Close()
	if err := experimentqueue.Migrate(context.Background(), db); err != nil {
		return fmt.Errorf("migrate experiment outbox: %w", err)
	}
	hostname, _ := os.Hostname()
	kafkaBrokers := strings.Split(env("KAFKA_BROKERS", "localhost:9092"), ",")
	experimentTopic := env("EXPERIMENT_KAFKA_TOPIC", "experiment-events")
	dispatcher, err := experimentqueue.NewDispatcher(
		db,
		compute,
		kafkaBrokers,
		experimentTopic,
		fmt.Sprintf("%s-%d", hostname, os.Getpid()),
		log,
	)
	if err != nil {
		return fmt.Errorf("configure experiment dispatcher: %w", err)
	}
	defer dispatcher.Close()
	consumer, err := experimentqueue.NewConsumer(
		db,
		kafkaBrokers,
		experimentTopic,
		env("EXPERIMENT_KAFKA_CONSUMER_GROUP", "experiment-audit"),
		log,
	)
	if err != nil {
		return fmt.Errorf("configure experiment consumer: %w", err)
	}
	defer consumer.Close()
	agentControl, err := controlplane.New(
		env("CONTROL_PLANE_URL", "http://localhost:8081"),
		env("CONTROL_PLANE_AUTH_SECRET", ""),
	)
	if err != nil {
		return err
	}
	api, err := businessapi.New(
		db,
		compute,
		agentControl,
		env("BUSINESS_DATA_ROOT", "../data"),
		env("BUSINESS_LOG_ROOT", "../logs"),
		env("BUSINESS_CONFIG_ROOT", "../configs"),
		env("BUSINESS_EVAL_ROOT", "../evals"),
	)
	if err != nil {
		return err
	}
	if err := api.Migrate(context.Background()); err != nil {
		return fmt.Errorf("migrate business API: %w", err)
	}

	server := &http.Server{
		Addr:              env("BUSINESS_API_ADDR", ":8080"),
		Handler:           api.Handler(env("FRONTEND_ORIGIN", "http://localhost:5173")),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	go dispatcher.Run(ctx)
	go consumer.Run(ctx)
	serveErr := make(chan error, 1)
	go func() {
		log.Info("Gin business API listening", "addr", server.Addr)
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serveErr <- err
		}
	}()
	select {
	case <-ctx.Done():
	case err := <-serveErr:
		return err
	}
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()
	return server.Shutdown(shutdownCtx)
}

// healthcheck 请求本机健康端点，供容器运行时判断实例是否存活。
func healthcheck() error {
	client := &http.Client{Timeout: 2 * time.Second}
	response, err := client.Get(env("BUSINESS_API_HEALTHCHECK_URL", "http://127.0.0.1:8080/readyz"))
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("readiness returned %s", response.Status)
	}
	return nil
}

// env 读取非空环境变量，否则返回默认值。
func env(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

// envInt 读取整数环境变量，格式非法时安全退回默认值。
func envInt(key string, fallback int) int {
	value, err := strconv.Atoi(strings.TrimSpace(os.Getenv(key)))
	if err != nil || value < 1 {
		return fallback
	}
	return value
}

// envDuration 读取 Go duration 格式的环境变量。
func envDuration(key string, fallback time.Duration) time.Duration {
	value, err := time.ParseDuration(strings.TrimSpace(os.Getenv(key)))
	if err != nil || value <= 0 {
		return fallback
	}
	return value
}
