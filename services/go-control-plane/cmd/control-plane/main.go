// 文件用途：启动 Control Plane，装配 MySQL、Redis 锁、Kafka Outbox、Python gRPC、Worker 和 HTTP 服务。
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"syscall"
	"time"

	"knowledge-agent/go-control-plane/internal/config"
	"knowledge-agent/go-control-plane/internal/distlock"
	"knowledge-agent/go-control-plane/internal/httpapi"
	"knowledge-agent/go-control-plane/internal/outbox"
	"knowledge-agent/go-control-plane/internal/python"
	"knowledge-agent/go-control-plane/internal/redisclient"
	"knowledge-agent/go-control-plane/internal/service"
	"knowledge-agent/go-control-plane/internal/store"
	"knowledge-agent/go-control-plane/internal/telemetry"
	"knowledge-agent/go-control-plane/internal/worker"
)

// main 根据命令行参数执行健康检查或启动 Control Plane。
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
		log.Error("control plane stopped", "error", err)
		os.Exit(1)
	}
}

// healthcheck 请求本机健康端点，供容器编排系统探测服务。
func healthcheck() error {
	url := os.Getenv("CONTROL_PLANE_HEALTHCHECK_URL")
	if url == "" {
		url = "http://127.0.0.1:8080/readyz"
	}
	client := &http.Client{Timeout: 2 * time.Second}
	response, err := client.Get(url)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("readiness returned %s", response.Status)
	}
	return nil
}

// run 加载配置和迁移，连接全部依赖，启动 Worker、Outbox Dispatcher 与 HTTP Server。
func run(log *slog.Logger) error {
	cfg, err := config.Load()
	if err != nil {
		return fmt.Errorf("load configuration: %w", err)
	}
	st, err := store.OpenMySQL(cfg.DatabaseDSN)
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}
	defer st.Close()
	migrations, err := loadMigrations(cfg.MigrationPath)
	if err != nil {
		return fmt.Errorf("read migrations: %w", err)
	}
	migrationCtx, cancelMigration := context.WithTimeout(context.Background(), 2*time.Minute)
	err = st.MigrateAll(migrationCtx, migrations)
	cancelMigration()
	if err != nil {
		return fmt.Errorf("apply migration: %w", err)
	}

	var redisClient *redisclient.Client
	var redisLocker *distlock.Manager
	// A short renewable DB lease lets another worker recover a crashed run
	// promptly; it is independent from the overall execution timeout.
	serviceOptions := []service.Option{service.WithLeaseDuration(cfg.RunLeaseDuration)}
	if cfg.RedisAddress != "" {
		redisClient, err = redisclient.New(cfg.RedisAddress, 3*time.Second)
		if err != nil {
			return fmt.Errorf("configure redis: %w", err)
		}
		defer redisClient.Close()
		pingCtx, pingCancel := context.WithTimeout(context.Background(), 3*time.Second)
		err = redisClient.Ping(pingCtx)
		pingCancel()
		if err != nil {
			return fmt.Errorf("connect redis: %w", err)
		}
		redisLocker, err = distlock.New(redisClient, cfg.RedisLockPrefix, cfg.RedisLockTTL)
		if err != nil {
			return fmt.Errorf("configure redis lock: %w", err)
		}
		serviceOptions = append(serviceOptions, service.WithReadyCheck(redisClient.Ping))
	}
	kafkaPublisher, err := outbox.NewKafka(cfg.KafkaBrokers, cfg.KafkaTopic)
	if err != nil {
		return fmt.Errorf("configure Kafka publisher: %w", err)
	}
	defer kafkaPublisher.Close()
	kafkaCtx, kafkaCancel := context.WithTimeout(context.Background(), 5*time.Second)
	err = kafkaPublisher.Ping(kafkaCtx)
	kafkaCancel()
	if err != nil {
		return fmt.Errorf("connect Kafka publisher: %w", err)
	}
	serviceOptions = append(serviceOptions, service.WithReadyCheck(kafkaPublisher.Ping))

	py, err := python.New(cfg.PythonGRPCTarget, cfg.InternalToken, cfg.PythonTimeout)
	if err != nil {
		return fmt.Errorf("configure Python gRPC client: %w", err)
	}
	defer py.Close()
	svc := service.New(st, py, serviceOptions...)
	metrics := &telemetry.Metrics{}
	api := httpapi.New(svc, cfg.AuthSecret, log, metrics)
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	workerID := hostname() + "-" + fmt.Sprint(os.Getpid())
	workerOptions := make([]worker.Option, 0, 1)
	if redisLocker != nil {
		workerOptions = append(workerOptions, worker.WithLocker(redisLocker))
	}
	w := worker.New(st, svc, workerID, cfg.WorkerConcurrency, cfg.WorkerPollInterval, cfg.WorkerRunTimeout, log, workerOptions...)
	var background sync.WaitGroup
	background.Add(1)
	go func() {
		defer background.Done()
		w.Run(ctx)
	}()
	dispatcher := outbox.NewDispatcher(st, kafkaPublisher, workerID, log)
	background.Add(1)
	go func() {
		defer background.Done()
		dispatcher.Run(ctx)
	}()

	srv := &http.Server{
		Addr:              cfg.Address,
		Handler:           api.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}
	serveErr := make(chan error, 1)
	go func() {
		log.Info("control plane listening", "addr", srv.Addr, "worker_concurrency", cfg.WorkerConcurrency, "redis_enabled", cfg.RedisAddress != "", "kafka_topic", cfg.KafkaTopic)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serveErr <- err
		}
	}()

	var stoppedBy error
	select {
	case <-ctx.Done():
	case stoppedBy = <-serveErr:
		stop()
	}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		return fmt.Errorf("shutdown HTTP server: %w", err)
	}
	backgroundDone := make(chan struct{})
	go func() {
		background.Wait()
		close(backgroundDone)
	}()
	select {
	case <-backgroundDone:
	case <-shutdownCtx.Done():
		return fmt.Errorf("shutdown background workers: %w", shutdownCtx.Err())
	}
	if stoppedBy != nil {
		return fmt.Errorf("serve HTTP: %w", stoppedBy)
	}
	log.Info("control plane stopped gracefully")
	return nil
}

// hostname 返回当前主机名，用于构造可追踪的 Worker ID。
func hostname() string {
	host, err := os.Hostname()
	if err != nil || host == "" {
		return "control-plane"
	}
	return host
}

// loadMigrations 从单个 SQL 文件或目录加载并按文件名排序迁移。
func loadMigrations(path string) ([]store.Migration, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if !info.IsDir() {
		content, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		return []store.Migration{{Version: filepath.Base(path), SQL: string(content)}}, nil
	}
	entries, err := os.ReadDir(path)
	if err != nil {
		return nil, err
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
	migrations := make([]store.Migration, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".sql") {
			continue
		}
		content, err := os.ReadFile(filepath.Join(path, entry.Name()))
		if err != nil {
			return nil, err
		}
		migrations = append(migrations, store.Migration{Version: entry.Name(), SQL: string(content)})
	}
	if len(migrations) == 0 {
		return nil, fmt.Errorf("no .sql migrations found in %s", path)
	}
	return migrations, nil
}
