// 文件用途：从环境变量加载并校验 Control Plane 的数据库、gRPC、Redis、Kafka 和 Worker 配置。
package config

import (
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"
	"time"
)

// Config 汇总 Control Plane 启动所需的全部外部配置。
type Config struct {
	Address            string
	DatabaseDSN        string
	AuthSecret         string
	InternalToken      string
	MigrationPath      string
	PythonGRPCTarget   string
	PythonTimeout      time.Duration
	RedisAddress       string
	RedisLockPrefix    string
	RedisLockTTL       time.Duration
	KafkaBrokers       []string
	KafkaTopic         string
	WorkerConcurrency  int
	WorkerPollInterval time.Duration
	RunLeaseDuration   time.Duration
	WorkerRunTimeout   time.Duration
	ShutdownTimeout    time.Duration
}

// Load 读取环境变量、应用默认值并执行整体配置校验。
func Load() (Config, error) {
	cfg := Config{
		Address:            env("CONTROL_PLANE_ADDR", ":8080"),
		DatabaseDSN:        strings.TrimSpace(os.Getenv("CONTROL_PLANE_DATABASE_DSN")),
		AuthSecret:         os.Getenv("CONTROL_PLANE_AUTH_SECRET"),
		InternalToken:      os.Getenv("CONTROL_PLANE_INTERNAL_TOKEN"),
		MigrationPath:      env("CONTROL_PLANE_MIGRATION", "migrations"),
		PythonGRPCTarget:   env("PYTHON_AGENT_GRPC_TARGET", "localhost:50051"),
		RedisAddress:       redisAddress(),
		RedisLockPrefix:    env("REDIS_LOCK_PREFIX", "knowledge-agent:lock:"),
		RedisLockTTL:       30 * time.Second,
		KafkaBrokers:       csvEnv("KAFKA_BROKERS"),
		KafkaTopic:         env("KAFKA_TOPIC", "agent-events"),
		WorkerConcurrency:  4,
		PythonTimeout:      90 * time.Second,
		WorkerPollInterval: 500 * time.Millisecond,
		RunLeaseDuration:   90 * time.Second,
		WorkerRunTimeout:   10 * time.Minute,
		ShutdownTimeout:    10 * time.Second,
	}
	var err error
	if cfg.PythonTimeout, err = duration("CONTROL_PLANE_PYTHON_TIMEOUT", cfg.PythonTimeout); err != nil {
		return Config{}, err
	}
	if cfg.WorkerPollInterval, err = duration("CONTROL_PLANE_WORKER_POLL_INTERVAL", cfg.WorkerPollInterval); err != nil {
		return Config{}, err
	}
	if cfg.RunLeaseDuration, err = duration("CONTROL_PLANE_RUN_LEASE", cfg.RunLeaseDuration); err != nil {
		return Config{}, err
	}
	if cfg.WorkerRunTimeout, err = duration("CONTROL_PLANE_WORKER_RUN_TIMEOUT", cfg.WorkerRunTimeout); err != nil {
		return Config{}, err
	}
	if cfg.ShutdownTimeout, err = duration("CONTROL_PLANE_SHUTDOWN_TIMEOUT", cfg.ShutdownTimeout); err != nil {
		return Config{}, err
	}
	if cfg.RedisLockTTL, err = duration("REDIS_LOCK_TTL", cfg.RedisLockTTL); err != nil {
		return Config{}, err
	}
	if raw := os.Getenv("CONTROL_PLANE_WORKER_CONCURRENCY"); raw != "" {
		cfg.WorkerConcurrency, err = strconv.Atoi(raw)
		if err != nil {
			return Config{}, fmt.Errorf("CONTROL_PLANE_WORKER_CONCURRENCY: %w", err)
		}
	}
	if err := cfg.validate(); err != nil {
		return Config{}, err
	}
	return cfg, nil
}

// validate 检查必填项、地址格式、密钥强度、并发范围和超时关系。
func (c Config) validate() error {
	if c.DatabaseDSN == "" {
		return fmt.Errorf("CONTROL_PLANE_DATABASE_DSN is required")
	}
	if len(c.AuthSecret) < 16 {
		return fmt.Errorf("CONTROL_PLANE_AUTH_SECRET must contain at least 16 characters")
	}
	if len(c.InternalToken) < 16 {
		return fmt.Errorf("CONTROL_PLANE_INTERNAL_TOKEN must contain at least 16 characters")
	}
	if c.RedisAddress == "" {
		return fmt.Errorf("REDIS_URL is required for distributed locking")
	}
	if len(c.KafkaBrokers) == 0 {
		return fmt.Errorf("KAFKA_BROKERS is required for outbox delivery")
	}
	if c.KafkaTopic == "" {
		return fmt.Errorf("KAFKA_TOPIC is required for outbox delivery")
	}
	if _, _, err := net.SplitHostPort(c.PythonGRPCTarget); err != nil {
		return fmt.Errorf("PYTHON_AGENT_GRPC_TARGET must be host:port: %w", err)
	}
	if c.WorkerConcurrency < 1 || c.WorkerConcurrency > 128 {
		return fmt.Errorf("CONTROL_PLANE_WORKER_CONCURRENCY must be between 1 and 128")
	}
	if c.PythonTimeout <= 0 || c.WorkerPollInterval <= 0 || c.RunLeaseDuration <= 0 || c.WorkerRunTimeout <= 0 || c.ShutdownTimeout <= 0 {
		return fmt.Errorf("configured durations must be positive")
	}
	if c.RunLeaseDuration < 3*time.Second {
		return fmt.Errorf("CONTROL_PLANE_RUN_LEASE must be at least 3s")
	}
	if c.WorkerRunTimeout <= c.RunLeaseDuration {
		return fmt.Errorf("CONTROL_PLANE_WORKER_RUN_TIMEOUT must exceed CONTROL_PLANE_RUN_LEASE")
	}
	if c.WorkerRunTimeout <= c.PythonTimeout {
		return fmt.Errorf("CONTROL_PLANE_WORKER_RUN_TIMEOUT must exceed CONTROL_PLANE_PYTHON_TIMEOUT")
	}
	if c.RedisLockTTL < 3*time.Second {
		return fmt.Errorf("REDIS_LOCK_TTL must be at least 3s")
	}
	return nil
}

// redisAddress 优先读取 REDIS_URL，并兼容旧的 REDIS_ADDR 配置。
func redisAddress() string {
	if value := strings.TrimSpace(os.Getenv("REDIS_URL")); value != "" {
		return value
	}
	return strings.TrimSpace(os.Getenv("REDIS_ADDR"))
}

// env 读取去除空白后的环境变量，否则使用默认值。
func env(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

// csvEnv 将逗号分隔的环境变量解析为非空字符串列表。
func csvEnv(key string) []string {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return nil
	}
	values := make([]string, 0)
	for _, value := range strings.Split(raw, ",") {
		if value = strings.TrimSpace(value); value != "" {
			values = append(values, value)
		}
	}
	return values
}

// duration 读取 Go duration 格式配置，并在格式错误时返回带字段名的错误。
func duration(key string, fallback time.Duration) (time.Duration, error) {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return fallback, nil
	}
	value, err := time.ParseDuration(raw)
	if err != nil {
		return 0, fmt.Errorf("%s: %w", key, err)
	}
	return value, nil
}
