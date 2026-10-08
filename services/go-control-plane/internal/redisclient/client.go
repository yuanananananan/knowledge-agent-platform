// 文件用途：封装 go-redis 客户端，为分布式锁提供受限命令接口和连接健康检查。
package redisclient

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

// Client is the project's small adapter around go-redis. The underlying
// client provides connection pooling, protocol handling, authentication,
// logical databases, TLS, deadlines, and automatic connection recovery.
// Client 包装 go-redis 通用客户端。
type Client struct {
	client *redis.Client
}

// New 解析 Redis URL，并统一配置连接、读写和连接池超时。
func New(rawURL string, timeout time.Duration) (*Client, error) {
	if !strings.Contains(rawURL, "://") {
		rawURL = "redis://" + rawURL
	}
	options, err := redis.ParseURL(rawURL)
	if err != nil {
		return nil, fmt.Errorf("parse redis URL: %w", err)
	}
	if timeout <= 0 {
		timeout = 3 * time.Second
	}
	options.DialTimeout = timeout
	options.ReadTimeout = timeout
	options.WriteTimeout = timeout
	options.PoolTimeout = timeout
	options.MinIdleConns = 2
	return &Client{client: redis.NewClient(options)}, nil
}

// Do 执行分布式锁所需的 Redis 命令，并规范返回值类型。
func (c *Client) Do(ctx context.Context, args ...string) (any, error) {
	command := make([]any, len(args))
	for index, argument := range args {
		command[index] = argument
	}
	result, err := c.client.Do(ctx, command...).Result()
	if errors.Is(err, redis.Nil) {
		return nil, nil
	}
	return result, err
}

// Ping 检查 Redis 服务是否可用。
func (c *Client) Ping(ctx context.Context) error {
	return c.client.Ping(ctx).Err()
}

// Close 关闭 Redis 客户端和连接池。
func (c *Client) Close() error { return c.client.Close() }
