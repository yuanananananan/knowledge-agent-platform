// 文件用途：基于 Redis SET NX PX 和 Lua 脚本实现可续租、可安全释放的分布式锁。
package distlock

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"strconv"
	"sync/atomic"
	"time"

	"knowledge-agent/go-control-plane/internal/domain"
)

// ErrBusy 表示锁已被其他执行者持有；ErrLost 表示执行期间失去了锁所有权。
var (
	ErrBusy = domain.ErrLockBusy
	ErrLost = errors.New("distributed lock was lost")
)

// renewScript 只有在随机令牌仍匹配时才延长 TTL，避免误续租他人的锁。
const renewScript = `if redis.call("GET", KEYS[1]) == ARGV[1] then return redis.call("PEXPIRE", KEYS[1], ARGV[2]) else return 0 end`

// releaseScript 只有锁仍属于当前持有者时才删除键。
const releaseScript = `if redis.call("GET", KEYS[1]) == ARGV[1] then return redis.call("DEL", KEYS[1]) else return 0 end`

// Manager 管理 Redis 锁的命名空间、TTL 和底层命令客户端。
type Manager struct {
	client commander
	prefix string
	ttl    time.Duration
}

// commander 是分布式锁需要的最小 Redis 命令接口，便于测试替换。
type commander interface {
	Do(context.Context, ...string) (any, error)
}

// New 校验客户端和最小 TTL 后创建锁管理器。
func New(client commander, prefix string, ttl time.Duration) (*Manager, error) {
	if client == nil {
		return nil, errors.New("redis client is required")
	}
	if ttl < 3*time.Second {
		return nil, errors.New("distributed lock TTL must be at least 3 seconds")
	}
	return &Manager{client: client, prefix: prefix, ttl: ttl}, nil
}

// WithLock 获取资源锁、后台续租、执行回调，并使用令牌安全释放锁。
func (m *Manager) WithLock(ctx context.Context, resource string, fn func(context.Context) error) error {
	token, err := randomToken()
	if err != nil {
		return err
	}
	key := m.prefix + resource
	value, err := m.client.Do(ctx, "SET", key, token, "NX", "PX", strconv.FormatInt(m.ttl.Milliseconds(), 10))
	if err != nil {
		return fmt.Errorf("acquire redis lock: %w", err)
	}
	if value == nil {
		return ErrBusy
	}
	if value != "OK" {
		return fmt.Errorf("unexpected redis SET response %v", value)
	}

	lockCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	var lost atomic.Bool
	done := make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.NewTicker(m.ttl / 3)
		defer ticker.Stop()
		for {
			select {
			case <-lockCtx.Done():
				return
			case <-ticker.C:
				result, renewErr := m.client.Do(lockCtx, "EVAL", renewScript, "1", key, token, strconv.FormatInt(m.ttl.Milliseconds(), 10))
				if renewErr != nil || result != int64(1) {
					if lockCtx.Err() != nil {
						return
					}
					lost.Store(true)
					cancel()
					return
				}
			}
		}
	}()

	operationErr := fn(lockCtx)
	cancel()
	<-done
	releaseCtx, releaseCancel := context.WithTimeout(context.Background(), 2*time.Second)
	releaseResult, releaseErr := m.client.Do(releaseCtx, "EVAL", releaseScript, "1", key, token)
	releaseCancel()
	if lost.Load() {
		return errors.Join(operationErr, ErrLost)
	}
	if releaseErr != nil {
		return errors.Join(operationErr, fmt.Errorf("release redis lock: %w", releaseErr))
	}
	if releaseResult != int64(1) {
		return errors.Join(operationErr, ErrLost)
	}
	return operationErr
}

// randomToken 生成用于证明锁所有权的随机令牌。
func randomToken() (string, error) {
	data := make([]byte, 16)
	if _, err := rand.Read(data); err != nil {
		return "", err
	}
	return hex.EncodeToString(data), nil
}
