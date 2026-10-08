// 文件用途：轮询事务 Outbox，将数据库事件可靠发布到 Kafka，并对失败发布执行退避重试。
package outbox

import (
	"context"
	"errors"
	"log/slog"
	"time"

	"knowledge-agent/go-control-plane/internal/domain"
	"knowledge-agent/go-control-plane/internal/store"
)

// Dispatcher 保存 Outbox Store、发布器、Worker 身份和日志器。
type Dispatcher struct {
	store     *store.MySQL
	publisher Publisher
	worker    string
	log       *slog.Logger
}

// Publisher 抽象事件发布目标，生产环境实现为 Kafka。
type Publisher interface {
	Publish(context.Context, int64, string, string, string) error
}

// NewDispatcher 创建事务 Outbox 后台分发器。
func NewDispatcher(s *store.MySQL, p Publisher, worker string, log *slog.Logger) *Dispatcher {
	return &Dispatcher{store: s, publisher: p, worker: worker, log: log}
}

// Run 每 250ms 认领一条消息；成功确认，失败按指数退避重新排队。
func (d *Dispatcher) Run(ctx context.Context) {
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	for {
		msg, err := d.store.ClaimOutbox(ctx, d.worker, 10*time.Second)
		if err == nil {
			if err = d.publisher.Publish(ctx, msg.ID, msg.AggregateID, msg.EventType, string(msg.Payload)); err == nil {
				_ = d.store.MarkOutboxPublished(ctx, msg.ID)
			} else {
				delay := time.Duration(1<<min(msg.Attempts, 6)) * time.Second
				_ = d.store.RetryOutbox(ctx, msg.ID, delay)
				d.log.Warn("outbox publish failed", "id", msg.ID, "error", err)
			}
		} else if !errors.Is(err, domain.ErrNotFound) {
			d.log.Error("claim outbox", "error", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}
