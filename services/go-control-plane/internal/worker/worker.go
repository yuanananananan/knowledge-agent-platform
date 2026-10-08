// 文件用途：轮询待执行 run，通过并发信号量、超时和分布式锁驱动 Control Plane 处理任务。
package worker

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"

	"knowledge-agent/go-control-plane/internal/domain"
	"knowledge-agent/go-control-plane/internal/store"
)

// Processor 抽象单个 run 的业务处理器，实际实现为 Service。
type Processor interface {
	ProcessRun(context.Context, string, string) error
}

// Locker 抽象跨实例分布式锁。
type Locker interface {
	WithLock(context.Context, string, func(context.Context) error) error
}

// Worker 保存任务 Store、处理器、轮询和并发控制配置。
type Worker struct {
	store       store.Store
	processor   Processor
	id          string
	interval    time.Duration
	runTimeout  time.Duration
	concurrency int
	log         *slog.Logger
	locker      Locker
}

// Option 使用函数式选项配置 Worker。
type Option func(*Worker)

// WithLocker 为 Worker 增加 Redis 分布式锁，避免多实例重复执行同一 run。
func WithLocker(locker Locker) Option { return func(worker *Worker) { worker.locker = locker } }

// New 创建轮询型 Worker。
func New(s store.Store, p Processor, id string, concurrency int, interval, runTimeout time.Duration, log *slog.Logger, options ...Option) *Worker {
	worker := &Worker{store: s, processor: p, id: id, concurrency: concurrency, interval: interval, runTimeout: runTimeout, log: log}
	for _, option := range options {
		option(worker)
	}
	return worker
}

// Run 按固定间隔执行一轮拉取，直到父 Context 取消。
func (w *Worker) Run(ctx context.Context) {
	ticker := time.NewTicker(w.interval)
	defer ticker.Stop()
	for {
		w.once(ctx)
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// once 拉取有限数量候选 run，并用信号量限制同时执行数量。
func (w *Worker) once(ctx context.Context) {
	ids, err := w.store.ListPendingRunIDs(ctx, w.concurrency*4)
	if err != nil {
		w.log.Error("list pending runs", "error", err)
		return
	}
	semaphore := make(chan struct{}, w.concurrency)
	var wg sync.WaitGroup
	for _, id := range ids {
		select {
		case semaphore <- struct{}{}:
		case <-ctx.Done():
			wg.Wait()
			return
		}
		wg.Add(1)
		go func(runID string) {
			defer wg.Done()
			defer func() { <-semaphore }()
			runCtx, cancel := context.WithTimeout(ctx, w.runTimeout)
			defer cancel()
			process := func(processCtx context.Context) error {
				return w.processor.ProcessRun(processCtx, runID, w.id)
			}
			var err error
			if w.locker != nil {
				err = w.locker.WithLock(runCtx, "run:"+runID, process)
			} else {
				err = process(runCtx)
			}
			if err != nil && !errors.Is(err, domain.ErrConflict) && !errors.Is(err, domain.ErrInvalidState) && !errors.Is(err, domain.ErrLockBusy) && !errors.Is(err, domain.ErrInProgress) {
				w.log.Error("process run", "run_id", runID, "error", err)
			}
		}(id)
	}
	wg.Wait()
}
