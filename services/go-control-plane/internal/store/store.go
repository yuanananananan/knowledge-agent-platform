// 文件用途：定义 Control Plane 持久化层接口以及数据库迁移数据结构。
package store

import (
	"context"
	"encoding/json"
	"time"

	"knowledge-agent/go-control-plane/internal/domain"
)

// Store 定义 Service 所依赖的完整持久化契约，生产实现为 MySQL。
type Store interface {
	// Run 生命周期、租约、状态迁移和事件读取。
	CreateRun(context.Context, domain.Run) error
	ListRuns(context.Context, int64, int, int) ([]domain.Run, error)
	ListRunsByConversation(context.Context, int64, string, int, int) ([]domain.Run, error)
	ListPendingRunIDs(context.Context, int) ([]string, error)
	GetRun(context.Context, int64, string) (domain.Run, error)
	ClaimRun(context.Context, string, string, time.Duration) (domain.Run, error)
	TransitionRun(context.Context, string, string, int64, domain.RunStatus, domain.RunStatus, string, string, json.RawMessage, string, json.RawMessage) error
	AppendRunEvent(context.Context, string, string, int64, string, json.RawMessage, time.Duration) error
	RenewRun(context.Context, string, string, int64, time.Duration) error
	CancelRun(context.Context, int64, string) error
	ListEvents(context.Context, int64, string, int64, int) ([]domain.Event, error)
	// Action 提案的幂等创建、认领和终态提交。
	CreateAction(context.Context, domain.Action) (domain.Action, bool, error)
	ListActions(context.Context, int64, int, int) ([]domain.Action, error)
	GetAction(context.Context, int64, string) (domain.Action, error)
	ClaimAction(context.Context, int64, string) (domain.Action, error)
	FinishAction(context.Context, string, domain.ActionStatus, json.RawMessage, string) error
	RejectAction(context.Context, int64, string) error
	// 连接生命周期和就绪检查。
	Ping(context.Context) error
	Close() error
}
