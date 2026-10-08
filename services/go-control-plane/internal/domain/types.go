// 文件用途：定义 Control Plane 的运行状态机、事件、动作实体和统一领域错误。
package domain

import (
	"encoding/json"
	"errors"
	"time"
)

// RunStatus 表示 Agent run 状态机中的一个状态。
type RunStatus string

// Run 状态按照 pending → collecting_evidence → reasoning → 终态推进。
const (
	RunPending    RunStatus = "pending"
	RunCollecting RunStatus = "collecting_evidence"
	RunReasoning  RunStatus = "reasoning"
	RunSucceeded  RunStatus = "succeeded"
	RunFailed     RunStatus = "failed"
	RunCancelled  RunStatus = "cancelled"
)

// terminal 保存不会再发生后续状态迁移的终态集合。
var terminal = map[RunStatus]bool{RunSucceeded: true, RunFailed: true, RunCancelled: true}

// Terminal 判断当前状态是否为成功、失败或取消。
func (s RunStatus) Terminal() bool { return terminal[s] }

// CanTransition 校验一次 run 状态迁移是否符合预定义状态机。
func CanTransition(from, to RunStatus) bool {
	allowed := map[RunStatus]map[RunStatus]bool{
		RunPending:    {RunCollecting: true, RunCancelled: true, RunFailed: true},
		RunCollecting: {RunReasoning: true, RunCancelled: true, RunFailed: true},
		RunReasoning:  {RunSucceeded: true, RunCancelled: true, RunFailed: true},
	}
	return allowed[from][to]
}

// Run 是 Control Plane 中可持久化、可租约认领的 Agent 执行任务。
type Run struct {
	ID             string          `json:"run_id"`
	ConversationID string          `json:"conversation_id"`
	UserID         int64           `json:"user_id,omitempty"`
	Message        string          `json:"message,omitempty"`
	Status         RunStatus       `json:"status"`
	Answer         string          `json:"answer,omitempty"`
	Error          string          `json:"error,omitempty"`
	Version        int64           `json:"version"`
	FencingToken   int64           `json:"-"`
	CreatedAt      time.Time       `json:"created_at"`
	UpdatedAt      time.Time       `json:"updated_at"`
	Metadata       json.RawMessage `json:"metadata,omitempty"`
}

// Event 是带单调 sequence 的持久化运行事件。
type Event struct {
	RunID     string          `json:"run_id"`
	Sequence  int64           `json:"sequence"`
	Type      string          `json:"type"`
	Payload   json.RawMessage `json:"payload"`
	CreatedAt time.Time       `json:"created_at"`
}

// ActionStatus 表示高风险动作从待审批到执行终态的状态。
type ActionStatus string

// Action 的状态包含待审批、执行中、成功、失败和拒绝。
const (
	ActionPending   ActionStatus = "pending_approval"
	ActionExecuting ActionStatus = "executing"
	ActionSucceeded ActionStatus = "succeeded"
	ActionFailed    ActionStatus = "failed"
	ActionRejected  ActionStatus = "rejected"
)

// Action 描述经过验证、需要人工审批后才能执行的业务写操作。
type Action struct {
	ID             string          `json:"proposal_id"`
	UserID         int64           `json:"user_id,omitempty"`
	ConversationID string          `json:"conversation_id,omitempty"`
	Type           string          `json:"action_type"`
	Payload        json.RawMessage `json:"payload"`
	Reason         string          `json:"reason,omitempty"`
	Impact         string          `json:"impact"`
	IdempotencyKey string          `json:"idempotency_key,omitempty"`
	Status         ActionStatus    `json:"status"`
	Result         json.RawMessage `json:"result,omitempty"`
	Error          string          `json:"error,omitempty"`
	ExpiresAt      time.Time       `json:"expires_at"`
	CreatedAt      time.Time       `json:"created_at"`
	UpdatedAt      time.Time       `json:"updated_at"`
}

// 统一领域错误用于跨 HTTP、Store 和 gRPC 层稳定映射错误语义。
var (
	ErrNotFound     = errors.New("not found")
	ErrConflict     = errors.New("conflict")
	ErrInvalidState = errors.New("invalid state transition")
	ErrInvalidInput = errors.New("invalid input")
	ErrDependency   = errors.New("dependency unavailable")
	ErrLockBusy     = errors.New("distributed lock is held")
	ErrInProgress   = errors.New("remote operation is already in progress")
)
