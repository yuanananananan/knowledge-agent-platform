// 文件用途：实现 Control Plane 核心业务，包括 run 状态机、租约恢复、流事件持久化和动作审批。
package service

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"knowledge-agent/go-control-plane/internal/domain"
	"knowledge-agent/go-control-plane/internal/python"
	"knowledge-agent/go-control-plane/internal/store"
)

// Python 定义 Service 执行推理、流式事件、动作和健康检查所需的下游能力。
type Python interface {
	// Run 通过 Python Agent 执行一次非流式推理，并返回最终答案。
	Run(context.Context, python.RunRequest) (python.RunResponse, error)
	// RunStream 通过 server-streaming gRPC 执行 Agent；每收到一个事件就调用 handle。
	RunStream(context.Context, python.RunRequest, func(python.RunEvent) error) error
	// ExecuteAction 执行已经通过人工审批的写操作。
	ExecuteAction(context.Context, python.ActionRequest) (python.ActionResponse, error)
	// Ready 检查 Python Agent 服务当前是否可用。
	Ready(context.Context) error
}

const (
	// modelDeltaBatchMaxBytes 限制单个模型增量事件的最大合并大小，避免事件过大。
	modelDeltaBatchMaxBytes = 512
	// modelDeltaBatchMaxWait 控制连续 token 在收到后续 chunk 时允许等待的最长批处理窗口。
	modelDeltaBatchMaxWait = 75 * time.Millisecond
)

// modelDeltaBatch 将连续的小块 model.delta 合并，减少每个 token 单独提交 MySQL 的压力。
type modelDeltaBatch struct {
	content   strings.Builder
	startedAt time.Time
}

// Add 追加一块模型文本，并返回当前批次是否已经达到落库条件。
func (b *modelDeltaBatch) Add(now time.Time, content string) bool {
	if content == "" {
		return false
	}
	if b.content.Len() == 0 {
		b.startedAt = now
	}
	b.content.WriteString(content)
	return b.content.Len() >= modelDeltaBatchMaxBytes || now.Sub(b.startedAt) >= modelDeltaBatchMaxWait
}

// Take 取出当前批次并重置缓冲区；没有内容时返回空字符串。
func (b *modelDeltaBatch) Take() string {
	if b.content.Len() == 0 {
		return ""
	}
	content := b.content.String()
	b.content.Reset()
	b.startedAt = time.Time{}
	return content
}

// Service 是 Control Plane 的核心业务层，负责运行状态机、租约、取消和动作审批。
type Service struct {
	// store 持久化 run、事件和 action，是业务状态的事实来源。
	store store.Store
	// python 是访问 Python Agent 私有 gRPC 服务的客户端抽象。
	python Python
	// now 可在测试中替换，从而稳定地验证时间相关逻辑。
	now func() time.Time
	// leaseDuration 是 Worker 对一个 run 的租约有效期。
	leaseDuration time.Duration
	// readyChecks 保存 Redis、Kafka 等额外依赖的就绪检查函数。
	readyChecks []func(context.Context) error
	// mu 保护 activeCancels，避免取消和执行协程并发读写 map。
	mu sync.Mutex
	// activeCancels 保存当前进程正在执行的 run 及其取消函数。
	activeCancels map[string]context.CancelFunc
}

// Option 使用函数式选项修改 Service 的可选配置。
type Option func(*Service)

// WithLeaseDuration 配置运行租约时长；非正数会被忽略并保留默认值。
func WithLeaseDuration(duration time.Duration) Option {
	return func(s *Service) {
		if duration > 0 {
			s.leaseDuration = duration
		}
	}
}

// WithReadyCheck 为 Service 增加一个外部依赖就绪检查。
func WithReadyCheck(check func(context.Context) error) Option {
	return func(s *Service) {
		if check != nil {
			s.readyChecks = append(s.readyChecks, check)
		}
	}
}

// New 创建 Service，并设置默认的三分钟租约和进程内取消表。
func New(s store.Store, p Python, options ...Option) *Service {
	service := &Service{store: s, python: p, now: func() time.Time { return time.Now().UTC() }, leaseDuration: 3 * time.Minute, activeCancels: make(map[string]context.CancelFunc)}
	// 依次应用调用方提供的配置选项。
	for _, option := range options {
		option(service)
	}
	return service
}

// id 生成带业务前缀的随机公开 ID，例如 run_xxx、conv_xxx 和 act_xxx。
func id(prefix string) string {
	// 16 个随机字节编码后得到 32 位十六进制字符串。
	b := make([]byte, 16)
	_, _ = rand.Read(b)
	return prefix + hex.EncodeToString(b)
}

// payload 将任意 Go 值编码成用于数据库事件的 JSON；传入值应当可序列化。
func payload(v any) json.RawMessage { b, _ := json.Marshal(v); return b }

// CreateRun 校验请求、生成运行标识，并创建初始状态为 pending 的持久化任务。
func (s *Service) CreateRun(ctx context.Context, uid int64, conversation, message string) (domain.Run, error) {
	// 统一清理消息两端空白，防止纯空白问题进入执行队列。
	message = strings.TrimSpace(message)
	// 限制用户、消息和会话 ID，保护后端和数据库免受异常输入影响。
	if uid <= 0 || message == "" || len(message) > 20000 || len(conversation) > 64 {
		return domain.Run{}, fmt.Errorf("%w: invalid run request", domain.ErrInvalidInput)
	}
	// 首次对话没有 conversation_id 时，由 Control Plane 生成稳定会话 ID。
	if conversation == "" {
		conversation = id("conv_")
	}
	now := s.now()
	// 创建 run 只负责持久化排队，不在当前 HTTP 请求中直接执行模型。
	r := domain.Run{ID: id("run_"), ConversationID: conversation, UserID: uid, Message: message, Status: domain.RunPending, CreatedAt: now, UpdatedAt: now}
	// Store 会在事务中写入 run.created 事件，并阻止同一会话存在多个活跃 run。
	if err := s.store.CreateRun(ctx, r); err != nil {
		return domain.Run{}, err
	}
	return r, nil
}

// ListRuns 分页查询当前用户的运行记录；提供 conversation 时只查询该会话。
func (s *Service) ListRuns(ctx context.Context, uid int64, conversation string, limit, offset int) ([]domain.Run, error) {
	// 统一校验并规范分页参数。
	limit, offset, err := page(limit, offset)
	if err != nil {
		return nil, err
	}
	if conversation != "" {
		// 会话维度查询用于恢复页面中最近一次 run。
		return s.store.ListRunsByConversation(ctx, uid, conversation, limit, offset)
	}
	return s.store.ListRuns(ctx, uid, limit, offset)
}

// GetRun 按用户和 run ID 查询运行，用户条件用于阻止越权读取。
func (s *Service) GetRun(ctx context.Context, uid int64, id string) (domain.Run, error) {
	return s.store.GetRun(ctx, uid, id)
}

// CancelRun 将非终态 run 持久化为 cancelled，并取消本进程中正在执行的 gRPC。
func (s *Service) CancelRun(ctx context.Context, uid int64, id string) (domain.Run, error) {
	// 先提交数据库状态，确保取消事实不会只停留在当前进程内存中。
	if err := s.store.CancelRun(ctx, uid, id); err != nil {
		return domain.Run{}, err
	}
	// 查找本进程内对应的执行 Context；任务也可能由其他 Control Plane 实例持有。
	s.mu.Lock()
	cancel := s.activeCancels[id]
	s.mu.Unlock()
	if cancel != nil {
		// Context 取消会沿 RunStream 传到 gRPC 和 Python generator。
		cancel()
	}
	// 返回数据库中的最终取消状态。
	return s.store.GetRun(ctx, uid, id)
}

// Events 返回 sequence 大于 after 的事件，每次最多读取 100 条。
func (s *Service) Events(ctx context.Context, uid int64, id string, after int64) ([]domain.Event, error) {
	return s.store.ListEvents(ctx, uid, id, after, 100)
}

// ProcessRun 由后台 Worker 调用，负责认领并完整执行一个 Agent run。
func (s *Service) ProcessRun(ctx context.Context, runID, worker string) error {
	// 原子认领 run：写入 worker、设置租约并增加 fencing token。
	r, err := s.store.ClaimRun(ctx, runID, worker, s.leaseDuration)
	if err != nil {
		return err
	}
	// 首次领取时把状态从 pending 推进到 collecting_evidence，并记录状态事件。
	if r.Status == domain.RunPending {
		if err = s.move(ctx, r.ID, worker, r.FencingToken, domain.RunPending, domain.RunCollecting, "run.collecting", nil, "", nil); err != nil {
			return err
		}
		r.Status = domain.RunCollecting
	}
	// 证据收集阶段完成后进入 reasoning，之后才允许接收 Python 流事件。
	if r.Status == domain.RunCollecting {
		if err = s.move(ctx, r.ID, worker, r.FencingToken, domain.RunCollecting, domain.RunReasoning, "run.reasoning", nil, "", nil); err != nil {
			return err
		}
	}
	// 创建本次执行专用的可取消 Context。
	runCtx, cancel := context.WithCancel(ctx)
	// 保存取消函数，使 HTTP 取消接口能够终止当前进程中的执行。
	s.mu.Lock()
	s.activeCancels[r.ID] = cancel
	s.mu.Unlock()
	defer func() {
		// 无论成功、失败还是超时，都释放 Context 并清除进程内索引。
		cancel()
		s.mu.Lock()
		delete(s.activeCancels, r.ID)
		s.mu.Unlock()
	}()

	// 独立协程定期续租，防止长时间推理被其他 Worker 当成失效任务重新认领。
	renewDone := make(chan struct{})
	go func() {
		// 每经过租约时长的三分之一续租一次，并保证最短间隔为一秒。
		interval := s.leaseDuration / 3
		if interval < time.Second {
			interval = time.Second
		}
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		for {
			select {
			case <-renewDone:
				// 主执行完成后主动停止续租协程。
				return
			case <-runCtx.Done():
				// 运行被取消或超时后停止续租。
				return
			case <-ticker.C:
				// 续租必须匹配 worker 和 fencing token，旧 Worker 无法延长新租约。
				if err := s.store.RenewRun(runCtx, r.ID, worker, r.FencingToken, s.leaseDuration); err != nil {
					// 一旦失去租约立即取消执行，避免僵尸 Worker 继续产生副作用。
					cancel()
					return
				}
			}
		}
	}()
	// 函数退出时通知续租协程结束。
	defer close(renewDone)

	// terminal 用于确认 Python 事件流是否明确发送了完成或失败事件。
	terminal := false
	// 连续的 model.delta 先在内存中做短暂合并，避免每个 token 都开启一次数据库事务。
	deltaBatch := modelDeltaBatch{}
	flushModelDeltas := func(flushCtx context.Context) error {
		content := deltaBatch.Take()
		if content == "" {
			return nil
		}
		return s.store.AppendRunEvent(flushCtx, r.ID, worker, r.FencingToken, "model.delta", payload(map[string]any{"content": content}), s.leaseDuration)
	}
	// 调用 Python Agent 的 server-streaming gRPC，并同步处理每个结构化事件。
	callErr := s.python.RunStream(runCtx, python.RunRequest{RunID: r.ID, ConversationID: r.ConversationID, UserID: r.UserID, Message: r.Message}, func(event python.RunEvent) error {
		switch event.Type {
		case "model.delta":
			// 只有标准的文本增量参与合并；异常 payload 仍原样持久化，避免信息丢失。
			var body struct {
				Content string `json:"content"`
			}
			if err := json.Unmarshal(event.Payload, &body); err != nil || body.Content == "" {
				if flushErr := flushModelDeltas(runCtx); flushErr != nil {
					return flushErr
				}
				return s.store.AppendRunEvent(runCtx, r.ID, worker, r.FencingToken, event.Type, event.Payload, s.leaseDuration)
			}
			// 达到大小或时间阈值时，把合并后的文本作为一个 model.delta 事件落库。
			if deltaBatch.Add(s.now(), body.Content) {
				return flushModelDeltas(runCtx)
			}
			return nil
		case "run.completed":
			// 终态之前必须先落库剩余文本，保持前端看到的事件顺序完整。
			if err := flushModelDeltas(runCtx); err != nil {
				return err
			}
			// 终态完成事件中携带最终 answer。
			var body struct {
				Answer string `json:"answer"`
			}
			_ = json.Unmarshal(event.Payload, &body)
			terminal = true
			// 状态和 run.completed 事件在同一数据库事务中提交。
			return s.move(runCtx, r.ID, worker, r.FencingToken, domain.RunReasoning, domain.RunSucceeded, "run.completed", map[string]any{"answer": body.Answer}, "", nil)
		case "run.failed":
			// 失败事件前同样刷新已生成的部分答案，便于页面展示失败前的输出。
			if err := flushModelDeltas(runCtx); err != nil {
				return err
			}
			// Python 失败事件可能同时提供错误类型和适合展示的消息。
			var body struct {
				Error   string `json:"error"`
				Message string `json:"message"`
			}
			_ = json.Unmarshal(event.Payload, &body)
			message := body.Message
			if message == "" {
				// 没有友好消息时退回到底层错误名称。
				message = body.Error
			}
			terminal = true
			// 将 run 转为 failed，并原样保存 Python 返回的失败 payload。
			return s.move(runCtx, r.ID, worker, r.FencingToken, domain.RunReasoning, domain.RunFailed, "run.failed", json.RawMessage(event.Payload), message, nil)
		default:
			// 工具或工作流事件前先刷新文本，确保跨事件类型的原始顺序不变。
			if err := flushModelDeltas(runCtx); err != nil {
				return err
			}
			// tool.completed 等结构化事件仍逐条落库，并顺便延长租约。
			return s.store.AppendRunEvent(runCtx, r.ID, worker, r.FencingToken, event.Type, event.Payload, s.leaseDuration)
		}
	})
	// 流在没有终态事件的情况下正常结束时，也先保存最后一个未满批次。
	if callErr == nil {
		callErr = flushModelDeltas(runCtx)
	}
	if callErr != nil {
		if errors.Is(callErr, context.Canceled) {
			// 主动取消或失去租约时，数据库状态由取消方或后续恢复者负责处理。
			return nil
		}
		if errors.Is(callErr, domain.ErrInProgress) {
			// Python 明确报告相同运行仍在执行时，保留可识别的冲突错误。
			return callErr
		}
		// 依赖错误发生后，使用不受原取消信号影响的 Context 尽力记录失败终态。
		_ = s.move(context.WithoutCancel(ctx), r.ID, worker, r.FencingToken, domain.RunReasoning, domain.RunFailed, "run.failed", map[string]any{"error": callErr.Error(), "message": callErr.Error()}, callErr.Error(), nil)
		return callErr
	}
	// 流正常关闭却没有终态事件属于协议异常，不能把 run 永久留在 reasoning。
	if !terminal {
		return s.move(ctx, r.ID, worker, r.FencingToken, domain.RunReasoning, domain.RunFailed, "run.failed", map[string]any{"error": "stream ended without terminal event", "message": "Agent 事件流意外结束"}, "stream ended without terminal event", nil)
	}
	return nil
}

// move 校验并执行一次 run 状态迁移，同时在同一事务中写入对应事件。
func (s *Service) move(ctx context.Context, id, worker string, fence int64, from, to domain.RunStatus, event string, eventPayload any, runErr string, meta json.RawMessage) error {
	answer := ""
	// 完成事件的 answer 同时写入 run 主表，便于非流式接口直接查询最终结果。
	if m, ok := eventPayload.(map[string]any); ok {
		answer, _ = m["answer"].(string)
	}
	// 调用方未提供 payload 时，至少记录迁移后的状态。
	if eventPayload == nil {
		eventPayload = map[string]any{"status": to}
	}
	// Store 会检查旧状态、worker、fencing token 和租约是否仍然有效。
	return s.store.TransitionRun(ctx, id, worker, fence, from, to, answer, runErr, meta, event, payload(eventPayload))
}

// CreateAction 校验并持久化一个待人工审批的高风险动作提案。
func (s *Service) CreateAction(ctx context.Context, uid int64, conversation, typ string, raw json.RawMessage, reason, impact, key string) (domain.Action, bool, error) {
	// 清理文本字段，避免仅包含空格的动作类型或影响说明通过校验。
	typ = strings.TrimSpace(typ)
	reason = strings.TrimSpace(reason)
	impact = strings.TrimSpace(impact)
	var object map[string]any
	// payload 必须是 JSON 对象，同时限制各字段长度。
	if uid <= 0 || typ == "" || len(typ) > 64 || len(conversation) > 64 || len(reason) > 4000 || impact == "" || len(impact) > 4000 || len(key) > 256 || !json.Valid(raw) || json.Unmarshal(raw, &object) != nil || object == nil {
		return domain.Action{}, false, fmt.Errorf("%w: invalid action", domain.ErrInvalidInput)
	}
	if key == "" {
		// 未提供幂等键时，根据用户、动作类型和 payload 生成稳定哈希。
		sum := sha256.Sum256(append([]byte(fmt.Sprintf("%d:%s:", uid, typ)), raw...))
		key = hex.EncodeToString(sum[:])
	} else {
		// 用户提供的幂等键也加入 user ID 后再哈希，避免不同用户互相冲突。
		sum := sha256.Sum256([]byte(fmt.Sprintf("%d:%s", uid, key)))
		key = hex.EncodeToString(sum[:])
	}
	now := s.now()
	// 提案默认等待审批，并在 30 分钟后过期。
	a := domain.Action{ID: id("act_"), UserID: uid, ConversationID: conversation, Type: typ, Payload: raw, Reason: reason, Impact: impact, IdempotencyKey: key, Status: domain.ActionPending, ExpiresAt: now.Add(30 * time.Minute), CreatedAt: now, UpdatedAt: now}
	// Store 根据唯一幂等键决定创建新提案还是返回已有提案。
	return s.store.CreateAction(ctx, a)
}

// ListActions 分页查询当前用户的动作提案。
func (s *Service) ListActions(ctx context.Context, uid int64, limit, offset int) ([]domain.Action, error) {
	limit, offset, err := page(limit, offset)
	if err != nil {
		return nil, err
	}
	return s.store.ListActions(ctx, uid, limit, offset)
}

// GetAction 查询当前用户的单个动作提案。
func (s *Service) GetAction(ctx context.Context, uid int64, id string) (domain.Action, error) {
	return s.store.GetAction(ctx, uid, id)
}

// ApproveAction 原子认领提案，调用 Python 执行动作，再持久化最终结果。
func (s *Service) ApproveAction(ctx context.Context, uid int64, id string) (domain.Action, error) {
	// ClaimAction 把 pending_approval 转成 executing，防止重复审批并发执行。
	a, err := s.store.ClaimAction(ctx, uid, id)
	if err != nil {
		return domain.Action{}, err
	}
	// 使用提案 ID 作为下游幂等键，即使请求重试也不会重复产生业务副作用。
	resp, callErr := s.python.ExecuteAction(ctx, python.ActionRequest{UserID: uid, ActionType: a.Type, Payload: a.Payload, Reason: a.Reason, IdempotencyKey: a.ID})
	// 即使原 HTTP 请求已经取消，也给最终状态写入最多五秒的收尾时间。
	finalizeCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
	defer cancel()
	if callErr != nil {
		// Python 执行失败时必须把 executing 收敛为 failed。
		if finishErr := s.store.FinishAction(finalizeCtx, id, domain.ActionFailed, nil, callErr.Error()); finishErr != nil {
			// 同时保留业务执行错误和状态落库错误，便于排查不一致。
			return domain.Action{}, errors.Join(callErr, fmt.Errorf("persist failed action: %w", finishErr))
		}
		return domain.Action{}, callErr
	}
	// 执行成功后保存结构化结果并将状态改为 succeeded。
	if err = s.store.FinishAction(finalizeCtx, id, domain.ActionSucceeded, resp.Result, ""); err != nil {
		return domain.Action{}, err
	}
	// 返回最终持久化后的完整提案，而不是只返回 Python 的临时响应。
	return s.store.GetAction(finalizeCtx, uid, id)
}

// RejectAction 将待审批提案标记为 rejected，不调用 Python 执行。
func (s *Service) RejectAction(ctx context.Context, uid int64, id string) error {
	return s.store.RejectAction(ctx, uid, id)
}

// Ready 依次检查数据库、Python Agent 和额外注册的外部依赖。
func (s *Service) Ready(ctx context.Context) error {
	// 数据库存储不可用时服务不能正常创建、恢复或推进任务。
	if err := s.store.Ping(ctx); err != nil {
		return err
	}
	// Python Agent 不可用时，Control Plane 无法实际执行 run 和 action。
	if err := s.python.Ready(ctx); err != nil {
		return err
	}
	// 检查 Redis、Kafka 等通过 Option 注入的依赖。
	for _, check := range s.readyChecks {
		if err := check(ctx); err != nil {
			return err
		}
	}
	return nil
}

// IsNotFound 判断错误链是否包含统一的领域层“未找到”错误。
func IsNotFound(err error) bool { return errors.Is(err, domain.ErrNotFound) }

// page 规范分页参数，默认每页 20 条，最多允许 100 条。
func page(limit, offset int) (int, int, error) {
	// limit=0 表示调用方没有显式指定，使用默认值。
	if limit == 0 {
		limit = 20
	}
	// 拒绝非法页大小和负偏移量，避免无界查询。
	if limit < 1 || limit > 100 || offset < 0 {
		return 0, 0, fmt.Errorf("%w: invalid pagination", domain.ErrInvalidInput)
	}
	return limit, offset, nil
}
