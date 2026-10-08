// 文件用途：实现 Control Plane 到 Python Agent 的 gRPC 客户端，包括流式运行、动作执行和健康检查。
package python

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"

	agentv1 "knowledge-agent/go-control-plane/gen/agent/v1"
	"knowledge-agent/go-control-plane/internal/domain"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// internalTokenMetadataKey 是 Control Plane 调用 Python 私有 gRPC 的认证头。
const internalTokenMetadataKey = "x-control-plane-token"

// Client 封装 Python Agent 的 gRPC Channel、Stub、令牌和普通调用超时。
type Client struct {
	connection *grpc.ClientConn
	rpc        agentv1.AgentExecutionServiceClient
	token      string
	timeout    time.Duration
}

// RunRequest 描述一次 Agent 运行需要的稳定标识、用户、会话和消息。
type RunRequest struct {
	RunID          string
	ConversationID string
	UserID         int64
	Message        string
}

// RunResponse 保存非流式执行的最终答案和元数据。
type RunResponse struct {
	Answer         string
	ConversationID string
	Metadata       json.RawMessage
}

// RunEvent 是从 server-streaming gRPC 收到的结构化增量事件。
type RunEvent struct {
	Type     string
	Payload  json.RawMessage
	Terminal bool
}

// ActionRequest 描述已经批准、允许 Python 执行的业务动作。
type ActionRequest struct {
	UserID         int64
	ActionType     string
	Payload        json.RawMessage
	Reason         string
	IdempotencyKey string
}

// ActionResponse 保存动作执行产生的结构化结果。
type ActionResponse struct {
	Result json.RawMessage
}

// New 建立 gRPC Channel，并配置单条消息大小限制。
func New(target, token string, timeout time.Duration) (*Client, error) {
	connection, err := grpc.NewClient(
		target,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(
			grpc.MaxCallRecvMsgSize(4<<20),
			grpc.MaxCallSendMsgSize(4<<20),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("create Python gRPC client: %w", err)
	}
	return &Client{
		connection: connection,
		rpc:        agentv1.NewAgentExecutionServiceClient(connection),
		token:      token,
		timeout:    timeout,
	}, nil
}

// Close 释放 gRPC 连接。
func (c *Client) Close() error {
	return c.connection.Close()
}

// Run 调用非流式 ExecuteRun，并校验返回的 metadata JSON。
func (c *Client) Run(ctx context.Context, in RunRequest) (RunResponse, error) {
	callCtx, cancel := c.callContext(ctx)
	defer cancel()
	response, err := c.rpc.ExecuteRun(callCtx, &agentv1.ExecuteRunRequest{
		RunId:          in.RunID,
		ConversationId: in.ConversationID,
		UserId:         in.UserID,
		Message:        in.Message,
	})
	if err != nil {
		return RunResponse{}, classifyRPCError(ctx, "ExecuteRun", err)
	}
	metadataJSON := json.RawMessage(response.GetMetadataJson())
	if len(metadataJSON) == 0 {
		metadataJSON = json.RawMessage(`{}`)
	}
	if !json.Valid(metadataJSON) {
		return RunResponse{}, fmt.Errorf("%w: ExecuteRun returned invalid metadata JSON", domain.ErrDependency)
	}
	return RunResponse{
		Answer:         response.GetAnswer(),
		ConversationID: response.GetConversationId(),
		Metadata:       metadataJSON,
	}, nil
}

// RunStream 持续接收 ExecuteRunStream 事件，并同步交给 handle 落库处理。
func (c *Client) RunStream(ctx context.Context, in RunRequest, handle func(RunEvent) error) error {
	// The worker owns the run deadline. Do not impose the shorter unary RPC
	// timeout on a healthy long-running stream.
	callCtx := metadata.AppendToOutgoingContext(ctx, internalTokenMetadataKey, c.token)
	stream, err := c.rpc.ExecuteRunStream(callCtx, &agentv1.ExecuteRunRequest{
		RunId: in.RunID, ConversationId: in.ConversationID, UserId: in.UserID, Message: in.Message,
	})
	if err != nil {
		return classifyRPCError(ctx, "ExecuteRunStream", err)
	}
	for {
		event, recvErr := stream.Recv()
		if errors.Is(recvErr, io.EOF) {
			return nil
		}
		if recvErr != nil {
			return classifyRPCError(ctx, "ExecuteRunStream", recvErr)
		}
		body := json.RawMessage(event.GetPayloadJson())
		if len(body) == 0 {
			body = json.RawMessage(`{}`)
		}
		if !json.Valid(body) {
			return fmt.Errorf("%w: ExecuteRunStream returned invalid payload JSON", domain.ErrDependency)
		}
		if err := handle(RunEvent{Type: event.GetType(), Payload: body, Terminal: event.GetTerminal()}); err != nil {
			return err
		}
	}
}

// Ready 在两秒内调用 Python 健康检查。
func (c *Client) Ready(ctx context.Context) error {
	callCtx, cancel := context.WithTimeout(metadata.AppendToOutgoingContext(ctx, internalTokenMetadataKey, c.token), 2*time.Second)
	defer cancel()
	response, err := c.rpc.Check(callCtx, &agentv1.HealthCheckRequest{})
	if err != nil {
		return classifyRPCError(ctx, "Check", err)
	}
	if response.GetStatus() != "serving" {
		return fmt.Errorf("%w: Python gRPC is not serving", domain.ErrDependency)
	}
	return nil
}

// ExecuteAction 调用 Python 执行审批后的动作，并校验结果 JSON。
func (c *Client) ExecuteAction(ctx context.Context, in ActionRequest) (ActionResponse, error) {
	callCtx, cancel := c.callContext(ctx)
	defer cancel()
	response, err := c.rpc.ExecuteAction(callCtx, &agentv1.ExecuteActionRequest{
		UserId:         in.UserID,
		ActionType:     in.ActionType,
		PayloadJson:    in.Payload,
		Reason:         in.Reason,
		IdempotencyKey: in.IdempotencyKey,
	})
	if err != nil {
		return ActionResponse{}, classifyRPCError(ctx, "ExecuteAction", err)
	}
	resultJSON := json.RawMessage(response.GetResultJson())
	if len(resultJSON) == 0 {
		resultJSON = json.RawMessage(`{}`)
	}
	if !json.Valid(resultJSON) {
		return ActionResponse{}, fmt.Errorf("%w: ExecuteAction returned invalid result JSON", domain.ErrDependency)
	}
	return ActionResponse{Result: resultJSON}, nil
}

// callContext 添加认证元数据，并为普通一元 RPC 设置统一超时。
func (c *Client) callContext(ctx context.Context) (context.Context, context.CancelFunc) {
	ctx = metadata.AppendToOutgoingContext(ctx, internalTokenMetadataKey, c.token)
	return context.WithTimeout(ctx, c.timeout)
}

// classifyRPCError 将 gRPC 状态码转换为 Control Plane 领域错误。
func classifyRPCError(parent context.Context, method string, err error) error {
	if parentErr := parent.Err(); parentErr != nil {
		return parentErr
	}
	switch status.Code(err) {
	case codes.Aborted, codes.AlreadyExists:
		return fmt.Errorf("%w: Python gRPC %s: %v", domain.ErrInProgress, method, err)
	case codes.Canceled:
		return context.Canceled
	case codes.DeadlineExceeded:
		return fmt.Errorf("%w: Python gRPC %s timed out: %v", domain.ErrDependency, method, err)
	default:
		if errors.Is(err, context.Canceled) {
			return context.Canceled
		}
		return fmt.Errorf("%w: Python gRPC %s failed: %v", domain.ErrDependency, method, err)
	}
}
