// 文件用途：实现 Business API 到 Python Agent 的私有 gRPC 客户端，用于计算派发和辅助能力调用。
package python

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	agentv1 "knowledge-agent/go-business-api/gen/agent/v1"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
)

// internalTokenMetadataKey 是 Go 服务调用 Python 私有 gRPC 时使用的认证元数据键。
const internalTokenMetadataKey = "x-control-plane-token"

// DispatchRequest 描述需要交给 Celery 执行的计算任务。
type DispatchRequest struct {
	TaskType string
	RecordID int64
	TaskID   string
	Payload  json.RawMessage
}

// ValidatedAction 保存 Python 校验和规范化后的动作参数及影响说明。
type ValidatedAction struct {
	Payload json.RawMessage
	Impact  string
}

// RenderedImage 保存 Python 转换后的图片字节、MIME 类型和文件名。
type RenderedImage struct {
	Content     []byte
	ContentType string
	Filename    string
}

// Client 封装 Python Agent 的 gRPC 连接、认证令牌和调用超时。
type Client struct {
	connection *grpc.ClientConn
	rpc        agentv1.AgentExecutionServiceClient
	token      string
	timeout    time.Duration
}

// New 建立到 Python Agent 的 gRPC Channel 并创建类型安全客户端。
func New(target, token string, timeout time.Duration) (*Client, error) {
	connection, err := grpc.NewClient(
		target,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(32<<20), grpc.MaxCallSendMsgSize(8<<20)),
	)
	if err != nil {
		return nil, fmt.Errorf("create Python gRPC client: %w", err)
	}
	return &Client{connection: connection, rpc: agentv1.NewAgentExecutionServiceClient(connection), token: token, timeout: timeout}, nil
}

// Close 释放底层 gRPC 连接。
func (c *Client) Close() error { return c.connection.Close() }

// DispatchCompute 请求 Python 将训练、验证、NAS 或评测任务派发给 Celery。
func (c *Client) DispatchCompute(ctx context.Context, in DispatchRequest) error {
	callCtx, cancel := c.callContext(ctx)
	defer cancel()
	_, err := c.rpc.DispatchCompute(callCtx, &agentv1.DispatchComputeRequest{TaskType: in.TaskType, RecordId: in.RecordID, TaskId: in.TaskID, PayloadJson: in.Payload})
	if err != nil {
		return fmt.Errorf("Python gRPC DispatchCompute failed: %w", err)
	}
	return nil
}

// Knowledge 调用 Python RAG 引擎的状态、搜索或重建操作。
func (c *Client) Knowledge(ctx context.Context, operation string, payload json.RawMessage) (json.RawMessage, error) {
	callCtx, cancel := c.callContext(ctx)
	defer cancel()
	response, err := c.rpc.Knowledge(callCtx, &agentv1.KnowledgeRequest{Operation: operation, PayloadJson: payload})
	if err != nil {
		return nil, fmt.Errorf("Python gRPC Knowledge failed: %w", err)
	}
	if !json.Valid(response.ResultJson) {
		return nil, fmt.Errorf("Python gRPC Knowledge returned invalid JSON")
	}
	return response.ResultJson, nil
}

// ValidateAction 在动作进入审批表之前让 Python 执行业务参数校验和规范化。
func (c *Client) ValidateAction(ctx context.Context, userID int64, actionType string, payload json.RawMessage) (ValidatedAction, error) {
	callCtx, cancel := c.callContext(ctx)
	defer cancel()
	response, err := c.rpc.ValidateAction(callCtx, &agentv1.ValidateActionRequest{UserId: userID, ActionType: actionType, PayloadJson: payload})
	if err != nil {
		return ValidatedAction{}, fmt.Errorf("Python gRPC ValidateAction failed: %w", err)
	}
	if !json.Valid(response.PayloadJson) || response.Impact == "" {
		return ValidatedAction{}, fmt.Errorf("Python gRPC ValidateAction returned an invalid response")
	}
	return ValidatedAction{Payload: response.PayloadJson, Impact: response.Impact}, nil
}

// RenderImage 将受信任的数据文件转换为浏览器可以展示的 PNG。
func (c *Client) RenderImage(ctx context.Context, path string) (RenderedImage, error) {
	callCtx, cancel := c.callContext(ctx)
	defer cancel()
	response, err := c.rpc.RenderImage(callCtx, &agentv1.RenderImageRequest{Path: path})
	if err != nil {
		return RenderedImage{}, fmt.Errorf("Python gRPC RenderImage failed: %w", err)
	}
	return RenderedImage{Content: response.Content, ContentType: response.ContentType, Filename: response.Filename}, nil
}

// CancelCompute 请求 Celery 撤销指定后台任务。
func (c *Client) CancelCompute(ctx context.Context, taskID string, terminate bool) error {
	callCtx, cancel := c.callContext(ctx)
	defer cancel()
	_, err := c.rpc.CancelCompute(callCtx, &agentv1.CancelComputeRequest{TaskId: taskID, Terminate: terminate})
	if err != nil {
		return fmt.Errorf("Python gRPC CancelCompute failed: %w", err)
	}
	return nil
}

// callContext 为私有调用附加认证元数据并设置统一超时。
func (c *Client) callContext(ctx context.Context) (context.Context, context.CancelFunc) {
	ctx = metadata.AppendToOutgoingContext(ctx, internalTokenMetadataKey, c.token)
	return context.WithTimeout(ctx, c.timeout)
}
