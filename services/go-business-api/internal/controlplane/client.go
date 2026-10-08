// 文件用途：实现 Business API 到 Control Plane 的签名 HTTP 客户端及 SSE 长连接客户端。
package controlplane

import (
	"bytes"
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// Client is the authenticated internal HTTP client used by the public Go API
// to talk to the Agent control plane. Browser credentials never cross this
// boundary; the gateway signs the already-authenticated user id instead.
// Client 保存 Control Plane 地址、HMAC 密钥和短请求 HTTP 客户端。
type Client struct {
	baseURL *url.URL
	secret  []byte
	http    *http.Client
}

// Run 是 Business API 需要使用的 Control Plane 运行摘要。
type Run struct {
	ID             string          `json:"run_id"`
	ConversationID string          `json:"conversation_id"`
	Status         string          `json:"status"`
	Answer         string          `json:"answer,omitempty"`
	Error          string          `json:"error,omitempty"`
	Payload        json.RawMessage `json:"-"`
}

// Action 是 Control Plane 动作提案的公开客户端模型。
type Action struct {
	ID             string          `json:"proposal_id"`
	ConversationID string          `json:"conversation_id,omitempty"`
	Type           string          `json:"action_type"`
	Payload        json.RawMessage `json:"payload"`
	Reason         string          `json:"reason,omitempty"`
	Impact         string          `json:"impact"`
	Status         string          `json:"status"`
	Result         json.RawMessage `json:"result,omitempty"`
	Error          string          `json:"error,omitempty"`
	ExpiresAt      time.Time       `json:"expires_at"`
	CreatedAt      time.Time       `json:"created_at"`
	UpdatedAt      time.Time       `json:"updated_at"`
}

// HTTPError 保留 Control Plane 返回的状态码和响应体，供网关原样映射。
type HTTPError struct {
	Status int
	Body   []byte
}

// Error 实现 error 接口。
func (e *HTTPError) Error() string {
	return fmt.Sprintf("control plane returned HTTP %d", e.Status)
}

// New 校验绝对地址和 HMAC 密钥后创建客户端。
func New(rawURL, secret string) (*Client, error) {
	parsed, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return nil, errors.New("CONTROL_PLANE_URL must be an absolute URL")
	}
	if len(secret) < 16 {
		return nil, errors.New("CONTROL_PLANE_AUTH_SECRET must contain at least 16 characters")
	}
	return &Client{
		baseURL: parsed,
		secret:  []byte(secret),
		http:    &http.Client{Timeout: 15 * time.Second},
	}, nil
}

// signedRequest 使用用户 ID、时间戳和共享密钥为内部请求生成 HMAC 身份头。
func (c *Client) signedRequest(ctx context.Context, userID int64, method, path string, body []byte) (*http.Request, error) {
	if userID <= 0 {
		return nil, errors.New("invalid user id")
	}
	endpoint, err := url.Parse(strings.TrimRight(c.baseURL.String(), "/") + path)
	if err != nil {
		return nil, err
	}
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, endpoint.String(), reader)
	if err != nil {
		return nil, err
	}
	stamp := strconv.FormatInt(time.Now().Unix(), 10)
	rawUser := strconv.FormatInt(userID, 10)
	mac := hmac.New(sha256.New, c.secret)
	_, _ = mac.Write([]byte(rawUser + ":" + stamp))
	req.Header.Set("X-User-ID", rawUser)
	req.Header.Set("X-Auth-Timestamp", stamp)
	req.Header.Set("X-Auth-Signature", hex.EncodeToString(mac.Sum(nil)))
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	return req, nil
}

// doJSON 执行签名 JSON 请求，并统一处理响应大小、状态码和反序列化。
func (c *Client) doJSON(ctx context.Context, userID int64, method, path string, input any, output any) error {
	var body []byte
	var err error
	if input != nil {
		body, err = json.Marshal(input)
		if err != nil {
			return err
		}
	}
	req, err := c.signedRequest(ctx, userID, method, path, body)
	if err != nil {
		return err
	}
	response, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	payload, err := io.ReadAll(io.LimitReader(response.Body, 4<<20))
	if err != nil {
		return err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return &HTTPError{Status: response.StatusCode, Body: payload}
	}
	if output == nil || len(payload) == 0 {
		return nil
	}
	return json.Unmarshal(payload, output)
}

// CreateRun 在 Control Plane 创建一个初始状态为 pending 的 Agent run。
func (c *Client) CreateRun(ctx context.Context, userID int64, message, conversationID string) (Run, error) {
	request := map[string]string{"message": message}
	if conversationID != "" {
		request["conversation_id"] = conversationID
	}
	var run Run
	err := c.doJSON(ctx, userID, http.MethodPost, "/v1/runs", request, &run)
	return run, err
}

// GetRun 查询指定 run 的当前状态和最终结果。
func (c *Client) GetRun(ctx context.Context, userID int64, runID string) (Run, error) {
	var run Run
	err := c.doJSON(ctx, userID, http.MethodGet, "/v1/runs/"+url.PathEscape(runID), nil, &run)
	return run, err
}

// CancelRun 请求 Control Plane 取消非终态 run。
func (c *Client) CancelRun(ctx context.Context, userID int64, runID string, output any) error {
	return c.doJSON(ctx, userID, http.MethodDelete, "/v1/runs/"+url.PathEscape(runID), nil, output)
}

// ListRuns 分页查询用户或指定会话下的运行记录。
func (c *Client) ListRuns(ctx context.Context, userID int64, conversationID string, limit int, output any) error {
	query := url.Values{}
	query.Set("limit", strconv.Itoa(limit))
	query.Set("offset", "0")
	if conversationID != "" {
		query.Set("conversation_id", conversationID)
	}
	return c.doJSON(ctx, userID, http.MethodGet, "/v1/runs?"+query.Encode(), nil, output)
}

// EventSnapshot 获取游标之后的一批持久化事件和 run 状态。
func (c *Client) EventSnapshot(ctx context.Context, userID int64, runID string, after int64, output any) error {
	path := "/v1/runs/" + url.PathEscape(runID) + "/events/snapshot?after=" + strconv.FormatInt(after, 10)
	return c.doJSON(ctx, userID, http.MethodGet, path, nil, output)
}

// ListActions 查询当前用户的动作提案。
func (c *Client) ListActions(ctx context.Context, userID int64, limit int, output any) error {
	query := url.Values{}
	query.Set("limit", strconv.Itoa(limit))
	query.Set("offset", "0")
	return c.doJSON(ctx, userID, http.MethodGet, "/v1/actions?"+query.Encode(), nil, output)
}

// CreateAction 创建待人工审批的动作提案。
func (c *Client) CreateAction(ctx context.Context, userID int64, input any, output any) error {
	return c.doJSON(ctx, userID, http.MethodPost, "/v1/actions", input, output)
}

// GetAction 查询单个动作提案。
func (c *Client) GetAction(ctx context.Context, userID int64, proposalID string) (Action, error) {
	var response struct {
		Proposal Action `json:"proposal"`
	}
	err := c.doJSON(ctx, userID, http.MethodGet, "/v1/actions/"+url.PathEscape(proposalID), nil, &response)
	return response.Proposal, err
}

// ApproveAction 审批并触发动作执行。
func (c *Client) ApproveAction(ctx context.Context, userID int64, proposalID string) (Action, error) {
	var response struct {
		Proposal Action `json:"proposal"`
	}
	err := c.doJSON(ctx, userID, http.MethodPost, "/v1/actions/"+url.PathEscape(proposalID)+"/approve", map[string]any{}, &response)
	return response.Proposal, err
}

// RejectAction 拒绝动作提案。
func (c *Client) RejectAction(ctx context.Context, userID int64, proposalID string) error {
	return c.doJSON(ctx, userID, http.MethodPost, "/v1/actions/"+url.PathEscape(proposalID)+"/reject", map[string]any{}, nil)
}

// OpenEventStream leaves the body open so the caller can relay SSE while
// tracking the last durable sequence number. The caller must close Body.
func (c *Client) OpenEventStream(ctx context.Context, userID int64, runID string, after int64) (*http.Response, error) {
	path := "/v1/runs/" + url.PathEscape(runID) + "/events?after=" + strconv.FormatInt(after, 10)
	req, err := c.signedRequest(ctx, userID, http.MethodGet, path, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "text/event-stream")
	// Streaming requests must not inherit the short JSON request timeout.
	response, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	if response.StatusCode < 200 || response.StatusCode >= 300 {
		payload, _ := io.ReadAll(io.LimitReader(response.Body, 4096))
		response.Body.Close()
		return nil, &HTTPError{Status: response.StatusCode, Body: payload}
	}
	return response, nil
}
