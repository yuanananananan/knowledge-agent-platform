// 文件用途：验证 Agent 网关的流式创建、事件转发、恢复和路由行为。
package businessapi

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"knowledge-agent/go-business-api/internal/controlplane"

	"github.com/gin-gonic/gin"
)

type fakeAgentControl struct {
	createdMessage string
}

func (f *fakeAgentControl) CreateRun(_ context.Context, _ int64, message, conversation string) (controlplane.Run, error) {
	f.createdMessage = message
	return controlplane.Run{ID: "run-1", ConversationID: conversation, Status: "pending"}, nil
}
func (f *fakeAgentControl) GetRun(context.Context, int64, string) (controlplane.Run, error) {
	return controlplane.Run{ID: "run-1", ConversationID: "conv-1", Status: "succeeded"}, nil
}
func (f *fakeAgentControl) CancelRun(context.Context, int64, string, any) error     { return nil }
func (f *fakeAgentControl) ListRuns(context.Context, int64, string, int, any) error { return nil }
func (f *fakeAgentControl) EventSnapshot(context.Context, int64, string, int64, any) error {
	return nil
}
func (f *fakeAgentControl) ListActions(context.Context, int64, int, any) error  { return nil }
func (f *fakeAgentControl) CreateAction(context.Context, int64, any, any) error { return nil }
func (f *fakeAgentControl) GetAction(context.Context, int64, string) (controlplane.Action, error) {
	return controlplane.Action{}, nil
}
func (f *fakeAgentControl) ApproveAction(context.Context, int64, string) (controlplane.Action, error) {
	return controlplane.Action{}, nil
}
func (f *fakeAgentControl) RejectAction(context.Context, int64, string) error { return nil }
func (f *fakeAgentControl) OpenEventStream(context.Context, int64, string, int64) (*http.Response, error) {
	body := "id: 1\nevent: model.delta\ndata: {\"run_id\":\"run-1\",\"sequence\":1,\"type\":\"model.delta\",\"payload\":{\"content\":\"ok\"}}\n\n" +
		"id: 2\nevent: run.completed\ndata: {\"run_id\":\"run-1\",\"sequence\":2,\"type\":\"run.completed\",\"payload\":{\"answer\":\"ok\"}}\n\n"
	return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(body)), Header: make(http.Header)}, nil
}

func TestAgentChatStreamBypassesFlaskProxy(t *testing.T) {
	gin.SetMode(gin.TestMode)
	control := &fakeAgentControl{}
	server := &Server{agentControl: control}
	router := gin.New()
	router.Use(func(c *gin.Context) {
		c.Set("user_id", int64(42))
		c.Next()
	})
	router.Any("/api/agent/*path", server.agentGateway)

	request := httptest.NewRequest(http.MethodPost, "/api/agent/chat/stream", strings.NewReader(`{"message":"hello","conversation_id":"conv-1"}`))
	request.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	router.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusOK {
		t.Fatalf("unexpected status %d: %s", recorder.Code, recorder.Body.String())
	}
	if control.createdMessage != "hello" {
		t.Fatalf("run was not created directly: %q", control.createdMessage)
	}
	body := recorder.Body.String()
	for _, expected := range []string{"run.accepted", "model.delta", "run.completed"} {
		if !strings.Contains(body, expected) {
			t.Fatalf("missing %q in SSE body: %s", expected, body)
		}
	}
}
