// 文件用途：验证 Control Plane 客户端的身份签名、请求结构和事件游标传递。
package controlplane

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (fn roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return fn(request)
}

func response(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}

func TestCreateRunSignsIdentityAndPreservesConversation(t *testing.T) {
	secret := "test-secret-at-least-16"
	transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.Method != http.MethodPost || r.URL.Path != "/v1/runs" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.String())
		}
		if r.Header.Get("X-User-ID") != "42" || r.Header.Get("X-Auth-Signature") == "" || r.Header.Get("X-Auth-Timestamp") == "" {
			t.Fatalf("missing signed identity headers: %#v", r.Header)
		}
		var body map[string]string
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body["message"] != "hello" || body["conversation_id"] != "conv-1" {
			t.Fatalf("unexpected body: %#v", body)
		}
		return response(http.StatusAccepted, `{"run_id":"run-1","conversation_id":"conv-1","status":"pending"}`), nil
	})

	client, err := New("http://control-plane.test", secret)
	if err != nil {
		t.Fatal(err)
	}
	client.http.Transport = transport
	run, err := client.CreateRun(context.Background(), 42, "hello", "conv-1")
	if err != nil {
		t.Fatal(err)
	}
	if run.ID != "run-1" || run.ConversationID != "conv-1" || run.Status != "pending" {
		t.Fatalf("unexpected run: %#v", run)
	}
}

func TestEventSnapshotKeepsQueryString(t *testing.T) {
	transport := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.Path != "/v1/runs/run-1/events/snapshot" || r.URL.Query().Get("after") != "7" {
			t.Fatalf("unexpected URL: %s", r.URL.String())
		}
		return response(http.StatusOK, `{"events":[]}`), nil
	})

	client, err := New("http://control-plane.test", "test-secret-at-least-16")
	if err != nil {
		t.Fatal(err)
	}
	client.http.Transport = transport
	var result map[string]any
	if err := client.EventSnapshot(context.Background(), 9, "run-1", 7, &result); err != nil {
		t.Fatal(err)
	}
}
