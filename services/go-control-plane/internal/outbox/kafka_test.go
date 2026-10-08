// 文件用途：验证 Kafka 发布器的配置校验和事件 JSON 编码契约。
package outbox

import (
	"encoding/json"
	"testing"
)

func TestNewKafkaValidatesConfiguration(t *testing.T) {
	if _, err := NewKafka(nil, "agent-events"); err == nil {
		t.Fatal("expected missing broker validation error")
	}
	if _, err := NewKafka([]string{"localhost:9092"}, " "); err == nil {
		t.Fatal("expected missing topic validation error")
	}
}

func TestKafkaEventJSONKeepsPayloadAsJSON(t *testing.T) {
	encoded, err := json.Marshal(kafkaEvent{
		EventID:     42,
		AggregateID: "run_123",
		EventType:   "run.completed",
		Payload:     json.RawMessage(`{"answer":"ok"}`),
	})
	if err != nil {
		t.Fatal(err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded["event_id"] != float64(42) {
		t.Fatalf("unexpected event_id: %#v", decoded["event_id"])
	}
	payload, ok := decoded["payload"].(map[string]any)
	if !ok || payload["answer"] != "ok" {
		t.Fatalf("payload was not embedded as JSON: %#v", decoded["payload"])
	}
}
