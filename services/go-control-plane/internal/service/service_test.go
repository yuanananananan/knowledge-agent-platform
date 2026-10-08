// 文件用途：验证 Control Plane Service 的模型增量批处理边界和内容完整性。
package service

import (
	"strings"
	"testing"
	"time"
)

// TestModelDeltaBatchFlushesAtByteLimit 验证达到字节阈值时触发落库。
func TestModelDeltaBatchFlushesAtByteLimit(t *testing.T) {
	var batch modelDeltaBatch
	now := time.Unix(1, 0)
	if batch.Add(now, strings.Repeat("a", modelDeltaBatchMaxBytes-1)) {
		t.Fatal("batch flushed before reaching byte limit")
	}
	if !batch.Add(now, "b") {
		t.Fatal("batch did not flush at byte limit")
	}
	if got := batch.Take(); len(got) != modelDeltaBatchMaxBytes {
		t.Fatalf("unexpected batch length: got %d want %d", len(got), modelDeltaBatchMaxBytes)
	}
}

// TestModelDeltaBatchFlushesAtWaitLimit 验证持续产生 token 时达到等待窗口会触发落库。
func TestModelDeltaBatchFlushesAtWaitLimit(t *testing.T) {
	var batch modelDeltaBatch
	started := time.Unix(1, 0)
	if batch.Add(started, "你") {
		t.Fatal("first delta should stay buffered")
	}
	if batch.Add(started.Add(modelDeltaBatchMaxWait-time.Millisecond), "好") {
		t.Fatal("batch flushed before wait limit")
	}
	if !batch.Add(started.Add(modelDeltaBatchMaxWait), "！") {
		t.Fatal("batch did not flush at wait limit")
	}
	if got := batch.Take(); got != "你好！" {
		t.Fatalf("unexpected merged content: %q", got)
	}
	if got := batch.Take(); got != "" {
		t.Fatalf("batch was not reset: %q", got)
	}
}
