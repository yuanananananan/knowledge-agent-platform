// 文件用途：实现 run 创建、查询、取消、SSE 事件流和事件快照 HTTP 接口。
package httpapi

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"knowledge-agent/go-control-plane/internal/domain"
)

// ready 在两秒预算内检查数据库、Python、Redis 和 Kafka 等依赖。
func (s *Server) ready(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
	defer cancel()
	if err := s.svc.Ready(ctx); err != nil {
		s.log.Warn("readiness check failed", "error", err)
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "not_ready"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready"})
}

// createRun 解析请求并创建一个持久化 pending run。
func (s *Server) createRun(w http.ResponseWriter, r *http.Request) {
	var in struct {
		ConversationID string `json:"conversation_id"`
		Message        string `json:"message"`
	}
	if decode(r, &in) != nil {
		bad(w)
		return
	}
	run, err := s.svc.CreateRun(r.Context(), userID(r), in.ConversationID, in.Message)
	if err != nil {
		s.writeError(w, err)
		return
	}
	s.metrics.Runs.Add(1)
	writeJSON(w, http.StatusAccepted, run)
}

// listRuns 按用户或会话分页查询运行记录。
func (s *Server) listRuns(w http.ResponseWriter, r *http.Request) {
	limit, offset, err := pagination(r)
	if err != nil {
		s.writeError(w, err)
		return
	}
	runs, err := s.svc.ListRuns(r.Context(), userID(r), r.URL.Query().Get("conversation_id"), limit, offset)
	if err != nil {
		s.writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"runs": runs, "limit": limit, "offset": offset})
}

// getRun 返回指定 run 的当前状态。
func (s *Server) getRun(w http.ResponseWriter, r *http.Request) {
	run, err := s.svc.GetRun(r.Context(), userID(r), r.PathValue("id"))
	if err != nil {
		s.writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, run)
}

// cancelRun 持久化取消状态并通知当前进程中的执行协程。
func (s *Server) cancelRun(w http.ResponseWriter, r *http.Request) {
	run, err := s.svc.CancelRun(r.Context(), userID(r), r.PathValue("id"))
	if err != nil {
		s.writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, run)
}

// events 以 SSE 增量发送 after/Last-Event-ID 之后的持久化事件。
func (s *Server) events(w http.ResponseWriter, r *http.Request) {
	after := int64(0)
	rawAfter := r.Header.Get("Last-Event-ID")
	if query := r.URL.Query().Get("after"); query != "" {
		rawAfter = query
	}
	if rawAfter != "" {
		var err error
		after, err = strconv.ParseInt(rawAfter, 10, 64)
		if err != nil || after < 0 {
			s.writeError(w, fmt.Errorf("%w: invalid event cursor", domain.ErrInvalidInput))
			return
		}
	}
	if _, err := s.svc.GetRun(r.Context(), userID(r), r.PathValue("id")); err != nil {
		s.writeError(w, err)
		return
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "stream_unsupported"})
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("X-Accel-Buffering", "no")
	deadline := time.NewTimer(30 * time.Second)
	defer deadline.Stop()
	tick := time.NewTicker(400 * time.Millisecond)
	defer tick.Stop()
	for {
		events, err := s.svc.Events(r.Context(), userID(r), r.PathValue("id"), after)
		if err != nil {
			return
		}
		for _, event := range events {
			data, _ := json.Marshal(event)
			_, _ = fmt.Fprintf(w, "id: %d\nevent: %s\ndata: %s\n\n", event.Sequence, event.Type, data)
			after = event.Sequence
			flusher.Flush()
		}
		run, err := s.svc.GetRun(r.Context(), userID(r), r.PathValue("id"))
		if err != nil || run.Status.Terminal() {
			return
		}
		select {
		case <-r.Context().Done():
			return
		case <-deadline.C:
			_, _ = fmt.Fprint(w, ": reconnect\n\n")
			flusher.Flush()
			return
		case <-tick.C:
			if len(events) == 0 {
				_, _ = fmt.Fprint(w, ": heartbeat\n\n")
				flusher.Flush()
			}
		}
	}
}

// eventSnapshot 返回一批游标后的事件和 run 状态，供页面恢复使用。
func (s *Server) eventSnapshot(w http.ResponseWriter, r *http.Request) {
	after, err := strconv.ParseInt(r.URL.Query().Get("after"), 10, 64)
	if r.URL.Query().Get("after") == "" {
		after = 0
		err = nil
	}
	if err != nil || after < 0 {
		s.writeError(w, fmt.Errorf("%w: invalid event cursor", domain.ErrInvalidInput))
		return
	}
	events, err := s.svc.Events(r.Context(), userID(r), r.PathValue("id"), after)
	if err != nil {
		s.writeError(w, err)
		return
	}
	run, err := s.svc.GetRun(r.Context(), userID(r), r.PathValue("id"))
	if err != nil {
		s.writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"run": run, "events": events})
}
