// 文件用途：实现 Control Plane 动作提案的创建、查询、审批和拒绝 HTTP 接口。
package httpapi

import (
	"encoding/json"
	"net/http"
)

// createAction 解析并创建待审批动作提案。
func (s *Server) createAction(w http.ResponseWriter, r *http.Request) {
	var in struct {
		ConversationID string          `json:"conversation_id"`
		Type           string          `json:"action_type"`
		Payload        json.RawMessage `json:"payload"`
		Reason         string          `json:"reason"`
		Impact         string          `json:"impact"`
		IdempotencyKey string          `json:"idempotency_key"`
	}
	if decode(r, &in) != nil {
		bad(w)
		return
	}
	action, created, err := s.svc.CreateAction(r.Context(), userID(r), in.ConversationID, in.Type, in.Payload, in.Reason, in.Impact, in.IdempotencyKey)
	if err != nil {
		s.writeError(w, err)
		return
	}
	if created {
		s.metrics.Actions.Add(1)
		writeJSON(w, http.StatusCreated, map[string]any{"created": true, "proposal": action})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"created": false, "proposal": action})
}

// listActions 分页返回当前用户的动作提案。
func (s *Server) listActions(w http.ResponseWriter, r *http.Request) {
	limit, offset, err := pagination(r)
	if err != nil {
		s.writeError(w, err)
		return
	}
	actions, err := s.svc.ListActions(r.Context(), userID(r), limit, offset)
	if err != nil {
		s.writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"actions": actions, "limit": limit, "offset": offset})
}

// getAction 返回当前用户有权访问的单个动作提案。
func (s *Server) getAction(w http.ResponseWriter, r *http.Request) {
	action, err := s.svc.GetAction(r.Context(), userID(r), r.PathValue("id"))
	if err != nil {
		s.writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"proposal": action})
}

// approveAction 原子认领并执行指定动作提案。
func (s *Server) approveAction(w http.ResponseWriter, r *http.Request) {
	action, err := s.svc.ApproveAction(r.Context(), userID(r), r.PathValue("id"))
	if err != nil {
		s.writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"proposal": action})
}

// rejectAction 将尚未执行的动作提案标记为拒绝。
func (s *Server) rejectAction(w http.ResponseWriter, r *http.Request) {
	if err := s.svc.RejectAction(r.Context(), userID(r), r.PathValue("id")); err != nil {
		s.writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "rejected"})
}
