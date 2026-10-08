// 文件用途：提供 Agent 动作提案的创建、查询、审批、拒绝和执行结果校验接口。
package businessapi

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"

	"knowledge-agent/go-business-api/internal/controlplane"

	"github.com/gin-gonic/gin"
)

// agentActions 根据 HTTP 方法分派动作列表查询或创建请求。
func (s *Server) agentActions(c *gin.Context) {
	if c.Request.Method == http.MethodPost {
		s.createAgentAction(c)
		return
	}
	limit, err := strconv.Atoi(c.DefaultQuery("limit", "20"))
	if err != nil || limit < 1 || limit > 100 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "limit 必须是 1 到 100 的整数"})
		return
	}
	var upstream struct {
		Actions []controlplane.Action `json:"actions"`
	}
	if err := s.agentControl.ListActions(c.Request.Context(), userID(c), limit, &upstream); err != nil {
		s.writeControlPlaneError(c, err)
		return
	}
	status := strings.TrimSpace(c.Query("status"))
	conversationID := strings.TrimSpace(c.Query("conversation_id"))
	items := make([]gin.H, 0, len(upstream.Actions))
	for _, action := range upstream.Actions {
		if status != "" && action.Status != status {
			continue
		}
		if conversationID != "" && action.ConversationID != conversationID {
			continue
		}
		items = append(items, publicAction(action))
	}
	c.JSON(http.StatusOK, gin.H{"items": items})
}

// createAgentAction 校验动作输入，并在 Control Plane 中创建待审批提案。
func (s *Server) createAgentAction(c *gin.Context) {
	var input struct {
		ConversationID string          `json:"conversation_id"`
		ActionType     string          `json:"action_type"`
		Payload        json.RawMessage `json:"payload"`
		Reason         string          `json:"reason"`
		IdempotencyKey string          `json:"idempotency_key"`
	}
	if c.ShouldBindJSON(&input) != nil || strings.TrimSpace(input.ActionType) == "" || len(input.ConversationID) > 64 || len(input.Reason) > 4000 || len(input.IdempotencyKey) > 256 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_action", "code": "invalid_action"})
		return
	}
	if len(input.Payload) == 0 {
		input.Payload = json.RawMessage(`{}`)
	}
	validated, err := s.compute.ValidateAction(c.Request.Context(), userID(c), strings.TrimSpace(input.ActionType), input.Payload)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error(), "code": "invalid_action"})
		return
	}
	request := gin.H{
		"conversation_id": input.ConversationID, "action_type": strings.TrimSpace(input.ActionType),
		"payload": json.RawMessage(validated.Payload), "reason": strings.TrimSpace(input.Reason),
		"impact": validated.Impact, "idempotency_key": input.IdempotencyKey,
	}
	var response struct {
		Created  bool                `json:"created"`
		Proposal controlplane.Action `json:"proposal"`
	}
	if err := s.agentControl.CreateAction(c.Request.Context(), userID(c), request, &response); err != nil {
		s.writeControlPlaneError(c, err)
		return
	}
	status := http.StatusOK
	if response.Created {
		status = http.StatusCreated
	}
	c.JSON(status, gin.H{"created": response.Created, "proposal": publicAction(response.Proposal)})
}

// agentActionDetail 返回动作提案及其业务执行验证结果。
func (s *Server) agentActionDetail(c *gin.Context, proposalID string) {
	action, ok := s.loadAgentAction(c, proposalID)
	if !ok {
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"proposal":     publicAction(action),
		"verification": s.verifyAgentAction(c, action),
	})
}

// approveAgentAction 请求 Control Plane 审批并执行提案。
func (s *Server) approveAgentAction(c *gin.Context, proposalID string) {
	if !validProposalID(proposalID) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_proposal_id"})
		return
	}
	action, err := s.agentControl.ApproveAction(c.Request.Context(), userID(c), proposalID)
	if err != nil {
		s.writeControlPlaneError(c, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"proposal":     publicAction(action),
		"verification": s.verifyAgentAction(c, action),
	})
}

// rejectAgentAction 拒绝尚未执行的动作提案。
func (s *Server) rejectAgentAction(c *gin.Context, proposalID string) {
	if !validProposalID(proposalID) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_proposal_id"})
		return
	}
	if err := s.agentControl.RejectAction(c.Request.Context(), userID(c), proposalID); err != nil {
		s.writeControlPlaneError(c, err)
		return
	}
	action, ok := s.loadAgentAction(c, proposalID)
	if !ok {
		return
	}
	c.JSON(http.StatusOK, gin.H{"proposal": publicAction(action)})
}

// loadAgentAction 校验提案 ID 并读取当前用户有权访问的提案。
func (s *Server) loadAgentAction(c *gin.Context, proposalID string) (controlplane.Action, bool) {
	if !validProposalID(proposalID) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_proposal_id"})
		return controlplane.Action{}, false
	}
	action, err := s.agentControl.GetAction(c.Request.Context(), userID(c), proposalID)
	if err != nil {
		s.writeControlPlaneError(c, err)
		return controlplane.Action{}, false
	}
	return action, true
}

// validProposalID 对公开提案 ID 做长度和字符集白名单校验。
func validProposalID(value string) bool {
	if value == "" || len(value) > 128 {
		return false
	}
	for _, char := range value {
		if (char < 'a' || char > 'z') && (char < 'A' || char > 'Z') && (char < '0' || char > '9') && char != '_' && char != '-' {
			return false
		}
	}
	return true
}

// publicAction 将内部 Action 转换为稳定的公开 JSON 字段。
func publicAction(action controlplane.Action) gin.H {
	payload := map[string]any{}
	if len(action.Payload) > 0 {
		_ = json.Unmarshal(action.Payload, &payload)
	}
	var result any
	if len(action.Result) > 0 && string(action.Result) != "null" {
		_ = json.Unmarshal(action.Result, &result)
	}
	var conversationID any
	if action.ConversationID != "" {
		conversationID = action.ConversationID
	}
	var reason any
	if action.Reason != "" {
		reason = action.Reason
	}
	var errorMessage any
	if action.Error != "" {
		errorMessage = action.Error
	}
	var executedAt any
	if action.Status == "succeeded" || action.Status == "failed" {
		executedAt = action.UpdatedAt
	}
	return gin.H{
		"proposal_id": action.ID, "conversation_id": conversationID,
		"action_type": action.Type, "payload": payload, "reason": reason,
		"impact": action.Impact, "status": action.Status, "result": result,
		"error_message": errorMessage, "expires_at": action.ExpiresAt,
		"approved_at": nil, "executed_at": executedAt,
		"created_at": action.CreatedAt, "updated_at": action.UpdatedAt,
	}
}

// verifyAgentAction 查询业务表，确认已成功动作的实际副作用是否存在。
func (s *Server) verifyAgentAction(c *gin.Context, action controlplane.Action) gin.H {
	if action.Status != "succeeded" {
		return gin.H{"verified": false, "target": "pending", "status": action.Status}
	}
	var execution struct {
		Result map[string]any `json:"result"`
	}
	if json.Unmarshal(action.Result, &execution) != nil || execution.Result == nil {
		return gin.H{"verified": false, "target": "execution_record"}
	}
	switch action.Type {
	case "create_train_job", "retry_train_job", "clone_train_job", "stop_train_job":
		jobID, _ := execution.Result["job_id"].(string)
		if jobID == "" && action.Type == "stop_train_job" {
			var payload map[string]any
			_ = json.Unmarshal(action.Payload, &payload)
			jobID, _ = payload["job_id"].(string)
		}
		var status string
		var progress sql.NullFloat64
		var rawResult []byte
		err := s.db.QueryRowContext(c.Request.Context(), `SELECT status,progress,result FROM train_jobs WHERE job_id=? AND user_id=? AND isDelete=0`, jobID, userID(c)).Scan(&status, &progress, &rawResult)
		if err != nil {
			return gin.H{"verified": false, "target": "train_job", "job_id": jobID}
		}
		verification := gin.H{"verified": true, "target": "train_job", "job_id": jobID, "status": status, "progress": progress.Float64}
		if status == "completed" || status == "failed" {
			verification["result"] = decodeNullableJSON(rawResult)
		}
		return verification
	case "start_validation":
		validationID, _ := execution.Result["val_job_id"].(string)
		var status string
		var rawResult []byte
		err := s.db.QueryRowContext(c.Request.Context(), `SELECT status,result FROM test_records WHERE val_job_id=? AND user_id=? AND isDelete=0`, validationID, userID(c)).Scan(&status, &rawResult)
		if err != nil {
			return gin.H{"verified": false, "target": "validation", "val_job_id": validationID}
		}
		verification := gin.H{"verified": true, "target": "validation", "val_job_id": validationID, "status": status}
		if status == "completed" || status == "failed" {
			verification["result"] = decodeNullableJSON(rawResult)
		}
		return verification
	default:
		return gin.H{"verified": false, "target": "unknown"}
	}
}

// decodeNullableJSON 将数据库可空 JSON 转换为对象；空值返回 nil。
func decodeNullableJSON(raw []byte) any {
	if len(raw) == 0 {
		return nil
	}
	var value any
	if json.Unmarshal(raw, &value) != nil {
		return nil
	}
	return value
}
