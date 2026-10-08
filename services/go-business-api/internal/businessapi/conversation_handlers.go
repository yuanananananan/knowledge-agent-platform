// 文件用途：管理 Agent 会话及其消息、检查点、运行和删除时的级联清理。
package businessapi

import (
	"database/sql"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

// conversationIDPattern 限制会话 ID 字符集，防止非法路径和数据库标识。
var conversationIDPattern = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

// agentConversationRow 表示从数据库读取的会话概要。
type agentConversationRow struct {
	ID                string
	Title             string
	CheckpointVersion int
	CreatedAt         time.Time
	UpdatedAt         time.Time
}

// conversationJSON 将数据库行转换为公开 API 使用的 JSON 格式。
func conversationJSON(row agentConversationRow) gin.H {
	return gin.H{
		"conversation_id":    row.ID,
		"title":              row.Title,
		"checkpoint_version": row.CheckpointVersion,
		"created_at":         row.CreatedAt.UTC().Format(time.RFC3339),
		"updated_at":         row.UpdatedAt.UTC().Format(time.RFC3339),
	}
}

// agentConversations 根据 HTTP 方法分派会话列表或创建操作。
func (s *Server) agentConversations(c *gin.Context) {
	if c.Request.Method == http.MethodPost {
		s.createAgentConversation(c)
		return
	}
	limit := 50
	if raw := c.Query("limit"); raw != "" {
		parsed, err := strconv.Atoi(raw)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "limit 必须是整数"})
			return
		}
		limit = parsed
	}
	if limit < 1 {
		limit = 1
	}
	if limit > 100 {
		limit = 100
	}
	rows, err := s.db.QueryContext(c, `SELECT conversation_id,title,checkpoint_version,created_at,updated_at
FROM agent_conversations WHERE user_id=? AND conversation_id NOT LIKE 'eval\\_%' ESCAPE '\\'
ORDER BY updated_at DESC LIMIT ?`, userID(c), limit)
	if err != nil {
		fail(c, err)
		return
	}
	defer rows.Close()
	items := []gin.H{}
	for rows.Next() {
		var row agentConversationRow
		if err := rows.Scan(&row.ID, &row.Title, &row.CheckpointVersion, &row.CreatedAt, &row.UpdatedAt); err != nil {
			fail(c, err)
			return
		}
		items = append(items, conversationJSON(row))
	}
	if err := rows.Err(); err != nil {
		fail(c, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{"items": items})
}

// createAgentConversation 生成唯一会话 ID 并持久化初始会话。
func (s *Server) createAgentConversation(c *gin.Context) {
	var input struct {
		ConversationID string `json:"conversation_id"`
		Title          string `json:"title"`
	}
	if err := c.ShouldBindJSON(&input); err != nil && !errors.Is(err, io.EOF) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_request"})
		return
	}
	if input.ConversationID == "" {
		input.ConversationID = "conv_" + randomToken()[:32]
	}
	if !conversationIDPattern.MatchString(input.ConversationID) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "conversation_id 只能包含字母、数字、下划线和连字符，且最长 64 位"})
		return
	}
	input.Title = truncateRunes(strings.TrimSpace(input.Title), 80)
	if input.Title == "" {
		input.Title = "新对话"
	}

	var owner int64
	var existing agentConversationRow
	err := s.db.QueryRowContext(c, `SELECT user_id,conversation_id,title,checkpoint_version,created_at,updated_at
FROM agent_conversations WHERE conversation_id=?`, input.ConversationID).Scan(
		&owner, &existing.ID, &existing.Title, &existing.CheckpointVersion, &existing.CreatedAt, &existing.UpdatedAt,
	)
	if err == nil {
		if owner != userID(c) {
			c.JSON(http.StatusConflict, gin.H{"error": "conversation_id 已被占用"})
			return
		}
		c.JSON(http.StatusCreated, gin.H{"conversation": conversationJSON(existing)})
		return
	}
	if !errors.Is(err, sql.ErrNoRows) {
		fail(c, err)
		return
	}

	_, err = s.db.ExecContext(c, `INSERT INTO agent_conversations
(conversation_id,user_id,title,checkpoint_version,created_at,updated_at)
VALUES(?,?,?,0,UTC_TIMESTAMP(6),UTC_TIMESTAMP(6))`, input.ConversationID, userID(c), input.Title)
	if err != nil {
		fail(c, err)
		return
	}
	if err := s.db.QueryRowContext(c, `SELECT conversation_id,title,checkpoint_version,created_at,updated_at
FROM agent_conversations WHERE conversation_id=? AND user_id=?`, input.ConversationID, userID(c)).Scan(
		&existing.ID, &existing.Title, &existing.CheckpointVersion, &existing.CreatedAt, &existing.UpdatedAt,
	); err != nil {
		fail(c, err)
		return
	}
	c.JSON(http.StatusCreated, gin.H{"conversation": conversationJSON(existing)})
}

// agentConversation 返回会话详情和最新检查点中的消息。
func (s *Server) agentConversation(c *gin.Context, conversationID string) {
	if !conversationIDPattern.MatchString(conversationID) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_conversation_id"})
		return
	}
	if c.Request.Method == http.MethodDelete {
		s.deleteAgentConversation(c, conversationID)
		return
	}
	var row agentConversationRow
	if err := s.db.QueryRowContext(c, `SELECT conversation_id,title,checkpoint_version,created_at,updated_at
FROM agent_conversations WHERE conversation_id=? AND user_id=?`, conversationID, userID(c)).Scan(
		&row.ID, &row.Title, &row.CheckpointVersion, &row.CreatedAt, &row.UpdatedAt,
	); errors.Is(err, sql.ErrNoRows) {
		c.JSON(http.StatusNotFound, gin.H{"error": "对话不存在"})
		return
	} else if err != nil {
		fail(c, err)
		return
	}
	var raw []byte
	err := s.db.QueryRowContext(c, `SELECT state FROM agent_checkpoints
WHERE conversation_id=? AND user_id=? ORDER BY version DESC LIMIT 1`, conversationID, userID(c)).Scan(&raw)
	messages := []any{}
	if err == nil {
		var state struct {
			Messages []any `json:"messages"`
		}
		if json.Unmarshal(raw, &state) == nil && state.Messages != nil {
			messages = state.Messages
		}
	} else if !errors.Is(err, sql.ErrNoRows) {
		fail(c, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{"conversation": conversationJSON(row), "messages": messages})
}

// deleteAgentConversation 校验密码后事务性删除会话及关联运行数据。
func (s *Server) deleteAgentConversation(c *gin.Context, conversationID string) {
	var input struct {
		Password string `json:"password"`
	}
	if c.ShouldBindJSON(&input) != nil || input.Password == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "请输入当前账号密码"})
		return
	}
	var passwordHash string
	if err := s.db.QueryRowContext(c, `SELECT userPassword FROM user WHERE id=? AND isDelete=0`, userID(c)).Scan(&passwordHash); err != nil || !verifyPassword(passwordHash, input.Password) {
		c.JSON(http.StatusForbidden, gin.H{"error": "密码错误，无法删除会话"})
		return
	}
	var exists int
	if err := s.db.QueryRowContext(c, `SELECT 1 FROM agent_conversations WHERE conversation_id=? AND user_id=?`, conversationID, userID(c)).Scan(&exists); errors.Is(err, sql.ErrNoRows) {
		c.JSON(http.StatusNotFound, gin.H{"error": "对话不存在"})
		return
	} else if err != nil {
		fail(c, err)
		return
	}
	var runs map[string]any
	if err := s.agentControl.ListRuns(c.Request.Context(), userID(c), conversationID, 100, &runs); err != nil {
		s.writeControlPlaneError(c, err)
		return
	}
	if rawRuns, ok := runs["runs"].([]any); ok {
		for _, raw := range rawRuns {
			item, _ := raw.(map[string]any)
			status, _ := item["status"].(string)
			if status == "pending" || status == "collecting_evidence" || status == "reasoning" {
				c.JSON(http.StatusConflict, gin.H{"error": "会话仍在运行，请先停止生成再删除"})
				return
			}
		}
	}

	tx, err := s.db.BeginTx(c, nil)
	if err != nil {
		fail(c, err)
		return
	}
	defer tx.Rollback()
	checkpointResult, err := tx.ExecContext(c, `DELETE FROM agent_checkpoints WHERE conversation_id=? AND user_id=?`, conversationID, userID(c))
	if err != nil {
		fail(c, err)
		return
	}
	// Legacy Python-owned runtime rows may still exist from runs created before
	// the Go control plane migration. Remove them when the tables are present.
	_, _ = tx.ExecContext(c, `DELETE e FROM agent_trace_events e JOIN agent_runs r ON r.run_id=e.run_id
WHERE r.conversation_id=? AND r.user_id=? AND e.user_id=?`, conversationID, userID(c), userID(c))
	runResult, _ := tx.ExecContext(c, `DELETE FROM agent_runs WHERE conversation_id=? AND user_id=?`, conversationID, userID(c))
	conversationResult, err := tx.ExecContext(c, `DELETE FROM agent_conversations WHERE conversation_id=? AND user_id=?`, conversationID, userID(c))
	if err != nil {
		fail(c, err)
		return
	}
	if err := tx.Commit(); err != nil {
		fail(c, err)
		return
	}
	checkpoints, _ := checkpointResult.RowsAffected()
	legacyRuns := int64(0)
	if runResult != nil {
		legacyRuns, _ = runResult.RowsAffected()
	}
	conversations, _ := conversationResult.RowsAffected()
	c.JSON(http.StatusOK, gin.H{
		"message": "会话已永久删除", "conversation_id": conversationID,
		"deleted": gin.H{"checkpoints": checkpoints, "runs": legacyRuns, "conversations": conversations},
	})
}

// truncateRunes 按 Unicode 字符而不是字节安全截断文本。
func truncateRunes(value string, limit int) string {
	runes := []rune(value)
	if len(runes) <= limit {
		return value
	}
	return string(runes[:limit])
}
