// 文件用途：代理知识库状态、检索和重建操作，并规范请求与响应格式。
package businessapi

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/gin-gonic/gin"
)

// agentKnowledge 校验请求后把知识库操作代理给 Python gRPC 服务。
func (s *Server) agentKnowledge(c *gin.Context, operation string) {
	payload := map[string]any{}
	if operation == "rebuild" {
		role := c.GetString("user_role")
		if role != "admin" && role != "administrator" {
			c.JSON(http.StatusForbidden, gin.H{"message": "Forbidden: insufficient permissions"})
			return
		}
	}
	if operation == "search" {
		if err := c.ShouldBindJSON(&payload); err != nil && !errors.Is(err, io.EOF) {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_request"})
			return
		}
		query, _ := payload["query"].(string)
		query = strings.TrimSpace(query)
		if query == "" || len(query) > 4000 {
			c.JSON(http.StatusBadRequest, gin.H{"error": "query 不能为空且不能超过 4000 个字符"})
			return
		}
		payload["query"] = query
		if err := validateKnowledgeFilters(payload); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}
	}
	encoded, _ := json.Marshal(payload)
	result, err := s.compute.Knowledge(c.Request.Context(), operation, encoded)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": "knowledge_service_unavailable"})
		return
	}
	c.Data(http.StatusOK, "application/json; charset=utf-8", result)
}

// validateKnowledgeFilters 限制检索过滤字段、类型和元数据键，防止任意查询扩散。
func validateKnowledgeFilters(payload map[string]any) error {
	raw, exists := payload["metadata_filters"]
	if !exists || raw == nil {
		return nil
	}
	filters, ok := raw.(map[string]any)
	if !ok {
		return errors.New("metadata_filters 必须是对象")
	}
	if len(filters) > 10 {
		return errors.New("metadata_filters 最多包含10个字段")
	}
	for key, value := range filters {
		if len(key) > 64 || !safeMetadataKey(key) {
			return errors.New("metadata_filters 包含非法字段")
		}
		switch value.(type) {
		case string, float64, bool:
		default:
			return errors.New("metadata_filters 的值只能是字符串、数字或布尔值")
		}
	}
	return nil
}

// safeMetadataKey 判断元数据字段名是否只包含允许字符。
func safeMetadataKey(value string) bool {
	if value == "" {
		return false
	}
	for _, char := range value {
		if (char < 'a' || char > 'z') && (char < 'A' || char > 'Z') && (char < '0' || char > '9') && char != '_' {
			return false
		}
	}
	return true
}
