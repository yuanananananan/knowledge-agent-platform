// 文件用途：实现 Agent 公开网关，包括聊天建单、SSE 流转发、断点续传、运行查询和取消。
package businessapi

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"knowledge-agent/go-business-api/internal/controlplane"

	"github.com/gin-gonic/gin"
)

// agentGateway 统一接管公开的 /api/agent/* 路由。
// Go 层负责鉴权、持久化和流程编排；只有模型推理等 AI 工作才通过私有 gRPC 交给 Python。
func (s *Server) agentGateway(c *gin.Context) {
	// path 是去掉 /api/agent 前缀后的通配符路径，例如 /chat/stream。
	path := c.Param("path")
	// 同时根据 HTTP 方法和子路径，把请求分派给具体处理函数。
	switch {
	case c.Request.Method == http.MethodPost && path == "/chat/stream":
		// 创建一次运行，并通过 SSE 持续返回增量事件。
		s.agentChatStream(c)
	case c.Request.Method == http.MethodPost && path == "/chat":
		// 创建一次运行，并在最多 60 秒内轮询等待最终答案。
		s.agentChat(c)
	case path == "/conversations" && (c.Request.Method == http.MethodGet || c.Request.Method == http.MethodPost):
		// GET 查询会话列表，POST 创建会话。
		s.agentConversations(c)
	case strings.HasPrefix(path, "/conversations/") && !strings.Contains(strings.TrimPrefix(path, "/conversations/"), "/") && (c.Request.Method == http.MethodGet || c.Request.Method == http.MethodDelete):
		// 只匹配 /conversations/{id}，额外的斜杠表示它属于更深层的子路由。
		s.agentConversation(c, strings.TrimPrefix(path, "/conversations/"))
	case c.Request.Method == http.MethodGet && strings.HasPrefix(path, "/runs/") && strings.HasSuffix(path, "/stream"):
		// 根据客户端提供的事件游标，恢复一个已经存在的 SSE 流。
		s.resumeAgentRun(c, strings.TrimSuffix(strings.TrimPrefix(path, "/runs/"), "/stream"))
	case c.Request.Method == http.MethodGet && strings.HasPrefix(path, "/runs/") && strings.HasSuffix(path, "/events"):
		// 返回一次性的运行事件快照，不保持长连接。
		s.agentRunEvents(c, strings.TrimSuffix(strings.TrimPrefix(path, "/runs/"), "/events"))
	case c.Request.Method == http.MethodDelete && strings.HasPrefix(path, "/runs/") && !strings.Contains(strings.TrimPrefix(path, "/runs/"), "/"):
		// 取消指定的后台 Agent 运行。
		s.cancelAgentRun(c, strings.TrimPrefix(path, "/runs/"))
	case c.Request.Method == http.MethodGet && strings.HasPrefix(path, "/conversations/") && strings.HasSuffix(path, "/runs"):
		// 提取 /conversations/{conversationID}/runs 中的会话 ID。
		conversationID := strings.TrimSuffix(strings.TrimPrefix(path, "/conversations/"), "/runs")
		s.agentConversationRuns(c, conversationID)
	case path == "/actions" && (c.Request.Method == http.MethodGet || c.Request.Method == http.MethodPost):
		// 查询或创建需要人工审批的 Agent 动作。
		s.agentActions(c)
	case c.Request.Method == http.MethodGet && strings.HasPrefix(path, "/actions/") && !strings.Contains(strings.TrimPrefix(path, "/actions/"), "/"):
		// 查询一个动作提案的详情。
		s.agentActionDetail(c, strings.TrimPrefix(path, "/actions/"))
	case c.Request.Method == http.MethodPost && strings.HasPrefix(path, "/actions/") && strings.HasSuffix(path, "/approve"):
		// 审批并执行一个动作提案。
		s.approveAgentAction(c, strings.TrimSuffix(strings.TrimPrefix(path, "/actions/"), "/approve"))
	case c.Request.Method == http.MethodPost && strings.HasPrefix(path, "/actions/") && strings.HasSuffix(path, "/reject"):
		// 拒绝一个动作提案。
		s.rejectAgentAction(c, strings.TrimSuffix(strings.TrimPrefix(path, "/actions/"), "/reject"))
	case path == "/golden-cases" && (c.Request.Method == http.MethodGet || c.Request.Method == http.MethodPost):
		// 查询或创建 Agent 的 Golden Case 测试用例。
		s.agentGoldenCases(c)
	case path == "/evaluations" && (c.Request.Method == http.MethodGet || c.Request.Method == http.MethodPost):
		// 查询评测记录或创建新的评测任务。
		s.agentEvaluations(c)
	case c.Request.Method == http.MethodGet && strings.HasPrefix(path, "/evaluations/") && !strings.Contains(strings.TrimPrefix(path, "/evaluations/"), "/"):
		// 查询一个评测任务的详情。
		s.agentEvaluationDetail(c, strings.TrimPrefix(path, "/evaluations/"))
	case c.Request.Method == http.MethodGet && path == "/monitor/summary":
		// 返回 Agent 运行和后台任务的监控汇总。
		s.agentMonitorSummary(c)
	case c.Request.Method == http.MethodGet && path == "/knowledge/status":
		// 查询知识库状态。
		s.agentKnowledge(c, "status")
	case c.Request.Method == http.MethodPost && path == "/knowledge/search":
		// 执行知识库检索。
		s.agentKnowledge(c, "search")
	case c.Request.Method == http.MethodPost && path == "/knowledge/reindex":
		// 重建知识库索引。
		s.agentKnowledge(c, "rebuild")
	case c.Request.Method == http.MethodGet && path == "/capabilities":
		// 返回 Agent 当前支持的能力列表。
		s.agentCapabilities(c)
	case c.Request.Method == http.MethodPost && path == "/insights/compare":
		// 比较多个实验的指标和结果。
		s.compareAgentExperiments(c)
	case c.Request.Method == http.MethodGet && strings.HasPrefix(path, "/insights/analyze/"):
		// 分析指定实验，并从路径末尾提取实验 ID。
		s.analyzeAgentExperiment(c, strings.TrimPrefix(path, "/insights/analyze/"))
	case c.Request.Method == http.MethodGet && strings.HasPrefix(path, "/datasets/") && strings.HasSuffix(path, "/quality"):
		// 检查指定数据集的质量。
		s.agentDatasetQuality(c, strings.TrimSuffix(strings.TrimPrefix(path, "/datasets/"), "/quality"))
	case c.Request.Method == http.MethodGet && path == "/monitor/security":
		// 返回 Agent 安全相关的监控信息。
		s.agentSecurityMonitor(c)
	default:
		// 没有任何规则匹配时，返回统一的 404 响应。
		c.JSON(http.StatusNotFound, gin.H{"error": "agent_endpoint_not_found"})
	}
}

// agentChat 提供非流式聊天接口：先创建后台 run，再短轮询等待最终结果。
func (s *Server) agentChat(c *gin.Context) {
	// 使用匿名结构体约束请求 JSON，只接收消息和可选会话 ID。
	var input struct {
		Message        string `json:"message"`
		ConversationID string `json:"conversation_id"`
	}
	// JSON 缺失、类型错误或格式非法时直接拒绝请求。
	if c.ShouldBindJSON(&input) != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_request"})
		return
	}
	// 去除消息两端空白，避免把纯空格当成有效问题。
	input.Message = strings.TrimSpace(input.Message)
	// 限制输入和会话 ID 的大小，避免异常大请求进入后台执行链路。
	if input.Message == "" || len(input.Message) > 20_000 || len(input.ConversationID) > 64 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "message 不能为空且不能超过 20000 个字符"})
		return
	}
	// 把已认证用户 ID 和请求内容交给 Control Plane，创建持久化运行记录。
	run, err := s.agentControl.CreateRun(c.Request.Context(), userID(c), input.Message, input.ConversationID)
	if err != nil {
		s.writeControlPlaneError(c, err)
		return
	}
	// 非流式接口最多同步等待 60 秒，每 500 毫秒查询一次运行状态。
	deadline := time.NewTimer(60 * time.Second)
	ticker := time.NewTicker(500 * time.Millisecond)
	// 无论函数从哪个分支退出，都释放底层定时器资源。
	defer deadline.Stop()
	defer ticker.Stop()
	for {
		select {
		case <-c.Request.Context().Done():
			// 浏览器断开或请求被取消后，不再继续轮询。
			return
		case <-deadline.C:
			// 60 秒内没有完成并不代表失败：返回 202，让客户端后续按 run ID 查询。
			c.JSON(http.StatusAccepted, gin.H{"run_id": run.ID, "conversation_id": run.ConversationID, "status": run.Status})
			return
		case <-ticker.C:
			// 从 Control Plane 获取持久化的最新状态，而不是依赖当前进程内存。
			run, err = s.agentControl.GetRun(c.Request.Context(), userID(c), run.ID)
			if err != nil {
				s.writeControlPlaneError(c, err)
				return
			}
			switch run.Status {
			case "succeeded":
				// 成功后返回最终答案。
				c.JSON(http.StatusOK, gin.H{"answer": run.Answer, "conversation_id": run.ConversationID, "run_id": run.ID})
				return
			case "failed", "cancelled":
				// 执行失败或被取消时，把运行错误返回给客户端。
				c.JSON(http.StatusBadGateway, gin.H{"error": run.Error, "run_id": run.ID})
				return
			}
		}
	}
}

// agentChatStream 创建后台 run，并通过 SSE 把持久化事件实时转发给浏览器。
func (s *Server) agentChatStream(c *gin.Context) {
	var input struct {
		Message        string `json:"message"`
		ConversationID string `json:"conversation_id"`
	}
	// 解析并校验输入，规则与非流式接口保持一致。
	if c.ShouldBindJSON(&input) != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_request"})
		return
	}
	input.Message = strings.TrimSpace(input.Message)
	if input.Message == "" || len(input.Message) > 20_000 || len(input.ConversationID) > 64 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "message 不能为空且不能超过 20000 个字符"})
		return
	}
	run, err := s.agentControl.CreateRun(c.Request.Context(), userID(c), input.Message, input.ConversationID)
	if err != nil {
		s.writeControlPlaneError(c, err)
		return
	}

	// 先切换为 SSE 响应，再发送一个网关生成的 sequence=0 接受事件。
	// 这个事件让前端尽早获得 run ID，后续可使用该 ID 进行断点恢复或取消。
	sseHeaders(c)
	writeSSE(c, 0, "run.accepted", gin.H{
		"run_id": run.ID, "sequence": 0, "type": "run.accepted",
		"payload": gin.H{"conversation_id": run.ConversationID},
	})
	// 立即刷新响应缓冲区，避免 run.accepted 被代理或服务器暂存。
	c.Writer.Flush()
	// 从持久化事件序号 0 之后开始转发该 run 的事件。
	s.relayAgentRun(c, run.ID, 0)
}

// resumeAgentRun 根据客户端最后收到的事件序号恢复 SSE，不重新创建或执行 run。
func (s *Server) resumeAgentRun(c *gin.Context, runID string) {
	if runID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_run_id"})
		return
	}
	// after 表示客户端已经确认收到的最后一个 sequence；默认从头读取。
	after, err := strconv.ParseInt(c.DefaultQuery("after", "0"), 10, 64)
	if err != nil || after < 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_event_cursor"})
		return
	}
	// 只转发 sequence > after 的事件，防止重连后重复展示旧增量。
	sseHeaders(c)
	s.relayAgentRun(c, runID, after)
}

// relayAgentRun 把 Control Plane 的内部 SSE 原样中继给浏览器，并维护断点游标。
func (s *Server) relayAgentRun(c *gin.Context, runID string, after int64) {
	// 上游会定期主动关闭 SSE；只要 run 未结束，就携带最新游标重新连接。
	for {
		// 使用当前用户身份和 after 游标打开 Control Plane 的内部事件流。
		response, err := s.agentControl.OpenEventStream(c.Request.Context(), userID(c), runID, after)
		if err != nil {
			writeSSEFailure(c, runID, err)
			return
		}
		// SSE 是按行组织的文本协议，Scanner 适合逐行读取并原样转发。
		scanner := bufio.NewScanner(response.Body)
		// 初始缓冲区 16 KiB，并允许单行事件最大扩展到 1 MiB。
		scanner.Buffer(make([]byte, 16*1024), 1<<20)
		for scanner.Scan() {
			line := scanner.Text()
			// 从上游的 id: 行提取持久化事件 sequence，作为下一次重连游标。
			if strings.HasPrefix(line, "id:") {
				if sequence, parseErr := strconv.ParseInt(strings.TrimSpace(strings.TrimPrefix(line, "id:")), 10, 64); parseErr == nil {
					// 游标只能向前推进，防止异常或重复事件导致游标回退。
					after = max(after, sequence)
				}
			}
			// 保持上游 SSE 文本不变，将当前行写入浏览器响应。
			_, _ = io.WriteString(c.Writer, line+"\n")
			if line == "" {
				// SSE 使用空行标记一个事件结束；此时立即 Flush，降低展示延迟。
				c.Writer.Flush()
			}
		}
		// 一轮上游 SSE 结束后及时关闭 Body，释放 HTTP 连接资源。
		closeErr := response.Body.Close()
		// 如果读取异常且不是浏览器主动断开，把网关错误转换成 SSE 失败事件。
		if scanErr := scanner.Err(); scanErr != nil && c.Request.Context().Err() == nil {
			writeSSEFailure(c, runID, scanErr)
			return
		}
		// 同样处理关闭响应体时发生的异常。
		if closeErr != nil && c.Request.Context().Err() == nil {
			writeSSEFailure(c, runID, closeErr)
			return
		}
		// 浏览器已经断开时无需查询状态或重新连接上游。
		if c.Request.Context().Err() != nil {
			return
		}
		// 上游正常结束后查询 run 状态，判断是终态关闭还是需要续接。
		run, err := s.agentControl.GetRun(c.Request.Context(), userID(c), runID)
		if err != nil {
			writeSSEFailure(c, runID, err)
			return
		}
		// 终态 run 不会再产生新事件，中继到此结束。
		if run.Status == "succeeded" || run.Status == "failed" || run.Status == "cancelled" {
			return
		}
		// 非终态会回到循环顶部，以最新 after 游标重新建立 SSE。
	}
}

// agentRunEvents 返回某个游标之后的事件快照，适合页面初始化和历史查看。
func (s *Server) agentRunEvents(c *gin.Context, runID string) {
	after, err := strconv.ParseInt(c.DefaultQuery("after", "0"), 10, 64)
	if err != nil || after < 0 || runID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_request"})
		return
	}
	// Control Plane 的响应结构包含 run 当前状态和事件数组，因此使用通用 map 透传。
	var result map[string]any
	if err := s.agentControl.EventSnapshot(c.Request.Context(), userID(c), runID, after, &result); err != nil {
		s.writeControlPlaneError(c, err)
		return
	}
	c.JSON(http.StatusOK, result)
}

// agentConversationRuns 查询一个会话最近的运行记录。
func (s *Server) agentConversationRuns(c *gin.Context, conversationID string) {
	if conversationID == "" || len(conversationID) > 64 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_conversation_id"})
		return
	}
	var result map[string]any
	// 当前固定查询最近 20 条，由 Control Plane 负责用户隔离和数据读取。
	if err := s.agentControl.ListRuns(c.Request.Context(), userID(c), conversationID, 20, &result); err != nil {
		s.writeControlPlaneError(c, err)
		return
	}
	// 绕过 Flask 后仍保持旧公开接口的 {"items": [...]} 响应结构。
	if runs, ok := result["runs"]; ok {
		c.JSON(http.StatusOK, gin.H{"items": runs})
		return
	}
	c.JSON(http.StatusOK, gin.H{"items": []any{}})
}

// cancelAgentRun 请求 Control Plane 取消指定 run，并返回取消后的状态。
func (s *Server) cancelAgentRun(c *gin.Context, runID string) {
	if runID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_run_id"})
		return
	}
	var result map[string]any
	if err := s.agentControl.CancelRun(c.Request.Context(), userID(c), runID, &result); err != nil {
		s.writeControlPlaneError(c, err)
		return
	}
	c.JSON(http.StatusOK, result)
}

// writeControlPlaneError 将内部 Control Plane 错误映射成公开 API 响应。
func (s *Server) writeControlPlaneError(c *gin.Context, err error) {
	var upstream *controlplane.HTTPError
	// 上游明确返回了 HTTP 状态码时，保留其状态码和 JSON 响应体。
	if errors.As(err, &upstream) {
		contentType := "application/json; charset=utf-8"
		c.Data(upstream.Status, contentType, upstream.Body)
		return
	}
	// 网络中断、连接失败等无法映射的依赖错误统一返回 502。
	c.JSON(http.StatusBadGateway, gin.H{"error": "control_plane_unavailable"})
}

// sseHeaders 设置流式响应必须的响应头。
func sseHeaders(c *gin.Context) {
	// 声明 Server-Sent Events MIME 类型。
	c.Header("Content-Type", "text/event-stream")
	// 禁止客户端和中间代理缓存事件流。
	c.Header("Cache-Control", "no-cache")
	// 禁止 Nginx 聚合响应，确保事件能够及时到达浏览器。
	c.Header("X-Accel-Buffering", "no")
}

// writeSSE 按 SSE 文本协议写入一个完整事件；调用方决定何时 Flush。
func writeSSE(c *gin.Context, sequence int64, eventType string, payload any) {
	// data 字段必须是单行可传输的 JSON。
	encoded, _ := json.Marshal(payload)
	// 空行表示当前 SSE 事件结束。
	_, _ = fmt.Fprintf(c.Writer, "id: %d\nevent: %s\ndata: %s\n\n", sequence, eventType, encoded)
}

// writeSSEFailure 把网关或上游流错误包装成前端认识的 run.failed 事件。
func writeSSEFailure(c *gin.Context, runID string, err error) {
	// sequence=-1 表示这是网关临时生成的错误，而不是数据库中的持久化事件序号。
	writeSSE(c, -1, "run.failed", gin.H{
		"run_id": runID, "sequence": -1, "type": "run.failed",
		"payload": gin.H{"message": "Go control-plane 事件流不可用", "error": err.Error()},
	})
	// 错误需要立刻送达浏览器，不能继续留在响应缓冲区中。
	c.Writer.Flush()
}
