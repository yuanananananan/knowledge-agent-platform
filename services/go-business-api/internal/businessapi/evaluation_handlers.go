// 文件用途：管理 Agent Golden Case、评测任务、评测结果和超时任务收敛。
package businessapi

import (
	"database/sql"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"knowledge-agent/go-business-api/internal/experimentqueue"
	"knowledge-agent/go-business-api/internal/python"

	"github.com/gin-gonic/gin"
)

// goldenCase 对应数据库中的单条版本化评测用例。
type goldenCase struct {
	CaseID            string          `json:"case_id"`
	Name              string          `json:"name"`
	Category          string          `json:"category"`
	Prompt            string          `json:"prompt"`
	RequiredKeywords  json.RawMessage `json:"required_keywords"`
	ForbiddenKeywords json.RawMessage `json:"forbidden_keywords"`
	ExpectedIntent    sql.NullString  `json:"-"`
	ExpectedTools     json.RawMessage `json:"expected_tools"`
	Enabled           bool            `json:"enabled"`
}

// evalRun 表示一次 Golden Case 批量评测的概要状态。
type evalRun struct {
	ID          string          `json:"eval_run_id"`
	Status      string          `json:"status"`
	Total       int             `json:"total"`
	Passed      int             `json:"passed"`
	Score       float64         `json:"score"`
	Metrics     json.RawMessage `json:"metrics"`
	CreatedAt   time.Time       `json:"created_at"`
	CompletedAt sql.NullTime    `json:"-"`
}

// agentGoldenCases 根据 HTTP 方法查询或创建 Golden Case。
func (s *Server) agentGoldenCases(c *gin.Context) {
	if c.Request.Method == http.MethodPost {
		s.createGoldenCase(c)
		return
	}
	ids, err := s.versionedGoldenCaseIDs()
	if err != nil {
		fail(c, err)
		return
	}
	items := make([]gin.H, 0, len(ids))
	for _, caseID := range ids {
		var item goldenCase
		err := s.db.QueryRowContext(c.Request.Context(), `SELECT case_id,name,category,prompt,required_keywords,forbidden_keywords,expected_intent,expected_tools,enabled FROM agent_golden_cases WHERE case_id=?`, caseID).Scan(
			&item.CaseID, &item.Name, &item.Category, &item.Prompt, &item.RequiredKeywords,
			&item.ForbiddenKeywords, &item.ExpectedIntent, &item.ExpectedTools, &item.Enabled,
		)
		if errors.Is(err, sql.ErrNoRows) {
			continue
		}
		if err != nil {
			fail(c, err)
			return
		}
		items = append(items, publicGoldenCase(item))
	}
	c.JSON(http.StatusOK, gin.H{"items": items})
}

// createGoldenCase 校验预期意图、关键词和工具后持久化评测用例。
func (s *Server) createGoldenCase(c *gin.Context) {
	var input struct {
		CaseID            string   `json:"case_id"`
		Name              string   `json:"name"`
		Category          string   `json:"category"`
		Prompt            string   `json:"prompt"`
		RequiredKeywords  []string `json:"required_keywords"`
		ForbiddenKeywords []string `json:"forbidden_keywords"`
		ExpectedIntent    string   `json:"expected_intent"`
		ExpectedTools     []string `json:"expected_tools"`
	}
	if c.ShouldBindJSON(&input) != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_request"})
		return
	}
	input.Prompt = strings.TrimSpace(input.Prompt)
	if input.Prompt == "" || len(input.Prompt) > 20_000 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "prompt 不能为空"})
		return
	}
	if input.CaseID == "" {
		input.CaseID = "golden_" + randomID()[:16]
	}
	if !validProposalID(input.CaseID) || len(input.ExpectedIntent) > 32 || len(input.RequiredKeywords) > 100 || len(input.ForbiddenKeywords) > 100 || len(input.ExpectedTools) > 100 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_golden_case"})
		return
	}
	if input.Name == "" {
		input.Name = "未命名样例"
	}
	if input.Category == "" {
		input.Category = "general"
	}
	if len([]rune(input.Name)) > 160 || len(input.Category) > 64 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_golden_case"})
		return
	}
	required, _ := json.Marshal(input.RequiredKeywords)
	forbidden, _ := json.Marshal(input.ForbiddenKeywords)
	tools, _ := json.Marshal(input.ExpectedTools)
	var intent any
	if strings.TrimSpace(input.ExpectedIntent) != "" {
		intent = strings.TrimSpace(input.ExpectedIntent)
	}
	_, err := s.db.ExecContext(c.Request.Context(), `INSERT INTO agent_golden_cases(case_id,name,category,prompt,required_keywords,forbidden_keywords,expected_intent,expected_tools,enabled) VALUES(?,?,?,?,?,?,?,?,1)`, input.CaseID, input.Name, input.Category, input.Prompt, required, forbidden, intent, tools)
	if err != nil {
		c.JSON(http.StatusConflict, gin.H{"error": "case_id 已存在"})
		return
	}
	c.JSON(http.StatusCreated, gin.H{"case": gin.H{
		"case_id": input.CaseID, "name": input.Name, "category": input.Category,
		"prompt": input.Prompt, "required_keywords": input.RequiredKeywords,
		"forbidden_keywords": input.ForbiddenKeywords, "expected_intent": intent,
		"expected_tools": input.ExpectedTools, "enabled": true,
	}})
}

// agentEvaluations 查询评测历史或启动新的后台评测。
func (s *Server) agentEvaluations(c *gin.Context) {
	if c.Request.Method == http.MethodPost {
		s.startAgentEvaluation(c)
		return
	}
	s.reconcileStaleEvaluations(c)
	rows, err := s.db.QueryContext(c.Request.Context(), `SELECT eval_run_id,status,total,passed,score,metrics,created_at,completed_at FROM agent_eval_runs WHERE user_id=? ORDER BY created_at DESC LIMIT 20`, userID(c))
	if err != nil {
		fail(c, err)
		return
	}
	defer rows.Close()
	items := make([]gin.H, 0)
	for rows.Next() {
		var item evalRun
		if err := rows.Scan(&item.ID, &item.Status, &item.Total, &item.Passed, &item.Score, &item.Metrics, &item.CreatedAt, &item.CompletedAt); err != nil {
			fail(c, err)
			return
		}
		items = append(items, publicEvalRun(item))
	}
	if err := rows.Err(); err != nil {
		fail(c, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{"items": items})
}

// startAgentEvaluation 创建评测记录并通过 Python/Celery 派发执行。
func (s *Server) startAgentEvaluation(c *gin.Context) {
	var input struct {
		CaseIDs []string `json:"case_ids"`
	}
	if err := c.ShouldBindJSON(&input); err != nil && !errors.Is(err, io.EOF) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_request"})
		return
	}
	caseIDs := input.CaseIDs
	if len(caseIDs) == 0 {
		var err error
		caseIDs, err = s.versionedGoldenCaseIDs()
		if err != nil {
			fail(c, err)
			return
		}
	}
	if len(caseIDs) == 0 || len(caseIDs) > 500 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "没有可运行的 Golden Dataset 样例"})
		return
	}
	seen := make(map[string]struct{}, len(caseIDs))
	cleanIDs := make([]string, 0, len(caseIDs))
	for _, caseID := range caseIDs {
		caseID = strings.TrimSpace(caseID)
		if !validProposalID(caseID) {
			c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_case_id"})
			return
		}
		if _, exists := seen[caseID]; !exists {
			seen[caseID] = struct{}{}
			cleanIDs = append(cleanIDs, caseID)
		}
	}
	args := make([]any, 0, len(cleanIDs))
	for _, caseID := range cleanIDs {
		args = append(args, caseID)
	}
	var enabled int
	query := `SELECT COUNT(*) FROM agent_golden_cases WHERE enabled=1 AND case_id IN (` + placeholders(len(cleanIDs)) + `)`
	if err := s.db.QueryRowContext(c.Request.Context(), query, args...).Scan(&enabled); err != nil {
		fail(c, err)
		return
	}
	if enabled == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "没有可运行的 Golden Dataset 样例"})
		return
	}
	evalID := "eval_" + randomID()
	metrics := []byte(`{"execution":"celery","completed_cases":0,"progress":0}`)
	tx, err := s.db.BeginTx(c.Request.Context(), nil)
	if err != nil {
		fail(c, err)
		return
	}
	defer tx.Rollback()
	if _, err := tx.ExecContext(c.Request.Context(), `INSERT INTO agent_eval_runs(eval_run_id,user_id,status,total,passed,score,metrics) VALUES(?,?,'pending',?,0,0,?)`, evalID, userID(c), enabled, metrics); err != nil {
		fail(c, err)
		return
	}
	payload, _ := json.Marshal(gin.H{"case_ids": cleanIDs})
	if err := experimentqueue.EnqueueDispatch(c.Request.Context(), tx, python.DispatchRequest{TaskType: "agent_evaluation", RecordID: userID(c), TaskID: evalID, Payload: payload}); err != nil {
		fail(c, err)
		return
	}
	if err := tx.Commit(); err != nil {
		fail(c, err)
		return
	}
	item := evalRun{ID: evalID, Status: "pending", Total: enabled, Metrics: metrics, CreatedAt: time.Now().UTC()}
	c.JSON(http.StatusAccepted, gin.H{"evaluation": publicEvalRun(item)})
}

// agentEvaluationDetail 返回一次评测的概要和逐用例结果。
func (s *Server) agentEvaluationDetail(c *gin.Context, evalID string) {
	if !validProposalID(evalID) {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_eval_run_id"})
		return
	}
	var item evalRun
	err := s.db.QueryRowContext(c.Request.Context(), `SELECT eval_run_id,status,total,passed,score,metrics,created_at,completed_at FROM agent_eval_runs WHERE eval_run_id=? AND user_id=?`, evalID, userID(c)).Scan(&item.ID, &item.Status, &item.Total, &item.Passed, &item.Score, &item.Metrics, &item.CreatedAt, &item.CompletedAt)
	if errors.Is(err, sql.ErrNoRows) {
		c.JSON(http.StatusNotFound, gin.H{"error": "评测记录不存在"})
		return
	}
	if err != nil {
		fail(c, err)
		return
	}
	rows, err := s.db.QueryContext(c.Request.Context(), `SELECT case_id,passed,score,answer,details,latency_ms FROM agent_eval_results WHERE eval_run_id=? ORDER BY id`, evalID)
	if err != nil {
		fail(c, err)
		return
	}
	results, err := scanRows(rows)
	if err != nil {
		fail(c, err)
		return
	}
	c.JSON(http.StatusOK, gin.H{"evaluation": publicEvalRun(item), "results": results})
}

// agentMonitorSummary 聚合 run 和后台任务的状态数量。
func (s *Server) agentMonitorSummary(c *gin.Context) {
	var upstream struct {
		Runs []struct {
			Status string `json:"status"`
		} `json:"runs"`
	}
	if err := s.agentControl.ListRuns(c.Request.Context(), userID(c), "", 100, &upstream); err != nil {
		s.writeControlPlaneError(c, err)
		return
	}
	runs := map[string]int{}
	for _, run := range upstream.Runs {
		runs[run.Status]++
	}
	rows, err := s.db.QueryContext(c.Request.Context(), `SELECT status,COUNT(*) FROM train_jobs WHERE user_id=? AND isDelete=0 GROUP BY status`, userID(c))
	if err != nil {
		fail(c, err)
		return
	}
	defer rows.Close()
	tasks := map[string]int{}
	for rows.Next() {
		var status string
		var count int
		if err := rows.Scan(&status, &count); err != nil {
			fail(c, err)
			return
		}
		tasks[status] = count
	}
	c.JSON(http.StatusOK, gin.H{"runs": runs, "tasks": tasks})
}

// versionedGoldenCaseIDs 返回当前启用且属于正式数据集版本的用例 ID。
func (s *Server) versionedGoldenCaseIDs() ([]string, error) {
	payload, err := os.ReadFile(filepath.Join(s.evalRoot, "golden_dataset.json"))
	if err != nil {
		return nil, err
	}
	var cases []struct {
		CaseID string `json:"case_id"`
	}
	if err := json.Unmarshal(payload, &cases); err != nil {
		return nil, err
	}
	ids := make([]string, 0, len(cases))
	for _, item := range cases {
		if item.CaseID != "" {
			ids = append(ids, item.CaseID)
		}
	}
	return ids, nil
}

// reconcileStaleEvaluations 将长时间无进展的 running 评测收敛为 failed。
func (s *Server) reconcileStaleEvaluations(c *gin.Context) {
	cutoff := time.Now().UTC().Add(-time.Hour)
	message := "评测执行进程已中断，请重新运行"
	_, _ = s.db.ExecContext(c.Request.Context(), `UPDATE agent_eval_runs e LEFT JOIN (SELECT eval_run_id,MAX(created_at) AS last_at FROM agent_eval_results GROUP BY eval_run_id) r ON r.eval_run_id=e.eval_run_id SET e.status='failed',e.metrics=JSON_SET(COALESCE(e.metrics,JSON_OBJECT()),'$.error',?),e.completed_at=UTC_TIMESTAMP(6) WHERE e.user_id=? AND e.status='running' AND COALESCE(r.last_at,e.created_at)<?`, message, userID(c), cutoff)
}

// publicGoldenCase 将内部评测用例转换为公开 JSON。
func publicGoldenCase(item goldenCase) gin.H {
	return gin.H{
		"case_id": item.CaseID, "name": item.Name, "category": item.Category,
		"prompt": item.Prompt, "required_keywords": decodeJSONArray(item.RequiredKeywords),
		"forbidden_keywords": decodeJSONArray(item.ForbiddenKeywords),
		"expected_intent":    nullableString(item.ExpectedIntent),
		"expected_tools":     decodeJSONArray(item.ExpectedTools), "enabled": item.Enabled,
	}
}

// publicEvalRun 将内部评测运行转换为公开 JSON。
func publicEvalRun(item evalRun) gin.H {
	var metrics any = map[string]any{}
	if len(item.Metrics) > 0 {
		_ = json.Unmarshal(item.Metrics, &metrics)
	}
	var completedAt any
	if item.CompletedAt.Valid {
		completedAt = item.CompletedAt.Time
	}
	return gin.H{"eval_run_id": item.ID, "status": item.Status, "total": item.Total, "passed": item.Passed, "score": item.Score, "metrics": metrics, "created_at": item.CreatedAt, "completed_at": completedAt}
}

// decodeJSONArray 安全解析 JSON 数组，空值或非法值返回空数组。
func decodeJSONArray(raw json.RawMessage) []any {
	items := make([]any, 0)
	_ = json.Unmarshal(raw, &items)
	return items
}

// nullableString 把 sql.NullString 转成 JSON 可表达的字符串或 nil。
func nullableString(value sql.NullString) any {
	if value.Valid {
		return value.String
	}
	return nil
}
