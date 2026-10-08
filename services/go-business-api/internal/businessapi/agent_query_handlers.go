// 文件用途：提供 Agent 会话、运行轨迹、监控、安全信息和能力清单等查询接口。
package businessapi

import (
	"database/sql"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
)

// agentCapabilities 返回前端可展示的 Agent 能力和安全边界。
func (s *Server) agentCapabilities(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"observation": []string{"datasets", "model_resources", "train_jobs", "nas_jobs", "job_logs", "experiment_lineage"},
		"analysis":    []string{"log_diagnosis", "config_diff", "metric_comparison", "model_ranking", "significance_analysis"},
		"planning":    []string{"resource_estimation", "next_experiment_recommendation", "rag_supported_experiment_design"},
		"knowledge":   []string{"hybrid_retrieval", "source_citations", "version_filtering"},
		"actions":     []string{"clone_train_job", "create_train_job", "retry_train_job", "start_validation", "stop_train_job"},
		"action_flow": []string{"draft", "approve", "execute", "verify"},
		"runtime":     []string{"durable_checkpoint", "sse_trace", "golden_dataset_eval"},
	})
}

// compareAgentExperiments 校验任务归属并比较多个训练实验的指标。
func (s *Server) compareAgentExperiments(c *gin.Context) {
	var input struct {
		JobIDs []string `json:"job_ids"`
	}
	if c.ShouldBindJSON(&input) != nil || len(input.JobIDs) < 2 || len(input.JobIDs) > 100 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "至少选择两个属于当前用户的训练任务"})
		return
	}
	args := []any{userID(c)}
	for _, id := range input.JobIDs {
		args = append(args, id)
	}
	rows, err := s.db.QueryContext(c.Request.Context(), `SELECT job_id,COALESCE(description,''),status,result FROM train_jobs WHERE user_id=? AND isDelete=0 AND job_id IN (`+placeholders(len(input.JobIDs))+`)`, args...)
	if err != nil {
		fail(c, err)
		return
	}
	defer rows.Close()
	items := make([]gin.H, 0)
	metricSet := map[string]struct{}{}
	for rows.Next() {
		var id, description, status string
		var result []byte
		if err := rows.Scan(&id, &description, &status, &result); err != nil {
			fail(c, err)
			return
		}
		metrics := resultMetrics(result)
		for key := range metrics {
			metricSet[key] = struct{}{}
		}
		items = append(items, gin.H{"job_id": id, "description": description, "status": status, "metrics": metrics})
	}
	if len(items) < 2 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "至少选择两个属于当前用户的训练任务"})
		return
	}
	metrics := make([]string, 0, len(metricSet))
	for key := range metricSet {
		metrics = append(metrics, key)
	}
	sort.Strings(metrics)
	c.JSON(http.StatusOK, gin.H{"items": items, "metrics": metrics})
}

// analyzeAgentExperiment 汇总单个训练任务的结果并生成规则化建议。
func (s *Server) analyzeAgentExperiment(c *gin.Context, jobID string) {
	if jobID == "" || len(jobID) > 128 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_job_id"})
		return
	}
	var result []byte
	err := s.db.QueryRowContext(c.Request.Context(), `SELECT result FROM train_jobs WHERE job_id=? AND user_id=? AND isDelete=0`, jobID, userID(c)).Scan(&result)
	if errors.Is(err, sql.ErrNoRows) {
		c.JSON(http.StatusNotFound, gin.H{"error": "训练任务不存在"})
		return
	}
	if err != nil {
		fail(c, err)
		return
	}
	metrics := resultMetrics(result)
	suggestions := make([]string, 0, 2)
	if value, ok := metrics["loss"]; ok && value > 1 {
		suggestions = append(suggestions, "损失较高，建议降低学习率或增加训练轮数")
	}
	if value, ok := metrics["oa"]; ok && value < 70 {
		suggestions = append(suggestions, "总体精度偏低，建议检查类别均衡和数据质量")
	}
	if len(suggestions) == 0 {
		suggestions = append(suggestions, "当前指标未发现明显异常，可作为后续实验基线")
	}
	c.JSON(http.StatusOK, gin.H{"job_id": jobID, "metrics": metrics, "summary": strings.Join(suggestions, "；"), "suggestions": suggestions})
}

// agentDatasetQuality 检查数据集目录、类别和划分是否满足训练要求。
func (s *Server) agentDatasetQuality(c *gin.Context, rawID string) {
	datasetID, err := strconv.ParseInt(rawID, 10, 64)
	if err != nil || datasetID <= 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid_dataset_id"})
		return
	}
	var name, storedPath string
	err = s.db.QueryRowContext(c.Request.Context(), `SELECT name,path FROM datasets WHERE id=? AND isDelete=0`, datasetID).Scan(&name, &storedPath)
	if errors.Is(err, sql.ErrNoRows) {
		c.JSON(http.StatusNotFound, gin.H{"error": "数据集不存在"})
		return
	}
	if err != nil {
		fail(c, err)
		return
	}
	root := storedPath
	if !filepath.IsAbs(root) {
		root, err = safePath(s.dataRoot, strings.TrimPrefix(filepath.ToSlash(root), "data/"))
		if err != nil {
			c.JSON(http.StatusForbidden, gin.H{"error": "invalid_dataset_path"})
			return
		}
	}
	issues := make([]string, 0)
	splits := gin.H{}
	classSets := map[string]map[string]struct{}{}
	for _, split := range []string{"train", "test"} {
		counts := map[string]int{}
		classSets[split] = map[string]struct{}{}
		entries, readErr := os.ReadDir(filepath.Join(root, split))
		if readErr != nil {
			issues = append(issues, "缺少 "+split+" 目录")
		} else {
			for _, entry := range entries {
				if !entry.IsDir() {
					continue
				}
				count := 0
				_ = filepath.WalkDir(filepath.Join(root, split, entry.Name()), func(_ string, item os.DirEntry, walkErr error) error {
					if walkErr == nil && !item.IsDir() {
						count++
					}
					return nil
				})
				counts[entry.Name()] = count
				classSets[split][entry.Name()] = struct{}{}
			}
		}
		total := 0
		for _, count := range counts {
			total += count
		}
		if len(counts) == 0 {
			issues = append(issues, split+" 分区没有样本")
		}
		splits[split] = gin.H{"classes": counts, "total": total}
	}
	if len(classSets["train"]) > 0 && len(classSets["test"]) > 0 && !sameStringSet(classSets["train"], classSets["test"]) {
		issues = append(issues, "train/test 类别目录不一致")
	}
	c.JSON(http.StatusOK, gin.H{"dataset_id": datasetID, "name": name, "healthy": len(issues) == 0, "issues": issues, "splits": splits})
}

// agentSecurityMonitor 汇总审计日志中的安全事件统计。
func (s *Server) agentSecurityMonitor(c *gin.Context) {
	role := c.GetString("user_role")
	if role != "admin" && role != "administrator" {
		c.JSON(http.StatusForbidden, gin.H{"message": "Forbidden: insufficient permissions"})
		return
	}
	rows, err := s.db.QueryContext(c.Request.Context(), `SELECT id,user_id,action,resource_id,success,created_at FROM audit_logs ORDER BY id DESC LIMIT 100`)
	writeRows(c, rows, err)
}

// resultMetrics 从任务结果 JSON 中提取可比较的数值指标。
func resultMetrics(raw []byte) map[string]float64 {
	var decoded map[string]any
	if json.Unmarshal(raw, &decoded) != nil {
		return map[string]float64{}
	}
	for _, key := range []string{"Metrics", "metrics"} {
		if nested, ok := decoded[key].(map[string]any); ok {
			decoded = nested
			break
		}
	}
	metrics := map[string]float64{}
	for key, value := range decoded {
		if number, ok := value.(float64); ok {
			metrics[strings.ToLower(key)] = number
		}
	}
	return metrics
}

// sameStringSet 判断两个字符串集合是否完全相同。
func sameStringSet(left, right map[string]struct{}) bool {
	if len(left) != len(right) {
		return false
	}
	for value := range left {
		if _, ok := right[value]; !ok {
			return false
		}
	}
	return true
}
