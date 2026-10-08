// 文件用途：处理分析任务的创建、查询和结果持久化相关 HTTP 请求。
package businessapi

import (
	"context"
	"encoding/json"
	"strconv"

	"github.com/gin-gonic/gin"
)

// createAnalysis 创建数据分析任务并派发后台计算。
func (s *Server) createAnalysis(c *gin.Context) {
	var in struct {
		Name      string `json:"analysis_name"`
		Baseline  string `json:"baseline_val_job_id" binding:"required"`
		Knowledge string `json:"knowledge_val_job_id" binding:"required"`
		Metric    string `json:"metric_key"`
		Remark    string `json:"remark"`
	}
	if c.ShouldBindJSON(&in) != nil {
		c.JSON(400, gin.H{"error": "invalid request"})
		return
	}
	if in.Metric == "" {
		in.Metric = "oa"
	}
	if in.Metric != "oa" && in.Metric != "f1" && in.Metric != "precision" && in.Metric != "recall" && in.Metric != "kappa" {
		c.JSON(400, gin.H{"error": "unsupported metric"})
		return
	}
	type joined struct {
		valID, trainID, backbone string
		adaptive                 bool
		result                   []byte
	}
	load := func(id string) (joined, error) {
		var row joined
		err := s.db.QueryRowContext(c, `SELECT v.val_job_id,v.train_job_id,COALESCE(b.name,''),t.adaptive,v.result FROM test_records v JOIN train_jobs t ON t.job_id=v.train_job_id LEFT JOIN backbones b ON b.id=t.backbone_id WHERE v.val_job_id=? AND v.user_id=? AND v.isDelete=0`, id, userID(c)).Scan(&row.valID, &row.trainID, &row.backbone, &row.adaptive, &row.result)
		return row, err
	}
	base, err := load(in.Baseline)
	if err != nil {
		c.JSON(404, gin.H{"error": "baseline validation not found"})
		return
	}
	knowledge, err := load(in.Knowledge)
	if err != nil {
		c.JSON(404, gin.H{"error": "knowledge validation not found"})
		return
	}
	metricRow := func(role string, row joined) gin.H {
		var result map[string]any
		_ = json.Unmarshal(row.result, &result)
		metrics, _ := result["Metrics"].(map[string]any)
		if metrics == nil {
			metrics = map[string]any{}
		}
		return gin.H{"model_id": row.trainID, "group_id": "", "role": role, "model_name": row.trainID, "backbone": row.backbone, "adaptive": row.adaptive, "train_job_id": row.trainID, "val_job_id": row.valID, "oa": metrics["OA"], "precision": metrics["Precision"], "recall": metrics["Recall"], "f1": metrics["F1"], "kappa": metrics["Kappa"]}
	}
	analysisID := "analysis_" + randomID()
	groupID := "group_" + randomID()
	rows := []gin.H{metricRow("baseline", base), metricRow("knowledge", knowledge)}
	for _, row := range rows {
		row["group_id"] = groupID
	}
	snapshot := gin.H{"group_id": groupID, "metric_key": in.Metric, "backbone": base.backbone, "rows": rows}
	snapshotJSON, _ := json.Marshal(snapshot)
	if in.Name == "" {
		in.Name = "模型有效性分析"
	}
	_, err = s.db.ExecContext(c, `INSERT INTO effectiveness_analysis(analysis_id,analysis_name,user_id,metric_key,backbone,baseline_val_job_id,knowledge_val_job_id,baseline_train_job_id,knowledge_train_job_id,status,result,remark,isDelete) VALUES(?,?,?,?,?,?,?,?,?,'completed',?,?,0)`, analysisID, in.Name, userID(c), in.Metric, base.backbone, base.valID, knowledge.valID, base.trainID, knowledge.trainID, snapshotJSON, in.Remark)
	if err != nil {
		fail(c, err)
		return
	}
	s.writeAnalysis(c, analysisID)
}

// listAnalyses 查询当前用户的分析任务列表。
func (s *Server) listAnalyses(c *gin.Context) {
	page, _ := strconv.Atoi(c.DefaultQuery("page", "1"))
	size, _ := strconv.Atoi(c.DefaultQuery("page_size", "10"))
	if page < 1 {
		page = 1
	}
	if size < 1 || size > 100 {
		size = 10
	}
	var total int
	_ = s.db.QueryRowContext(c, `SELECT COUNT(*) FROM effectiveness_analysis WHERE user_id=? AND isDelete=0`, userID(c)).Scan(&total)
	rows, err := s.db.QueryContext(c, `SELECT analysis_id,analysis_name,user_id,metric_key,backbone,baseline_val_job_id,knowledge_val_job_id,baseline_train_job_id,knowledge_train_job_id,status,result,remark,createTime,updateTime FROM effectiveness_analysis WHERE user_id=? AND isDelete=0 ORDER BY id DESC LIMIT ? OFFSET ?`, userID(c), size, (page-1)*size)
	if err != nil {
		fail(c, err)
		return
	}
	items, err := scanRows(rows)
	if err != nil {
		fail(c, err)
		return
	}
	c.JSON(200, gin.H{"code": 200, "message": "success", "data": gin.H{"items": items, "total": total, "page": page, "page_size": size}})
}

// getAnalysis 根据路由 ID 返回单个分析任务。
func (s *Server) getAnalysis(c *gin.Context) { s.writeAnalysis(c, c.Param("id")) }

// writeAnalysis 执行归属校验并序列化分析任务。
func (s *Server) writeAnalysis(c *gin.Context, id string) {
	rows, err := s.db.QueryContext(c, `SELECT analysis_id,analysis_name,user_id,metric_key,backbone,baseline_val_job_id,knowledge_val_job_id,baseline_train_job_id,knowledge_train_job_id,status,result,remark,createTime,updateTime FROM effectiveness_analysis WHERE analysis_id=? AND user_id=? AND isDelete=0`, id, userID(c))
	if err != nil {
		fail(c, err)
		return
	}
	items, err := scanRows(rows)
	if err != nil || len(items) == 0 {
		c.JSON(404, gin.H{"error": "analysis not found"})
		return
	}
	c.JSON(200, gin.H{"code": 200, "message": "success", "data": items[0]})
}

// deleteAnalysis 取消后台任务并删除当前用户拥有的分析记录。
func (s *Server) deleteAnalysis(c *gin.Context) {
	res, err := s.db.ExecContext(c, `UPDATE effectiveness_analysis SET isDelete=1 WHERE analysis_id=? AND user_id=? AND isDelete=0`, c.Param("id"), userID(c))
	if err != nil {
		fail(c, err)
		return
	}
	n, _ := res.RowsAffected()
	if n == 0 {
		c.JSON(404, gin.H{"error": "analysis not found"})
		return
	}
	c.JSON(200, gin.H{"code": 200, "message": "deleted", "data": gin.H{}})
}

// markFailed 在后台派发失败时将对应业务记录收敛为 failed。
func (s *Server) markFailed(ctx context.Context, table string, id int64, cause error) {
	raw, _ := json.Marshal(gin.H{"error": cause.Error()})
	_, _ = s.db.ExecContext(ctx, `UPDATE `+table+` SET status='failed',result=? WHERE id=?`, raw, id)
}
