// 文件用途：创建训练、验证、NAS 等计算任务，并通过 Python gRPC/Celery 派发后台执行。
package businessapi

import (
	"encoding/json"
	"strings"

	"knowledge-agent/go-business-api/internal/experimentqueue"
	"knowledge-agent/go-business-api/internal/python"

	"github.com/gin-gonic/gin"
)

// startTraining 校验资源归属、创建训练记录并派发 Celery 任务。
func (s *Server) startTraining(c *gin.Context) {
	var in map[string]any
	if c.ShouldBindJSON(&in) != nil {
		c.JSON(400, gin.H{"error": "invalid request"})
		return
	}
	dataset := number(in["dataset_id"])
	backbone := number(in["backbone_id"])
	classifier := number(in["classifier_id"])
	if dataset < 1 || backbone < 1 || classifier < 1 {
		c.JSON(400, gin.H{"error": "dataset_id, backbone_id and classifier_id are required"})
		return
	}
	task := randomID()
	config := buildConfig(in)
	var datasetName, datasetPath, modality, backboneName string
	if s.db.QueryRowContext(c, `SELECT name,path,modality FROM datasets WHERE id=? AND isDelete=0`, dataset).Scan(&datasetName, &datasetPath, &modality) != nil {
		c.JSON(400, gin.H{"error": "dataset not found"})
		return
	}
	if s.db.QueryRowContext(c, `SELECT name FROM backbones WHERE id=? AND isDelete=0`, backbone).Scan(&backboneName) != nil {
		c.JSON(400, gin.H{"error": "backbone not found"})
		return
	}
	dataConfig := config["data"].(map[string]any)
	dataConfig["name"] = datasetName
	dataConfig["root"] = datasetPath
	dataConfig["mode"] = modality
	config["model"].(map[string]any)["backbone"] = backboneName
	featureNames := []string{}
	if raw, ok := in["feature_ids"].([]any); ok && len(raw) > 0 {
		ids := make([]any, len(raw))
		for i, v := range raw {
			ids[i] = number(v)
		}
		rows, queryErr := s.db.QueryContext(c, `SELECT name FROM feature_extractors WHERE isDelete=0 AND id IN (`+placeholders(len(ids))+`)`, ids...)
		if queryErr == nil {
			defer rows.Close()
			for rows.Next() {
				var name string
				if rows.Scan(&name) == nil {
					featureNames = append(featureNames, name)
				}
			}
		}
	}
	config["feature_combinations"] = featureNames
	nodes, _ := json.Marshal(in["nodes"])
	features, _ := json.Marshal(in["feature_ids"])
	cfg, _ := json.Marshal(config)
	tx, err := s.db.BeginTx(c, nil)
	if err != nil {
		fail(c, err)
		return
	}
	defer tx.Rollback()
	res, err := tx.ExecContext(c, `INSERT INTO train_jobs(job_id,user_id,dataset_id,nodes,backbone_id,classifier_id,feature_ids,adaptive,status,progress,config,description,isDelete) VALUES(?,?,?,?,?,?,?,?, 'pending',0,?,?,0)`, task, userID(c), dataset, nodes, backbone, classifier, features, boolValue(in["is_nas"]), cfg, stringValue(in["description"]))
	if err != nil {
		fail(c, err)
		return
	}
	id, _ := res.LastInsertId()
	if err = experimentqueue.EnqueueDispatch(c, tx, python.DispatchRequest{TaskType: "training", RecordID: id, TaskID: task}); err != nil {
		fail(c, err)
		return
	}
	if err = tx.Commit(); err != nil {
		fail(c, err)
		return
	}
	c.JSON(202, gin.H{"message": "训练任务已进入可靠队列", "id": id, "job_id": task, "celery_task_id": task})
}

// startValidation 创建验证记录并派发模型验证任务。
func (s *Server) startValidation(c *gin.Context) {
	var in struct {
		TrainJobID string `json:"train_job_id" binding:"required"`
	}
	if c.ShouldBindJSON(&in) != nil {
		c.JSON(400, gin.H{"error": "train_job_id required"})
		return
	}
	var cfg []byte
	if s.db.QueryRowContext(c, `SELECT config FROM train_jobs WHERE job_id=? AND user_id=? AND isDelete=0`, in.TrainJobID, userID(c)).Scan(&cfg) != nil {
		c.JSON(404, gin.H{"error": "train job not found"})
		return
	}
	task := randomID()
	tx, err := s.db.BeginTx(c, nil)
	if err != nil {
		fail(c, err)
		return
	}
	defer tx.Rollback()
	res, err := tx.ExecContext(c, `INSERT INTO test_records(val_job_id,train_job_id,user_id,config,status,isDelete) VALUES(?,?,?,?,'pending',0)`, task, in.TrainJobID, userID(c), cfg)
	if err != nil {
		fail(c, err)
		return
	}
	id, _ := res.LastInsertId()
	if err = experimentqueue.EnqueueDispatch(c, tx, python.DispatchRequest{TaskType: "validation", RecordID: id, TaskID: task}); err != nil {
		fail(c, err)
		return
	}
	if err = tx.Commit(); err != nil {
		fail(c, err)
		return
	}
	c.JSON(202, gin.H{"message": "测试任务已进入可靠队列", "id": id, "val_job_id": task, "celery_task_id": task})
}

// startNAS 创建神经架构搜索记录并派发 NAS 后台任务。
func (s *Server) startNAS(c *gin.Context) {
	var in map[string]any
	if c.ShouldBindJSON(&in) != nil {
		c.JSON(400, gin.H{"error": "invalid request"})
		return
	}
	dataset := number(in["dataset_id"])
	modality := strings.ToLower(stringValue(in["modality"]))
	if dataset < 1 || (modality != "sar" && modality != "rd" && modality != "mon") {
		c.JSON(400, gin.H{"error": "invalid dataset or modality"})
		return
	}
	cfg, _ := json.Marshal(in["config"])
	task := randomID()
	tx, err := s.db.BeginTx(c, nil)
	if err != nil {
		fail(c, err)
		return
	}
	defer tx.Rollback()
	res, err := tx.ExecContext(c, `INSERT INTO nas_search_jobs(job_id,user_id,dataset_id,modality,description,config,status,progress,isDelete) VALUES(?,?,?,?,?,?,'pending',0,0)`, task, userID(c), dataset, modality, stringValue(in["description"]), cfg)
	if err != nil {
		fail(c, err)
		return
	}
	id, _ := res.LastInsertId()
	if err = experimentqueue.EnqueueDispatch(c, tx, python.DispatchRequest{TaskType: "nas", RecordID: id, TaskID: task}); err != nil {
		fail(c, err)
		return
	}
	if err = tx.Commit(); err != nil {
		fail(c, err)
		return
	}
	c.JSON(202, gin.H{"message": "NAS搜索任务已进入可靠队列", "id": id, "job_id": task, "celery_task_id": task})
}

// validateMany 批量校验请求引用的数据库资源是否存在且属于当前用户。
func (s *Server) validateMany(c *gin.Context) {
	var in struct {
		IDs []string `json:"job_ids" binding:"required"`
	}
	if c.ShouldBindJSON(&in) != nil || len(in.IDs) == 0 {
		c.JSON(400, gin.H{"error": "job_ids must be a non-empty list"})
		return
	}
	args := []any{userID(c)}
	for _, id := range in.IDs {
		args = append(args, id)
	}
	rows, err := s.db.QueryContext(c, `SELECT val_job_id,result FROM test_records WHERE user_id=? AND val_job_id IN (`+placeholders(len(in.IDs))+`)`, args...)
	if err != nil {
		fail(c, err)
		return
	}
	defer rows.Close()
	out := []any{}
	for rows.Next() {
		var id string
		var raw []byte
		_ = rows.Scan(&id, &raw)
		var result map[string]any
		_ = json.Unmarshal(raw, &result)
		out = append(out, gin.H{"id": id, "Metrics": result["Metrics"]})
	}
	c.JSON(200, gin.H{"results": out})
}
