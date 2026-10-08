// 文件用途：定义 Business API Server、依赖接口、中间件和全部公开 HTTP 路由。
package businessapi

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"strings"
	"time"

	"knowledge-agent/go-business-api/internal/controlplane"
	"knowledge-agent/go-business-api/internal/python"

	"github.com/gin-gonic/gin"
)

// sessionCookie 是浏览器保存登录会话令牌的 Cookie 名称。
const sessionCookie = "knowledge_session"

// Compute 抽象 Python 计算服务，便于 Server 解耦具体 gRPC 客户端并进行测试替换。
type Compute interface {
	DispatchCompute(context.Context, python.DispatchRequest) error
	CancelCompute(context.Context, string, bool) error
	Knowledge(context.Context, string, json.RawMessage) (json.RawMessage, error)
	ValidateAction(context.Context, int64, string, json.RawMessage) (python.ValidatedAction, error)
	RenderImage(context.Context, string) (python.RenderedImage, error)
}

// AgentControl 抽象 Control Plane 的 run、事件和动作提案能力。
type AgentControl interface {
	CreateRun(context.Context, int64, string, string) (controlplane.Run, error)
	GetRun(context.Context, int64, string) (controlplane.Run, error)
	CancelRun(context.Context, int64, string, any) error
	ListRuns(context.Context, int64, string, int, any) error
	EventSnapshot(context.Context, int64, string, int64, any) error
	OpenEventStream(context.Context, int64, string, int64) (*http.Response, error)
	ListActions(context.Context, int64, int, any) error
	CreateAction(context.Context, int64, any, any) error
	GetAction(context.Context, int64, string) (controlplane.Action, error)
	ApproveAction(context.Context, int64, string) (controlplane.Action, error)
	RejectAction(context.Context, int64, string) error
}

// Server 保存 Business API 所需的数据库、下游客户端和文件目录依赖。
type Server struct {
	db            *sql.DB
	compute       Compute
	agentControl  AgentControl
	dataRoot      string
	logRoot       string
	configRoot    string
	evalRoot      string
	sessionTTL    time.Duration
	sessionSecure bool
	// publicRegistration 控制匿名用户是否可以自行注册；生产环境应关闭。
	publicRegistration bool
}

// catalog 描述一种通用目录资源的表名和允许写入字段。
type catalog struct {
	table, singular string
	columns         []string
	adminWrite      bool
}

// New 校验依赖和根目录后创建 Business API Server。
func New(db *sql.DB, compute Compute, agentControl AgentControl, dataRoot, logRoot, configRoot, evalRoot string) (*Server, error) {
	if db == nil || compute == nil || agentControl == nil {
		return nil, errors.New("database, compute client and Agent control client are required")
	}
	return &Server{
		db: db, compute: compute, agentControl: agentControl,
		dataRoot: dataRoot, logRoot: logRoot, configRoot: configRoot, evalRoot: evalRoot,
		sessionTTL: 24 * time.Hour,
		sessionSecure: strings.EqualFold(os.Getenv("SESSION_COOKIE_SECURE"), "true"),
		publicRegistration: !strings.EqualFold(os.Getenv("ALLOW_PUBLIC_REGISTRATION"), "false"),
	}, nil
}

// Migrate 创建 Business API 自有表，并补齐旧业务表所需字段和索引。
func (s *Server) Migrate(ctx context.Context) error {
	statements := []string{`CREATE TABLE IF NOT EXISTS business_sessions (
session_id CHAR(64) PRIMARY KEY, user_id BIGINT NOT NULL, user_role VARCHAR(32) NOT NULL,
expires_at DATETIME(6) NOT NULL, created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
KEY ix_business_sessions_expiry(expires_at), KEY ix_business_sessions_user(user_id))`,
		`CREATE TABLE IF NOT EXISTS agent_conversations (
id BIGINT AUTO_INCREMENT PRIMARY KEY, conversation_id VARCHAR(64) NOT NULL UNIQUE,
user_id BIGINT NOT NULL, title VARCHAR(160) NOT NULL DEFAULT '新对话',
checkpoint_version INT NOT NULL DEFAULT 0,
created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
updated_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
KEY ix_agent_conversations_user(user_id), KEY ix_agent_conversations_updated(updated_at))`,
		`CREATE TABLE IF NOT EXISTS agent_checkpoints (
id BIGINT AUTO_INCREMENT PRIMARY KEY, checkpoint_id VARCHAR(64) NOT NULL UNIQUE,
conversation_id VARCHAR(64) NOT NULL, user_id BIGINT NOT NULL, version INT NOT NULL,
state JSON NOT NULL, created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
UNIQUE KEY uq_agent_checkpoint_version(conversation_id,version),
KEY ix_agent_checkpoints_user(user_id), KEY ix_agent_checkpoints_conversation(conversation_id))`,
		`CREATE TABLE IF NOT EXISTS agent_golden_cases (
id BIGINT AUTO_INCREMENT PRIMARY KEY, case_id VARCHAR(64) NOT NULL UNIQUE,
name VARCHAR(160) NOT NULL, category VARCHAR(64) NOT NULL, prompt TEXT NOT NULL,
required_keywords JSON NOT NULL, forbidden_keywords JSON NOT NULL,
expected_intent VARCHAR(32) NULL, expected_tools JSON NOT NULL,
enabled BOOLEAN NOT NULL DEFAULT TRUE,
created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
KEY ix_agent_golden_cases_category(category))`,
		`CREATE TABLE IF NOT EXISTS agent_eval_runs (
id BIGINT AUTO_INCREMENT PRIMARY KEY, eval_run_id VARCHAR(64) NOT NULL UNIQUE,
user_id BIGINT NOT NULL, status VARCHAR(24) NOT NULL DEFAULT 'running',
total INT NOT NULL DEFAULT 0, passed INT NOT NULL DEFAULT 0,
score DOUBLE NOT NULL DEFAULT 0, metrics JSON NOT NULL,
created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6), completed_at DATETIME(6) NULL,
KEY ix_agent_eval_runs_user(user_id), KEY ix_agent_eval_runs_created(created_at))`,
		`CREATE TABLE IF NOT EXISTS agent_eval_results (
id BIGINT AUTO_INCREMENT PRIMARY KEY, eval_run_id VARCHAR(64) NOT NULL,
case_id VARCHAR(64) NOT NULL, passed BOOLEAN NOT NULL, score DOUBLE NOT NULL,
answer TEXT NULL, details JSON NOT NULL, latency_ms INT NOT NULL DEFAULT 0,
created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
KEY ix_agent_eval_results_run(eval_run_id), KEY ix_agent_eval_results_case(case_id))`,
	}
	for _, statement := range statements {
		if _, err := s.db.ExecContext(ctx, statement); err != nil {
			return err
		}
	}
	return nil
}

// Handler 注册鉴权、CORS 和全部业务路由，返回可交给 http.Server 的处理器。
func (s *Server) Handler(allowedOrigin string) http.Handler {
	gin.SetMode(gin.ReleaseMode)
	r := gin.New()
	r.Use(gin.Recovery(), s.cors(allowedOrigin))
	r.GET("/health", func(c *gin.Context) { c.JSON(http.StatusOK, gin.H{"status": "ok", "service": "go-business-api"}) })
	r.GET("/readyz", func(c *gin.Context) {
		ctx, cancel := context.WithTimeout(c.Request.Context(), 2*time.Second)
		defer cancel()
		if err := s.db.PingContext(ctx); err != nil {
			c.JSON(http.StatusServiceUnavailable, gin.H{"status": "not_ready"})
			return
		}
		c.JSON(http.StatusOK, gin.H{"status": "ready"})
	})
	r.POST("/api/user/register", s.publicRegister)
	r.POST("/api/user/login", s.login)
	auth := r.Group("/api", s.authenticate)
	auth.POST("/user/logout", s.logout)
	auth.POST("/register", s.requireRole("admin"), s.createUser)
	auth.GET("/user", s.requireRole("admin"), s.listUsers)
	auth.GET("/user/", s.requireRole("admin"), s.listUsers)
	auth.POST("/user/", s.requireRole("admin"), s.createUser)
	auth.PUT("/user/:id", s.requireRole("admin"), s.updateUser)
	auth.DELETE("/user/:id", s.requireRole("admin"), s.softDelete("user"))

	s.registerCatalog(auth, "/datasets", catalog{"datasets", "Dataset", []string{"name", "modality", "path", "labels"}, false})
	auth.POST("/datasets/upload", s.uploadDataset)
	s.registerCatalog(auth, "/backbone", catalog{"backbones", "Backbone", []string{"name", "modality", "backbone_type", "params_schema", "artifact_path", "description"}, true})
	s.registerCatalog(auth, "/classifier", catalog{"classifiers", "Classifier", []string{"name", "params_schema", "description"}, true})
	s.registerCatalog(auth, "/feature_extractor", catalog{"feature_extractors", "FeatureExtractor", []string{"name", "modality", "params_schema", "description"}, true})
	auth.POST("/feature_extractor/names", s.featureNames)
	auth.POST("/feature_extractor/upload_json", s.requireRole("admin"), s.rejectDynamicFeatureCode)
	auth.GET("/feature_extractor/group/", s.listFeatureGroups)
	auth.GET("/feature_extractor/group/:id", s.getFeatureGroup)
	auth.GET("/feature_extractor/group/:id/", s.getFeatureGroup)
	auth.POST("/feature_extractor/group/", s.requireRole("admin"), s.createFeatureGroup)
	auth.PUT("/feature_extractor/group/:id", s.requireRole("admin"), s.updateFeatureGroup)
	auth.DELETE("/feature_extractor/group/:id", s.requireRole("admin"), s.softDelete("feature_groups"))
	auth.GET("/images/:id", s.listImages)
	auth.GET("/images/file", s.imageFile)
	auth.POST("/images/", s.requireRole("admin"), s.createCatalog(catalog{"images", "Image", []string{"dataset_id", "name", "modality", "path", "label"}, true}))
	auth.PUT("/images/:id", s.requireRole("admin"), s.updateCatalog(catalog{"images", "Image", []string{"dataset_id", "name", "modality", "path", "label"}, true}))
	auth.DELETE("/images/:id", s.requireRole("admin"), s.softDelete("images"))
	auth.GET("/config/schema", s.configSchema)

	auth.GET("/train/", s.listJobs("train_jobs"))
	auth.GET("/train/status/:id", s.jobStatus("train_jobs", "job_id"))
	auth.GET("/train/result/:id", s.jobField("train_jobs", "job_id", "result"))
	auth.GET("/train/config/:id", s.jobField("train_jobs", "job_id", "config"))
	auth.GET("/train/logs/:id", s.streamJobLog("train_jobs", "job_id"))
	auth.GET("/train/logs_file/:id", s.readJobLog("train_jobs", "job_id"))
	auth.POST("/train/start", s.startTraining)
	auth.DELETE("/train/:id", s.deleteJob("train_jobs", "job_id", "job_id"))
	auth.GET("/val/", s.listJobs("test_records"))
	auth.GET("/val/status/:id", s.validationStatus)
	auth.GET("/val/result/:id", s.jobField("test_records", "val_job_id", "result"))
	auth.GET("/val/logs_file/:id", s.readJobLog("test_records", "val_job_id"))
	auth.POST("/val/start", s.startValidation)
	auth.POST("/val/validate", s.validateMany)
	auth.DELETE("/val/:id", s.deleteJob("test_records", "val_job_id", "val_job_id"))
	auth.GET("/nas/search", s.listJobs("nas_search_jobs"))
	auth.GET("/nas/search/status/:id", s.jobStatus("nas_search_jobs", "job_id"))
	auth.GET("/nas/search/result/:id", s.jobRow("nas_search_jobs", "job_id"))
	auth.GET("/nas/logs/:id", s.readJobLog("nas_search_jobs", "job_id"))
	auth.GET("/nas/logs_file/:id", s.readJobLog("nas_search_jobs", "job_id"))
	auth.POST("/nas/search/start", s.startNAS)
	auth.DELETE("/nas/search/:id", s.deleteJob("nas_search_jobs", "id", "job_id"))
	auth.POST("/analysis/effectiveness/start", s.createAnalysis)
	auth.GET("/analysis/effectiveness", s.listAnalyses)
	auth.GET("/analysis/effectiveness/:id", s.getAnalysis)
	auth.DELETE("/analysis/effectiveness/:id", s.deleteAnalysis)
	auth.Any("/agent/*path", s.agentGateway)
	return r
}

// rejectDynamicFeatureCode keeps the legacy route stable without accepting
// arbitrary Python source code. Algorithms must be reviewed, packaged and
// deployed with the Python AI service; Go only manages their metadata.
