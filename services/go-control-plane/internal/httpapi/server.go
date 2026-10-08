// 文件用途：定义 Control Plane HTTP Server、HMAC 身份认证中间件、路由和请求指标记录。
package httpapi

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"log/slog"
	"net/http"
	"strconv"
	"strings"
	"time"

	"knowledge-agent/go-control-plane/internal/service"
	"knowledge-agent/go-control-plane/internal/telemetry"
)

// Server 保存 HTTP 路由、Service、认证密钥、日志和指标依赖。
type Server struct {
	svc     *service.Service
	secret  []byte
	log     *slog.Logger
	metrics *telemetry.Metrics
	mux     *http.ServeMux
}

// New 创建 Server 并注册全部路由。
func New(svc *service.Service, secret string, log *slog.Logger, metrics *telemetry.Metrics) *Server {
	s := &Server{svc: svc, secret: []byte(secret), log: log, metrics: metrics, mux: http.NewServeMux()}
	s.routes()
	return s
}

// Handler 返回套有认证、恢复和指标中间件的根处理器。
func (s *Server) Handler() http.Handler { return s.middleware(s.mux) }

// routes 注册健康检查、指标、run 和 action 的 HTTP 路由。
func (s *Server) routes() {
	s.mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	s.mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	s.mux.HandleFunc("GET /readyz", s.ready)
	s.mux.Handle("GET /metrics", s.metrics.Handler())
	s.mux.HandleFunc("POST /v1/runs", s.createRun)
	s.mux.HandleFunc("GET /v1/runs", s.listRuns)
	s.mux.HandleFunc("GET /v1/runs/{id}", s.getRun)
	s.mux.HandleFunc("DELETE /v1/runs/{id}", s.cancelRun)
	s.mux.HandleFunc("GET /v1/runs/{id}/events", s.events)
	s.mux.HandleFunc("GET /v1/runs/{id}/events/snapshot", s.eventSnapshot)
	s.mux.HandleFunc("POST /v1/actions", s.createAction)
	s.mux.HandleFunc("GET /v1/actions", s.listActions)
	s.mux.HandleFunc("GET /v1/actions/{id}", s.getAction)
	s.mux.HandleFunc("POST /v1/actions/{id}/approve", s.approveAction)
	s.mux.HandleFunc("POST /v1/actions/{id}/reject", s.rejectAction)
}

// ctxKey 使用私有类型避免用户 ID Context 键与其他包发生冲突。
type ctxKey string

// userKey 是认证后用户 ID 的 Context 键。
const userKey ctxKey = "user"

// middleware 为请求附加 request ID、恢复 panic、记录指标，并认证 /v1 路由。
func (s *Server) middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		status := http.StatusOK
		ww := &statusWriter{ResponseWriter: w, status: &status}
		requestID := r.Header.Get("X-Request-ID")
		if requestID == "" {
			requestID = strconv.FormatInt(time.Now().UnixNano(), 36)
		}
		if len(requestID) > 128 {
			requestID = requestID[:128]
		}
		ww.Header().Set("X-Request-ID", requestID)
		ww.Header().Set("X-Content-Type-Options", "nosniff")
		defer func() {
			if recovered := recover(); recovered != nil {
				status = http.StatusInternalServerError
				s.log.Error("http panic", "request_id", requestID, "panic", recovered)
				if !ww.wrote {
					writeJSON(ww, http.StatusInternalServerError, map[string]string{"error": "internal_error"})
				}
			}
			s.metrics.Observe(start, status >= 400)
			s.log.Info("http request", "request_id", requestID, "method", r.Method, "path", r.URL.Path, "status", status, "duration_ms", time.Since(start).Milliseconds())
		}()

		if strings.HasPrefix(r.URL.Path, "/v1/") {
			uid, err := s.authenticate(r)
			if err != nil {
				writeJSON(ww, http.StatusUnauthorized, map[string]string{"error": "unauthorized"})
				return
			}
			r = r.WithContext(context.WithValue(r.Context(), userKey, uid))
		}
		next.ServeHTTP(ww, r)
	})
}

// statusWriter 包装 ResponseWriter，用于记录最终 HTTP 状态并保留 Flush 能力。
type statusWriter struct {
	http.ResponseWriter
	status *int
	wrote  bool
}

// WriteHeader 记录首次写出的状态码。
func (w *statusWriter) WriteHeader(code int) {
	if w.wrote {
		return
	}
	w.wrote = true
	*w.status = code
	w.ResponseWriter.WriteHeader(code)
}

// Write 在尚未写头时隐式记录 200，再转发响应体。
func (w *statusWriter) Write(p []byte) (int, error) {
	if !w.wrote {
		w.WriteHeader(http.StatusOK)
	}
	return w.ResponseWriter.Write(p)
}

// Flush 将流式响应立即推送到底层连接。
func (w *statusWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// Authentication uses an HMAC-signed identity from the existing login gateway.
// Signature = hex(HMAC-SHA256(secret, userID+":"+unixTimestamp)).
// authenticate 校验 Business API 发送的用户 ID、时间戳和 HMAC 签名。
func (s *Server) authenticate(r *http.Request) (int64, error) {
	raw, stamp, sig := r.Header.Get("X-User-ID"), r.Header.Get("X-Auth-Timestamp"), r.Header.Get("X-Auth-Signature")
	uid, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || uid <= 0 {
		return 0, errors.New("invalid user")
	}
	ts, err := strconv.ParseInt(stamp, 10, 64)
	if err != nil || time.Since(time.Unix(ts, 0)) > 5*time.Minute || time.Until(time.Unix(ts, 0)) > time.Minute {
		return 0, errors.New("expired")
	}
	mac := hmac.New(sha256.New, s.secret)
	mac.Write([]byte(raw + ":" + stamp))
	expected := hex.EncodeToString(mac.Sum(nil))
	if !hmac.Equal([]byte(expected), []byte(sig)) {
		return 0, errors.New("signature")
	}
	return uid, nil
}

// userID 读取中间件写入请求 Context 的已认证用户 ID。
func userID(r *http.Request) int64 { return r.Context().Value(userKey).(int64) }
