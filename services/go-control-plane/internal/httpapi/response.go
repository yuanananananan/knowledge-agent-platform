// 文件用途：提供 Control Plane HTTP 层通用的 JSON 编解码、错误映射和分页参数处理。
package httpapi

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"

	"knowledge-agent/go-control-plane/internal/domain"
	"knowledge-agent/go-control-plane/internal/service"
)

// decode 严格解析大小受限的单个 JSON 对象，并拒绝尾随内容。
func decode(r *http.Request, v any) error {
	if contentType := strings.ToLower(r.Header.Get("Content-Type")); contentType != "" && !strings.HasPrefix(contentType, "application/json") {
		return errors.New("content type must be application/json")
	}
	d := json.NewDecoder(io.LimitReader(r.Body, 1<<20))
	d.DisallowUnknownFields()
	if err := d.Decode(v); err != nil {
		return err
	}
	var extra any
	if err := d.Decode(&extra); !errors.Is(err, io.EOF) {
		return errors.New("request body must contain one JSON value")
	}
	return nil
}

// writeJSON 设置安全响应头、状态码并编码 JSON。
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// bad 返回统一的无效请求响应。
func bad(w http.ResponseWriter) {
	writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid_request"})
}

// code 将领域错误映射为稳定的 HTTP 状态码。
func code(err error) int {
	if service.IsNotFound(err) {
		return http.StatusNotFound
	}
	if errors.Is(err, domain.ErrConflict) || errors.Is(err, domain.ErrInvalidState) {
		return http.StatusConflict
	}
	if errors.Is(err, domain.ErrInvalidInput) {
		return http.StatusBadRequest
	}
	if errors.Is(err, domain.ErrDependency) {
		return http.StatusBadGateway
	}
	return http.StatusInternalServerError
}

// writeError 记录服务端错误，并向客户端隐藏不应公开的内部细节。
func (s *Server) writeError(w http.ResponseWriter, err error) {
	status := code(err)
	public := "internal_error"
	switch status {
	case http.StatusBadRequest:
		public = "invalid_request"
	case http.StatusNotFound:
		public = "not_found"
	case http.StatusConflict:
		public = "conflict"
	case http.StatusBadGateway:
		public = "dependency_unavailable"
	}
	if status >= 500 {
		s.log.Error("request failed", "error", err)
	}
	writeJSON(w, status, map[string]string{"error": public})
}

// pagination 解析并校验 limit/offset 查询参数。
func pagination(r *http.Request) (int, int, error) {
	limit, offset := 20, 0
	var err error
	if raw := r.URL.Query().Get("limit"); raw != "" {
		limit, err = strconv.Atoi(raw)
		if err != nil {
			return 0, 0, fmt.Errorf("%w: invalid limit", domain.ErrInvalidInput)
		}
	}
	if raw := r.URL.Query().Get("offset"); raw != "" {
		offset, err = strconv.Atoi(raw)
		if err != nil {
			return 0, 0, fmt.Errorf("%w: invalid offset", domain.ErrInvalidInput)
		}
	}
	if limit < 1 || limit > 100 || offset < 0 {
		return 0, 0, fmt.Errorf("%w: invalid pagination", domain.ErrInvalidInput)
	}
	return limit, offset, nil
}
