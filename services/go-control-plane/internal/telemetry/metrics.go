// 文件用途：使用原子计数器采集 HTTP 请求、错误、run、action 和累计延迟指标。
package telemetry

import (
	"fmt"
	"net/http"
	"sync/atomic"
	"time"
)

// Metrics 保存无锁累加的进程级请求和延迟指标。
type Metrics struct {
	Requests, Errors, Runs, Actions atomic.Uint64
	latencyNanos                    atomic.Uint64
}

// Observe 记录一次 HTTP 请求、可选错误和耗时。
func (m *Metrics) Observe(start time.Time, failed bool) {
	m.Requests.Add(1)
	m.latencyNanos.Add(uint64(time.Since(start)))
	if failed {
		m.Errors.Add(1)
	}
}

// Handler 以 Prometheus 文本格式暴露当前计数器和平均延迟。
func (m *Metrics) Handler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		fmt.Fprintf(w, "control_plane_http_requests_total %d\ncontrol_plane_http_errors_total %d\ncontrol_plane_runs_created_total %d\ncontrol_plane_actions_created_total %d\ncontrol_plane_http_latency_seconds_sum %f\n", m.Requests.Load(), m.Errors.Load(), m.Runs.Load(), m.Actions.Load(), float64(m.latencyNanos.Load())/float64(time.Second))
	})
}
