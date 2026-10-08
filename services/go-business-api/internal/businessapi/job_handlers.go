// 文件用途：提供训练、验证和 NAS 任务的查询、删除、日志流以及状态转换接口。
package businessapi

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

// jobStatus 生成查询任务状态的通用 Handler。
func (s *Server) jobStatus(table, key string) gin.HandlerFunc {
	return func(c *gin.Context) {
		var status string
		var progress int
		if s.db.QueryRowContext(c, `SELECT status,COALESCE(progress,0) FROM `+table+` WHERE `+key+`=? AND isDelete=0 AND user_id=?`, c.Param("id"), userID(c)).Scan(&status, &progress) != nil {
			c.JSON(404, gin.H{"error": "job_id not found"})
			return
		}
		c.JSON(200, gin.H{"status": status, "progress": progress})
	}
}

// validationStatus 返回验证任务状态，并兼容 Celery 状态字段。
func (s *Server) validationStatus(c *gin.Context) {
	var status string
	if s.db.QueryRowContext(c, `SELECT status FROM test_records WHERE val_job_id=? AND isDelete=0 AND user_id=?`, c.Param("id"), userID(c)).Scan(&status) != nil {
		c.JSON(404, gin.H{"error": "job_id not found"})
		return
	}
	c.JSON(200, gin.H{"status": status, "progress": 0})
}

// jobField 生成读取任务单个字段的 Handler。
func (s *Server) jobField(table, key, field string) gin.HandlerFunc {
	return func(c *gin.Context) {
		var raw []byte
		if s.db.QueryRowContext(c, `SELECT `+field+` FROM `+table+` WHERE `+key+`=? AND isDelete=0 AND user_id=?`, c.Param("id"), userID(c)).Scan(&raw) != nil {
			c.JSON(404, gin.H{"error": "job_id not found"})
			return
		}
		var value any
		if len(raw) > 0 {
			_ = json.Unmarshal(raw, &value)
		}
		c.JSON(200, gin.H{field: value})
	}
}

// jobRow 生成读取完整任务记录的 Handler。
func (s *Server) jobRow(table, key string) gin.HandlerFunc {
	return func(c *gin.Context) {
		rows, err := s.db.QueryContext(c, `SELECT * FROM `+table+` WHERE `+key+`=? AND isDelete=0 AND user_id=?`, c.Param("id"), userID(c))
		if err != nil {
			fail(c, err)
			return
		}
		items, err := scanRows(rows)
		if err != nil || len(items) == 0 {
			c.JSON(404, gin.H{"error": "not found"})
			return
		}
		c.JSON(200, items[0])
	}
}

// deleteJob 生成取消后台计算并删除任务记录的 Handler。
func (s *Server) deleteJob(table, key, taskColumn string) gin.HandlerFunc {
	return func(c *gin.Context) {
		var taskID string
		err := s.db.QueryRowContext(c, `SELECT `+taskColumn+` FROM `+table+` WHERE `+key+`=? AND isDelete=0 AND user_id=?`, c.Param("id"), userID(c)).Scan(&taskID)
		if err != nil {
			c.JSON(404, gin.H{"error": "not found"})
			return
		}
		_ = s.compute.CancelCompute(c, taskID, true)
		_, err = s.db.ExecContext(c, `UPDATE `+table+` SET isDelete=1,status='deleted' WHERE `+key+`=? AND user_id=?`, c.Param("id"), userID(c))
		if err != nil {
			fail(c, err)
			return
		}
		c.JSON(200, gin.H{"message": "task deleted"})
	}
}

// ownedJob 检查路由中的任务是否存在且属于当前用户。
func (s *Server) ownedJob(c *gin.Context, table, key string) bool {
	var exists int
	err := s.db.QueryRowContext(c, `SELECT 1 FROM `+table+` WHERE `+key+`=? AND user_id=? AND isDelete=0`, c.Param("id"), userID(c)).Scan(&exists)
	if err != nil {
		c.JSON(404, gin.H{"error": "job not found"})
		return false
	}
	return true
}

// readJobLog 读取任务日志文件并返回已有内容。
func (s *Server) readJobLog(table, key string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if !s.ownedJob(c, table, key) {
			return
		}
		path, err := safePath(s.logRoot, c.Param("id")+".log")
		if err != nil {
			c.JSON(403, gin.H{"error": "invalid path"})
			return
		}
		data, err := os.ReadFile(path)
		if err != nil {
			c.JSON(404, gin.H{"message": "Log not found"})
			return
		}
		c.JSON(200, gin.H{"log": string(data)})
	}
}

// streamJobLog 以 SSE/流式方式持续发送新增任务日志。
func (s *Server) streamJobLog(table, key string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if !s.ownedJob(c, table, key) {
			return
		}
		path, err := safePath(s.logRoot, c.Param("id")+".log")
		if err != nil {
			c.JSON(403, gin.H{"error": "invalid path"})
			return
		}
		c.Header("Content-Type", "text/event-stream")
		c.Header("Cache-Control", "no-cache")
		c.Header("X-Accel-Buffering", "no")
		flusher, ok := c.Writer.(http.Flusher)
		if !ok {
			c.JSON(500, gin.H{"error": "streaming unsupported"})
			return
		}
		var offset int64
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			file, openErr := os.Open(path)
			if openErr == nil {
				_, _ = file.Seek(offset, io.SeekStart)
				scanner := bufio.NewScanner(file)
				scanner.Buffer(make([]byte, 64*1024), 1024*1024)
				for scanner.Scan() {
					_, _ = fmt.Fprintf(c.Writer, "data: %s\n\n", strings.TrimSuffix(scanner.Text(), "\r"))
				}
				offset, _ = file.Seek(0, io.SeekCurrent)
				_ = file.Close()
				flusher.Flush()
			}
			select {
			case <-c.Request.Context().Done():
				return
			case <-ticker.C:
			}
		}
	}
}
