// 文件用途：为数据集、骨干网络、分类器等通用目录资源生成一致的 CRUD 路由。
package businessapi

import (
	"strings"

	"github.com/gin-gonic/gin"
)

// registerCatalog 按资源描述一次性注册列表、新增、修改和软删除路由。
func (s *Server) registerCatalog(g *gin.RouterGroup, path string, spec catalog) {
	g.GET(path, s.listTable(spec.table, "id"))
	g.GET(path+"/", s.listTable(spec.table, "id"))
	write := []gin.HandlerFunc{}
	if spec.adminWrite {
		write = append(write, s.requireRole("admin"))
	}
	g.POST(path+"/", append(write, s.createCatalog(spec))...)
	g.PUT(path+"/:id", append(write, s.updateCatalog(spec))...)
	g.DELETE(path+"/:id", append(write, s.softDelete(spec.table))...)
}

// listTable 生成读取通用目录表的 Handler。
func (s *Server) listTable(table, order string) gin.HandlerFunc {
	return func(c *gin.Context) {
		rows, err := s.db.QueryContext(c, `SELECT * FROM `+table+` WHERE isDelete=0 ORDER BY `+order)
		writeRows(c, rows, err)
	}
}

// listJobs 生成按当前用户过滤任务列表的 Handler。
func (s *Server) listJobs(table string) gin.HandlerFunc {
	return func(c *gin.Context) {
		rows, err := s.db.QueryContext(c, `SELECT * FROM `+table+` WHERE isDelete=0 AND user_id=? ORDER BY id DESC`, userID(c))
		writeRows(c, rows, err)
	}
}

// createCatalog 生成只写入白名单字段的目录创建 Handler。
func (s *Server) createCatalog(spec catalog) gin.HandlerFunc {
	return func(c *gin.Context) {
		var in map[string]any
		if c.ShouldBindJSON(&in) != nil {
			c.JSON(400, gin.H{"error": "invalid request"})
			return
		}
		cols, args := allowed(in, spec.columns)
		if len(cols) == 0 {
			c.JSON(400, gin.H{"error": "no fields"})
			return
		}
		q := `INSERT INTO ` + spec.table + ` (` + strings.Join(cols, ",") + `) VALUES (` + placeholders(len(cols)) + `)`
		res, err := s.db.ExecContext(c, q, args...)
		if err != nil {
			fail(c, err)
			return
		}
		id, _ := res.LastInsertId()
		c.JSON(201, gin.H{"message": spec.singular + " created", "id": id})
	}
}

// updateCatalog 生成带字段白名单和归属校验的更新 Handler。
func (s *Server) updateCatalog(spec catalog) gin.HandlerFunc {
	return func(c *gin.Context) {
		var in map[string]any
		if c.ShouldBindJSON(&in) != nil {
			c.JSON(400, gin.H{"error": "invalid request"})
			return
		}
		cols, args := allowed(in, spec.columns)
		sets := make([]string, len(cols))
		for i, col := range cols {
			sets[i] = col + "=?"
		}
		if len(sets) == 0 {
			c.JSON(400, gin.H{"error": "no fields"})
			return
		}
		args = append(args, c.Param("id"))
		_, err := s.db.ExecContext(c, `UPDATE `+spec.table+` SET `+strings.Join(sets, ",")+` WHERE id=? AND isDelete=0`, args...)
		if err != nil {
			fail(c, err)
			return
		}
		c.JSON(200, gin.H{"message": spec.singular + " updated"})
	}
}

// softDelete 生成将 isDelete 标记为真的逻辑删除 Handler。
func (s *Server) softDelete(table string) gin.HandlerFunc {
	return func(c *gin.Context) {
		res, err := s.db.ExecContext(c, `UPDATE `+table+` SET isDelete=1 WHERE id=? AND isDelete=0`, c.Param("id"))
		if err != nil {
			fail(c, err)
			return
		}
		n, _ := res.RowsAffected()
		if n == 0 {
			c.JSON(404, gin.H{"error": "not found"})
			return
		}
		c.JSON(200, gin.H{"message": "deleted"})
	}
}
