// 文件用途：根据特征 ID 查询并返回特征提取器名称等轻量辅助信息。
package businessapi

import "github.com/gin-gonic/gin"

// featureNames 批量查询指定特征提取器 ID 对应的名称。
func (s *Server) featureNames(c *gin.Context) {
	var in struct {
		IDs []int64 `json:"ids"`
	}
	if c.ShouldBindJSON(&in) != nil || len(in.IDs) == 0 {
		c.JSON(400, gin.H{"error": "ids must be a list"})
		return
	}
	q := `SELECT name FROM feature_extractors WHERE isDelete=0 AND id IN (` + placeholders(len(in.IDs)) + `)`
	args := make([]any, len(in.IDs))
	for i, v := range in.IDs {
		args[i] = v
	}
	rows, err := s.db.QueryContext(c, q, args...)
	if err != nil {
		fail(c, err)
		return
	}
	defer rows.Close()
	names := []string{}
	for rows.Next() {
		var n string
		_ = rows.Scan(&n)
		names = append(names, n)
	}
	c.JSON(200, names)
}

// listFeatureGroups 查询当前用户的特征组合列表。
func (s *Server) listFeatureGroups(c *gin.Context) {
	rows, err := s.db.QueryContext(c, `SELECT g.id,g.name,COALESCE(g.description,''),g.createTime,COUNT(i.id) feature_count FROM feature_groups g LEFT JOIN feature_group_items i ON i.group_id=g.id WHERE g.isDelete=0 GROUP BY g.id ORDER BY g.id`)
	writeRows(c, rows, err)
}

// getFeatureGroup 返回一个属于当前用户的特征组合。
func (s *Server) getFeatureGroup(c *gin.Context) {
	var id int64
	var name string
	if s.db.QueryRowContext(c, `SELECT id,name FROM feature_groups WHERE id=? AND isDelete=0`, c.Param("id")).Scan(&id, &name) != nil {
		c.JSON(404, gin.H{"error": "not found"})
		return
	}
	rows, err := s.db.QueryContext(c, `SELECT f.id,f.name,f.modality,f.params_schema,f.description FROM feature_group_items i JOIN feature_extractors f ON f.id=i.feature_extractor_id WHERE i.group_id=? AND f.isDelete=0`, id)
	if err != nil {
		fail(c, err)
		return
	}
	items, err := scanRows(rows)
	if err != nil {
		fail(c, err)
		return
	}
	c.JSON(200, gin.H{"group_id": id, "group_name": name, "features": items})
}

// createFeatureGroup 校验名称和特征列表后创建组合。
func (s *Server) createFeatureGroup(c *gin.Context) {
	var in struct {
		Name        string  `json:"name" binding:"required"`
		Description string  `json:"description"`
		Features    []int64 `json:"features" binding:"required"`
	}
	if c.ShouldBindJSON(&in) != nil || len(in.Features) == 0 {
		c.JSON(400, gin.H{"error": "Missing name or features"})
		return
	}
	tx, err := s.db.BeginTx(c, nil)
	if err != nil {
		fail(c, err)
		return
	}
	defer tx.Rollback()
	res, err := tx.ExecContext(c, `INSERT INTO feature_groups(name,description) VALUES(?,?)`, in.Name, in.Description)
	if err != nil {
		fail(c, err)
		return
	}
	id, _ := res.LastInsertId()
	for _, fid := range in.Features {
		if _, err = tx.ExecContext(c, `INSERT INTO feature_group_items(group_id,feature_extractor_id) VALUES(?,?)`, id, fid); err != nil {
			fail(c, err)
			return
		}
	}
	if err = tx.Commit(); err != nil {
		fail(c, err)
		return
	}
	c.JSON(201, gin.H{"message": "FeatureGroup created", "group_id": id})
}

// updateFeatureGroup 修改当前用户拥有的特征组合。
func (s *Server) updateFeatureGroup(c *gin.Context) {
	spec := catalog{"feature_groups", "FeatureGroup", []string{"name", "description"}, true}
	s.updateCatalog(spec)(c)
}
