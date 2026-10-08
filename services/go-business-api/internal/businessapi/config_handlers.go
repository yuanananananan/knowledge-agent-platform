// 文件用途：读取模型训练配置模板和 YAML Schema，并向前端返回可编辑配置结构。
package businessapi

import (
	"encoding/json"
	"os"
	"strings"

	"github.com/gin-gonic/gin"
	"gopkg.in/yaml.v3"
)

// configSchema 安全读取配置模板，将 YAML 转换为前端使用的 JSON Schema。
func (s *Server) configSchema(c *gin.Context) {
	name := c.DefaultQuery("name", "sar_config.yaml")
	if !safeNamePattern.MatchString(name) || !strings.HasSuffix(name, "_config.yaml") {
		c.JSON(400, gin.H{"error": "invalid config name"})
		return
	}
	path, err := safePath(s.configRoot, name)
	if err != nil {
		c.JSON(400, gin.H{"error": "invalid config path"})
		return
	}
	data, err := os.ReadFile(path)
	if err != nil {
		c.JSON(404, gin.H{"error": "config not found"})
		return
	}
	var cfg map[string]any
	if yaml.Unmarshal(data, &cfg) != nil {
		c.JSON(500, gin.H{"error": "invalid config"})
		return
	}
	items := []gin.H{}
	var flatten func(map[string]any)
	flatten = func(section map[string]any) {
		for key, value := range section {
			if key == "feature_combinations" {
				continue
			}
			if nested, ok := value.(map[string]any); ok {
				flatten(nested)
				continue
			}
			typ := "string"
			switch value.(type) {
			case bool:
				typ = "boolean"
			case int, int64, float64:
				typ = "number"
			case []any:
				typ = "list"
			}
			if key == "img_size" {
				typ = "string"
				raw, _ := json.Marshal(value)
				value = string(raw)
			}
			items = append(items, gin.H{"name": key, "type": typ, "value": value, "description": key + " 配置项。"})
		}
	}
	flatten(cfg)
	c.JSON(200, gin.H{"extra": items})
}
