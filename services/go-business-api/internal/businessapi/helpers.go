// 文件用途：提供 Business API 共用的参数解析、ID 生成、哈希、路径安全和数据库辅助函数。
package businessapi

import (
	"crypto/md5"
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"golang.org/x/crypto/bcrypt"
	"golang.org/x/crypto/pbkdf2"
	"golang.org/x/crypto/scrypt"
)

// writeRows 统一处理 SQL 查询错误、扫描多行结果并返回 JSON。
func writeRows(c *gin.Context, rows *sql.Rows, err error) {
	if err != nil {
		fail(c, err)
		return
	}
	items, err := scanRows(rows)
	if err != nil {
		fail(c, err)
		return
	}
	c.JSON(200, items)
}

// scanRows 将列名未知的 SQL 结果动态转换为键值对象数组。
func scanRows(rows *sql.Rows) ([]map[string]any, error) {
	defer rows.Close()
	cols, _ := rows.Columns()
	out := []map[string]any{}
	for rows.Next() {
		values := make([]any, len(cols))
		ptrs := make([]any, len(cols))
		for i := range values {
			ptrs[i] = &values[i]
		}
		if err := rows.Scan(ptrs...); err != nil {
			return nil, err
		}
		item := map[string]any{}
		for i, v := range values {
			if b, ok := v.([]byte); ok {
				var decoded any
				if json.Unmarshal(b, &decoded) == nil && (len(b) > 0 && (b[0] == '{' || b[0] == '[')) {
					v = decoded
				} else {
					v = string(b)
				}
			}
			if t, ok := v.(time.Time); ok {
				v = t.UTC().Format(time.RFC3339)
			}
			item[cols[i]] = v
		}
		out = append(out, item)
	}
	return out, rows.Err()
}

// fail 隐藏内部错误细节并返回统一的 500 响应。
func fail(c *gin.Context, err error) { c.JSON(500, gin.H{"error": "internal_error"}) }

// allowed 从输入对象中提取允许写入数据库的白名单字段及对应值。
func allowed(in map[string]any, columns []string) ([]string, []any) {
	cols := []string{}
	args := []any{}
	for _, col := range columns {
		if value, ok := in[col]; ok {
			if m, ok := value.(map[string]any); ok {
				value, _ = json.Marshal(m)
			}
			cols = append(cols, col)
			args = append(args, value)
		}
	}
	return cols, args
}

// placeholders 生成参数化 SQL 使用的问号占位符列表。
func placeholders(n int) string {
	if n < 1 {
		return ""
	}
	return strings.TrimSuffix(strings.Repeat("?,", n), ",")
}

// randomToken 生成 256 位随机令牌并编码为十六进制。
func randomToken() string { b := make([]byte, 32); _, _ = rand.Read(b); return hex.EncodeToString(b) }

// randomID 从随机令牌中截取适合作为公开业务 ID 的长度。
func randomID() string { return randomToken()[:32] }

// hashToken 使用 SHA-256 保存令牌摘要，避免数据库保存明文令牌。
func hashToken(v string) string { sum := sha256.Sum256([]byte(v)); return hex.EncodeToString(sum[:]) }

// stringValue 将动态 JSON 字段安全转换成去除首尾空白的字符串。
func stringValue(v any) string { s, _ := v.(string); return strings.TrimSpace(s) }

// number 将 JSON 数字或字符串统一转换成 int64。
func number(v any) int64 {
	switch n := v.(type) {
	case float64:
		return int64(n)
	case int:
		return int64(n)
	case int64:
		return n
	}
	return 0
}

// boolValue 将动态 JSON 字段安全转换成布尔值。
func boolValue(v any) bool { b, _ := v.(bool); return b }

// safePath 将相对名称约束在指定根目录内，防止目录穿越。
func safePath(root, name string) (string, error) {
	base, err := filepath.Abs(root)
	if err != nil {
		return "", err
	}
	candidate := name
	if !filepath.IsAbs(candidate) {
		candidate = filepath.Join(base, candidate)
	}
	candidate, err = filepath.Abs(candidate)
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(base, candidate)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", errors.New("path escapes root")
	}
	return candidate, nil
}

// buildConfig 从请求中提取并组合训练配置对象。
func buildConfig(in map[string]any) map[string]any {
	cfg := map[string]any{"data": map[string]any{}, "model": map[string]any{"GNN": map[string]any{"is_GNN": boolValue(in["is_GNN"])}, "Cross_attention": map[string]any{}}, "training": map[string]any{"optimizer": map[string]any{}}, "feature_combinations": []any{}, "is_nas": boolValue(in["is_nas"]), "is_ssl": false}
	extra, _ := in["extra"].([]any)
	for _, raw := range extra {
		item, _ := raw.(map[string]any)
		key := stringValue(item["name"])
		value := item["value"]
		switch key {
		case "mode", "name", "root", "num_classes", "batch_size", "num_workers", "shuffle", "train_ratio", "img_size", "is_aug":
			cfg["data"].(map[string]any)[key] = value
		case "backbone", "pretrained", "output_layer", "d_model":
			cfg["model"].(map[string]any)[key] = value
		case "is_GNN", "k", "n_blocks", "n_nodes", "dropout":
			cfg["model"].(map[string]any)["GNN"].(map[string]any)[key] = value
		case "is_attention":
			cfg["model"].(map[string]any)["Cross_attention"].(map[string]any)[key] = value
		case "epochs", "device", "checkpoint_dir":
			cfg["training"].(map[string]any)[key] = value
		case "type", "lr", "weight_decay", "momentum":
			cfg["training"].(map[string]any)["optimizer"].(map[string]any)[key] = value
		}
	}
	return cfg
}

// verifyPassword 解析已编码密码并使用恒定时间比较验证摘要。
func verifyPassword(encoded, password string) bool {
	if bcrypt.CompareHashAndPassword([]byte(encoded), []byte(password)) == nil {
		return true
	}
	parts := strings.Split(encoded, "$")
	if len(parts) == 3 {
		method, salt, want := parts[0], parts[1], parts[2]
		if strings.HasPrefix(method, "scrypt:") {
			var n, r, p int
			if _, err := fmt.Sscanf(method, "scrypt:%d:%d:%d", &n, &r, &p); err == nil {
				got, err := scrypt.Key([]byte(password), []byte(salt), n, r, p, 64)
				return err == nil && hex.EncodeToString(got) == want
			}
		}
		if strings.HasPrefix(method, "pbkdf2:sha256:") {
			var rounds int
			if _, err := fmt.Sscanf(method, "pbkdf2:sha256:%d", &rounds); err == nil {
				return hex.EncodeToString(pbkdf2.Key([]byte(password), []byte(salt), rounds, 32, sha256.New)) == want
			}
		}
	}
	sum := md5.Sum([]byte(password))
	return hex.EncodeToString(sum[:]) == encoded
}

// hashPassword 使用随机盐和迭代哈希生成可持久化密码编码。
func hashPassword(password string) (string, error) {
	saltBytes := make([]byte, 16)
	if _, err := rand.Read(saltBytes); err != nil {
		return "", err
	}
	salt := hex.EncodeToString(saltBytes)
	derived, err := scrypt.Key([]byte(password), []byte(salt), 32768, 8, 1, 64)
	if err != nil {
		return "", err
	}
	return "scrypt:32768:8:1$" + salt + "$" + hex.EncodeToString(derived), nil
}
