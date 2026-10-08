// 文件用途：提供会话级兼容接口，并明确拒绝已下线的动态特征代码执行能力。
package businessapi

import (
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

// rejectDynamicFeatureCode 明确拒绝历史上的动态代码执行接口，收紧安全边界。
func (s *Server) rejectDynamicFeatureCode(c *gin.Context) {
	c.JSON(http.StatusGone, gin.H{
		"error":   "dynamic feature code upload is disabled",
		"message": "请将特征算法作为 Python AI 服务的受审代码发布，再通过特征算子接口登记元数据",
	})
}

// cors 生成只允许配置来源携带凭据访问的跨域中间件。
func (s *Server) cors(origin string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if origin != "" {
			c.Header("Access-Control-Allow-Origin", origin)
			c.Header("Vary", "Origin")
		}
		c.Header("Access-Control-Allow-Credentials", "true")
		c.Header("Access-Control-Allow-Headers", "Content-Type,Accept,Last-Event-ID")
		c.Header("Access-Control-Allow-Methods", "GET,POST,PUT,PATCH,DELETE,OPTIONS")
		if c.Request.Method == http.MethodOptions {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}
		c.Next()
	}
}

// stripUpstreamCORS 删除反向代理上游的 CORS 头，避免与本服务策略冲突。
func stripUpstreamCORS(response *http.Response) error {
	for name := range response.Header {
		if strings.HasPrefix(strings.ToLower(name), "access-control-") {
			response.Header.Del(name)
		}
	}
	return nil
}

// authenticate 校验会话 Cookie，把用户 ID 和角色写入 Gin Context。
func (s *Server) authenticate(c *gin.Context) {
	cookie, err := c.Cookie(sessionCookie)
	if err != nil {
		c.AbortWithStatusJSON(401, gin.H{"message": "Unauthorized: not logged in"})
		return
	}
	var uid int64
	var role string
	err = s.db.QueryRowContext(c, `SELECT s.user_id,u.userRole FROM business_sessions s JOIN user u ON u.id=s.user_id AND u.isDelete=0 WHERE s.session_id=? AND s.expires_at>UTC_TIMESTAMP()`, hashToken(cookie)).Scan(&uid, &role)
	if err != nil {
		c.AbortWithStatusJSON(401, gin.H{"message": "登录已过期，请重新登录"})
		return
	}
	c.Set("user_id", uid)
	c.Set("user_role", role)
	c.Next()
}

// requireRole 生成基于 Context 用户角色的授权中间件。
func (s *Server) requireRole(role string) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.GetString("user_role") != role {
			c.AbortWithStatusJSON(403, gin.H{"message": "Forbidden: insufficient permissions"})
			return
		}
		c.Next()
	}
}

// userID 从已认证请求上下文中读取当前用户 ID。
func userID(c *gin.Context) int64 { value, _ := c.Get("user_id"); id, _ := value.(int64); return id }

// register 创建用户、哈希密码并处理用户名唯一冲突。
func (s *Server) register(c *gin.Context) {
	var in struct {
		Account  string `json:"userAccount" binding:"required,max=256"`
		Name     string `json:"userName" binding:"required,max=256"`
		Password string `json:"userPassword" binding:"required,min=6,max=128"`
	}
	if c.ShouldBindJSON(&in) != nil {
		c.JSON(400, gin.H{"message": "invalid registration"})
		return
	}
	var exists int
	_ = s.db.QueryRowContext(c, `SELECT 1 FROM user WHERE userAccount=? AND isDelete=0`, in.Account).Scan(&exists)
	if exists == 1 {
		c.JSON(409, gin.H{"message": "Account already exists"})
		return
	}
	hashed, err := hashPassword(in.Password)
	if err != nil {
		fail(c, err)
		return
	}
	result, err := s.db.ExecContext(c, `INSERT INTO user(userName,userAccount,userPassword,userRole,isDelete) VALUES(?,?,?,'user',0)`, in.Name, in.Account, string(hashed))
	if err != nil {
		fail(c, err)
		return
	}
	id, _ := result.LastInsertId()
	c.JSON(201, gin.H{"message": "User registered successfully", "id": id})
}

// publicRegister 在生产环境关闭匿名注册，同时保留管理员创建用户的能力。
func (s *Server) publicRegister(c *gin.Context) {
	if !s.publicRegistration {
		c.JSON(http.StatusForbidden, gin.H{"message": "Public registration is disabled"})
		return
	}
	s.register(c)
}

// login 验证凭据、创建服务端会话并设置安全 Cookie。
func (s *Server) login(c *gin.Context) {
	var in struct {
		Account  string `json:"userAccount" binding:"required"`
		Password string `json:"userPassword" binding:"required"`
	}
	if c.ShouldBindJSON(&in) != nil {
		c.JSON(400, gin.H{"message": "invalid credentials"})
		return
	}
	var id int64
	var name, hash, role string
	err := s.db.QueryRowContext(c, `SELECT id,COALESCE(userName,''),userPassword,userRole FROM user WHERE userAccount=? AND isDelete=0`, in.Account).Scan(&id, &name, &hash, &role)
	if err != nil || !verifyPassword(hash, in.Password) {
		c.JSON(401, gin.H{"message": "Invalid credentials"})
		return
	}
	raw := randomToken()
	_, _ = s.db.ExecContext(c, `DELETE FROM business_sessions WHERE expires_at<=UTC_TIMESTAMP()`)
	_, err = s.db.ExecContext(c, `INSERT INTO business_sessions(session_id,user_id,user_role,expires_at) VALUES(?,?,?,?)`, hashToken(raw), id, role, time.Now().UTC().Add(s.sessionTTL))
	if err != nil {
		fail(c, err)
		return
	}
	http.SetCookie(c.Writer, &http.Cookie{Name: sessionCookie, Value: raw, Path: "/", HttpOnly: true, Secure: s.sessionSecure, SameSite: http.SameSiteLaxMode, MaxAge: int(s.sessionTTL.Seconds())})
	c.JSON(200, gin.H{"message": "Login successful", "user": gin.H{"id": id, "userName": name, "userAccount": in.Account, "userRole": role}})
}

// logout 删除数据库会话并清除浏览器 Cookie。
func (s *Server) logout(c *gin.Context) {
	if raw, err := c.Cookie(sessionCookie); err == nil {
		_, _ = s.db.ExecContext(c, `DELETE FROM business_sessions WHERE session_id=?`, hashToken(raw))
	}
	http.SetCookie(c.Writer, &http.Cookie{Name: sessionCookie, Path: "/", MaxAge: -1, HttpOnly: true, Secure: s.sessionSecure, SameSite: http.SameSiteLaxMode})
	c.JSON(200, gin.H{"message": "Logged out successfully"})
}

// listUsers 返回未删除用户列表，仅供授权角色访问。
func (s *Server) listUsers(c *gin.Context) {
	rows, err := s.db.QueryContext(c, `SELECT id,userName,userAccount,userRole FROM user WHERE isDelete=0 ORDER BY id`)
	writeRows(c, rows, err)
}

// createUser 复用注册流程创建后台用户。
func (s *Server) createUser(c *gin.Context) { s.register(c) }

// updateUser 更新用户资料、角色、密码或删除标记。
func (s *Server) updateUser(c *gin.Context) {
	var in map[string]any
	if c.ShouldBindJSON(&in) != nil {
		c.JSON(400, gin.H{"error": "invalid request"})
		return
	}
	sets := []string{}
	args := []any{}
	for _, key := range []string{"userName", "userRole"} {
		if v, ok := in[key]; ok {
			sets = append(sets, key+"=?")
			args = append(args, v)
		}
	}
	if p, ok := in["userPassword"].(string); ok && p != "" {
		h, err := hashPassword(p)
		if err != nil {
			fail(c, err)
			return
		}
		sets = append(sets, "userPassword=?")
		args = append(args, string(h))
	}
	if len(sets) == 0 {
		c.JSON(400, gin.H{"error": "no fields"})
		return
	}
	args = append(args, c.Param("id"))
	_, err := s.db.ExecContext(c, `UPDATE user SET `+strings.Join(sets, ",")+` WHERE id=? AND isDelete=0`, args...)
	if err != nil {
		fail(c, err)
		return
	}
	c.JSON(200, gin.H{"message": "User updated"})
}
