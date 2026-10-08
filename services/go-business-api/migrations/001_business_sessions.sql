-- 文件用途：创建 Business API 会话表，用于持久化登录会话及其过期时间。
CREATE TABLE IF NOT EXISTS business_sessions (
    session_id CHAR(64) PRIMARY KEY,
    user_id BIGINT NOT NULL,
    user_role VARCHAR(32) NOT NULL,
    expires_at DATETIME(6) NOT NULL,
    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    KEY ix_business_sessions_expiry (expires_at),
    KEY ix_business_sessions_user (user_id)
);
