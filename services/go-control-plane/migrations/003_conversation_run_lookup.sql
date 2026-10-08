-- 文件用途：增加用户和会话维度的 run 查询索引，加速会话恢复与历史记录读取。
CREATE INDEX idx_control_plane_runs_conversation_status
    ON control_plane_runs(user_id, conversation_id, status, created_at);
