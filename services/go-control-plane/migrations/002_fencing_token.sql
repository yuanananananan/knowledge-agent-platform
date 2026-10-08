-- 文件用途：为 run 增加 fencing token，阻止租约失效的旧 Worker 提交结果。
ALTER TABLE control_plane_runs
  ADD COLUMN fencing_token BIGINT NOT NULL DEFAULT 0 AFTER version;
