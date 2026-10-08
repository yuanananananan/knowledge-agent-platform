-- 文件用途：创建 Control Plane 的 run、事件、动作提案和事务 Outbox 核心表。
-- run 主表：保存请求、状态机、Worker 租约、最终答案和错误。
CREATE TABLE IF NOT EXISTS control_plane_runs (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  run_id VARCHAR(64) NOT NULL UNIQUE,
  conversation_id VARCHAR(64) NOT NULL,
  user_id BIGINT NOT NULL,
  message TEXT NOT NULL,
  status VARCHAR(32) NOT NULL,
  answer LONGTEXT NULL,
  error_message TEXT NULL,
  metadata JSON NULL,
  version BIGINT NOT NULL DEFAULT 0,
  worker_id VARCHAR(128) NULL,
  lease_expires_at DATETIME(6) NULL,
  created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  INDEX ix_cp_runs_user_created(user_id, created_at),
  INDEX ix_cp_runs_status_lease(status, lease_expires_at)
);

-- 事件表：按 run_id + sequence 保存可断点续读的有序事件日志。
CREATE TABLE IF NOT EXISTS control_plane_events (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  run_id VARCHAR(64) NOT NULL,
  sequence BIGINT NOT NULL,
  event_type VARCHAR(64) NOT NULL,
  payload JSON NOT NULL,
  created_at DATETIME(6) NOT NULL,
  UNIQUE KEY uq_cp_event_sequence(run_id, sequence)
);

-- 动作提案表：保存需要人工审批的高风险写操作及其幂等键。
CREATE TABLE IF NOT EXISTS control_plane_actions (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  proposal_id VARCHAR(64) NOT NULL UNIQUE,
  user_id BIGINT NOT NULL,
  conversation_id VARCHAR(64) NULL,
  action_type VARCHAR(64) NOT NULL,
  payload JSON NOT NULL,
  reason TEXT NULL,
  impact TEXT NOT NULL,
  idempotency_key CHAR(64) NOT NULL UNIQUE,
  status VARCHAR(32) NOT NULL,
  result JSON NULL,
  error_message TEXT NULL,
  expires_at DATETIME(6) NOT NULL,
  created_at DATETIME(6) NOT NULL,
  updated_at DATETIME(6) NOT NULL,
  INDEX ix_cp_actions_user_status(user_id, status)
);

-- Outbox 表：与业务事件同事务写入，后台可靠发布到 Kafka。
CREATE TABLE IF NOT EXISTS control_plane_outbox (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  aggregate_id VARCHAR(64) NOT NULL,
  event_type VARCHAR(64) NOT NULL,
  payload JSON NOT NULL,
  attempts INT NOT NULL DEFAULT 0,
  available_at DATETIME(6) NOT NULL,
  published_at DATETIME(6) NULL,
  worker_id VARCHAR(128) NULL,
  lease_expires_at DATETIME(6) NULL,
  created_at DATETIME(6) NOT NULL,
  INDEX ix_cp_outbox_pending(published_at, available_at, lease_expires_at)
);
