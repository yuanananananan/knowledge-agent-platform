// 文件用途：实现事务 Outbox 消息的认领、发布确认、退避重试及数据库连接辅助方法。
package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"time"
)

// Close 关闭数据库连接池。
func (m *MySQL) Close() error { return m.db.Close() }

// Ping 检查数据库连接是否可用。
func (m *MySQL) Ping(ctx context.Context) error { return m.db.PingContext(ctx) }

// nullableJSON 把空 JSON 转为 SQL NULL，非空 JSON 保留原始字节。
func nullableJSON(v json.RawMessage) any {
	if len(v) == 0 {
		return nil
	}
	return []byte(v)
}

// OutboxMessage 表示一条等待发布到 Kafka 的事务事件。
type OutboxMessage struct {
	ID                     int64
	AggregateID, EventType string
	Payload                json.RawMessage
	Attempts               int
}

// ClaimOutbox 使用 FOR UPDATE SKIP LOCKED 认领一条到期消息并设置租约。
func (m *MySQL) ClaimOutbox(ctx context.Context, worker string, lease time.Duration) (OutboxMessage, error) {
	tx, err := m.db.BeginTx(ctx, &sql.TxOptions{})
	if err != nil {
		return OutboxMessage{}, err
	}
	defer tx.Rollback()
	var msg OutboxMessage
	err = tx.QueryRowContext(ctx, `SELECT id,aggregate_id,event_type,payload,attempts FROM control_plane_outbox WHERE published_at IS NULL AND available_at<=UTC_TIMESTAMP() AND (lease_expires_at IS NULL OR lease_expires_at<UTC_TIMESTAMP()) ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED`).Scan(&msg.ID, &msg.AggregateID, &msg.EventType, &msg.Payload, &msg.Attempts)
	if err != nil {
		return OutboxMessage{}, translate(err)
	}
	_, err = tx.ExecContext(ctx, `UPDATE control_plane_outbox SET worker_id=?,lease_expires_at=? WHERE id=?`, worker, time.Now().UTC().Add(lease), msg.ID)
	if err != nil {
		return OutboxMessage{}, err
	}
	return msg, tx.Commit()
}

// MarkOutboxPublished 记录成功发布时间并清除租约。
func (m *MySQL) MarkOutboxPublished(ctx context.Context, id int64) error {
	_, err := m.db.ExecContext(ctx, `UPDATE control_plane_outbox SET published_at=UTC_TIMESTAMP(),lease_expires_at=NULL WHERE id=?`, id)
	return err
}

// RetryOutbox 增加尝试次数，把消息推迟到未来并释放当前租约。
func (m *MySQL) RetryOutbox(ctx context.Context, id int64, delay time.Duration) error {
	_, err := m.db.ExecContext(ctx, `UPDATE control_plane_outbox SET attempts=attempts+1,available_at=?,lease_expires_at=NULL WHERE id=?`, time.Now().UTC().Add(delay), id)
	return err
}
