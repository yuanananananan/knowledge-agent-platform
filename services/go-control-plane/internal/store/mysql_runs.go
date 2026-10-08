// 文件用途：实现 run、事件、租约、fencing token 和状态迁移在 MySQL 中的原子持久化。
package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"time"

	"knowledge-agent/go-control-plane/internal/domain"
)

// translate 将 sql.ErrNoRows 统一转换成领域层 ErrNotFound。
func translate(err error) error {
	if errors.Is(err, sql.ErrNoRows) {
		return domain.ErrNotFound
	}
	return err
}

// CreateRun 在事务中阻止同会话并发 run，并同时写入 run.created 和 Outbox。
func (m *MySQL) CreateRun(ctx context.Context, r domain.Run) error {
	tx, err := m.db.BeginTx(ctx, &sql.TxOptions{})
	if err != nil {
		return err
	}
	defer tx.Rollback()
	var activeID string
	err = tx.QueryRowContext(ctx, `SELECT run_id FROM control_plane_runs WHERE user_id=? AND conversation_id=? AND status IN (?,?,?) LIMIT 1 FOR UPDATE`, r.UserID, r.ConversationID, domain.RunPending, domain.RunCollecting, domain.RunReasoning).Scan(&activeID)
	if err == nil {
		return domain.ErrInProgress
	}
	if !errors.Is(err, sql.ErrNoRows) {
		return err
	}
	_, err = tx.ExecContext(ctx, `INSERT INTO control_plane_runs
 (run_id,conversation_id,user_id,message,status,version,created_at,updated_at)
 VALUES (?,?,?,?,?,0,?,?)`, r.ID, r.ConversationID, r.UserID, r.Message, r.Status, r.CreatedAt, r.UpdatedAt)
	if err != nil {
		return err
	}
	payload, _ := json.Marshal(map[string]any{"status": r.Status})
	if _, err = appendEventTx(ctx, tx, domain.Event{RunID: r.ID, Type: "run.created", Payload: payload}); err != nil {
		return err
	}
	return tx.Commit()
}

// ListRunsByConversation 按用户和会话倒序分页查询 run。
func (m *MySQL) ListRunsByConversation(ctx context.Context, uid int64, conversation string, limit, offset int) ([]domain.Run, error) {
	rows, err := m.db.QueryContext(ctx, `SELECT `+runColumns+` FROM control_plane_runs WHERE user_id=? AND conversation_id=? ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?`, uid, conversation, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]domain.Run, 0)
	for rows.Next() {
		r, scanErr := scanRun(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// ListRuns 按用户倒序分页查询全部 run。
func (m *MySQL) ListRuns(ctx context.Context, uid int64, limit, offset int) ([]domain.Run, error) {
	rows, err := m.db.QueryContext(ctx, `SELECT `+runColumns+` FROM control_plane_runs WHERE user_id=? ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?`, uid, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]domain.Run, 0)
	for rows.Next() {
		r, err := scanRun(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// ListPendingRunIDs 找出未租用或租约过期的非终态 run，供 Worker 拉取。
func (m *MySQL) ListPendingRunIDs(ctx context.Context, limit int) ([]string, error) {
	rows, err := m.db.QueryContext(ctx, `SELECT run_id FROM control_plane_runs WHERE status IN (?,?,?) AND (lease_expires_at IS NULL OR lease_expires_at<UTC_TIMESTAMP()) ORDER BY created_at LIMIT ?`, domain.RunPending, domain.RunCollecting, domain.RunReasoning, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

// scanRun 按固定列顺序将 SQL 行转换成领域 Run。
func scanRun(row interface{ Scan(...any) error }) (domain.Run, error) {
	var r domain.Run
	var answer, runErr sql.NullString
	var meta []byte
	err := row.Scan(&r.ID, &r.ConversationID, &r.UserID, &r.Message, &r.Status, &answer, &runErr, &r.Version, &r.FencingToken, &r.CreatedAt, &r.UpdatedAt, &meta)
	r.Answer, r.Error = answer.String, runErr.String
	r.Metadata = meta
	return r, translate(err)
}

// runColumns 必须与 scanRun 的扫描参数顺序保持一致。
const runColumns = `run_id,conversation_id,user_id,message,status,answer,error_message,version,fencing_token,created_at,updated_at,metadata`

// GetRun 使用用户 ID 限制读取范围，防止跨用户访问。
func (m *MySQL) GetRun(ctx context.Context, uid int64, id string) (domain.Run, error) {
	return scanRun(m.db.QueryRowContext(ctx, `SELECT `+runColumns+` FROM control_plane_runs WHERE run_id=? AND user_id=?`, id, uid))
}

// ClaimRun 原子写入 Worker、租约并递增 fencing token，确保只有一个有效执行者。
func (m *MySQL) ClaimRun(ctx context.Context, id, worker string, lease time.Duration) (domain.Run, error) {
	res, err := m.db.ExecContext(ctx, `UPDATE control_plane_runs SET worker_id=?,lease_expires_at=?,fencing_token=fencing_token+1,updated_at=UTC_TIMESTAMP() WHERE run_id=? AND status IN (?,?,?) AND (lease_expires_at IS NULL OR lease_expires_at<UTC_TIMESTAMP())`, worker, time.Now().UTC().Add(lease), id, domain.RunPending, domain.RunCollecting, domain.RunReasoning)
	if err != nil {
		return domain.Run{}, err
	}
	n, _ := res.RowsAffected()
	if n != 1 {
		return domain.Run{}, domain.ErrConflict
	}
	return scanRun(m.db.QueryRowContext(ctx, `SELECT `+runColumns+` FROM control_plane_runs WHERE run_id=?`, id))
}

// TransitionRun 校验状态机、租约与 fencing token，并事务性更新 run 和追加事件。
func (m *MySQL) TransitionRun(ctx context.Context, id, worker string, fence int64, from, to domain.RunStatus, answer, runErr string, meta json.RawMessage, eventType string, eventPayload json.RawMessage) error {
	if !domain.CanTransition(from, to) {
		return domain.ErrInvalidState
	}
	tx, err := m.db.BeginTx(ctx, &sql.TxOptions{})
	if err != nil {
		return err
	}
	defer tx.Rollback()
	res, err := tx.ExecContext(ctx, `UPDATE control_plane_runs SET status=?,answer=?,error_message=?,metadata=?,version=version+1,updated_at=UTC_TIMESTAMP() WHERE run_id=? AND status=? AND worker_id=? AND fencing_token=? AND lease_expires_at>UTC_TIMESTAMP()`, to, answer, runErr, nullableJSON(meta), id, from, worker, fence)
	if err != nil {
		return err
	}
	n, _ := res.RowsAffected()
	if n != 1 {
		return domain.ErrInvalidState
	}
	if _, err = appendEventTx(ctx, tx, domain.Event{RunID: id, Type: eventType, Payload: eventPayload}); err != nil {
		return err
	}
	return tx.Commit()
}

// AppendRunEvent 在验证当前执行者后延长租约，并事务性追加非终态事件。
func (m *MySQL) AppendRunEvent(ctx context.Context, id, worker string, fence int64, eventType string, eventPayload json.RawMessage, lease time.Duration) error {
	tx, err := m.db.BeginTx(ctx, &sql.TxOptions{})
	if err != nil {
		return err
	}
	defer tx.Rollback()
	res, err := tx.ExecContext(ctx, `UPDATE control_plane_runs SET lease_expires_at=?,updated_at=UTC_TIMESTAMP() WHERE run_id=? AND status=? AND worker_id=? AND fencing_token=?`, time.Now().UTC().Add(lease), id, domain.RunReasoning, worker, fence)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n != 1 {
		return domain.ErrInvalidState
	}
	if _, err = appendEventTx(ctx, tx, domain.Event{RunID: id, Type: eventType, Payload: eventPayload}); err != nil {
		return err
	}
	return tx.Commit()
}

// RenewRun 只允许当前 fencing token 的 Worker 延长活跃 run 租约。
func (m *MySQL) RenewRun(ctx context.Context, id, worker string, fence int64, lease time.Duration) error {
	res, err := m.db.ExecContext(ctx, `UPDATE control_plane_runs SET lease_expires_at=?,updated_at=UTC_TIMESTAMP() WHERE run_id=? AND status IN (?,?) AND worker_id=? AND fencing_token=?`, time.Now().UTC().Add(lease), id, domain.RunCollecting, domain.RunReasoning, worker, fence)
	if err != nil {
		return err
	}
	if n, _ := res.RowsAffected(); n != 1 {
		return domain.ErrInvalidState
	}
	return nil
}

// CancelRun 锁定 run、拒绝终态取消，并事务性写入 run.cancelled 事件。
func (m *MySQL) CancelRun(ctx context.Context, uid int64, id string) error {
	tx, err := m.db.BeginTx(ctx, &sql.TxOptions{})
	if err != nil {
		return err
	}
	defer tx.Rollback()
	var status domain.RunStatus
	if err = tx.QueryRowContext(ctx, `SELECT status FROM control_plane_runs WHERE run_id=? AND user_id=? FOR UPDATE`, id, uid).Scan(&status); err != nil {
		return translate(err)
	}
	if status.Terminal() {
		return domain.ErrConflict
	}
	if _, err = tx.ExecContext(ctx, `UPDATE control_plane_runs SET status=?,version=version+1,lease_expires_at=NULL,updated_at=UTC_TIMESTAMP() WHERE run_id=?`, domain.RunCancelled, id); err != nil {
		return err
	}
	payload, _ := json.Marshal(map[string]any{"status": domain.RunCancelled})
	if _, err = appendEventTx(ctx, tx, domain.Event{RunID: id, Type: "run.cancelled", Payload: payload}); err != nil {
		return err
	}
	return tx.Commit()
}

// appendEventTx 在同一事务中分配单调 sequence，同时写事件表和 Outbox 表。
func appendEventTx(ctx context.Context, tx *sql.Tx, e domain.Event) (domain.Event, error) {
	if err := tx.QueryRowContext(ctx, `SELECT COALESCE(MAX(sequence),0)+1 FROM control_plane_events WHERE run_id=? FOR UPDATE`, e.RunID).Scan(&e.Sequence); err != nil {
		return e, err
	}
	e.CreatedAt = time.Now().UTC()
	if _, err := tx.ExecContext(ctx, `INSERT INTO control_plane_events(run_id,sequence,event_type,payload,created_at) VALUES(?,?,?,?,?)`, e.RunID, e.Sequence, e.Type, e.Payload, e.CreatedAt); err != nil {
		return e, err
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO control_plane_outbox(aggregate_id,event_type,payload,created_at,available_at) VALUES(?,?,?,?,?)`, e.RunID, e.Type, e.Payload, e.CreatedAt, e.CreatedAt); err != nil {
		return e, err
	}
	return e, nil
}

// ListEvents 验证 run 归属后按 sequence 增量读取事件。
func (m *MySQL) ListEvents(ctx context.Context, uid int64, id string, after int64, limit int) ([]domain.Event, error) {
	var exists int
	if err := m.db.QueryRowContext(ctx, `SELECT 1 FROM control_plane_runs WHERE run_id=? AND user_id=?`, id, uid).Scan(&exists); err != nil {
		return nil, translate(err)
	}
	rows, err := m.db.QueryContext(ctx, `SELECT run_id,sequence,event_type,payload,created_at FROM control_plane_events WHERE run_id=? AND sequence>? ORDER BY sequence LIMIT ?`, id, after, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []domain.Event
	for rows.Next() {
		var e domain.Event
		if err = rows.Scan(&e.RunID, &e.Sequence, &e.Type, &e.Payload, &e.CreatedAt); err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	return out, rows.Err()
}
