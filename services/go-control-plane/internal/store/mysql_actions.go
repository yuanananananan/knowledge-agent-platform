// 文件用途：实现动作提案在 MySQL 中的创建、幂等查询、认领、完成和拒绝操作。
package store

import (
	"context"
	"database/sql"
	"encoding/json"

	"knowledge-agent/go-control-plane/internal/domain"
)

// scanAction 将统一列顺序的 SQL 行转换为领域 Action。
func scanAction(row interface{ Scan(...any) error }) (domain.Action, error) {
	var a domain.Action
	var conversation, reason, actionErr sql.NullString
	var result []byte
	err := row.Scan(&a.ID, &a.UserID, &conversation, &a.Type, &a.Payload, &reason, &a.Impact, &a.IdempotencyKey, &a.Status, &result, &actionErr, &a.ExpiresAt, &a.CreatedAt, &a.UpdatedAt)
	a.ConversationID, a.Reason, a.Error = conversation.String, reason.String, actionErr.String
	a.Result = result
	return a, translate(err)
}

// actionColumns 固定查询列顺序，必须与 scanAction 参数保持一致。
const actionColumns = `proposal_id,user_id,conversation_id,action_type,payload,reason,impact,idempotency_key,status,result,error_message,expires_at,created_at,updated_at`

// CreateAction 依赖唯一幂等键创建提案；冲突时返回已有提案。
func (m *MySQL) CreateAction(ctx context.Context, a domain.Action) (domain.Action, bool, error) {
	_, err := m.db.ExecContext(ctx, `INSERT IGNORE INTO control_plane_actions(proposal_id,user_id,conversation_id,action_type,payload,reason,impact,idempotency_key,status,expires_at,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)`, a.ID, a.UserID, a.ConversationID, a.Type, a.Payload, a.Reason, a.Impact, a.IdempotencyKey, a.Status, a.ExpiresAt, a.CreatedAt, a.UpdatedAt)
	if err != nil {
		return a, false, err
	}
	existing, err := scanAction(m.db.QueryRowContext(ctx, `SELECT `+actionColumns+` FROM control_plane_actions WHERE idempotency_key=?`, a.IdempotencyKey))
	return existing, existing.ID == a.ID, err
}

// ListActions 按创建时间倒序分页查询用户提案。
func (m *MySQL) ListActions(ctx context.Context, uid int64, limit, offset int) ([]domain.Action, error) {
	rows, err := m.db.QueryContext(ctx, `SELECT `+actionColumns+` FROM control_plane_actions WHERE user_id=? ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?`, uid, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]domain.Action, 0)
	for rows.Next() {
		a, err := scanAction(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, a)
	}
	return out, rows.Err()
}

// GetAction 使用用户 ID 和提案 ID 防止越权读取。
func (m *MySQL) GetAction(ctx context.Context, uid int64, id string) (domain.Action, error) {
	return scanAction(m.db.QueryRowContext(ctx, `SELECT `+actionColumns+` FROM control_plane_actions WHERE proposal_id=? AND user_id=?`, id, uid))
}

// ClaimAction 仅将未过期的 pending_approval 原子更新为 executing。
func (m *MySQL) ClaimAction(ctx context.Context, uid int64, id string) (domain.Action, error) {
	res, err := m.db.ExecContext(ctx, `UPDATE control_plane_actions SET status=?,updated_at=UTC_TIMESTAMP() WHERE proposal_id=? AND user_id=? AND status=? AND expires_at>UTC_TIMESTAMP()`, domain.ActionExecuting, id, uid, domain.ActionPending)
	if err != nil {
		return domain.Action{}, err
	}
	n, _ := res.RowsAffected()
	if n != 1 {
		return domain.Action{}, domain.ErrConflict
	}
	return m.GetAction(ctx, uid, id)
}

// FinishAction 只允许 executing 提案进入 succeeded 或 failed 终态。
func (m *MySQL) FinishAction(ctx context.Context, id string, status domain.ActionStatus, result json.RawMessage, actionErr string) error {
	res, err := m.db.ExecContext(ctx, `UPDATE control_plane_actions SET status=?,result=?,error_message=?,updated_at=UTC_TIMESTAMP() WHERE proposal_id=? AND status=?`, status, nullableJSON(result), actionErr, id, domain.ActionExecuting)
	if err != nil {
		return err
	}
	n, _ := res.RowsAffected()
	if n != 1 {
		return domain.ErrInvalidState
	}
	return nil
}

// RejectAction 只允许当前用户拒绝 pending_approval 提案。
func (m *MySQL) RejectAction(ctx context.Context, uid int64, id string) error {
	res, err := m.db.ExecContext(ctx, `UPDATE control_plane_actions SET status=?,updated_at=UTC_TIMESTAMP() WHERE proposal_id=? AND user_id=? AND status=?`, domain.ActionRejected, id, uid, domain.ActionPending)
	if err != nil {
		return err
	}
	n, _ := res.RowsAffected()
	if n != 1 {
		return domain.ErrConflict
	}
	return nil
}
