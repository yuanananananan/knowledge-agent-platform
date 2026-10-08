// 文件用途：用 MySQL 发件箱可靠投递实验计算任务，并把实验状态事件发布到 Kafka。
package experimentqueue

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"time"

	"knowledge-agent/go-business-api/internal/python"

	"github.com/segmentio/kafka-go"
)

const (
	kindDispatch = "dispatch"
	kindEvent    = "event"
)

// Compute 是后台投递器实际需要的 Python 计算服务能力。
type Compute interface {
	DispatchCompute(context.Context, python.DispatchRequest) error
}

// Message 表示一条已经被当前后台实例领取的发件箱消息。
type Message struct {
	ID          int64
	Kind        string
	AggregateID string
	TaskType    string
	EventType   string
	Payload     []byte
	Attempts    int
}

// Dispatcher 轮询发件箱：计算命令送入 Celery/Redis，状态事件送入 Kafka。
type Dispatcher struct {
	db       *sql.DB
	compute  Compute
	kafka    *kafka.Writer
	workerID string
	log      *slog.Logger
}

// Migrate 创建实验任务发件箱。业务记录与派发命令可在同一 MySQL 事务中写入。
func Migrate(ctx context.Context, db *sql.DB) error {
	statements := []string{`CREATE TABLE IF NOT EXISTS experiment_outbox (
id BIGINT AUTO_INCREMENT PRIMARY KEY,
message_kind VARCHAR(16) NOT NULL,
aggregate_id VARCHAR(64) NOT NULL,
task_type VARCHAR(32) NULL,
event_type VARCHAR(96) NOT NULL,
payload JSON NOT NULL,
attempts INT NOT NULL DEFAULT 0,
available_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
worker_id VARCHAR(96) NULL,
lease_expires_at DATETIME(6) NULL,
published_at DATETIME(6) NULL,
last_error VARCHAR(1000) NULL,
created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
UNIQUE KEY uq_experiment_outbox_event(message_kind,aggregate_id,event_type),
KEY ix_experiment_outbox_pending(published_at,available_at,lease_expires_at)
)`, `CREATE TABLE IF NOT EXISTS experiment_events (
id BIGINT AUTO_INCREMENT PRIMARY KEY,
event_id BIGINT NOT NULL,
aggregate_id VARCHAR(64) NOT NULL,
event_type VARCHAR(96) NOT NULL,
payload JSON NOT NULL,
occurred_at DATETIME(6) NOT NULL,
consumed_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
UNIQUE KEY uq_experiment_events_event_id(event_id),
KEY ix_experiment_events_aggregate(aggregate_id,event_id),
KEY ix_experiment_events_type(event_type,consumed_at)
)`}
	for _, statement := range statements {
		if _, err := db.ExecContext(ctx, statement); err != nil {
			return err
		}
	}
	return nil
}

// EnqueueDispatch 与实验业务记录共用 tx，消除“记录已创建但 Redis 队列未收到任务”的空档。
func EnqueueDispatch(ctx context.Context, tx *sql.Tx, request python.DispatchRequest) error {
	payload := request.Payload
	if len(payload) == 0 {
		payload = json.RawMessage(`{}`)
	}
	envelope, err := json.Marshal(struct {
		RecordID int64           `json:"record_id"`
		Payload  json.RawMessage `json:"payload"`
	}{RecordID: request.RecordID, Payload: payload})
	if err != nil {
		return err
	}
	_, err = tx.ExecContext(ctx, `INSERT INTO experiment_outbox(message_kind,aggregate_id,task_type,event_type,payload)
VALUES(?,?,?,?,?)`, kindDispatch, request.TaskID, request.TaskType, "compute.requested", envelope)
	return err
}

// NewDispatcher 创建实验消息后台投递器。同一实验编号作为 Kafka Key，保证同一实验事件有序。
func NewDispatcher(db *sql.DB, compute Compute, brokers []string, topic, workerID string, log *slog.Logger) (*Dispatcher, error) {
	cleaned := make([]string, 0, len(brokers))
	for _, broker := range brokers {
		if broker = strings.TrimSpace(broker); broker != "" {
			cleaned = append(cleaned, broker)
		}
	}
	if db == nil || compute == nil || len(cleaned) == 0 || strings.TrimSpace(topic) == "" {
		return nil, errors.New("database, compute client, Kafka brokers and experiment topic are required")
	}
	return &Dispatcher{
		db:       db,
		compute:  compute,
		workerID: workerID,
		log:      log,
		kafka: &kafka.Writer{
			Addr:                   kafka.TCP(cleaned...),
			Topic:                  strings.TrimSpace(topic),
			Balancer:               &kafka.Hash{},
			BatchSize:              1,
			RequiredAcks:           kafka.RequireAll,
			Async:                  false,
			AllowAutoTopicCreation: false,
		},
	}, nil
}

// Run 持续处理待投递消息；失败时指数退避，最长等待一分钟后再次尝试。
func (d *Dispatcher) Run(ctx context.Context) {
	ticker := time.NewTicker(250 * time.Millisecond)
	defer ticker.Stop()
	for {
		msg, err := d.claim(ctx)
		if err == nil {
			err = d.deliver(ctx, msg)
			if err == nil {
				_ = d.markPublished(ctx, msg.ID)
			} else {
				delay := time.Duration(1<<min(msg.Attempts, 6)) * time.Second
				_ = d.retry(ctx, msg.ID, delay, err)
				d.log.Warn("experiment outbox delivery failed", "id", msg.ID, "kind", msg.Kind, "error", err)
			}
		} else if !errors.Is(err, sql.ErrNoRows) {
			d.log.Error("claim experiment outbox", "error", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

func (d *Dispatcher) claim(ctx context.Context) (Message, error) {
	tx, err := d.db.BeginTx(ctx, nil)
	if err != nil {
		return Message{}, err
	}
	defer tx.Rollback()
	var msg Message
	err = tx.QueryRowContext(ctx, `SELECT id,message_kind,aggregate_id,COALESCE(task_type,''),event_type,payload,attempts
FROM experiment_outbox
WHERE published_at IS NULL AND available_at<=UTC_TIMESTAMP(6)
AND (lease_expires_at IS NULL OR lease_expires_at<UTC_TIMESTAMP(6))
ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED`).Scan(
		&msg.ID, &msg.Kind, &msg.AggregateID, &msg.TaskType, &msg.EventType, &msg.Payload, &msg.Attempts,
	)
	if err != nil {
		return Message{}, err
	}
	// 租约要长于 Business API 配置的两分钟 gRPC 超时，避免慢投递被另一实例同时领取。
	_, err = tx.ExecContext(ctx, `UPDATE experiment_outbox SET worker_id=?,lease_expires_at=? WHERE id=?`, d.workerID, time.Now().UTC().Add(3*time.Minute), msg.ID)
	if err != nil {
		return Message{}, err
	}
	return msg, tx.Commit()
}

func (d *Dispatcher) deliver(ctx context.Context, msg Message) error {
	switch msg.Kind {
	case kindDispatch:
		var envelope struct {
			RecordID int64           `json:"record_id"`
			Payload  json.RawMessage `json:"payload"`
		}
		if err := json.Unmarshal(msg.Payload, &envelope); err != nil || envelope.RecordID <= 0 {
			return errors.New("invalid compute dispatch envelope")
		}
		return d.compute.DispatchCompute(ctx, python.DispatchRequest{
			TaskType: msg.TaskType,
			TaskID:   msg.AggregateID,
			RecordID: envelope.RecordID,
			Payload:  envelope.Payload,
		})
	case kindEvent:
		value, err := json.Marshal(map[string]any{
			"event_id":       msg.ID,
			"aggregate_id":   msg.AggregateID,
			"event_type":     msg.EventType,
			"payload":        json.RawMessage(msg.Payload),
			"occurred_at":    time.Now().UTC().Format(time.RFC3339Nano),
			"schema_version": 1,
		})
		if err != nil {
			return err
		}
		return d.kafka.WriteMessages(ctx, kafka.Message{
			Key: []byte(msg.AggregateID), Value: value, Time: time.Now().UTC(),
			Headers: []kafka.Header{
				{Key: "event_id", Value: []byte(strconv.FormatInt(msg.ID, 10))},
				{Key: "event_type", Value: []byte(msg.EventType)},
			},
		})
	default:
		return fmt.Errorf("unsupported experiment outbox kind %q", msg.Kind)
	}
}

func (d *Dispatcher) markPublished(ctx context.Context, id int64) error {
	_, err := d.db.ExecContext(ctx, `UPDATE experiment_outbox SET published_at=UTC_TIMESTAMP(6),lease_expires_at=NULL,last_error=NULL WHERE id=?`, id)
	return err
}

func (d *Dispatcher) retry(ctx context.Context, id int64, delay time.Duration, cause error) error {
	message := cause.Error()
	if len(message) > 1000 {
		message = message[:1000]
	}
	_, err := d.db.ExecContext(ctx, `UPDATE experiment_outbox SET attempts=attempts+1,available_at=?,lease_expires_at=NULL,last_error=? WHERE id=?`, time.Now().UTC().Add(delay), message, id)
	return err
}

// Close 刷新并关闭 Kafka 写入器。
func (d *Dispatcher) Close() error { return d.kafka.Close() }
