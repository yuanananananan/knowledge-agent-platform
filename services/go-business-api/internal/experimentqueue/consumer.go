// 文件用途：消费 Kafka 实验事件，幂等写入审计表，供监控、统计和故障追查使用。
package experimentqueue

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
	"time"

	"github.com/segmentio/kafka-go"
)

// Consumer 是 experiment-events 的审计消费者；同一事件重复送达时由 event_id 去重。
type Consumer struct {
	db     *sql.DB
	reader *kafka.Reader
	log    *slog.Logger
}

type consumedEvent struct {
	EventID     int64           `json:"event_id"`
	AggregateID string          `json:"aggregate_id"`
	EventType   string          `json:"event_type"`
	Payload     json.RawMessage `json:"payload"`
	OccurredAt  time.Time       `json:"occurred_at"`
}

// NewConsumer 创建共享消费组；多实例部署时一条事件只交给其中一个实例处理。
func NewConsumer(db *sql.DB, brokers []string, topic, groupID string, log *slog.Logger) (*Consumer, error) {
	cleaned := make([]string, 0, len(brokers))
	for _, broker := range brokers {
		if broker = strings.TrimSpace(broker); broker != "" {
			cleaned = append(cleaned, broker)
		}
	}
	if db == nil || len(cleaned) == 0 || strings.TrimSpace(topic) == "" || strings.TrimSpace(groupID) == "" {
		return nil, errors.New("database, Kafka brokers, experiment topic and consumer group are required")
	}
	return &Consumer{
		db:  db,
		log: log,
		reader: kafka.NewReader(kafka.ReaderConfig{
			Brokers:     cleaned,
			Topic:       strings.TrimSpace(topic),
			GroupID:     strings.TrimSpace(groupID),
			MinBytes:    1,
			MaxBytes:    10e6,
			MaxWait:     time.Second,
			StartOffset: kafka.FirstOffset,
		}),
	}, nil
}

// Run 只有在数据库保存成功后才提交 Kafka 位点；进程中断会重投，并由唯一事件编号安全去重。
func (c *Consumer) Run(ctx context.Context) {
	for {
		message, err := c.reader.FetchMessage(ctx)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			c.log.Warn("fetch experiment Kafka event failed", "error", err)
			continue
		}
		if err := c.consume(ctx, message.Value); err != nil {
			c.log.Error("store experiment Kafka event failed", "partition", message.Partition, "offset", message.Offset, "error", err)
			time.Sleep(time.Second)
			continue
		}
		if err := c.reader.CommitMessages(ctx, message); err != nil && ctx.Err() == nil {
			c.log.Warn("commit experiment Kafka offset failed", "partition", message.Partition, "offset", message.Offset, "error", err)
		}
	}
}

func (c *Consumer) consume(ctx context.Context, raw []byte) error {
	var event consumedEvent
	if err := json.Unmarshal(raw, &event); err != nil {
		return err
	}
	if event.EventID <= 0 || event.AggregateID == "" || event.EventType == "" || event.OccurredAt.IsZero() || !json.Valid(event.Payload) {
		return errors.New("invalid experiment event envelope")
	}
	_, err := c.db.ExecContext(ctx, `INSERT INTO experiment_events(event_id,aggregate_id,event_type,payload,occurred_at)
VALUES(?,?,?,?,?) ON DUPLICATE KEY UPDATE event_id=VALUES(event_id)`, event.EventID, event.AggregateID, event.EventType, event.Payload, event.OccurredAt.UTC())
	return err
}

// Close 关闭 Kafka Reader 并退出消费组。
func (c *Consumer) Close() error { return c.reader.Close() }
