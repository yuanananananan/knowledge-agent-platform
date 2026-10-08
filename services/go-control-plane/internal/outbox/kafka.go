// 文件用途：封装 Kafka 同步生产者，以 run ID 分区并使用 acks=all 发布 Outbox 事件。
package outbox

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/segmentio/kafka-go"
)

// Kafka publishes transactional-outbox records to a Kafka topic. The
// aggregate ID is used as the message key so events for one run stay ordered
// within a partition.
// Kafka 保存 Broker、Topic 和同步 Writer。
type Kafka struct {
	brokers []string
	topic   string
	writer  *kafka.Writer
}

// kafkaEvent 是写入 Kafka value 的稳定事件信封。
type kafkaEvent struct {
	EventID     int64           `json:"event_id"`
	AggregateID string          `json:"aggregate_id"`
	EventType   string          `json:"event_type"`
	Payload     json.RawMessage `json:"payload"`
}

// NewKafka 清理配置并创建按 aggregate ID 哈希分区的同步生产者。
func NewKafka(brokers []string, topic string) (*Kafka, error) {
	cleaned := make([]string, 0, len(brokers))
	for _, broker := range brokers {
		if broker = strings.TrimSpace(broker); broker != "" {
			cleaned = append(cleaned, broker)
		}
	}
	topic = strings.TrimSpace(topic)
	if len(cleaned) == 0 {
		return nil, fmt.Errorf("at least one Kafka broker is required")
	}
	if topic == "" {
		return nil, fmt.Errorf("Kafka topic is required")
	}

	return &Kafka{
		brokers: cleaned,
		topic:   topic,
		writer: &kafka.Writer{
			Addr:                   kafka.TCP(cleaned...),
			Topic:                  topic,
			Balancer:               &kafka.Hash{},
			BatchSize:              1,
			RequiredAcks:           kafka.RequireAll,
			Async:                  false,
			AllowAutoTopicCreation: false,
		},
	}, nil
}

// Publish 将一条 Outbox 记录编码为 Kafka 消息，并等待 ISR 确认。
func (k *Kafka) Publish(ctx context.Context, id int64, aggregate, event, payload string) error {
	value, err := json.Marshal(kafkaEvent{
		EventID:     id,
		AggregateID: aggregate,
		EventType:   event,
		Payload:     json.RawMessage(payload),
	})
	if err != nil {
		return fmt.Errorf("encode Kafka event: %w", err)
	}

	err = k.writer.WriteMessages(ctx, kafka.Message{
		Key:   []byte(aggregate),
		Value: value,
		Time:  time.Now().UTC(),
		Headers: []kafka.Header{
			{Key: "event_id", Value: []byte(strconv.FormatInt(id, 10))},
			{Key: "event_type", Value: []byte(event)},
		},
	})
	if err != nil {
		return fmt.Errorf("publish Kafka event: %w", err)
	}
	return nil
}

// Ping verifies that at least one configured broker can be reached.
// Ping 连接任一 Broker 并读取 Topic 分区，验证发布依赖是否就绪。
func (k *Kafka) Ping(ctx context.Context) error {
	var lastErr error
	for _, broker := range k.brokers {
		conn, err := (&kafka.Dialer{}).DialContext(ctx, "tcp", broker)
		if err == nil {
			_, err = conn.ReadPartitions(k.topic)
			if err != nil {
				lastErr = err
				_ = conn.Close()
				continue
			}
			if closeErr := conn.Close(); closeErr != nil {
				return fmt.Errorf("close Kafka probe connection: %w", closeErr)
			}
			return nil
		}
		lastErr = err
	}
	return fmt.Errorf("connect to Kafka brokers: %w", lastErr)
}

// Close 刷新并关闭 Kafka Writer。
func (k *Kafka) Close() error {
	return k.writer.Close()
}
