// 文件用途：创建和配置 MySQL Store，执行迁移并提供事务级迁移锁与数据库指纹校验。
package store

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"time"

	_ "github.com/go-sql-driver/mysql"
)

// MySQL 是 Store 接口的数据库实现。
type MySQL struct{ db *sql.DB }

// OpenMySQL 打开连接池、设置容量与生命周期，并验证数据库可达性。
func OpenMySQL(dsn string) (*MySQL, error) {
	db, err := sql.Open("mysql", dsn)
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(20)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(4 * time.Minute)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, err
	}
	return &MySQL{db: db}, nil
}

// Migrate 执行一段兼容旧入口的单文件 Schema。
func (m *MySQL) Migrate(ctx context.Context, schema string) error {
	return m.MigrateAll(ctx, []Migration{{Version: "legacy", SQL: schema}})
}

// Migration 保存迁移名称和 SQL 内容；名称用于顺序及防篡改校验。
type Migration struct {
	Version string
	SQL     string
}

// MigrateAll 获取数据库级迁移锁，按顺序执行尚未应用的迁移并记录校验和。
func (m *MySQL) MigrateAll(ctx context.Context, migrations []Migration) error {
	connection, err := m.db.Conn(ctx)
	if err != nil {
		return err
	}
	defer connection.Close()
	var acquired int
	if err := connection.QueryRowContext(ctx, `SELECT GET_LOCK('knowledge-agent:control-plane:migrate', 30)`).Scan(&acquired); err != nil {
		return fmt.Errorf("acquire migration lock: %w", err)
	}
	if acquired != 1 {
		return errors.New("timed out acquiring migration lock")
	}
	defer func() {
		releaseCtx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		var released sql.NullInt64
		_ = connection.QueryRowContext(releaseCtx, `SELECT RELEASE_LOCK('knowledge-agent:control-plane:migrate')`).Scan(&released)
	}()
	if _, err := connection.ExecContext(ctx, `CREATE TABLE IF NOT EXISTS control_plane_schema_migrations (
version VARCHAR(128) PRIMARY KEY, checksum CHAR(64) NOT NULL, dirty BOOLEAN NOT NULL,
applied_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6))`); err != nil {
		return err
	}
	for _, migration := range migrations {
		digest := sha256.Sum256([]byte(migration.SQL))
		checksum := hex.EncodeToString(digest[:])
		var stored string
		var dirty bool
		err := connection.QueryRowContext(ctx, `SELECT checksum,dirty FROM control_plane_schema_migrations WHERE version=?`, migration.Version).Scan(&stored, &dirty)
		if err == nil {
			if dirty {
				return fmt.Errorf("migration %s is dirty; inspect the schema before retrying", migration.Version)
			}
			if stored != checksum {
				return fmt.Errorf("migration %s checksum changed", migration.Version)
			}
			continue
		}
		if !errors.Is(err, sql.ErrNoRows) {
			return err
		}
		if _, err = connection.ExecContext(ctx, `INSERT INTO control_plane_schema_migrations(version,checksum,dirty) VALUES(?,?,TRUE)`, migration.Version, checksum); err != nil {
			return err
		}
		if err = execSchema(ctx, connection, migration.SQL); err != nil {
			return fmt.Errorf("migration %s failed (marked dirty): %w", migration.Version, err)
		}
		if _, err = connection.ExecContext(ctx, `UPDATE control_plane_schema_migrations SET dirty=FALSE,applied_at=UTC_TIMESTAMP(6) WHERE version=?`, migration.Version); err != nil {
			return fmt.Errorf("mark migration %s clean: %w", migration.Version, err)
		}
	}
	return nil
}

// schemaExecutor 抽象能够执行 SQL 的连接或事务。
type schemaExecutor interface {
	ExecContext(context.Context, string, ...any) (sql.Result, error)
}

// execSchema 按分号拆分迁移文本并逐条执行非空语句。
func execSchema(ctx context.Context, executor schemaExecutor, schema string) error {
	lines := strings.Split(schema, "\n")
	for i, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "--") {
			lines[i] = ""
		}
	}
	schema = strings.Join(lines, "\n")
	for _, statement := range strings.Split(schema, ";") {
		statement = strings.TrimSpace(statement)
		if statement == "" {
			continue
		}
		if _, err := executor.ExecContext(ctx, statement); err != nil {
			return err
		}
	}
	return nil
}
