//! Forward-only schema migrations for log_store.

use rusqlite::Connection;

/// Current schema version (incremented with each migration).
pub const CURRENT_VERSION: u32 = 4;

const MIGRATIONS_V1: &str = r#"
CREATE TABLE IF NOT EXISTS summaries (
    request_id   TEXT PRIMARY KEY,
    state        TEXT    NOT NULL DEFAULT 'active',
    created_at   TEXT    NOT NULL,
    terminal_at  TEXT,
    route        TEXT,
    model        TEXT,
    provider     TEXT,
    engine       TEXT,
    status_code  INTEGER,
    error_msg    TEXT,
    tenant_id    TEXT,
    account_id   TEXT,
    user_id      TEXT
);

CREATE INDEX IF NOT EXISTS idx_summaries_created ON summaries (created_at DESC, request_id DESC);
CREATE INDEX IF NOT EXISTS idx_summaries_state ON summaries (state);

CREATE TABLE IF NOT EXISTS lifecycle_events (
    event_id     TEXT PRIMARY KEY,
    request_id   TEXT    NOT NULL REFERENCES summaries(request_id) ON DELETE CASCADE,
    occurred_at  TEXT    NOT NULL,
    payload_json TEXT    NOT NULL DEFAULT '{}',

    UNIQUE(request_id, event_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_terminal_event_one_per_request
ON lifecycle_events (request_id)
WHERE payload_json LIKE '%"type":"completed"%'
   OR payload_json LIKE '%"type":"failed"%'
   OR payload_json LIKE '%"type":"rejected"%'
   OR payload_json LIKE '%"type":"cancelled"%';

CREATE INDEX IF NOT EXISTS idx_lifecycle_events_occurred ON lifecycle_events (occurred_at DESC, event_id DESC);
CREATE INDEX IF NOT EXISTS idx_lifecycle_events_request ON lifecycle_events (request_id);

CREATE TABLE IF NOT EXISTS artifact_pointers (
    artifact_id  TEXT PRIMARY KEY,
    request_id   TEXT    NOT NULL REFERENCES summaries(request_id) ON DELETE CASCADE,
    occurred_at  TEXT    NOT NULL,
    kind         TEXT    NOT NULL,
    metadata_json TEXT,

    UNIQUE(request_id, artifact_id)
);

CREATE INDEX IF NOT EXISTS idx_artifact_pointers_occurred ON artifact_pointers (occurred_at DESC, artifact_id DESC);

CREATE TABLE IF NOT EXISTS proxy_records (
    attempt_id   TEXT PRIMARY KEY,
    request_id   TEXT    NOT NULL REFERENCES summaries(request_id) ON DELETE CASCADE,
    occurred_at  TEXT    NOT NULL,
    target       TEXT    NOT NULL,
    provider     TEXT,
    engine       TEXT,
    started_at   TEXT,
    completed_at TEXT,
    status_code  INTEGER,
    error_msg    TEXT,

    UNIQUE(request_id, attempt_id)
);

CREATE INDEX IF NOT EXISTS idx_proxy_records_occurred ON proxy_records (occurred_at DESC, attempt_id DESC);

CREATE TABLE IF NOT EXISTS audit_entries (
    sequence     INTEGER PRIMARY KEY AUTOINCREMENT CHECK (sequence > 0),
    entry_id     TEXT NOT NULL UNIQUE,
    request_id   TEXT REFERENCES summaries(request_id) ON DELETE SET NULL,
    occurred_at  TEXT NOT NULL,
    actor        TEXT NOT NULL,
    action       TEXT NOT NULL,
    detail_json  TEXT,

    UNIQUE(request_id, entry_id)
);

CREATE INDEX IF NOT EXISTS idx_audit_entries_occurred ON audit_entries (occurred_at DESC, entry_id DESC);

CREATE TABLE IF NOT EXISTS webhook_deliveries (
    delivery_id   TEXT PRIMARY KEY,
    request_id    TEXT REFERENCES summaries(request_id) ON DELETE SET NULL,
    occurred_at   TEXT    NOT NULL,
    target_url    TEXT    NOT NULL,
    attempt_number INTEGER NOT NULL,
    status_code   INTEGER,
    response_body TEXT,
    error_msg     TEXT,

    UNIQUE(request_id, delivery_id)
);

CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_occurred ON webhook_deliveries (occurred_at DESC, delivery_id DESC);

CREATE TABLE IF NOT EXISTS cleanup_runs (
    run_id        TEXT PRIMARY KEY,
    occurred_at   TEXT    NOT NULL,
    policy_name   TEXT    NOT NULL,
    cutoff_before TEXT    NOT NULL,
    deleted_count INTEGER NOT NULL DEFAULT 0,
    duration_ms   INTEGER
);

CREATE INDEX IF NOT EXISTS idx_cleanup_runs_occurred ON cleanup_runs (occurred_at DESC, run_id DESC);
"#;

const MIGRATIONS_V2: &str = r#"
ALTER TABLE artifact_pointers ADD COLUMN media_kind TEXT;
ALTER TABLE artifact_pointers ADD COLUMN checksum TEXT;
ALTER TABLE artifact_pointers ADD COLUMN bytes INTEGER NOT NULL DEFAULT 0;
ALTER TABLE artifact_pointers ADD COLUMN version INTEGER NOT NULL DEFAULT 1;
ALTER TABLE artifact_pointers ADD COLUMN redacted INTEGER NOT NULL DEFAULT 0;
ALTER TABLE artifact_pointers ADD COLUMN truncated INTEGER NOT NULL DEFAULT 0;
ALTER TABLE artifact_pointers ADD COLUMN stored_at TEXT;
ALTER TABLE artifact_pointers ADD COLUMN missing INTEGER NOT NULL DEFAULT 0;
ALTER TABLE artifact_pointers ADD COLUMN corrupt INTEGER NOT NULL DEFAULT 0;
"#;

const MIGRATIONS_V3: &str = r#"
DROP INDEX IF EXISTS idx_terminal_event_one_per_request;
CREATE UNIQUE INDEX idx_terminal_event_one_per_request
ON lifecycle_events (request_id)
WHERE payload_json LIKE '%"type":"completed"%'
   OR payload_json LIKE '%"type":"failed"%'
   OR payload_json LIKE '%"type":"rejected"%'
   OR payload_json LIKE '%"type":"cancelled"%'
   OR payload_json LIKE '%"type":"dropped"%';
"#;

const MIGRATIONS_V4: &str = r#"
CREATE TABLE pending_artifact_deletions (
    artifact_id TEXT PRIMARY KEY,
    request_id  TEXT NOT NULL
);
"#;

/// Apply pending migrations one version at a time. Each schema change and its
/// `user_version` bump commit atomically, so an interrupted migration can be
/// retried safely.
pub fn apply_migrations(conn: &Connection) -> Result<(), rusqlite::Error> {
    let current_ver: i32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;

    if current_ver > CURRENT_VERSION as i32 {
        return Err(rusqlite::Error::InvalidQuery);
    }
    if current_ver == CURRENT_VERSION as i32 {
        return Ok(());
    }

    for version in (current_ver + 1)..=CURRENT_VERSION as i32 {
        let sql = match version {
            1 => MIGRATIONS_V1,
            2 => MIGRATIONS_V2,
            3 => MIGRATIONS_V3,
            4 => MIGRATIONS_V4,
            _ => return Err(rusqlite::Error::InvalidQuery),
        };
        let transaction = conn.unchecked_transaction()?;
        transaction.execute_batch(sql)?;
        transaction.pragma_update(None, "user_version", version)?;
        transaction.commit()?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_future_schema_versions_without_modifying_them() {
        let connection = Connection::open_in_memory().unwrap();
        connection.pragma_update(None, "user_version", 99).unwrap();
        assert!(matches!(
            apply_migrations(&connection),
            Err(rusqlite::Error::InvalidQuery)
        ));
        let version: i32 = connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 99);
    }

    #[test]
    fn failed_migration_does_not_advance_schema_version() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute("CREATE TABLE summaries (request_id TEXT PRIMARY KEY)", [])
            .unwrap();
        assert!(apply_migrations(&connection).is_err());
        let version: i32 = connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(version, 0);
    }
}
