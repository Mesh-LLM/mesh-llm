//! Compatibility migration for local stores created by pre-integration builds.

use rusqlite::Connection;

pub(super) const LEGACY_VERSION: i32 = 10;

/// Match the abandoned v10 development schema by structure, not only by its
/// version marker. This keeps an unrelated future schema at version 10 from
/// being rewritten.
pub(super) fn matches(conn: &Connection) -> Result<bool, rusqlite::Error> {
    conn.query_row(
        r#"
        SELECT
            EXISTS(SELECT 1 FROM pragma_table_info('lifecycle_events') WHERE name = 'payload_json')
            AND NOT EXISTS(SELECT 1 FROM pragma_table_info('lifecycle_events') WHERE name = 'event_type')
            AND EXISTS(SELECT 1 FROM pragma_table_info('audit_entries') WHERE name = 'source')
            AND NOT EXISTS(SELECT 1 FROM pragma_table_info('audit_entries') WHERE name = 'sequence')
            AND EXISTS(SELECT 1 FROM pragma_table_info('maintenance_operations') WHERE name = 'intent_hash')
            AND EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'maintenance_previews')
        "#,
        [],
        |row| row.get(0),
    )
}

/// Replace the pre-integration schema transactionally while retaining request,
/// lifecycle, artifact, proxy, audit, webhook, and cleanup records. Obsolete
/// maintenance receipts are intentionally retired; their audit entries remain.
pub(super) fn migrate(conn: &Connection) -> Result<(), rusqlite::Error> {
    let transaction = conn.unchecked_transaction()?;
    transaction.execute_batch(LEGACY_PREPARE_SQL)?;
    transaction.execute_batch(super::MIGRATIONS_V1)?;
    transaction.execute_batch(super::MIGRATIONS_V2)?;
    transaction.execute_batch(super::MIGRATIONS_V3)?;
    transaction.execute_batch(LEGACY_COPY_SQL)?;
    transaction.execute_batch(&format!("PRAGMA user_version = {}", super::CURRENT_VERSION))?;
    transaction.commit()
}

const LEGACY_PREPARE_SQL: &str = r#"
PRAGMA defer_foreign_keys = ON;

DROP TRIGGER IF EXISTS maintenance_operation_targets_resolution_insert;
DROP TRIGGER IF EXISTS maintenance_operation_targets_resolution_update;

DROP TABLE IF EXISTS maintenance_operation_remaining_requests;
DROP TABLE IF EXISTS maintenance_preview_targets;
DROP TABLE IF EXISTS maintenance_operation_targets;
DROP TABLE IF EXISTS maintenance_operations;
DROP TABLE IF EXISTS maintenance_previews;
DROP TABLE IF EXISTS pending_artifact_deletions;

ALTER TABLE summaries RENAME TO legacy_v10_summaries;
ALTER TABLE lifecycle_events RENAME TO legacy_v10_lifecycle_events;
ALTER TABLE artifact_pointers RENAME TO legacy_v10_artifact_pointers;
ALTER TABLE proxy_records RENAME TO legacy_v10_proxy_records;
ALTER TABLE audit_entries RENAME TO legacy_v10_audit_entries;
ALTER TABLE webhook_deliveries RENAME TO legacy_v10_webhook_deliveries;
ALTER TABLE cleanup_runs RENAME TO legacy_v10_cleanup_runs;

DROP INDEX IF EXISTS idx_artifact_pointers_occurred;
DROP INDEX IF EXISTS idx_audit_entries_occurred;
DROP INDEX IF EXISTS idx_audit_entries_operation_id;
DROP INDEX IF EXISTS idx_cleanup_runs_occurred;
DROP INDEX IF EXISTS idx_lifecycle_events_occurred;
DROP INDEX IF EXISTS idx_lifecycle_events_request;
DROP INDEX IF EXISTS idx_maintenance_operation_remaining_requests;
DROP INDEX IF EXISTS idx_maintenance_operation_targets_operation;
DROP INDEX IF EXISTS idx_maintenance_preview_targets_operation;
DROP INDEX IF EXISTS idx_proxy_records_occurred;
DROP INDEX IF EXISTS idx_proxy_records_request_attempt;
DROP INDEX IF EXISTS idx_summaries_created;
DROP INDEX IF EXISTS idx_summaries_state;
DROP INDEX IF EXISTS idx_terminal_event_one_per_request;
DROP INDEX IF EXISTS idx_webhook_deliveries_occurred;
"#;

const LEGACY_COPY_SQL: &str = r#"
INSERT INTO summaries (
    request_id, state, created_at, terminal_at, route, model, provider, engine,
    status_code, error_msg, tenant_id, account_id, user_id
)
SELECT
    request_id,
    CASE
        WHEN state IN ('active', 'completed', 'failed', 'rejected', 'cancelled', 'dropped') THEN state
        ELSE 'failed'
    END,
    created_at, terminal_at, route, model, provider, engine, status_code,
    error_msg, tenant_id, account_id, user_id
FROM legacy_v10_summaries;

INSERT INTO lifecycle_events (
    event_id, request_id, occurred_at, payload_json, event_type, is_terminal
)
SELECT
    event_id,
    request_id,
    occurred_at,
    payload_json,
    CASE
        WHEN json_valid(payload_json) THEN COALESCE(json_extract(payload_json, '$.type'), 'unknown')
        ELSE 'unknown'
    END,
    CASE
        WHEN json_valid(payload_json)
         AND json_extract(payload_json, '$.type') IN ('completed', 'failed', 'rejected', 'cancelled', 'dropped')
        THEN 1
        ELSE 0
    END
FROM legacy_v10_lifecycle_events;

INSERT INTO artifact_pointers (
    artifact_id, request_id, occurred_at, kind, metadata_json, media_kind,
    checksum, bytes, version, redacted, truncated, stored_at, missing, corrupt,
    unavailable_reason
)
SELECT
    artifact_id, request_id, occurred_at, kind, metadata_json, media_kind,
    checksum, MAX(bytes, 0), MAX(version, 1),
    CASE WHEN redacted = 1 THEN 1 ELSE 0 END,
    CASE WHEN truncated = 1 THEN 1 ELSE 0 END,
    stored_at,
    CASE WHEN missing = 1 THEN 1 ELSE 0 END,
    CASE WHEN corrupt = 1 THEN 1 ELSE 0 END,
    NULL
FROM legacy_v10_artifact_pointers;

INSERT INTO proxy_records (
    attempt_id, request_id, occurred_at, target, provider, engine, started_at,
    completed_at, status_code, error_msg
)
SELECT
    attempt_id, request_id, occurred_at, target, provider, engine, started_at,
    completed_at, status_code, error_msg
FROM legacy_v10_proxy_records;

INSERT INTO audit_entries (
    sequence, entry_id, request_id, occurred_at, actor, action, detail_json
)
SELECT
    ROW_NUMBER() OVER (ORDER BY occurred_at, entry_id),
    entry_id,
    request_id,
    occurred_at,
    CASE
        WHEN actor IN ('logging_service', 'runtime', 'mesh', 'cli', 'logs_api') THEN actor
        ELSE 'logging_service'
    END,
    action,
    detail_json
FROM legacy_v10_audit_entries;

INSERT INTO webhook_deliveries (
    delivery_id, request_id, terminal_outcome, terminal_status_code, occurred_at,
    target_url, attempt_number, status_code, state, created_at, updated_at,
    max_attempts
)
SELECT
    delivery.delivery_id,
    delivery.request_id,
    CASE
        WHEN summary.state IN ('completed', 'failed', 'rejected', 'cancelled', 'dropped') THEN summary.state
        ELSE 'failed'
    END,
    CASE WHEN summary.status_code BETWEEN 100 AND 599 THEN summary.status_code ELSE NULL END,
    delivery.occurred_at,
    'configured_webhook',
    MIN(MAX(delivery.attempt_number, 0), 20),
    CASE WHEN delivery.status_code BETWEEN 100 AND 599 THEN delivery.status_code ELSE NULL END,
    CASE WHEN delivery.status_code BETWEEN 200 AND 299 THEN 'succeeded' ELSE 'dead_letter' END,
    delivery.occurred_at,
    delivery.occurred_at,
    MIN(MAX(delivery.attempt_number, 1), 20)
FROM legacy_v10_webhook_deliveries AS delivery
LEFT JOIN legacy_v10_summaries AS summary ON summary.request_id = delivery.request_id;

INSERT INTO cleanup_runs (
    run_id, occurred_at, policy_name, cutoff_before, deleted_count, duration_ms
)
SELECT
    run_id, occurred_at, policy_name, cutoff_before, MAX(deleted_count, 0),
    CASE WHEN duration_ms >= 0 THEN duration_ms ELSE NULL END
FROM legacy_v10_cleanup_runs;

DROP TABLE legacy_v10_lifecycle_events;
DROP TABLE legacy_v10_artifact_pointers;
DROP TABLE legacy_v10_proxy_records;
DROP TABLE legacy_v10_webhook_deliveries;
DROP TABLE legacy_v10_audit_entries;
DROP TABLE legacy_v10_cleanup_runs;
DROP TABLE legacy_v10_summaries;
"#;

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rusqlite::Connection;

    use crate::{LogStore, RealClock};

    #[test]
    fn fingerprinted_v10_store_upgrades_and_preserves_ledger_rows() {
        let root = tempfile::tempdir().expect("legacy database root");
        let database = root.path().join("log_store.db");
        let connection = Connection::open(&database).expect("open legacy database");
        connection
            .execute_batch(LEGACY_FIXTURE_SQL)
            .expect("seed legacy database");
        drop(connection);

        let store = LogStore::open(root.path(), Arc::new(RealClock)).expect("upgrade legacy store");

        assert_eq!(store.schema_version(), super::super::CURRENT_VERSION);
        assert!(store.get_summary("legacy-request").unwrap().is_some());
        let connection = store.conn();
        let event: (String, i64) = connection
            .query_row(
                "SELECT event_type, is_terminal FROM lifecycle_events WHERE event_id = 'legacy-event'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("read migrated event");
        assert_eq!(event, ("completed".to_string(), 1));
        let audit: (i64, String) = connection
            .query_row(
                "SELECT sequence, actor FROM audit_entries WHERE entry_id = 'legacy-audit'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("read migrated audit");
        assert_eq!(audit, (1, "logging_service".to_string()));
        let obsolete_tables: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name LIKE 'legacy_v10_%'",
                [],
                |row| row.get(0),
            )
            .expect("count legacy tables");
        assert_eq!(obsolete_tables, 0);
    }

    const LEGACY_FIXTURE_SQL: &str = r#"
    PRAGMA foreign_keys = ON;
    CREATE TABLE summaries (
        request_id TEXT PRIMARY KEY, state TEXT NOT NULL DEFAULT 'active', created_at TEXT NOT NULL,
        terminal_at TEXT, route TEXT, model TEXT, provider TEXT, engine TEXT, status_code INTEGER,
        error_msg TEXT, tenant_id TEXT, account_id TEXT, user_id TEXT
    );
    CREATE TABLE lifecycle_events (
        event_id TEXT PRIMARY KEY, request_id TEXT NOT NULL REFERENCES summaries(request_id) ON DELETE CASCADE,
        occurred_at TEXT NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}', UNIQUE(request_id, event_id)
    );
    CREATE TABLE artifact_pointers (
        artifact_id TEXT PRIMARY KEY, request_id TEXT NOT NULL REFERENCES summaries(request_id) ON DELETE CASCADE,
        occurred_at TEXT NOT NULL, kind TEXT NOT NULL, metadata_json TEXT, media_kind TEXT, checksum TEXT,
        bytes INTEGER NOT NULL DEFAULT 0, version INTEGER NOT NULL DEFAULT 1, redacted INTEGER NOT NULL DEFAULT 0,
        truncated INTEGER NOT NULL DEFAULT 0, stored_at TEXT, missing INTEGER NOT NULL DEFAULT 0,
        corrupt INTEGER NOT NULL DEFAULT 0, UNIQUE(request_id, artifact_id)
    );
    CREATE TABLE proxy_records (
        attempt_id TEXT PRIMARY KEY, request_id TEXT NOT NULL REFERENCES summaries(request_id) ON DELETE CASCADE,
        occurred_at TEXT NOT NULL, target TEXT NOT NULL, provider TEXT, engine TEXT, started_at TEXT,
        completed_at TEXT, status_code INTEGER, error_msg TEXT, attempt_number INTEGER NOT NULL DEFAULT 0,
        UNIQUE(request_id, attempt_id)
    );
    CREATE TABLE audit_entries (
        entry_id TEXT PRIMARY KEY, request_id TEXT REFERENCES summaries(request_id) ON DELETE SET NULL,
        occurred_at TEXT NOT NULL, actor TEXT NOT NULL, action TEXT NOT NULL, detail_json TEXT,
        source TEXT NOT NULL DEFAULT 'system', reason TEXT, result TEXT NOT NULL DEFAULT 'succeeded',
        operation_id TEXT, UNIQUE(request_id, entry_id)
    );
    CREATE TABLE webhook_deliveries (
        delivery_id TEXT PRIMARY KEY, request_id TEXT REFERENCES summaries(request_id) ON DELETE SET NULL,
        occurred_at TEXT NOT NULL, target_url TEXT NOT NULL, attempt_number INTEGER NOT NULL,
        status_code INTEGER, response_body TEXT, error_msg TEXT, UNIQUE(request_id, delivery_id)
    );
    CREATE TABLE cleanup_runs (
        run_id TEXT PRIMARY KEY, occurred_at TEXT NOT NULL, policy_name TEXT NOT NULL,
        cutoff_before TEXT NOT NULL, deleted_count INTEGER NOT NULL DEFAULT 0, duration_ms INTEGER
    );
    CREATE TABLE maintenance_previews (operation_id TEXT PRIMARY KEY);
    CREATE TABLE maintenance_operations (operation_id TEXT PRIMARY KEY, intent_hash TEXT NOT NULL);
    INSERT INTO summaries (
        request_id, state, created_at, terminal_at, route, model, provider, engine, status_code
    ) VALUES (
        'legacy-request', 'completed', '2026-08-03T03:50:26.386Z', '2026-08-03T03:50:26.387Z',
        'chat_completions', 'legacy-model', 'local', 'skippy', 200
    );
    INSERT INTO lifecycle_events (event_id, request_id, occurred_at, payload_json)
    VALUES (
        'legacy-event', 'legacy-request', '2026-08-03T03:50:26.387Z',
        '{"type":"completed","status_code":200}'
    );
    INSERT INTO audit_entries (entry_id, occurred_at, actor, action)
    VALUES ('legacy-audit', '2026-08-03T03:50:27Z', 'system', 'cleanup_preview');
    PRAGMA user_version = 10;
    "#;
}
