use rusqlite::Connection;

pub(super) const SOURCE_VERSION_THREE: u32 = 3;
pub(super) const SOURCE_VERSION_ELEVEN: u32 = 11;
pub(super) const SOURCE_VERSIONS: [u32; 2] = [SOURCE_VERSION_THREE, SOURCE_VERSION_ELEVEN];

#[derive(Debug, Eq, PartialEq)]
pub(super) struct SeededHistory {
    summary: String,
    lifecycle_event: String,
    audit_entry: String,
}

pub(super) fn connection() -> Connection {
    let connection = Connection::open_in_memory().expect("open released database");
    install(&connection);
    connection
}

pub(super) fn install(connection: &Connection) {
    install_at_marker(connection, SOURCE_VERSION_ELEVEN);
}

pub(super) fn install_at_marker(connection: &Connection, marker: u32) {
    assert!(SOURCE_VERSIONS.contains(&marker));
    connection
        .execute_batch(RELEASED_SCHEMA)
        .expect("install frozen released schema");
    connection
        .pragma_update(None, "user_version", marker)
        .expect("set frozen released schema marker");
}

pub(super) fn install_without_audit_autoincrement(connection: &Connection) {
    let schema =
        RELEASED_SCHEMA.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "INTEGER PRIMARY KEY");
    assert_ne!(
        schema, RELEASED_SCHEMA,
        "fixture mutation must change the schema"
    );
    connection
        .execute_batch(&schema)
        .expect("install frozen released schema without audit autoincrement");
    connection
        .pragma_update(None, "user_version", SOURCE_VERSION_ELEVEN)
        .expect("set frozen released schema marker");
}

pub(super) fn seeded_history(connection: &Connection) -> SeededHistory {
    SeededHistory {
        summary: connection
            .query_row(
                "SELECT json_array(request_id, state, created_at, terminal_at, route, model,
                                   provider, engine, status_code, error_msg, tenant_id, account_id,
                                   user_id)
                 FROM summaries WHERE request_id = 'released-request'",
                [],
                |row| row.get(0),
            )
            .expect("snapshot released summary"),
        lifecycle_event: connection
            .query_row(
                "SELECT json_array(event_id, request_id, occurred_at, payload_json, event_type,
                                   is_terminal)
                 FROM lifecycle_events WHERE event_id = 'released-event'",
                [],
                |row| row.get(0),
            )
            .expect("snapshot released lifecycle event"),
        audit_entry: connection
            .query_row(
                "SELECT json_array(sequence, entry_id, request_id, occurred_at, actor, action,
                                   detail_json)
                 FROM audit_entries WHERE entry_id = 'released-audit'",
                [],
                |row| row.get(0),
            )
            .expect("snapshot released audit entry"),
    }
}

pub(super) fn assert_seeded_history(connection: &Connection) {
    assert_eq!(
        connection
            .query_row(
                "SELECT state FROM summaries WHERE request_id = 'released-request'",
                [],
                |row| row.get::<_, String>(0),
            )
            .expect("released summary"),
        "completed"
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT event_type FROM lifecycle_events WHERE event_id = 'released-event'",
                [],
                |row| row.get::<_, String>(0),
            )
            .expect("released lifecycle event"),
        "completed"
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT action FROM audit_entries WHERE entry_id = 'released-audit'",
                [],
                |row| row.get::<_, String>(0),
            )
            .expect("released audit entry"),
        "request.completed"
    );
}

const RELEASED_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS summaries (
    request_id   TEXT PRIMARY KEY,
    state        TEXT    NOT NULL DEFAULT 'active'
        CHECK (state IN ('active', 'completed', 'failed', 'rejected', 'cancelled', 'dropped')),
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
    event_type   TEXT    NOT NULL DEFAULT 'unknown',
    is_terminal  INTEGER NOT NULL DEFAULT 0
        CHECK (is_terminal IN (0, 1)),
    UNIQUE(request_id, event_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_terminal_event_one_per_request
ON lifecycle_events (request_id)
WHERE is_terminal = 1;
CREATE INDEX IF NOT EXISTS idx_lifecycle_events_occurred ON lifecycle_events (occurred_at DESC, event_id DESC);
CREATE INDEX IF NOT EXISTS idx_lifecycle_events_request ON lifecycle_events (request_id);
CREATE INDEX IF NOT EXISTS idx_lifecycle_events_request_terminal ON lifecycle_events (request_id, is_terminal);

CREATE TABLE IF NOT EXISTS artifact_pointers (
    artifact_id  TEXT PRIMARY KEY,
    request_id   TEXT    NOT NULL REFERENCES summaries(request_id) ON DELETE CASCADE,
    occurred_at  TEXT    NOT NULL,
    kind         TEXT    NOT NULL,
    metadata_json TEXT,
    media_kind   TEXT,
    checksum     TEXT,
    bytes        INTEGER NOT NULL DEFAULT 0 CHECK (bytes >= 0),
    version      INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
    redacted     INTEGER NOT NULL DEFAULT 0 CHECK (redacted IN (0, 1)),
    truncated    INTEGER NOT NULL DEFAULT 0 CHECK (truncated IN (0, 1)),
    stored_at    TEXT,
    missing      INTEGER NOT NULL DEFAULT 0 CHECK (missing IN (0, 1)),
    corrupt      INTEGER NOT NULL DEFAULT 0 CHECK (corrupt IN (0, 1)),
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
    terminal_outcome TEXT NOT NULL
        CHECK (terminal_outcome IN ('completed', 'failed', 'rejected', 'cancelled', 'dropped')),
    terminal_status_code INTEGER
        CHECK (terminal_status_code IS NULL OR terminal_status_code BETWEEN 100 AND 599),
    occurred_at   TEXT    NOT NULL,
    target_url    TEXT    NOT NULL,
    attempt_number INTEGER NOT NULL DEFAULT 0
        CHECK (attempt_number BETWEEN 0 AND 20),
    status_code   INTEGER
        CHECK (status_code IS NULL OR status_code BETWEEN 100 AND 599),
    response_body TEXT,
    error_msg     TEXT,
    state         TEXT    NOT NULL DEFAULT 'succeeded'
        CHECK (state IN ('pending', 'in_flight', 'succeeded', 'retry', 'dead_letter', 'manual_retry')),
    created_at    TEXT    NOT NULL DEFAULT '',
    updated_at    TEXT    NOT NULL DEFAULT '',
    next_attempt_at TEXT,
    lease_expires_at TEXT,
    claim_generation INTEGER NOT NULL DEFAULT 0 CHECK (claim_generation >= 0),
    max_attempts  INTEGER NOT NULL DEFAULT 1 CHECK (max_attempts BETWEEN 1 AND 20),
    last_error_code TEXT
        CHECK (last_error_code IS NULL OR last_error_code IN
            ('timeout', 'transport', 'http_4xx', 'http_5xx', 'configuration')),
    UNIQUE(request_id, delivery_id)
);

CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_occurred ON webhook_deliveries (occurred_at DESC, delivery_id DESC);
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_eligible
ON webhook_deliveries (state, next_attempt_at, lease_expires_at, created_at, delivery_id);

CREATE TABLE IF NOT EXISTS cleanup_runs (
    run_id        TEXT PRIMARY KEY,
    occurred_at   TEXT    NOT NULL,
    policy_name   TEXT    NOT NULL,
    cutoff_before TEXT    NOT NULL,
    deleted_count INTEGER NOT NULL DEFAULT 0 CHECK (deleted_count >= 0),
    duration_ms   INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0)
);

CREATE INDEX IF NOT EXISTS idx_cleanup_runs_occurred ON cleanup_runs (occurred_at DESC, run_id DESC);

CREATE TABLE IF NOT EXISTS maintenance_operations (
    operation_id TEXT PRIMARY KEY,
    action TEXT NOT NULL CHECK (action IN ('cleanup', 'delete_one')),
    cutoff_before TEXT NOT NULL,
    request_limit INTEGER NOT NULL CHECK (request_limit BETWEEN 1 AND 100),
    reason TEXT NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('previewed', 'completed', 'partial')),
    planned_requests INTEGER NOT NULL CHECK (planned_requests >= 0),
    planned_events INTEGER NOT NULL CHECK (planned_events >= 0),
    planned_artifacts INTEGER NOT NULL CHECK (planned_artifacts >= 0),
    planned_proxy_records INTEGER NOT NULL CHECK (planned_proxy_records >= 0),
    planned_database_rows INTEGER NOT NULL CHECK (planned_database_rows >= 0),
    executed_requests INTEGER NOT NULL DEFAULT 0 CHECK (executed_requests >= 0),
    executed_events INTEGER NOT NULL DEFAULT 0 CHECK (executed_events >= 0),
    executed_artifacts INTEGER NOT NULL DEFAULT 0 CHECK (executed_artifacts >= 0),
    executed_proxy_records INTEGER NOT NULL DEFAULT 0 CHECK (executed_proxy_records >= 0),
    executed_database_rows INTEGER NOT NULL DEFAULT 0 CHECK (executed_database_rows >= 0),
    has_more INTEGER NOT NULL CHECK (has_more IN (0, 1)),
    created_at TEXT NOT NULL,
    completed_at TEXT,
    selection_fingerprint TEXT NOT NULL DEFAULT '',
    artifact_files_removed INTEGER NOT NULL DEFAULT 0 CHECK (artifact_files_removed >= 0),
    artifact_files_failed INTEGER NOT NULL DEFAULT 0 CHECK (artifact_files_failed >= 0),
    artifact_file_failure_class TEXT
        CHECK (artifact_file_failure_class IS NULL OR artifact_file_failure_class IN ('io', 'unsafe_path')),
    preview_audit_id TEXT,
    execution_audit_id TEXT,
    cleanup_filters_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS maintenance_operation_targets (
    operation_id TEXT NOT NULL REFERENCES maintenance_operations(operation_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    request_id TEXT NOT NULL,
    PRIMARY KEY (operation_id, request_id),
    UNIQUE (operation_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_maintenance_operation_targets_operation
ON maintenance_operation_targets (operation_id, ordinal);

CREATE TABLE IF NOT EXISTS pending_artifact_deletions (
    artifact_id TEXT PRIMARY KEY,
    request_id  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pending_artifact_deletions_request
ON pending_artifact_deletions (request_id, artifact_id);

CREATE INDEX IF NOT EXISTS idx_summaries_terminal_order
ON summaries (state, COALESCE(terminal_at, created_at), request_id);
CREATE INDEX IF NOT EXISTS idx_summaries_route_created
ON summaries (route, created_at DESC, request_id DESC);
CREATE INDEX IF NOT EXISTS idx_summaries_model_created
ON summaries (model, created_at DESC, request_id DESC);
CREATE INDEX IF NOT EXISTS idx_summaries_provider_created
ON summaries (provider, created_at DESC, request_id DESC);
CREATE INDEX IF NOT EXISTS idx_summaries_engine_created
ON summaries (engine, created_at DESC, request_id DESC);
CREATE INDEX IF NOT EXISTS idx_summaries_status_created
ON summaries (status_code, created_at DESC, request_id DESC);
CREATE INDEX IF NOT EXISTS idx_summaries_state_created
ON summaries (state, created_at DESC, request_id DESC);
CREATE INDEX IF NOT EXISTS idx_lifecycle_events_request_occurred
ON lifecycle_events (request_id, occurred_at ASC, event_id ASC);
CREATE INDEX IF NOT EXISTS idx_artifact_pointers_request_occurred
ON artifact_pointers (request_id, occurred_at ASC, artifact_id ASC);
CREATE INDEX IF NOT EXISTS idx_proxy_records_request_occurred
ON proxy_records (request_id, occurred_at DESC, attempt_id DESC);
CREATE INDEX IF NOT EXISTS idx_proxy_records_provider_occurred
ON proxy_records (provider, occurred_at DESC, attempt_id DESC);
CREATE INDEX IF NOT EXISTS idx_proxy_records_engine_occurred
ON proxy_records (engine, occurred_at DESC, attempt_id DESC);
CREATE INDEX IF NOT EXISTS idx_proxy_records_status_occurred
ON proxy_records (status_code, occurred_at DESC, attempt_id DESC);
CREATE INDEX IF NOT EXISTS idx_audit_entries_actor_occurred
ON audit_entries (actor, occurred_at DESC, entry_id DESC);
CREATE INDEX IF NOT EXISTS idx_audit_entries_severity_occurred
ON audit_entries (
    CASE WHEN json_valid(detail_json) THEN json_extract(detail_json, '$.severity') END,
    occurred_at DESC,
    entry_id DESC
);
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_ready
ON webhook_deliveries (COALESCE(next_attempt_at, created_at), created_at, delivery_id)
WHERE state IN ('pending', 'retry', 'manual_retry');
CREATE INDEX IF NOT EXISTS idx_webhook_deliveries_expired_lease
ON webhook_deliveries (lease_expires_at, created_at, delivery_id)
WHERE state = 'in_flight';

ALTER TABLE artifact_pointers ADD COLUMN unavailable_reason TEXT
    CHECK (unavailable_reason IS NULL OR unavailable_reason IN (
        'streaming_response_not_assembled',
        'response_body_not_bounded',
        'capture_content_limit_exceeded',
        'capture_memory_budget_exceeded',
        'artifact_capture_disabled',
        'artifact_capture_failed'
    ));

INSERT INTO summaries (request_id, state, created_at, terminal_at)
VALUES ('released-request', 'completed', '2026-08-01T00:00:00Z', '2026-08-01T00:00:01Z');
INSERT INTO lifecycle_events
    (event_id, request_id, occurred_at, payload_json, event_type, is_terminal)
VALUES
    ('released-event', 'released-request', '2026-08-01T00:00:01Z', '{}', 'completed', 1);
INSERT INTO audit_entries (entry_id, request_id, occurred_at, actor, action, detail_json)
VALUES
    ('released-audit', 'released-request', '2026-08-01T00:00:02Z', 'system', 'request.completed', '{}');
"#;
