use rusqlite::Connection;

const INITIAL_SCHEMA: &str = r#"
CREATE TABLE summaries (
    request_id TEXT PRIMARY KEY, state TEXT NOT NULL DEFAULT 'active'
        CHECK (state IN ('active', 'completed', 'failed', 'rejected', 'cancelled', 'dropped')),
    created_at TEXT NOT NULL, terminal_at TEXT, route TEXT, model TEXT, provider TEXT,
    engine TEXT, status_code INTEGER, error_msg TEXT, tenant_id TEXT, account_id TEXT, user_id TEXT,
    caller_endpoint_id TEXT, caller_addr TEXT, caller_path_type TEXT
);
CREATE INDEX idx_summaries_created ON summaries (created_at DESC, request_id DESC);
CREATE INDEX idx_summaries_state ON summaries (state);
CREATE INDEX idx_summaries_terminal_order
ON summaries (state, COALESCE(terminal_at, created_at), request_id);
CREATE INDEX idx_summaries_route_created ON summaries (route, created_at DESC, request_id DESC);
CREATE INDEX idx_summaries_model_created ON summaries (model, created_at DESC, request_id DESC);
CREATE INDEX idx_summaries_provider_created ON summaries (provider, created_at DESC, request_id DESC);
CREATE INDEX idx_summaries_engine_created ON summaries (engine, created_at DESC, request_id DESC);
CREATE INDEX idx_summaries_status_created ON summaries (status_code, created_at DESC, request_id DESC);
CREATE INDEX idx_summaries_state_created ON summaries (state, created_at DESC, request_id DESC);

CREATE TABLE lifecycle_events (
    event_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL REFERENCES summaries(request_id) ON DELETE CASCADE,
    occurred_at TEXT NOT NULL, payload_json TEXT NOT NULL DEFAULT '{}',
    event_type TEXT NOT NULL DEFAULT 'unknown', is_terminal INTEGER NOT NULL DEFAULT 0
        CHECK (is_terminal IN (0, 1)), UNIQUE(request_id, event_id)
);
CREATE UNIQUE INDEX idx_terminal_event_one_per_request
ON lifecycle_events (request_id) WHERE is_terminal = 1;
CREATE INDEX idx_lifecycle_events_occurred ON lifecycle_events (occurred_at DESC, event_id DESC);
CREATE INDEX idx_lifecycle_events_request ON lifecycle_events (request_id);
CREATE INDEX idx_lifecycle_events_request_terminal ON lifecycle_events (request_id, is_terminal);
CREATE INDEX idx_lifecycle_events_request_occurred
ON lifecycle_events (request_id, occurred_at ASC, event_id ASC);

CREATE TABLE artifact_pointers (
    artifact_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL REFERENCES summaries(request_id) ON DELETE CASCADE,
    occurred_at TEXT NOT NULL, kind TEXT NOT NULL, metadata_json TEXT, media_kind TEXT,
    checksum TEXT, bytes INTEGER NOT NULL DEFAULT 0 CHECK (bytes >= 0),
    version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
    redacted INTEGER NOT NULL DEFAULT 0 CHECK (redacted IN (0, 1)),
    truncated INTEGER NOT NULL DEFAULT 0 CHECK (truncated IN (0, 1)), stored_at TEXT,
    missing INTEGER NOT NULL DEFAULT 0 CHECK (missing IN (0, 1)),
    corrupt INTEGER NOT NULL DEFAULT 0 CHECK (corrupt IN (0, 1)),
    unavailable_reason TEXT CHECK (unavailable_reason IS NULL OR unavailable_reason IN (
        'streaming_response_not_assembled', 'response_body_not_bounded',
        'capture_content_limit_exceeded', 'capture_memory_budget_exceeded',
        'artifact_capture_disabled', 'artifact_capture_failed'
    )), UNIQUE(request_id, artifact_id)
);
CREATE INDEX idx_artifact_pointers_occurred ON artifact_pointers (occurred_at DESC, artifact_id DESC);
CREATE INDEX idx_artifact_pointers_request_occurred
ON artifact_pointers (request_id, occurred_at ASC, artifact_id ASC);

CREATE TABLE proxy_records (
    attempt_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL REFERENCES summaries(request_id) ON DELETE CASCADE,
    occurred_at TEXT NOT NULL, target TEXT NOT NULL, provider TEXT, engine TEXT,
    started_at TEXT, completed_at TEXT, status_code INTEGER, error_msg TEXT,
    UNIQUE(request_id, attempt_id)
);
CREATE INDEX idx_proxy_records_occurred ON proxy_records (occurred_at DESC, attempt_id DESC);
CREATE INDEX idx_proxy_records_request_occurred
ON proxy_records (request_id, occurred_at DESC, attempt_id DESC);
CREATE INDEX idx_proxy_records_provider_occurred
ON proxy_records (provider, occurred_at DESC, attempt_id DESC);
CREATE INDEX idx_proxy_records_engine_occurred
ON proxy_records (engine, occurred_at DESC, attempt_id DESC);
CREATE INDEX idx_proxy_records_status_occurred
ON proxy_records (status_code, occurred_at DESC, attempt_id DESC);

CREATE TABLE audit_entries (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT CHECK (sequence > 0), entry_id TEXT NOT NULL UNIQUE,
    request_id TEXT REFERENCES summaries(request_id) ON DELETE SET NULL, occurred_at TEXT NOT NULL,
    actor TEXT NOT NULL, action TEXT NOT NULL, detail_json TEXT, UNIQUE(request_id, entry_id)
);
CREATE INDEX idx_audit_entries_occurred ON audit_entries (occurred_at DESC, entry_id DESC);
CREATE INDEX idx_audit_entries_actor_occurred
ON audit_entries (actor, occurred_at DESC, entry_id DESC);
CREATE INDEX idx_audit_entries_severity_occurred ON audit_entries (
    CASE WHEN json_valid(detail_json) THEN json_extract(detail_json, '$.severity') END,
    occurred_at DESC, entry_id DESC
);

CREATE TABLE webhook_deliveries (
    delivery_id TEXT PRIMARY KEY, request_id TEXT REFERENCES summaries(request_id) ON DELETE SET NULL,
    terminal_outcome TEXT NOT NULL
        CHECK (terminal_outcome IN ('completed', 'failed', 'rejected', 'cancelled', 'dropped')),
    terminal_status_code INTEGER CHECK (terminal_status_code IS NULL OR terminal_status_code BETWEEN 100 AND 599),
    occurred_at TEXT NOT NULL, target_url TEXT NOT NULL,
    attempt_number INTEGER NOT NULL DEFAULT 0 CHECK (attempt_number BETWEEN 0 AND 20),
    status_code INTEGER CHECK (status_code IS NULL OR status_code BETWEEN 100 AND 599),
    response_body TEXT, error_msg TEXT,
    state TEXT NOT NULL DEFAULT 'succeeded'
        CHECK (state IN ('pending', 'in_flight', 'succeeded', 'retry', 'dead_letter', 'manual_retry')),
    created_at TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL DEFAULT '', next_attempt_at TEXT,
    lease_expires_at TEXT, claim_generation INTEGER NOT NULL DEFAULT 0 CHECK (claim_generation >= 0),
    max_attempts INTEGER NOT NULL DEFAULT 1 CHECK (max_attempts BETWEEN 1 AND 20),
    last_error_code TEXT CHECK (last_error_code IS NULL OR last_error_code IN
        ('timeout', 'transport', 'http_4xx', 'http_5xx', 'configuration')),
    UNIQUE(request_id, delivery_id)
);
CREATE INDEX idx_webhook_deliveries_occurred ON webhook_deliveries (occurred_at DESC, delivery_id DESC);
CREATE INDEX idx_webhook_deliveries_eligible
ON webhook_deliveries (state, next_attempt_at, lease_expires_at, created_at, delivery_id);
CREATE INDEX idx_webhook_deliveries_ready
ON webhook_deliveries (COALESCE(next_attempt_at, created_at), created_at, delivery_id)
WHERE state IN ('pending', 'retry', 'manual_retry');
CREATE INDEX idx_webhook_deliveries_expired_lease
ON webhook_deliveries (lease_expires_at, created_at, delivery_id) WHERE state = 'in_flight';

CREATE TABLE cleanup_runs (
    run_id TEXT PRIMARY KEY, occurred_at TEXT NOT NULL, policy_name TEXT NOT NULL,
    cutoff_before TEXT NOT NULL, deleted_count INTEGER NOT NULL DEFAULT 0 CHECK (deleted_count >= 0),
    duration_ms INTEGER CHECK (duration_ms IS NULL OR duration_ms >= 0)
);
CREATE INDEX idx_cleanup_runs_occurred ON cleanup_runs (occurred_at DESC, run_id DESC);

CREATE TABLE maintenance_operations (
    operation_id TEXT PRIMARY KEY, action TEXT NOT NULL CHECK (action IN ('cleanup', 'delete_one')),
    cutoff_before TEXT NOT NULL, request_limit INTEGER NOT NULL CHECK (request_limit BETWEEN 1 AND 100),
    reason TEXT NOT NULL, state TEXT NOT NULL CHECK (state IN ('previewed', 'completed', 'partial')),
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
    has_more INTEGER NOT NULL CHECK (has_more IN (0, 1)), created_at TEXT NOT NULL,
    completed_at TEXT, selection_fingerprint TEXT NOT NULL DEFAULT '',
    artifact_files_removed INTEGER NOT NULL DEFAULT 0 CHECK (artifact_files_removed >= 0),
    artifact_files_failed INTEGER NOT NULL DEFAULT 0 CHECK (artifact_files_failed >= 0),
    artifact_file_failure_class TEXT
        CHECK (artifact_file_failure_class IS NULL OR artifact_file_failure_class IN ('io', 'unsafe_path')),
    preview_audit_id TEXT, execution_audit_id TEXT, cleanup_filters_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE maintenance_operation_targets (
    operation_id TEXT NOT NULL REFERENCES maintenance_operations(operation_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0), request_id TEXT NOT NULL,
    PRIMARY KEY (operation_id, request_id), UNIQUE (operation_id, ordinal)
);
CREATE INDEX idx_maintenance_operation_targets_operation
ON maintenance_operation_targets (operation_id, ordinal);
CREATE TABLE pending_artifact_deletions (
    artifact_id TEXT PRIMARY KEY, request_id TEXT NOT NULL
);
CREATE INDEX idx_pending_artifact_deletions_request
ON pending_artifact_deletions (request_id, artifact_id);
"#;

pub(super) fn initialize(connection: &Connection) -> Result<(), rusqlite::Error> {
    connection.execute_batch(INITIAL_SCHEMA)
}
