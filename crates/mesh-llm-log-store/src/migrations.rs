//! Forward-only schema migrations for log_store.

use rusqlite::Connection;

mod legacy_v10;

/// Current forward-only schema version for the integrated local logging feature.
///
/// Versions 4 through 10 were used by pre-integration development builds. The
/// version remains monotonic so a v10 development store can be identified and
/// upgraded without mistaking it for an unknown future schema.
pub const CURRENT_VERSION: u32 = 11;

const MIGRATIONS_V1: &str = r#"
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
"#;

// Query and retention indexes are intentionally added in a second migration:
// V1 is already durable in the field, and these indexes must be available to
// upgraded stores without rebuilding their logging history.
const MIGRATIONS_V2: &str = r#"
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
"#;

// Artifact bodies can be intentionally unavailable even though their
// metadata record is healthy. Keep that state separate from the existing
// missing/corrupt file-health flags so API consumers never have to infer an
// intentional omission from a null checksum.
const MIGRATIONS_V3: &str = r#"
ALTER TABLE artifact_pointers ADD COLUMN unavailable_reason TEXT
    CHECK (unavailable_reason IS NULL OR unavailable_reason IN (
        'streaming_response_not_assembled',
        'response_body_not_bounded',
        'capture_content_limit_exceeded',
        'capture_memory_budget_exceeded',
        'artifact_capture_disabled',
        'artifact_capture_failed'
    ));
"#;

/// Install the complete V1 schema and its version marker atomically.
pub fn apply_migrations(conn: &Connection) -> Result<(), rusqlite::Error> {
    let current_ver: i32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;

    if !accepted_schema_version(conn, current_ver as u32)? {
        return Err(rusqlite::Error::InvalidQuery);
    }

    match current_ver {
        0 => {
            apply_migration_transactionally(conn, 1, MIGRATIONS_V1)?;
            apply_migration_transactionally(conn, 2, MIGRATIONS_V2)?;
            apply_migration_transactionally(conn, 3, MIGRATIONS_V3)?;
            mark_current(conn)
        }
        1 => {
            apply_migration_transactionally(conn, 2, MIGRATIONS_V2)?;
            apply_migration_transactionally(conn, 3, MIGRATIONS_V3)?;
            mark_current(conn)
        }
        2 => {
            apply_migration_transactionally(conn, 3, MIGRATIONS_V3)?;
            mark_current(conn)
        }
        3 => mark_current(conn),
        legacy_v10::LEGACY_VERSION => legacy_v10::migrate(conn),
        _ => Ok(()),
    }
}

/// Return the version pair when the database cannot be migrated safely.
///
/// The v10 development lineage is accepted only when its structural
/// fingerprint matches. This check is read-only so callers can surface an
/// actionable compatibility state before any migration is attempted.
pub(crate) fn incompatible_schema(
    conn: &Connection,
) -> Result<Option<(u32, u32)>, rusqlite::Error> {
    let found: u32 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    let supported = CURRENT_VERSION;
    let compatible = accepted_schema_version(conn, found)?;
    Ok((!compatible).then_some((found, supported)))
}

/// Whether `version` is a schema the store can open: a blank-slate version
/// the migration steps can upgrade, the current version, or the
/// fingerprint-guarded v10 development lineage.
fn accepted_schema_version(conn: &Connection, version: u32) -> Result<bool, rusqlite::Error> {
    Ok(matches!(version, 0..=3)
        || version == CURRENT_VERSION
        || (version == legacy_v10::LEGACY_VERSION as u32 && legacy_v10::matches(conn)?))
}

fn mark_current(conn: &Connection) -> Result<(), rusqlite::Error> {
    apply_migration_transactionally(conn, CURRENT_VERSION as i32, "")
}

fn apply_migration_transactionally(
    conn: &Connection,
    version: i32,
    migration: &str,
) -> Result<(), rusqlite::Error> {
    let transaction = conn.unchecked_transaction()?;
    transaction.execute_batch(migration)?;
    transaction.execute_batch(&format!("PRAGMA user_version = {version}"))?;
    transaction.commit()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table_columns(connection: &Connection, table: &str) -> Vec<String> {
        let mut statement = connection
            .prepare(&format!("PRAGMA table_info('{table}')"))
            .expect("prepare table info");
        statement
            .query_map([], |row| row.get(1))
            .expect("query table info")
            .collect::<Result<Vec<_>, _>>()
            .expect("collect table columns")
    }

    fn schema_object_names(connection: &Connection, object_type: &str) -> Vec<String> {
        let mut statement = connection
            .prepare(
                "SELECT name FROM sqlite_master \
                 WHERE type = ?1 AND name NOT LIKE 'sqlite_%' ORDER BY name",
            )
            .expect("prepare schema object query");
        statement
            .query_map([object_type], |row| row.get(0))
            .expect("query schema objects")
            .collect::<Result<Vec<_>, _>>()
            .expect("collect schema objects")
    }

    fn schema_sql(connection: &Connection, object_type: &str, name: &str) -> String {
        connection
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type = ?1 AND name = ?2",
                [object_type, name],
                |row| row.get(0),
            )
            .expect("schema SQL")
    }

    const EXPECTED_TABLES: &[&str] = &[
        "artifact_pointers",
        "audit_entries",
        "cleanup_runs",
        "lifecycle_events",
        "maintenance_operation_targets",
        "maintenance_operations",
        "pending_artifact_deletions",
        "proxy_records",
        "summaries",
        "webhook_deliveries",
    ];

    const EXPECTED_TABLE_COLUMNS: &[(&str, &[&str])] = &[
        (
            "summaries",
            &[
                "request_id",
                "state",
                "created_at",
                "terminal_at",
                "route",
                "model",
                "provider",
                "engine",
                "status_code",
                "error_msg",
                "tenant_id",
                "account_id",
                "user_id",
            ],
        ),
        (
            "lifecycle_events",
            &[
                "event_id",
                "request_id",
                "occurred_at",
                "payload_json",
                "event_type",
                "is_terminal",
            ],
        ),
        (
            "artifact_pointers",
            &[
                "artifact_id",
                "request_id",
                "occurred_at",
                "kind",
                "metadata_json",
                "media_kind",
                "checksum",
                "bytes",
                "version",
                "redacted",
                "truncated",
                "stored_at",
                "missing",
                "corrupt",
                "unavailable_reason",
            ],
        ),
        (
            "proxy_records",
            &[
                "attempt_id",
                "request_id",
                "occurred_at",
                "target",
                "provider",
                "engine",
                "started_at",
                "completed_at",
                "status_code",
                "error_msg",
            ],
        ),
        (
            "audit_entries",
            &[
                "sequence",
                "entry_id",
                "request_id",
                "occurred_at",
                "actor",
                "action",
                "detail_json",
            ],
        ),
        (
            "webhook_deliveries",
            &[
                "delivery_id",
                "request_id",
                "terminal_outcome",
                "terminal_status_code",
                "occurred_at",
                "target_url",
                "attempt_number",
                "status_code",
                "response_body",
                "error_msg",
                "state",
                "created_at",
                "updated_at",
                "next_attempt_at",
                "lease_expires_at",
                "claim_generation",
                "max_attempts",
                "last_error_code",
            ],
        ),
        (
            "cleanup_runs",
            &[
                "run_id",
                "occurred_at",
                "policy_name",
                "cutoff_before",
                "deleted_count",
                "duration_ms",
            ],
        ),
        (
            "maintenance_operations",
            &[
                "operation_id",
                "action",
                "cutoff_before",
                "request_limit",
                "reason",
                "state",
                "planned_requests",
                "planned_events",
                "planned_artifacts",
                "planned_proxy_records",
                "planned_database_rows",
                "executed_requests",
                "executed_events",
                "executed_artifacts",
                "executed_proxy_records",
                "executed_database_rows",
                "has_more",
                "created_at",
                "completed_at",
                "selection_fingerprint",
                "artifact_files_removed",
                "artifact_files_failed",
                "artifact_file_failure_class",
                "preview_audit_id",
                "execution_audit_id",
                "cleanup_filters_json",
            ],
        ),
        (
            "maintenance_operation_targets",
            &["operation_id", "ordinal", "request_id"],
        ),
        ("pending_artifact_deletions", &["artifact_id", "request_id"]),
    ];

    const EXPECTED_INDEXES: &[&str] = &[
        "idx_artifact_pointers_occurred",
        "idx_artifact_pointers_request_occurred",
        "idx_audit_entries_actor_occurred",
        "idx_audit_entries_occurred",
        "idx_audit_entries_severity_occurred",
        "idx_cleanup_runs_occurred",
        "idx_lifecycle_events_occurred",
        "idx_lifecycle_events_request",
        "idx_lifecycle_events_request_occurred",
        "idx_lifecycle_events_request_terminal",
        "idx_maintenance_operation_targets_operation",
        "idx_pending_artifact_deletions_request",
        "idx_proxy_records_engine_occurred",
        "idx_proxy_records_occurred",
        "idx_proxy_records_provider_occurred",
        "idx_proxy_records_request_occurred",
        "idx_proxy_records_status_occurred",
        "idx_summaries_created",
        "idx_summaries_engine_created",
        "idx_summaries_model_created",
        "idx_summaries_provider_created",
        "idx_summaries_route_created",
        "idx_summaries_state",
        "idx_summaries_state_created",
        "idx_summaries_status_created",
        "idx_summaries_terminal_order",
        "idx_terminal_event_one_per_request",
        "idx_webhook_deliveries_eligible",
        "idx_webhook_deliveries_expired_lease",
        "idx_webhook_deliveries_occurred",
        "idx_webhook_deliveries_ready",
    ];

    fn assert_schema_identity(connection: &Connection) {
        assert_eq!(
            CURRENT_VERSION, 11,
            "the canonical schema follows the pre-integration v10 lineage"
        );
        assert_eq!(
            connection
                .query_row("PRAGMA user_version", [], |row| row.get::<_, u32>(0))
                .expect("schema version"),
            11
        );
        assert_eq!(schema_object_names(connection, "table"), EXPECTED_TABLES);
        assert_eq!(schema_object_names(connection, "index"), EXPECTED_INDEXES);
    }

    fn assert_schema_columns(connection: &Connection) {
        for (table, expected) in EXPECTED_TABLE_COLUMNS {
            assert_eq!(table_columns(connection, table), *expected, "{table}");
        }
    }

    fn assert_schema_constraints(connection: &Connection) {
        let lifecycle = schema_sql(connection, "table", "lifecycle_events");
        assert!(lifecycle.contains("CHECK (is_terminal IN (0, 1))"));
        assert!(
            schema_sql(connection, "index", "idx_terminal_event_one_per_request")
                .contains("WHERE is_terminal = 1")
        );
        let webhook = schema_sql(connection, "table", "webhook_deliveries");
        for constraint in [
            "terminal_outcome IN",
            "terminal_status_code BETWEEN 100 AND 599",
            "status_code BETWEEN 100 AND 599",
            "state IN",
            "attempt_number BETWEEN 0 AND 20",
            "max_attempts BETWEEN 1 AND 20",
            "claim_generation >= 0",
            "last_error_code IN",
        ] {
            assert!(
                webhook.contains(constraint),
                "missing webhook constraint {constraint}"
            );
        }
        assert!(
            schema_sql(connection, "table", "summaries")
                .contains("state IN ('active', 'completed', 'failed'")
        );
        let artifacts = schema_sql(connection, "table", "artifact_pointers");
        for constraint in [
            "bytes >= 0",
            "version >= 1",
            "redacted IN (0, 1)",
            "truncated IN (0, 1)",
            "missing IN (0, 1)",
            "corrupt IN (0, 1)",
        ] {
            assert!(
                artifacts.contains(constraint),
                "missing artifact constraint {constraint}"
            );
        }
        let maintenance = schema_sql(connection, "table", "maintenance_operations");
        for constraint in [
            "action IN ('cleanup', 'delete_one')",
            "request_limit BETWEEN 1 AND 100",
            "state IN ('previewed', 'completed', 'partial')",
            "has_more IN (0, 1)",
            "artifact_file_failure_class IN ('io', 'unsafe_path')",
        ] {
            assert!(
                maintenance.contains(constraint),
                "missing maintenance constraint {constraint}"
            );
        }
        assert!(
            !schema_sql(connection, "table", "pending_artifact_deletions").contains("REFERENCES"),
            "pending work must outlive deleted pointer and summary owners"
        );
    }

    fn assert_fresh_only_migration_sql() {
        assert!(!MIGRATIONS_V1.contains("ALTER TABLE"));
        assert!(!MIGRATIONS_V1.contains("UPDATE "));
        assert!(!MIGRATIONS_V1.contains("DROP INDEX"));
    }

    #[test]
    fn fresh_database_is_exactly_complete_current_schema() {
        let connection = Connection::open_in_memory().expect("open database");
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .expect("enable foreign keys");
        apply_migrations(&connection).expect("apply fresh schema");

        assert_schema_identity(&connection);

        assert_schema_columns(&connection);
        assert_schema_constraints(&connection);
        assert_fresh_only_migration_sql();
    }

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
    fn rejects_unrecognized_v10_schema_without_modifying_it() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE sentinel (value TEXT); \
                 INSERT INTO sentinel VALUES ('unchanged'); \
                 PRAGMA user_version = 10;",
            )
            .unwrap();

        assert!(matches!(
            apply_migrations(&connection),
            Err(rusqlite::Error::InvalidQuery)
        ));
        assert_eq!(
            connection
                .query_row("PRAGMA user_version", [], |row| row.get::<_, i32>(0))
                .unwrap(),
            10
        );
        assert_eq!(
            connection
                .query_row("SELECT value FROM sentinel", [], |row| row
                    .get::<_, String>(0))
                .unwrap(),
            "unchanged"
        );
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

    #[test]
    fn forced_v1_failure_rolls_back_partial_schema_and_user_version() {
        let connection = Connection::open_in_memory().expect("open database");
        let forced_failure = r#"
            CREATE TABLE must_rollback (id INTEGER PRIMARY KEY);
            CREATE TABLE invalid_sql (
        "#;

        assert!(apply_migration_transactionally(&connection, 1, forced_failure).is_err());
        assert_eq!(
            connection
                .query_row("PRAGMA user_version", [], |row| row.get::<_, u32>(0))
                .expect("schema version"),
            0
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'must_rollback'",
                    [],
                    |row| row.get::<_, u32>(0),
                )
                .expect("partial table count"),
            0
        );
    }

    #[test]
    fn schema_v1_upgrades_to_current_and_current_schema_is_a_noop() {
        let connection = Connection::open_in_memory().expect("open database");
        connection
            .execute_batch(MIGRATIONS_V1)
            .expect("install V1 schema");
        connection.pragma_update(None, "user_version", 1).unwrap();
        apply_migrations(&connection).expect("upgrade V1 to current");
        assert_schema_identity(&connection);

        connection
            .execute_batch(
                "CREATE TABLE sentinel (value TEXT); INSERT INTO sentinel VALUES ('unchanged');",
            )
            .expect("seed sentinel");
        apply_migrations(&connection).expect("current schema is a no-op");
        assert_eq!(
            connection
                .query_row("SELECT value FROM sentinel", [], |row| row
                    .get::<_, String>(0))
                .expect("sentinel"),
            "unchanged"
        );
    }

    #[test]
    fn seeded_v2_artifact_row_survives_v3_with_nullable_unavailable_reason() {
        let root = tempfile::tempdir().expect("database root");
        let connection = Connection::open(root.path().join("log_store.db")).expect("open database");
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .expect("enable foreign keys");
        connection
            .execute_batch(MIGRATIONS_V1)
            .expect("install V1 schema");
        connection
            .execute_batch(MIGRATIONS_V2)
            .expect("install V2 schema");
        connection.pragma_update(None, "user_version", 2).unwrap();
        connection
            .execute(
                "INSERT INTO summaries (request_id, created_at) \
                 VALUES (?1, ?2)",
                [
                    "00000000-0000-4000-8000-000000000401",
                    "2025-01-01T00:00:00.000000000Z",
                ],
            )
            .expect("seed V2 summary");
        connection
            .execute(
                "INSERT INTO artifact_pointers \
                 (artifact_id, request_id, occurred_at, kind, media_kind, checksum, bytes, version, redacted) \
                 VALUES (?1, ?2, ?3, 'response', 'application/json', ?4, 2, 1, 1)",
                rusqlite::params![
                    "00000000-0000-4000-8000-000000000402",
                    "00000000-0000-4000-8000-000000000401",
                    "2025-01-01T00:00:00.100000000Z",
                    "ab"
                ],
            )
            .expect("seed V2 artifact");

        apply_migrations(&connection).expect("upgrade V2 to V3");
        assert_schema_identity(&connection);
        let row = connection
            .query_row(
                "SELECT request_id, kind, media_kind, checksum, bytes, redacted, unavailable_reason \
                 FROM artifact_pointers WHERE artifact_id = ?1",
                ["00000000-0000-4000-8000-000000000402"],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, Option<String>>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, i32>(5)?,
                        row.get::<_, Option<String>>(6)?,
                    ))
                },
            )
            .expect("read migrated artifact row");
        assert_eq!(
            row,
            (
                "00000000-0000-4000-8000-000000000401".to_string(),
                "response".to_string(),
                Some("application/json".to_string()),
                Some("ab".to_string()),
                2,
                1,
                None,
            )
        );
        drop(connection);
        let store = crate::LogStore::reopen_at(root.path(), std::sync::Arc::new(crate::RealClock))
            .expect("reopen migrated store through the production path");
        let page = store
            .query_artifacts(
                "00000000-0000-4000-8000-000000000401",
                &crate::PageQuery {
                    limit: 10,
                    cursor: None,
                    sort: crate::QuerySort::Ascending,
                },
            )
            .expect("read migrated artifact through repository query");
        assert_eq!(page.items.len(), 1);
        assert_eq!(
            page.items[0].artifact_id,
            "00000000-0000-4000-8000-000000000402"
        );
        assert_eq!(page.items[0].unavailable_reason, None);
        assert!(!page.items[0].missing);
        assert!(!page.items[0].corrupt);
    }

    #[test]
    fn initial_webhook_schema_bounds_terminal_status_codes() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch("PRAGMA foreign_keys = ON;")
            .unwrap();
        apply_migrations(&connection).unwrap();
        connection
            .execute(
                "INSERT INTO summaries (request_id, created_at) VALUES ('request-1', '2026-08-04T12:00:00Z')",
                [],
            )
            .unwrap();

        let error = connection
            .execute(
                "INSERT INTO webhook_deliveries \
                 (delivery_id, request_id, terminal_outcome, terminal_status_code, occurred_at, target_url, attempt_number) \
                 VALUES ('delivery-1', 'request-1', 'completed', 99, '2026-08-04T12:00:00Z', 'configured_webhook', 0)",
                [],
            )
            .expect_err("invalid terminal status must violate the V1 schema check");
        assert!(matches!(
            error,
            rusqlite::Error::SqliteFailure(_, Some(message))
                if message.contains("terminal_status_code")
        ));
    }
}
