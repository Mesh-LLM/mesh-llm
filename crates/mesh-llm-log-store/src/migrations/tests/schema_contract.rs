use super::*;

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

#[test]
fn fresh_database_is_exactly_complete_current_schema() {
    let connection = Connection::open_in_memory().expect("open database");
    connection
        .execute_batch("PRAGMA foreign_keys = ON;")
        .expect("enable foreign keys");

    apply_migrations(&connection).expect("apply fresh schema");

    assert_eq!(CURRENT_VERSION, 1);
    assert!(MIGRATIONS.is_empty());
    assert_eq!(
        connection
            .pragma_query_value(None, "application_id", |row| row.get::<_, u32>(0))
            .expect("application identity"),
        APPLICATION_ID
    );
    assert_eq!(
        connection
            .pragma_query_value(None, "user_version", |row| row.get::<_, u32>(0))
            .expect("schema version"),
        1
    );
    let application_tables = schema_object_names(&connection, "table")
        .into_iter()
        .filter(|name| name != "_mesh_llm_log_store_lineage")
        .collect::<Vec<_>>();
    assert_eq!(application_tables, EXPECTED_TABLES);
    assert_eq!(
        private_schema_object_names(&connection),
        ["_mesh_llm_log_store_lineage"]
    );
    assert!(lineage::is_valid(&connection).expect("inspect lineage marker"));
    assert!(
        schema_sql(&connection, "table", "_mesh_llm_log_store_lineage").contains("WITHOUT ROWID")
    );
    assert_eq!(schema_object_names(&connection, "index"), EXPECTED_INDEXES);
    assert_eq!(
        table_columns(&connection, "summaries"),
        [
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
            "caller_endpoint_id",
            "caller_addr",
            "caller_path_type",
        ]
    );
    assert_eq!(
        table_columns(&connection, "artifact_pointers")
            .last()
            .map(String::as_str),
        Some("unavailable_reason")
    );
    assert!(schema_sql(&connection, "table", "lifecycle_events").contains("is_terminal IN (0, 1)"));
    assert!(
        schema_sql(&connection, "index", "idx_terminal_event_one_per_request")
            .contains("WHERE is_terminal = 1")
    );
    assert!(
        schema_sql(&connection, "table", "webhook_deliveries")
            .contains("terminal_status_code BETWEEN 100 AND 599")
    );
    assert!(!schema_sql(&connection, "table", "pending_artifact_deletions").contains("REFERENCES"));
}

fn private_schema_object_names(connection: &Connection) -> Vec<String> {
    let mut statement = connection
        .prepare(
            "SELECT name FROM sqlite_schema
             WHERE name GLOB '_mesh_llm_log_store_*' ORDER BY name",
        )
        .expect("prepare private schema objects");
    statement
        .query_map([], |row| row.get(0))
        .expect("query private schema objects")
        .collect::<Result<Vec<_>, _>>()
        .expect("collect private schema objects")
}

#[test]
fn initial_webhook_schema_bounds_terminal_status_codes() {
    let connection = Connection::open_in_memory().expect("open database");
    apply_migrations(&connection).expect("apply schema");
    connection
        .execute(
            "INSERT INTO summaries (request_id, created_at) VALUES ('request-1', '2026-08-04T12:00:00Z')",
            [],
        )
        .expect("seed request");

    let error = connection
        .execute(
            "INSERT INTO webhook_deliveries \
             (delivery_id, request_id, terminal_outcome, terminal_status_code, occurred_at, target_url, attempt_number) \
             VALUES ('delivery-1', 'request-1', 'completed', 99, '2026-08-04T12:00:00Z', 'configured_webhook', 0)",
            [],
        )
        .expect_err("reject status below 100");

    assert!(
        matches!(error, rusqlite::Error::SqliteFailure(_, Some(message)) if message.contains("terminal_status_code"))
    );
}
