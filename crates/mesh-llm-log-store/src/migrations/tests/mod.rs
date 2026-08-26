use super::*;
use rusqlite::Connection;

mod compatibility;
mod released_schema_fixture;
mod released_schema_import;
mod runner;
mod schema_contract;
mod sqlite_autoincrement;

pub(super) const EXPECTED_INDEXES: &[&str] = &[
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

pub(super) fn schema_object_names(connection: &Connection, object_type: &str) -> Vec<String> {
    let mut statement = connection
        .prepare(
            "SELECT name FROM sqlite_master \
             WHERE type = ?1 AND name NOT LIKE 'sqlite_%' ORDER BY name",
        )
        .expect("prepare schema objects");
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
