pub(super) struct Table {
    pub(super) name: &'static str,
    pub(super) columns: &'static str,
    pub(super) foreign_keys: &'static str,
    pub(super) implicit_indexes: &'static str,
}

pub(super) const TABLES: &[Table] = &[
    Table {
        name: "artifact_pointers",
        columns: "artifact_id:TEXT:0:-:1:0|request_id:TEXT:1:-:0:0|occurred_at:TEXT:1:-:0:0|kind:TEXT:1:-:0:0|metadata_json:TEXT:0:-:0:0|media_kind:TEXT:0:-:0:0|checksum:TEXT:0:-:0:0|bytes:INTEGER:1:0:0:0|version:INTEGER:1:1:0:0|redacted:INTEGER:1:0:0:0|truncated:INTEGER:1:0:0:0|stored_at:TEXT:0:-:0:0|missing:INTEGER:1:0:0:0|corrupt:INTEGER:1:0:0:0|unavailable_reason:TEXT:0:-:0:0",
        foreign_keys: "summaries:request_id:request_id:NO ACTION:CASCADE:NONE",
        implicit_indexes: "artifact_pointers:sqlite_autoindex_artifact_pointers_1:1:pk:0:0:artifact_id:0:BINARY\nartifact_pointers:sqlite_autoindex_artifact_pointers_2:1:u:0:1:request_id:0:BINARY|0:artifact_id:0:BINARY",
    },
    Table {
        name: "audit_entries",
        columns: "sequence:INTEGER:0:-:1:0|entry_id:TEXT:1:-:0:0|request_id:TEXT:0:-:0:0|occurred_at:TEXT:1:-:0:0|actor:TEXT:1:-:0:0|action:TEXT:1:-:0:0|detail_json:TEXT:0:-:0:0",
        foreign_keys: "summaries:request_id:request_id:NO ACTION:SET NULL:NONE",
        implicit_indexes: "audit_entries:sqlite_autoindex_audit_entries_1:1:u:0:1:entry_id:0:BINARY\naudit_entries:sqlite_autoindex_audit_entries_2:1:u:0:2:request_id:0:BINARY|1:entry_id:0:BINARY",
    },
    Table {
        name: "cleanup_runs",
        columns: "run_id:TEXT:0:-:1:0|occurred_at:TEXT:1:-:0:0|policy_name:TEXT:1:-:0:0|cutoff_before:TEXT:1:-:0:0|deleted_count:INTEGER:1:0:0:0|duration_ms:INTEGER:0:-:0:0",
        foreign_keys: "",
        implicit_indexes: "cleanup_runs:sqlite_autoindex_cleanup_runs_1:1:pk:0:0:run_id:0:BINARY",
    },
    Table {
        name: "lifecycle_events",
        columns: "event_id:TEXT:0:-:1:0|request_id:TEXT:1:-:0:0|occurred_at:TEXT:1:-:0:0|payload_json:TEXT:1:'{}':0:0|event_type:TEXT:1:'unknown':0:0|is_terminal:INTEGER:1:0:0:0",
        foreign_keys: "summaries:request_id:request_id:NO ACTION:CASCADE:NONE",
        implicit_indexes: "lifecycle_events:sqlite_autoindex_lifecycle_events_1:1:pk:0:0:event_id:0:BINARY\nlifecycle_events:sqlite_autoindex_lifecycle_events_2:1:u:0:1:request_id:0:BINARY|0:event_id:0:BINARY",
    },
    Table {
        name: "maintenance_operation_targets",
        columns: "operation_id:TEXT:1:-:1:0|ordinal:INTEGER:1:-:0:0|request_id:TEXT:1:-:2:0",
        foreign_keys: "maintenance_operations:operation_id:operation_id:NO ACTION:CASCADE:NONE",
        implicit_indexes: "maintenance_operation_targets:sqlite_autoindex_maintenance_operation_targets_1:1:pk:0:0:operation_id:0:BINARY|2:request_id:0:BINARY\nmaintenance_operation_targets:sqlite_autoindex_maintenance_operation_targets_2:1:u:0:0:operation_id:0:BINARY|1:ordinal:0:BINARY",
    },
    Table {
        name: "maintenance_operations",
        columns: "operation_id:TEXT:0:-:1:0|action:TEXT:1:-:0:0|cutoff_before:TEXT:1:-:0:0|request_limit:INTEGER:1:-:0:0|reason:TEXT:1:-:0:0|state:TEXT:1:-:0:0|planned_requests:INTEGER:1:-:0:0|planned_events:INTEGER:1:-:0:0|planned_artifacts:INTEGER:1:-:0:0|planned_proxy_records:INTEGER:1:-:0:0|planned_database_rows:INTEGER:1:-:0:0|executed_requests:INTEGER:1:0:0:0|executed_events:INTEGER:1:0:0:0|executed_artifacts:INTEGER:1:0:0:0|executed_proxy_records:INTEGER:1:0:0:0|executed_database_rows:INTEGER:1:0:0:0|has_more:INTEGER:1:-:0:0|created_at:TEXT:1:-:0:0|completed_at:TEXT:0:-:0:0|selection_fingerprint:TEXT:1:'':0:0|artifact_files_removed:INTEGER:1:0:0:0|artifact_files_failed:INTEGER:1:0:0:0|artifact_file_failure_class:TEXT:0:-:0:0|preview_audit_id:TEXT:0:-:0:0|execution_audit_id:TEXT:0:-:0:0|cleanup_filters_json:TEXT:1:'{}':0:0",
        foreign_keys: "",
        implicit_indexes: "maintenance_operations:sqlite_autoindex_maintenance_operations_1:1:pk:0:0:operation_id:0:BINARY",
    },
    Table {
        name: "pending_artifact_deletions",
        columns: "artifact_id:TEXT:0:-:1:0|request_id:TEXT:1:-:0:0",
        foreign_keys: "",
        implicit_indexes: "pending_artifact_deletions:sqlite_autoindex_pending_artifact_deletions_1:1:pk:0:0:artifact_id:0:BINARY",
    },
    Table {
        name: "proxy_records",
        columns: "attempt_id:TEXT:0:-:1:0|request_id:TEXT:1:-:0:0|occurred_at:TEXT:1:-:0:0|target:TEXT:1:-:0:0|provider:TEXT:0:-:0:0|engine:TEXT:0:-:0:0|started_at:TEXT:0:-:0:0|completed_at:TEXT:0:-:0:0|status_code:INTEGER:0:-:0:0|error_msg:TEXT:0:-:0:0",
        foreign_keys: "summaries:request_id:request_id:NO ACTION:CASCADE:NONE",
        implicit_indexes: "proxy_records:sqlite_autoindex_proxy_records_1:1:pk:0:0:attempt_id:0:BINARY\nproxy_records:sqlite_autoindex_proxy_records_2:1:u:0:1:request_id:0:BINARY|0:attempt_id:0:BINARY",
    },
    Table {
        name: "summaries",
        columns: "request_id:TEXT:0:-:1:0|state:TEXT:1:'active':0:0|created_at:TEXT:1:-:0:0|terminal_at:TEXT:0:-:0:0|route:TEXT:0:-:0:0|model:TEXT:0:-:0:0|provider:TEXT:0:-:0:0|engine:TEXT:0:-:0:0|status_code:INTEGER:0:-:0:0|error_msg:TEXT:0:-:0:0|tenant_id:TEXT:0:-:0:0|account_id:TEXT:0:-:0:0|user_id:TEXT:0:-:0:0",
        foreign_keys: "",
        implicit_indexes: "summaries:sqlite_autoindex_summaries_1:1:pk:0:0:request_id:0:BINARY",
    },
    Table {
        name: "webhook_deliveries",
        columns: "delivery_id:TEXT:0:-:1:0|request_id:TEXT:0:-:0:0|terminal_outcome:TEXT:1:-:0:0|terminal_status_code:INTEGER:0:-:0:0|occurred_at:TEXT:1:-:0:0|target_url:TEXT:1:-:0:0|attempt_number:INTEGER:1:0:0:0|status_code:INTEGER:0:-:0:0|response_body:TEXT:0:-:0:0|error_msg:TEXT:0:-:0:0|state:TEXT:1:'succeeded':0:0|created_at:TEXT:1:'':0:0|updated_at:TEXT:1:'':0:0|next_attempt_at:TEXT:0:-:0:0|lease_expires_at:TEXT:0:-:0:0|claim_generation:INTEGER:1:0:0:0|max_attempts:INTEGER:1:1:0:0|last_error_code:TEXT:0:-:0:0",
        foreign_keys: "summaries:request_id:request_id:NO ACTION:SET NULL:NONE",
        implicit_indexes: "webhook_deliveries:sqlite_autoindex_webhook_deliveries_1:1:pk:0:0:delivery_id:0:BINARY\nwebhook_deliveries:sqlite_autoindex_webhook_deliveries_2:1:u:0:1:request_id:0:BINARY|0:delivery_id:0:BINARY",
    },
];

pub(super) struct Index {
    pub(super) name: &'static str,
    pub(super) table: &'static str,
    pub(super) unique: bool,
    pub(super) partial: bool,
    pub(super) columns: &'static str,
}

macro_rules! index {
    ($name:literal, $table:literal, $columns:literal) => {
        Index {
            name: $name,
            table: $table,
            unique: false,
            partial: false,
            columns: $columns,
        }
    };
}

pub(super) const INDEXES: &[Index] = &[
    index!(
        "idx_artifact_pointers_occurred",
        "artifact_pointers",
        "2:occurred_at:1:BINARY|0:artifact_id:1:BINARY"
    ),
    index!(
        "idx_artifact_pointers_request_occurred",
        "artifact_pointers",
        "1:request_id:0:BINARY|2:occurred_at:0:BINARY|0:artifact_id:0:BINARY"
    ),
    index!(
        "idx_audit_entries_actor_occurred",
        "audit_entries",
        "4:actor:0:BINARY|3:occurred_at:1:BINARY|1:entry_id:1:BINARY"
    ),
    index!(
        "idx_audit_entries_occurred",
        "audit_entries",
        "3:occurred_at:1:BINARY|1:entry_id:1:BINARY"
    ),
    index!(
        "idx_audit_entries_severity_occurred",
        "audit_entries",
        "-2:-:0:BINARY|3:occurred_at:1:BINARY|1:entry_id:1:BINARY"
    ),
    index!(
        "idx_cleanup_runs_occurred",
        "cleanup_runs",
        "1:occurred_at:1:BINARY|0:run_id:1:BINARY"
    ),
    index!(
        "idx_lifecycle_events_occurred",
        "lifecycle_events",
        "2:occurred_at:1:BINARY|0:event_id:1:BINARY"
    ),
    index!(
        "idx_lifecycle_events_request",
        "lifecycle_events",
        "1:request_id:0:BINARY"
    ),
    index!(
        "idx_lifecycle_events_request_occurred",
        "lifecycle_events",
        "1:request_id:0:BINARY|2:occurred_at:0:BINARY|0:event_id:0:BINARY"
    ),
    index!(
        "idx_lifecycle_events_request_terminal",
        "lifecycle_events",
        "1:request_id:0:BINARY|5:is_terminal:0:BINARY"
    ),
    index!(
        "idx_maintenance_operation_targets_operation",
        "maintenance_operation_targets",
        "0:operation_id:0:BINARY|1:ordinal:0:BINARY"
    ),
    index!(
        "idx_pending_artifact_deletions_request",
        "pending_artifact_deletions",
        "1:request_id:0:BINARY|0:artifact_id:0:BINARY"
    ),
    index!(
        "idx_proxy_records_engine_occurred",
        "proxy_records",
        "5:engine:0:BINARY|2:occurred_at:1:BINARY|0:attempt_id:1:BINARY"
    ),
    index!(
        "idx_proxy_records_occurred",
        "proxy_records",
        "2:occurred_at:1:BINARY|0:attempt_id:1:BINARY"
    ),
    index!(
        "idx_proxy_records_provider_occurred",
        "proxy_records",
        "4:provider:0:BINARY|2:occurred_at:1:BINARY|0:attempt_id:1:BINARY"
    ),
    index!(
        "idx_proxy_records_request_occurred",
        "proxy_records",
        "1:request_id:0:BINARY|2:occurred_at:1:BINARY|0:attempt_id:1:BINARY"
    ),
    index!(
        "idx_proxy_records_status_occurred",
        "proxy_records",
        "8:status_code:0:BINARY|2:occurred_at:1:BINARY|0:attempt_id:1:BINARY"
    ),
    index!(
        "idx_summaries_created",
        "summaries",
        "2:created_at:1:BINARY|0:request_id:1:BINARY"
    ),
    index!(
        "idx_summaries_engine_created",
        "summaries",
        "7:engine:0:BINARY|2:created_at:1:BINARY|0:request_id:1:BINARY"
    ),
    index!(
        "idx_summaries_model_created",
        "summaries",
        "5:model:0:BINARY|2:created_at:1:BINARY|0:request_id:1:BINARY"
    ),
    index!(
        "idx_summaries_provider_created",
        "summaries",
        "6:provider:0:BINARY|2:created_at:1:BINARY|0:request_id:1:BINARY"
    ),
    index!(
        "idx_summaries_route_created",
        "summaries",
        "4:route:0:BINARY|2:created_at:1:BINARY|0:request_id:1:BINARY"
    ),
    index!("idx_summaries_state", "summaries", "1:state:0:BINARY"),
    index!(
        "idx_summaries_state_created",
        "summaries",
        "1:state:0:BINARY|2:created_at:1:BINARY|0:request_id:1:BINARY"
    ),
    index!(
        "idx_summaries_status_created",
        "summaries",
        "8:status_code:0:BINARY|2:created_at:1:BINARY|0:request_id:1:BINARY"
    ),
    index!(
        "idx_summaries_terminal_order",
        "summaries",
        "1:state:0:BINARY|-2:-:0:BINARY|0:request_id:0:BINARY"
    ),
    Index {
        name: "idx_terminal_event_one_per_request",
        table: "lifecycle_events",
        unique: true,
        partial: true,
        columns: "1:request_id:0:BINARY",
    },
    index!(
        "idx_webhook_deliveries_eligible",
        "webhook_deliveries",
        "10:state:0:BINARY|13:next_attempt_at:0:BINARY|14:lease_expires_at:0:BINARY|11:created_at:0:BINARY|0:delivery_id:0:BINARY"
    ),
    Index {
        name: "idx_webhook_deliveries_expired_lease",
        table: "webhook_deliveries",
        unique: false,
        partial: true,
        columns: "14:lease_expires_at:0:BINARY|11:created_at:0:BINARY|0:delivery_id:0:BINARY",
    },
    index!(
        "idx_webhook_deliveries_occurred",
        "webhook_deliveries",
        "4:occurred_at:1:BINARY|0:delivery_id:1:BINARY"
    ),
    Index {
        name: "idx_webhook_deliveries_ready",
        table: "webhook_deliveries",
        unique: false,
        partial: true,
        columns: "-2:-:0:BINARY|11:created_at:0:BINARY|0:delivery_id:0:BINARY",
    },
];

#[cfg(test)]
mod tests {
    use super::{INDEXES, TABLES};

    #[test]
    fn tables_are_lexicographically_ordered() {
        assert!(TABLES.windows(2).all(|pair| pair[0].name < pair[1].name));
    }

    #[test]
    fn indexes_are_lexicographically_ordered() {
        assert!(INDEXES.windows(2).all(|pair| pair[0].name < pair[1].name));
    }
}
