use rusqlite::Connection;

pub(super) mod token;

struct Predicate {
    object_type: &'static str,
    name: &'static str,
    check_count: usize,
    required: &'static [&'static str],
}

pub(super) fn matches(connection: &Connection) -> Result<bool, rusqlite::Error> {
    for object in PREDICATES {
        let sql = connection.query_row(
            "SELECT sql FROM sqlite_schema WHERE type = ?1 AND name = ?2",
            [object.object_type, object.name],
            |row| row.get::<_, String>(0),
        )?;
        let tokens = token::tokenize(&sql);
        let expected_checks = object.required[..object.check_count]
            .iter()
            .map(|predicate| token::tokenize(predicate))
            .collect::<Vec<_>>();
        let exact_partial = object
            .required
            .iter()
            .find(|predicate| predicate.trim_start().starts_with("WHERE "))
            .is_none_or(|predicate| token::tail(&tokens, "WHERE") == token::tokenize(predicate));
        if token::checks(&tokens) != Some(expected_checks)
            || !exact_partial
            || object
                .required
                .iter()
                .any(|predicate| !token::contains(&tokens, &token::tokenize(predicate)))
        {
            return Ok(false);
        }
    }
    Ok(true)
}

const PREDICATES: &[Predicate] = &[
    Predicate {
        object_type: "table",
        name: "summaries",
        check_count: 1,
        required: &[
            "state IN ('active', 'completed', 'failed', 'rejected', 'cancelled', 'dropped')",
        ],
    },
    Predicate {
        object_type: "table",
        name: "lifecycle_events",
        check_count: 1,
        required: &["is_terminal IN (0, 1)", "UNIQUE(request_id, event_id)"],
    },
    Predicate {
        object_type: "table",
        name: "artifact_pointers",
        check_count: 7,
        required: &[
            "bytes >= 0",
            "version >= 1",
            "redacted IN (0, 1)",
            "truncated IN (0, 1)",
            "missing IN (0, 1)",
            "corrupt IN (0, 1)",
            "unavailable_reason IS NULL OR unavailable_reason IN ('streaming_response_not_assembled', 'response_body_not_bounded', 'capture_content_limit_exceeded', 'capture_memory_budget_exceeded', 'artifact_capture_disabled', 'artifact_capture_failed')",
            "UNIQUE(request_id, artifact_id)",
        ],
    },
    Predicate {
        object_type: "table",
        name: "proxy_records",
        check_count: 0,
        required: &["UNIQUE(request_id, attempt_id)"],
    },
    Predicate {
        object_type: "table",
        name: "audit_entries",
        check_count: 1,
        required: &[
            "sequence > 0",
            "entry_id TEXT NOT NULL UNIQUE",
            "UNIQUE(request_id, entry_id)",
        ],
    },
    Predicate {
        object_type: "table",
        name: "webhook_deliveries",
        check_count: 8,
        required: &[
            "terminal_outcome IN ('completed', 'failed', 'rejected', 'cancelled', 'dropped')",
            "terminal_status_code IS NULL OR terminal_status_code BETWEEN 100 AND 599",
            "attempt_number BETWEEN 0 AND 20",
            "status_code IS NULL OR status_code BETWEEN 100 AND 599",
            "state IN ('pending', 'in_flight', 'succeeded', 'retry', 'dead_letter', 'manual_retry')",
            "claim_generation >= 0",
            "max_attempts BETWEEN 1 AND 20",
            "last_error_code IS NULL OR last_error_code IN ('timeout', 'transport', 'http_4xx', 'http_5xx', 'configuration')",
            "UNIQUE(request_id, delivery_id)",
        ],
    },
    Predicate {
        object_type: "table",
        name: "cleanup_runs",
        check_count: 2,
        required: &[
            "deleted_count >= 0",
            "duration_ms IS NULL OR duration_ms >= 0",
        ],
    },
    Predicate {
        object_type: "table",
        name: "maintenance_operations",
        check_count: 17,
        required: &[
            "action IN ('cleanup', 'delete_one')",
            "request_limit BETWEEN 1 AND 100",
            "state IN ('previewed', 'completed', 'partial')",
            "planned_requests >= 0",
            "planned_events >= 0",
            "planned_artifacts >= 0",
            "planned_proxy_records >= 0",
            "planned_database_rows >= 0",
            "executed_requests >= 0",
            "executed_events >= 0",
            "executed_artifacts >= 0",
            "executed_proxy_records >= 0",
            "executed_database_rows >= 0",
            "has_more IN (0, 1)",
            "artifact_files_removed >= 0",
            "artifact_files_failed >= 0",
            "artifact_file_failure_class IS NULL OR artifact_file_failure_class IN ('io', 'unsafe_path')",
        ],
    },
    Predicate {
        object_type: "table",
        name: "maintenance_operation_targets",
        check_count: 1,
        required: &[
            "ordinal >= 0",
            "PRIMARY KEY (operation_id, request_id)",
            "UNIQUE (operation_id, ordinal)",
        ],
    },
    Predicate {
        object_type: "index",
        name: "idx_summaries_terminal_order",
        check_count: 0,
        required: &["COALESCE(terminal_at, created_at)"],
    },
    Predicate {
        object_type: "index",
        name: "idx_audit_entries_severity_occurred",
        check_count: 0,
        required: &[
            "CASE WHEN json_valid(detail_json) THEN json_extract(detail_json, '$.severity') END",
        ],
    },
    Predicate {
        object_type: "index",
        name: "idx_terminal_event_one_per_request",
        check_count: 0,
        required: &["WHERE is_terminal = 1"],
    },
    Predicate {
        object_type: "index",
        name: "idx_webhook_deliveries_ready",
        check_count: 0,
        required: &[
            "COALESCE(next_attempt_at, created_at)",
            "WHERE state IN ('pending', 'retry', 'manual_retry')",
        ],
    },
    Predicate {
        object_type: "index",
        name: "idx_webhook_deliveries_expired_lease",
        check_count: 0,
        required: &["WHERE state = 'in_flight'"],
    },
];
