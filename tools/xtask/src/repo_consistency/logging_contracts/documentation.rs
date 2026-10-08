use super::source;
use std::collections::BTreeSet;

const PAGE: &str = "mesh/website/src/docs/pages/logging-api.md";
const GUIDE: &str = "mesh/docs/LOGGING.md";

fn normalized(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}
fn terms(text: &str, expected: &[&str]) {
    for term in expected {
        assert!(text.contains(term), "missing logging contract {term}");
    }
}

#[test]
fn logging_docs_operator_sections_are_present_in_authored_page() {
    let page = source(PAGE);
    let headings: BTreeSet<_> = page
        .lines()
        .filter_map(|line| {
            let title = line.strip_prefix("## ")?;
            Some(
                title
                    .to_lowercase()
                    .split(|c: char| !c.is_ascii_alphanumeric())
                    .filter(|part| !part.is_empty())
                    .collect::<Vec<_>>()
                    .join("-"),
            )
        })
        .collect();
    for required in [
        "scope-and-trust-boundary",
        "status-and-health",
        "read-api",
        "live-sse",
        "export-and-artifact-controls",
        "cleanup-and-deletion",
        "terminal-webhooks",
        "privacy-and-errors",
        "configuration-and-limits",
        "compatibility",
    ] {
        assert!(headings.contains(required), "missing section {required}");
    }
}

#[test]
fn logging_docs_read_mutation_and_stream_routes_are_documented() {
    terms(
        &source(PAGE),
        &[
            "GET /api/status",
            "GET /api/logs/requests",
            "GET /api/logs/requests/{requestId}",
            "GET /api/logs/requests/{requestId}/events",
            "GET /api/logs/requests/{requestId}/artifacts",
            "GET /api/logs/artifacts/{artifactId}",
            "GET /api/logs/proxy",
            "GET /api/logs/audit",
            "GET /api/logs/events",
            "POST /api/logs/requests/export",
            "POST /api/logs/cleanup/preview",
            "POST /api/logs/cleanup/run",
            "POST /api/logs/requests/{requestId}/delete",
            "POST /api/logs/webhooks/{deliveryId}/retry",
        ],
    );
}

#[test]
fn logging_docs_cleanup_route_exclusion_is_exact_and_section_owned() {
    let page = source(PAGE);
    let section = page
        .split_once("## Cleanup and deletion")
        .expect("cleanup section")
        .1
        .split_once("## Terminal webhooks")
        .expect("webhook section")
        .0;
    terms(
        &normalized(section),
        &[
            "\u{60}excluderoute\u{60}",
            "omits rows whose route exactly matches its value",
        ],
    );
}

#[test]
fn logging_docs_recovery_privacy_and_configuration_remain_explicit() {
    terms(
        &source(PAGE),
        &[
            "loopback",
            "\u{60}Host\u{60}",
            "\u{60}Origin\u{60}",
            "nextCursor",
            "Last-Event-ID",
            "v1:",
            "replay_gap",
            "stream_error",
            "operationId",
            "selection fingerprint",
            "request_id",
            "status_code",
            "scheduled",
            "already_scheduled",
            "dead-letter",
            "metadata_only",
            "redacted_artifacts",
            "completed",
            "failed",
            "rejected",
            "cancelled",
            "dropped",
            "logging.enabled",
            "logging.retention_ttl_secs",
            "logging.retention_max_rows",
            "logging.replay_capacity",
            "logging.queue_capacity",
            "logging.artifact.capture_mode",
            "logging.export_limit_bytes",
            "logging.cleanup_cadence_secs",
            "logging.webhook.enabled",
            "logging.webhook.url",
            "logging.webhook.max_attempts",
            "logging.webhook.timeout_secs",
            "logging.webhook.dead_letter_retention_secs",
        ],
    );
}

#[test]
fn logging_docs_stream_error_audit_recovery_is_anchored_in_both_guides() {
    for path in [PAGE, GUIDE] {
        terms(
            &source(path),
            &[
                "invalid_event",
                "audit_reconcile_failed",
                "a1:",
                "GET /api/logs/audit",
            ],
        );
    }
}

#[test]
fn logging_docs_navigation_and_api_reference_link_to_authored_page() {
    for path in [
        "mesh/website/src/_data/docs.js",
        "mesh/website/src/docs/pages/api-reference.md",
    ] {
        terms(&source(path), &["/docs/pages/logging-api/"]);
    }
}

#[test]
fn logging_docs_direct_peer_audit_and_transitive_drop_have_distinct_scope() {
    for path in [PAGE, GUIDE] {
        terms(
            &normalized(&source(path)),
            &[
                "\u{60}gossip_incompatible_version_rejected\u{60} is emitted when a direct peer is",
                "a transitive announcement below the local version floor is dropped without emitting this",
            ],
        );
    }
}

#[test]
fn logging_docs_schema_incompatibility_is_typed_and_version_neutral() {
    for path in [PAGE, GUIDE] {
        let text = normalized(&source(path));
        terms(
            &text,
            &[
                "logging_schema_incompatible",
                "schema_version",
                "supported_schema_version",
                "left unchanged",
                "inference remains available",
                "log_store.db-wal",
                "log_store.db-shm",
                "pragma user_version",
            ],
        );
        assert!(
            text.contains("logging metadata is unavailable")
                || text.contains("logging metadata becomes unavailable")
        );
        for forbidden in [
            "legacy rows",
            "legacy entries",
            "previously retained",
            "older schema",
            "newer schema",
        ] {
            assert!(
                !text.contains(forbidden),
                "schema contract assumes temporal direction: {forbidden}"
            );
        }
    }
}

#[test]
fn logging_docs_stable_503_table_has_exact_error_code_set() {
    let page = source(PAGE);
    let rows = page
        .lines()
        .filter(|line| line.starts_with('|'))
        .map(|line| line.split('|').map(str::trim).collect::<Vec<_>>())
        .filter(|cells| cells.get(1) == Some(&"503"))
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), 1, "one stable 503 table owner is required");
    assert_eq!(
        rows[0].len(),
        4,
        "503 row must have status and code columns"
    );
    let codes: BTreeSet<_> = rows[0][2]
        .split('\u{60}')
        .enumerate()
        .filter_map(|(index, value)| (index % 2 == 1).then_some(value))
        .collect();
    assert_eq!(
        codes,
        BTreeSet::from([
            "artifact_deletion_unavailable",
            "export_timed_out",
            "maintenance_cancelled",
            "logging_unavailable",
            "store_unavailable",
            "logging_schema_incompatible",
        ])
    );
}
