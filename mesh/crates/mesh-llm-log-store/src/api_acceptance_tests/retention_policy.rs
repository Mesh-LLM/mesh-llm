use super::*;

#[test]
fn max_row_prune_is_deterministic_preserves_active_and_returns_owned_artifact_pointers() {
    let (store, _, _tmp) = open_store();
    for request_id in ["active", "tie-a", "tie-b", "newest"] {
        store
            .insert_summary(
                request_id,
                None,
                None,
                None,
                None,
                "2025-01-01T00:00:00Z",
                None,
                None,
                None,
            )
            .expect("insert summary");
    }
    for request_id in ["tie-a", "tie-b"] {
        store
            .write_terminal_event(
                request_id,
                &format!("event-{request_id}"),
                r#"{"type":"completed"}"#,
                "completed",
                None,
                "2025-02-01T00:00:00Z",
            )
            .expect("write terminal event");
    }
    store
        .write_terminal_event(
            "newest",
            "event-newest",
            r#"{"type":"completed"}"#,
            "completed",
            None,
            "2025-03-01T00:00:00Z",
        )
        .expect("write newest terminal event");
    store
        .insert_artifact_pointer(
            "artifact-tie-a",
            "tie-a",
            "2025-02-01T00:00:00Z",
            "request_body",
            None,
        )
        .expect("insert pointer");

    let (deleted, pointers) = store
        .cascade_prune_terminal_summaries_to_max_rows(2)
        .expect("prune terminal history");
    assert_eq!(deleted, 3, "summary, terminal event, and pointer");
    assert_eq!(
        pointers,
        vec![super::repositories::CascadeArtifactPointer {
            artifact_id: "artifact-tie-a".to_string(),
            request_id: "tie-a".to_string(),
        }]
    );
    assert!(store.get_summary("active").unwrap().is_some());
    assert!(store.get_summary("tie-a").unwrap().is_none());
    assert!(store.get_summary("tie-b").unwrap().is_some());
    assert!(store.get_summary("newest").unwrap().is_some());
    assert_eq!(store.count_table("artifact_pointers").unwrap(), 0);
}

#[test]
fn ttl_and_max_row_retention_queue_every_artifact_before_owner_cascade() {
    let (store, _, _tmp) = open_store();
    for (request_id, occurred_at) in [
        ("ttl-owner", "2025-01-01T00:00:00Z"),
        ("cap-owner", "2025-03-01T00:00:00Z"),
        ("retained-owner", "2025-04-01T00:00:00Z"),
    ] {
        store
            .insert_summary(
                request_id,
                None,
                None,
                None,
                None,
                occurred_at,
                None,
                None,
                None,
            )
            .expect("summary");
        store
            .write_terminal_event(
                request_id,
                &format!("terminal-{request_id}"),
                r#"{"type":"completed"}"#,
                "completed",
                None,
                occurred_at,
            )
            .expect("terminal");
        store
            .insert_artifact_pointer(
                &format!("artifact-{request_id}"),
                request_id,
                occurred_at,
                "response",
                None,
            )
            .expect("artifact pointer");
    }

    store
        .cascade_cleanup_before("2025-02-01T00:00:00Z")
        .expect("TTL retention");
    assert_eq!(
        pending_artifact_deletions(&store),
        vec![("artifact-ttl-owner".to_owned(), "ttl-owner".to_owned())]
    );
    assert!(store.get_summary("ttl-owner").unwrap().is_none());

    store
        .cascade_prune_terminal_summaries_to_max_rows(1)
        .expect("max-row retention");
    assert_eq!(
        pending_artifact_deletions(&store),
        vec![
            ("artifact-cap-owner".to_owned(), "cap-owner".to_owned()),
            ("artifact-ttl-owner".to_owned(), "ttl-owner".to_owned()),
        ]
    );
    assert!(store.get_summary("cap-owner").unwrap().is_none());
    assert!(store.get_summary("retained-owner").unwrap().is_some());
}

#[test]
fn max_row_prune_is_idempotent_after_retention_is_satisfied() {
    let (store, _, _tmp) = open_store();
    store
        .insert_summary(
            "completed",
            None,
            None,
            None,
            None,
            "2025-01-01T00:00:00Z",
            None,
            None,
            None,
        )
        .expect("insert summary");
    store
        .write_terminal_event(
            "completed",
            "event-completed",
            r#"{"type":"completed"}"#,
            "completed",
            None,
            "2025-01-01T00:00:01Z",
        )
        .expect("write terminal event");

    assert_eq!(
        store
            .cascade_prune_terminal_summaries_to_max_rows(1)
            .expect("initial no-op"),
        (0, Vec::new())
    );
}

#[test]
fn max_row_prune_survives_store_restart_with_the_retained_summaries_intact() {
    let (store, clock, tmp) = open_store();
    for (request_id, occurred_at) in [
        ("oldest", "2025-01-01T00:00:01Z"),
        ("newer", "2025-01-01T00:00:02Z"),
    ] {
        store
            .insert_summary(
                request_id,
                None,
                None,
                None,
                None,
                occurred_at,
                None,
                None,
                None,
            )
            .expect("insert summary");
        store
            .write_terminal_event(
                request_id,
                &format!("event-{request_id}"),
                r#"{"type":"completed"}"#,
                "completed",
                None,
                occurred_at,
            )
            .expect("write terminal");
    }
    store
        .cascade_prune_terminal_summaries_to_max_rows(1)
        .expect("prune before restart");
    drop(store);

    let reopened = LogStore::reopen_at(tmp.path(), clock).expect("reopen store");
    assert!(reopened.get_summary("oldest").unwrap().is_none());
    assert!(reopened.get_summary("newer").unwrap().is_some());
    assert_eq!(
        reopened
            .cascade_prune_terminal_summaries_to_max_rows(1)
            .expect("idempotent after restart"),
        (0, Vec::new())
    );
}

#[test]
fn retention_policy_uses_summary_ownership_and_per_table_ttl_after_reopen() {
    let (store, clock, tmp) = open_store();
    let old = "2025-01-01T00:00:00Z";
    let fresh = "2025-03-01T00:00:00Z";
    let cutoff = "2025-02-01T00:00:00Z";

    for request_id in ["expired-terminal", "retained-terminal", "active"] {
        store
            .insert_summary(request_id, None, None, None, None, old, None, None, None)
            .expect("insert summary");
    }
    store
        .write_terminal_event(
            "expired-terminal",
            "expired-terminal-event",
            r#"{"type":"completed"}"#,
            "completed",
            None,
            old,
        )
        .expect("write expired terminal");
    store
        .write_terminal_event(
            "retained-terminal",
            "retained-terminal-event",
            r#"{"type":"completed"}"#,
            "completed",
            None,
            fresh,
        )
        .expect("write retained terminal");
    store
        .conn()
        .execute(
            "INSERT INTO lifecycle_events (event_id, request_id, occurred_at, payload_json)\n\
             VALUES ('retained-old-event', 'retained-terminal', ?1, '{\"type\":\"chunk\"}')",
            rusqlite::params![old],
        )
        .expect("insert retained old event");
    store
        .insert_proxy_record(
            "expired-proxy",
            "expired-terminal",
            old,
            "target",
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("insert expired proxy");
    store
        .insert_proxy_record(
            "retained-proxy",
            "retained-terminal",
            old,
            "target",
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("insert retained proxy");
    store
        .insert_artifact_pointer("expired-artifact", "expired-terminal", old, "request", None)
        .expect("insert expired pointer");
    store
        .insert_artifact_pointer("active-artifact", "active", old, "request", None)
        .expect("insert active pointer");
    store
        .insert_audit_entry("old-audit", None, old, "operator", "test", None)
        .expect("insert audit");
    store
        .insert_webhook_delivery("old-webhook", None, old, 1, None)
        .expect("insert webhook");
    store
        .insert_cleanup_run("old-cleanup", old, "test", cutoff, 0, None)
        .expect("insert cleanup run");

    let result = store
        .apply_retention_policy(cutoff, 100)
        .expect("apply retention");
    assert_eq!(result.ttl_deleted_count, 11);
    assert_eq!(result.max_rows_deleted_count, 0);
    assert_eq!(
        result.artifact_pointers,
        vec![super::repositories::CascadeArtifactPointer {
            artifact_id: "expired-artifact".to_string(),
            request_id: "expired-terminal".to_string(),
        }]
    );
    assert!(store.get_summary("expired-terminal").unwrap().is_none());
    assert!(store.get_summary("retained-terminal").unwrap().is_none());
    assert!(store.get_summary("active").unwrap().is_some());
    assert!(
        store
            .get_artifact_pointer("active-artifact")
            .unwrap()
            .is_some()
    );
    assert!(result.table_results.iter().all(|entry| {
        matches!(
            entry.table,
            super::repositories::RetentionTable::Summaries
                | super::repositories::RetentionTable::LifecycleEvents
                | super::repositories::RetentionTable::ArtifactPointers
                | super::repositories::RetentionTable::ProxyRecords
                | super::repositories::RetentionTable::AuditEntries
                | super::repositories::RetentionTable::WebhookDeliveries
                | super::repositories::RetentionTable::CleanupRuns
        )
    }));
    assert_eq!(store.count_table("proxy_records").unwrap(), 0);
    assert_eq!(store.count_table("audit_entries").unwrap(), 0);
    assert_eq!(store.count_table("webhook_deliveries").unwrap(), 0);
    assert_eq!(store.count_table("cleanup_runs").unwrap(), 0);

    drop(store);
    let reopened = LogStore::reopen_at(tmp.path(), clock).expect("reopen store");
    assert_eq!(
        reopened
            .apply_retention_policy(cutoff, 100)
            .expect("idempotent retention after reopen"),
        super::repositories::RetentionCleanupResult {
            ttl_deleted_count: 0,
            max_rows_deleted_count: 0,
            artifact_pointers: vec![super::repositories::CascadeArtifactPointer {
                artifact_id: "expired-artifact".to_string(),
                request_id: "expired-terminal".to_string(),
            }],
            table_results: super::repositories::RetentionTable::ALL
                .into_iter()
                .map(|table| super::repositories::RetentionTableResult {
                    table,
                    ttl_deleted_count: 0,
                    max_rows_deleted_count: 0,
                })
                .collect(),
        }
    );
}

#[test]
fn webhook_dead_letter_retention_uses_transition_time_and_preserves_generic_policy() {
    use super::repositories::{RetentionPolicy, RetentionTable};

    let (store, clock, tmp) = open_store();
    let generic_cutoff = "2025-01-01T00:00:00Z";
    let generic_cleanup_cutoff = "2025-03-01T00:00:00Z";
    let dead_letter_cutoff = "2025-04-01T00:00:00Z";
    let occurred_at = "2025-02-01T00:00:00Z";
    let insert = |delivery_id: &str, state: &str, updated_at: &str| {
        store
            .conn()
            .execute(
                r#"
                INSERT INTO webhook_deliveries
                    (delivery_id, request_id, terminal_outcome, occurred_at, target_url, attempt_number, response_body, error_msg,
                     state, created_at, updated_at, next_attempt_at, lease_expires_at, claim_generation, max_attempts, last_error_code)
                VALUES (?1, NULL, 'completed', ?2, 'configured_webhook', 1, NULL, NULL, ?3, ?2, ?4, NULL, NULL, 0, 3, NULL)
                "#,
                rusqlite::params![delivery_id, occurred_at, state, updated_at],
            )
            .expect("insert webhook delivery");
    };
    insert("expired-dead-letter", "dead_letter", "2025-03-31T23:59:59Z");
    insert("fresh-dead-letter", "dead_letter", "2025-04-01T00:00:00Z");
    for (delivery_id, state) in [
        ("pending-delivery", "pending"),
        ("retry-delivery", "retry"),
        ("in-flight-delivery", "in_flight"),
        ("manual-retry-delivery", "manual_retry"),
        ("succeeded-delivery", "succeeded"),
    ] {
        insert(delivery_id, state, "2025-03-31T23:59:59Z");
    }

    let policy = RetentionPolicy::uniform(generic_cutoff, 100)
        .expect("generic retention policy")
        .with_webhook_dead_letter_cutoff(dead_letter_cutoff);
    let result = store
        .apply_retention_policy_map(&policy)
        .expect("dead-letter retention");
    assert_eq!(result.ttl_deleted_count, 1);
    assert_eq!(
        result
            .table_results
            .iter()
            .find(|entry| entry.table == RetentionTable::WebhookDeliveries)
            .expect("webhook table result")
            .ttl_deleted_count,
        1
    );
    assert!(
        store
            .webhook_delivery("expired-dead-letter")
            .expect("expired lookup")
            .is_none()
    );
    assert_eq!(
        store
            .webhook_delivery("fresh-dead-letter")
            .expect("fresh lookup")
            .expect("fresh dead letter retained")
            .state,
        WebhookDeliveryState::DeadLetter
    );
    for (delivery_id, expected_state) in [
        ("pending-delivery", WebhookDeliveryState::Pending),
        ("retry-delivery", WebhookDeliveryState::Retry),
        ("in-flight-delivery", WebhookDeliveryState::InFlight),
        ("manual-retry-delivery", WebhookDeliveryState::ManualRetry),
        ("succeeded-delivery", WebhookDeliveryState::Succeeded),
    ] {
        assert_eq!(
            store
                .webhook_delivery(delivery_id)
                .expect("non-dead-letter lookup")
                .expect("non-dead-letter retained")
                .state,
            expected_state,
            "{delivery_id} must not use the dead-letter window"
        );
    }

    drop(store);
    let reopened = LogStore::reopen_at(tmp.path(), clock).expect("reopen store");
    assert_eq!(
        reopened
            .apply_retention_policy_map(&policy)
            .expect("idempotent dead-letter retention")
            .ttl_deleted_count,
        0
    );
    let generic_result = reopened
        .apply_retention_policy_map(
            &RetentionPolicy::uniform(generic_cleanup_cutoff, 100)
                .expect("generic retention policy"),
        )
        .expect("generic webhook retention remains available");
    assert_eq!(generic_result.ttl_deleted_count, 6);
    assert_eq!(
        reopened
            .count_table("webhook_deliveries")
            .expect("count webhooks"),
        0
    );
}

#[test]
fn per_table_retention_caps_are_deterministic_owner_safe_and_restart_safe() {
    use super::repositories::{RetentionPolicy, RetentionTable, RetentionTablePolicy};

    let (store, clock, tmp) = open_store();
    let fresh = "2025-03-01T00:00:00Z";
    for request_id in ["active", "tie-a", "tie-b"] {
        store
            .insert_summary(request_id, None, None, None, None, fresh, None, None, None)
            .expect("insert summary");
    }
    for request_id in ["tie-a", "tie-b"] {
        store
            .write_terminal_event(
                request_id,
                &format!("terminal-{request_id}"),
                r#"{"type":"completed"}"#,
                "completed",
                None,
                fresh,
            )
            .expect("terminal event");
        store
            .insert_artifact_pointer(
                &format!("artifact-{request_id}"),
                request_id,
                fresh,
                "request",
                None,
            )
            .expect("owned pointer");
        store
            .insert_proxy_record(
                &format!("proxy-{request_id}"),
                request_id,
                fresh,
                "target",
                None,
                None,
                None,
                None,
                None,
                None,
            )
            .expect("owned proxy");
    }
    store
        .insert_artifact_pointer("artifact-active", "active", fresh, "request", None)
        .expect("active pointer");
    for index in 0..2 {
        let id = index.to_string();
        store
            .insert_audit_entry(
                &format!("audit-{id}"),
                None,
                fresh,
                "operator",
                "retention-test",
                None,
            )
            .expect("audit");
        store
            .insert_webhook_delivery(&format!("webhook-{id}"), None, fresh, 1, None)
            .expect("webhook");
        store
            .insert_cleanup_run(&format!("run-{id}"), fresh, "test", fresh, 0, None)
            .expect("cleanup receipt");
    }

    let table_policies = RetentionTable::ALL
        .into_iter()
        .map(|table| {
            (
                table,
                RetentionTablePolicy {
                    cutoff_occurred_at: "2025-01-01T00:00:00Z".to_string(),
                    max_rows: if table == RetentionTable::Summaries {
                        2
                    } else {
                        1
                    },
                },
            )
        })
        .collect::<BTreeMap<_, _>>();
    let result = store
        .apply_retention_policy_map(&RetentionPolicy::new(table_policies).expect("complete map"))
        .expect("per-table retention");

    // Same-time owner selection is deterministic by ID: the pointer cap picks
    // tie-a first, while active rows and their artifact remain protected.
    assert!(store.get_summary("tie-a").unwrap().is_none());
    assert!(store.get_summary("active").unwrap().is_some());
    assert!(
        store
            .get_artifact_pointer("artifact-active")
            .unwrap()
            .is_some()
    );
    assert_eq!(store.count_table("audit_entries").unwrap(), 1);
    assert_eq!(store.count_table("webhook_deliveries").unwrap(), 1);
    assert_eq!(store.count_table("cleanup_runs").unwrap(), 1);
    assert!(
        result
            .table_results
            .iter()
            .all(|entry| RetentionTable::ALL.contains(&entry.table))
    );
    assert!(
        result
            .table_results
            .iter()
            .any(|entry| entry.table == RetentionTable::ArtifactPointers
                && entry.max_rows_deleted_count > 0)
    );

    drop(store);
    let reopened = LogStore::reopen_at(tmp.path(), clock).expect("reopen");
    assert!(reopened.get_summary("active").unwrap().is_some());
    assert!(
        reopened
            .get_artifact_pointer("artifact-active")
            .unwrap()
            .is_some()
    );
    assert!(reopened.count_table("audit_entries").unwrap() <= 1);
    assert!(reopened.count_table("webhook_deliveries").unwrap() <= 1);
    assert!(reopened.count_table("cleanup_runs").unwrap() <= 1);
}

#[test]
fn per_table_retention_rejects_missing_or_unbounded_table_policies() {
    use super::repositories::{RetentionPolicy, RetentionTable, RetentionTablePolicy};

    let missing = BTreeMap::new();
    assert!(RetentionPolicy::new(missing).is_err());
    let zero = RetentionTable::ALL
        .into_iter()
        .map(|table| {
            (
                table,
                RetentionTablePolicy {
                    cutoff_occurred_at: "2025-01-01T00:00:00Z".to_string(),
                    max_rows: if table == RetentionTable::AuditEntries {
                        0
                    } else {
                        1
                    },
                },
            )
        })
        .collect();
    assert!(RetentionPolicy::new(zero).is_err());
}
