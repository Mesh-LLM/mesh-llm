use super::*;

#[test]
fn preview_snapshots_bounded_targets_and_rejects_invalid_typed_input() {
    let (_root, artifacts) = fixture();
    let store = artifacts.store_ref();
    seed_terminal(store, "old-a", "2025-01-01T00:00:00Z");
    seed_terminal(store, "old-b", "2025-01-02T00:00:00Z");
    let request = request(1, "2025-02-01T00:00:00Z");
    let receipt = store
        .preview_cleanup(&request, &NeverCancelled)
        .expect("preview");
    let replay = store
        .preview_cleanup(&request, &NeverCancelled)
        .expect("preview replay");
    assert_eq!(receipt.planned.requests, 1);
    assert!(receipt.has_more);
    assert_eq!(receipt.state, MaintenanceReceiptState::Previewed);
    assert_eq!(replay, receipt);
    let audit_id = receipt
        .preview_audit_id
        .as_deref()
        .expect("preview receipt audit ID");
    assert_eq!(audit_action_for(store, audit_id), "log_cleanup_preview");
    assert!(MaintenanceReason::try_from("\n").is_err());
    assert!(
        CleanupScope::new(
            MaintenanceTimestamp::try_from("2025-01-01T00:00:00Z").unwrap(),
            101
        )
        .is_err()
    );
}

#[test]
fn preview_cleanup_sanitizes_audit_detail_inside_transaction() {
    // Given: a valid maintenance request whose reason contains a credential and home path.
    let (_root, artifacts) = fixture();
    let store = artifacts.store_ref();
    let home = std::env::var("HOME").expect("HOME should be available to the test");
    let request = CleanupPreviewRequest {
        reason: MaintenanceReason::try_from(
            format!("Bearer maintenance-credential at {home}/private/audit.json").as_str(),
        )
        .expect("valid maintenance reason"),
        ..request(32, "2025-02-01T00:00:00Z")
    };

    // When: preview writes its audit row in the receipt transaction.
    let receipt = store
        .preview_cleanup(&request, &NeverCancelled)
        .expect("preview cleanup");
    let raw_detail: String = store
        .conn()
        .query_row(
            "SELECT detail_json FROM audit_entries WHERE entry_id = ?1",
            [receipt
                .preview_audit_id
                .as_deref()
                .expect("preview audit ID")],
            |row| row.get(0),
        )
        .expect("raw maintenance audit detail");

    // Then: sensitive originals are absent and safe transaction fields remain structured.
    assert!(!raw_detail.contains("Bearer"));
    assert!(!raw_detail.contains("maintenance-credential"));
    assert!(!raw_detail.contains(&home));
    let stored: serde_json::Value =
        serde_json::from_str(&raw_detail).expect("maintenance detail remains valid JSON");
    assert_eq!(stored["actor"], "trusted_local_operator");
    assert_eq!(stored["source"], "logs_api");
    assert_eq!(stored["result"], "previewed");
    assert_eq!(stored["operationId"], request.operation_id.to_string());
}

#[test]
fn cleanup_targets_visible_requests_instead_of_hidden_management_traffic() {
    let (_root, artifacts) = fixture();
    let store = artifacts.store_ref();
    seed_terminal_with_metadata(
        store,
        "visible-request",
        "2025-01-01T00:00:00.000740000Z",
        "chat_completions",
        "model-a",
        "mesh",
        "skippy",
        "completed",
    );
    seed_terminal_with_metadata(
        store,
        "hidden-management-request",
        "2025-01-01T00:00:00.000500000Z",
        "management_get_status",
        "model-a",
        "management_api",
        "management_get_status",
        "completed",
    );
    let request = request_with_limit(31, "2025-01-01T00:00:00.001Z", 10);

    let preview = store
        .preview_cleanup(&request, &NeverCancelled)
        .expect("preview visible cleanup");
    assert_eq!(preview.planned.requests, 1);
    assert_eq!(
        store
            .conn()
            .query_row(
                "SELECT request_id FROM maintenance_operation_targets WHERE operation_id = ?1",
                [request.operation_id.to_string()],
                |row| row.get::<_, String>(0),
            )
            .expect("visible cleanup target"),
        "visible-request"
    );

    let completed = artifacts
        .execute_cleanup(request.operation_id, &request.reason, &NeverCancelled)
        .expect("execute visible cleanup");
    assert_eq!(completed.executed.requests, 1);
    assert!(store.query_request("visible-request").unwrap().is_none());
    assert!(
        store
            .query_request("hidden-management-request")
            .unwrap()
            .is_some()
    );
}

#[test]
fn stale_preview_only_cleanup_receipt_is_ttl_eligible() {
    let (_root, artifacts) = fixture();
    let store = artifacts.store_ref();
    seed_terminal(store, "old-preview-retention", "2025-01-01T00:00:00Z");
    let request = request(30, "2025-02-01T00:00:00Z");
    let receipt = store
        .preview_cleanup(&request, &NeverCancelled)
        .expect("preview cleanup");
    assert_eq!(receipt.state, MaintenanceReceiptState::Previewed);

    assert_eq!(
        store
            .cleanup_maintenance_receipts("2999-01-01T00:00:00Z", i64::MAX as u64)
            .expect("expire stale preview"),
        1
    );
    assert_eq!(store.count_table("maintenance_operations").unwrap(), 0);
    assert_eq!(
        store.count_table("maintenance_operation_targets").unwrap(),
        0
    );
}

#[test]
fn preview_filters_terminal_ledger_scope_and_rejects_changed_replay() {
    let (_root, artifacts) = fixture();
    let store = artifacts.store_ref();
    let matching = (
        "matching",
        "2025-01-02T00:00:00Z",
        "route-a",
        "model-a",
        "provider-a",
        "engine-a",
        "completed",
    );
    let candidates = [
        matching,
        (
            "before-from",
            "2024-12-31T23:59:59Z",
            "route-a",
            "model-a",
            "provider-a",
            "engine-a",
            "completed",
        ),
        (
            "after-to",
            "2025-02-01T00:00:01Z",
            "route-a",
            "model-a",
            "provider-a",
            "engine-a",
            "completed",
        ),
        (
            "other-route",
            "2025-01-02T00:00:00Z",
            "route-b",
            "model-a",
            "provider-a",
            "engine-a",
            "completed",
        ),
        (
            "other-model",
            "2025-01-02T00:00:00Z",
            "route-a",
            "model-b",
            "provider-a",
            "engine-a",
            "completed",
        ),
        (
            "other-provider",
            "2025-01-02T00:00:00Z",
            "route-a",
            "model-a",
            "provider-b",
            "engine-a",
            "completed",
        ),
        (
            "other-engine",
            "2025-01-02T00:00:00Z",
            "route-a",
            "model-a",
            "provider-a",
            "engine-b",
            "completed",
        ),
        (
            "other-outcome",
            "2025-01-02T00:00:00Z",
            "route-a",
            "model-a",
            "provider-a",
            "engine-a",
            "failed",
        ),
    ];
    for (request_id, created_at, route, model, provider, engine, outcome) in candidates {
        seed_terminal_with_metadata(
            store, request_id, created_at, route, model, provider, engine, outcome,
        );
    }
    store
        .insert_summary(
            "active-match",
            Some("model-a"),
            Some("route-a"),
            Some("provider-a"),
            Some("engine-a"),
            "2025-01-02T00:00:00Z",
            None,
            None,
            None,
        )
        .expect("active summary");
    let filters = CleanupFilters::new(
        Some(MaintenanceTimestamp::try_from("2025-01-01T00:00:00Z").expect("from")),
        Some(MaintenanceTimestamp::try_from("2025-02-01T00:00:00Z").expect("to")),
        Some("route-a".to_owned()),
        Some("model-a".to_owned()),
        Some("provider-a".to_owned()),
        Some("engine-a".to_owned()),
        Some(CleanupOutcome::Completed),
    )
    .expect("filters");
    let request = CleanupPreviewRequest {
        operation_id: MaintenanceOperationId::new(Uuid::from_u128(0x50)),
        scope: CleanupScope::new(
            MaintenanceTimestamp::try_from("2025-03-01T00:00:00Z").expect("cutoff"),
            10,
        )
        .expect("scope")
        .with_filters(filters),
        reason: MaintenanceReason::try_from("operator cleanup").expect("reason"),
    };
    let receipt = store
        .preview_cleanup(&request, &NeverCancelled)
        .expect("filtered preview");
    assert_eq!(receipt.planned.requests, 1);
    assert_eq!(receipt.scope.filters().route(), Some("route-a"));
    assert_eq!(
        receipt.scope.filters().outcome(),
        Some(CleanupOutcome::Completed)
    );
    assert_eq!(
        store
            .conn()
            .query_row(
                "SELECT request_id FROM maintenance_operation_targets WHERE operation_id = ?1",
                [request.operation_id.to_string()],
                |row| row.get::<_, String>(0),
            )
            .expect("exact target"),
        "matching"
    );
    assert_eq!(
        store
            .preview_cleanup(&request, &NeverCancelled)
            .expect("same scope replay"),
        receipt
    );
    let changed_scope = CleanupPreviewRequest {
        scope: request.scope.clone().with_filters(
            CleanupFilters::new(
                Some(MaintenanceTimestamp::try_from("2025-01-01T00:00:00Z").expect("from")),
                Some(MaintenanceTimestamp::try_from("2025-02-01T00:00:00Z").expect("to")),
                Some("route-a".to_owned()),
                Some("model-b".to_owned()),
                Some("provider-a".to_owned()),
                Some("engine-a".to_owned()),
                Some(CleanupOutcome::Completed),
            )
            .expect("changed filters"),
        ),
        ..request.clone()
    };
    assert!(matches!(
        store.preview_cleanup(&changed_scope, &NeverCancelled),
        Err(LogStoreError::MaintenanceOperationConflict)
    ));
    assert!(
        CleanupFilters::new(
            None,
            None,
            Some("/private/model?token=secret".to_owned()),
            None,
            None,
            None,
            None,
        )
        .is_err()
    );
    assert!(CleanupOutcome::try_from("active").is_err());
}

#[test]
fn cleanup_receipt_preserves_typed_lookup_and_intent_validation() {
    let (_root, artifacts) = fixture();
    let store = artifacts.store_ref();
    let request = request(2, "2025-02-01T00:00:00Z");

    assert!(matches!(
        store.cleanup_receipt(request.operation_id, &request.reason),
        Err(LogStoreError::MaintenanceOperationNotFound)
    ));

    let preview = store
        .preview_cleanup(&request, &NeverCancelled)
        .expect("preview");
    assert_eq!(
        store
            .cleanup_receipt(request.operation_id, &request.reason)
            .expect("matching receipt"),
        preview
    );

    let different_reason =
        MaintenanceReason::try_from("different operator reason").expect("different valid reason");
    assert!(matches!(
        store.cleanup_receipt(request.operation_id, &different_reason),
        Err(LogStoreError::MaintenanceOperationConflict)
    ));
}

#[test]
fn execute_replays_completed_receipt_and_cascades_artifact_owner() {
    let (_root, artifacts) = fixture();
    let store = artifacts.store_ref();
    seed_terminal(store, "old-artifact", "2025-01-01T00:00:00Z");
    artifacts
        .write_artifact(
            "artifact-1",
            "old-artifact",
            "response",
            "2025-01-01T00:00:01Z",
            b"redacted",
            None,
            1,
            true,
            false,
            128,
            128,
        )
        .expect("artifact");
    let request = request(2, "2025-02-01T00:00:00Z");
    let preview = store
        .preview_cleanup(&request, &NeverCancelled)
        .expect("preview");
    let first = artifacts
        .execute_cleanup(request.operation_id, &request.reason, &NeverCancelled)
        .expect("execute");
    let replay = artifacts
        .execute_cleanup(request.operation_id, &request.reason, &NeverCancelled)
        .expect("replay");
    assert_eq!(first, replay);
    assert_eq!(first.state, MaintenanceReceiptState::Completed);
    assert_eq!(first.planned, preview.planned);
    assert_eq!(first.executed.requests, 1);
    assert_eq!(store.query_request("old-artifact").unwrap(), None);
    assert!(artifacts.read_artifact("artifact-1").is_err());
    let preview_audit_id = preview
        .preview_audit_id
        .as_deref()
        .expect("preview receipt audit ID");
    let execute_audit_id = first
        .execution_audit_id
        .as_deref()
        .expect("execute receipt audit ID");
    assert_ne!(preview_audit_id, execute_audit_id);
    assert_eq!(
        audit_action_for(store, execute_audit_id),
        "log_cleanup_execute"
    );
    let execute_audits: i64 = store
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM audit_entries WHERE action = 'log_cleanup_execute'",
            [],
            |row| row.get(0),
        )
        .expect("audits");
    assert_eq!(execute_audits, 1);
    let detail: String = store
        .conn()
        .query_row(
            "SELECT detail_json FROM audit_entries WHERE action = 'log_cleanup_execute'",
            [],
            |row| row.get(0),
        )
        .expect("audit detail");
    assert!(detail.contains("trusted_local_operator") && detail.contains("logs_api"));
    assert!(detail.contains(&request.operation_id.to_string()));
}

fn assert_partial_cleanup_retry_state(
    artifacts: &ArtifactFileStore,
    receipt: &MaintenanceReceipt,
    deleted_request: &str,
    failed_request: &str,
    unrelated_request: &str,
    failed_artifact: &str,
) {
    assert_eq!(receipt.state, MaintenanceReceiptState::Partial);
    assert_eq!(receipt.artifact_deletion.removed, 1);
    assert_eq!(receipt.artifact_deletion.failed, 1);
    assert_eq!(
        receipt.artifact_deletion.failure_class,
        Some(ArtifactDeletionFailureClass::Io)
    );
    let store = artifacts.store_ref();
    assert!(
        store
            .query_request(deleted_request)
            .expect("deleted request query")
            .is_none()
    );
    assert!(
        store
            .query_request(failed_request)
            .expect("failed request query")
            .is_some()
    );
    assert!(
        store
            .query_artifact(failed_artifact)
            .expect("failed artifact query")
            .is_some()
    );
    assert!(
        store
            .query_request(unrelated_request)
            .expect("unrelated request query")
            .is_some()
    );
    assert!(!format!("{receipt:?}").contains("artifacts/"));
    assert_eq!(
        pending_artifact_deletions(store),
        vec![(failed_artifact.to_owned(), failed_request.to_owned())],
        "only the failed cleanup artifact remains queued"
    );
}

fn assert_completed_cleanup_retry_state(
    artifacts: &ArtifactFileStore,
    partial: &MaintenanceReceipt,
    completed: &MaintenanceReceipt,
    replay: &MaintenanceReceipt,
    failed_request: &str,
    unrelated_artifact: &str,
) {
    assert_eq!(completed, replay);
    assert_eq!(completed.state, MaintenanceReceiptState::Completed);
    assert_eq!(completed.executed, completed.planned);
    assert_eq!(completed.artifact_deletion.removed, 2);
    assert_eq!(completed.artifact_deletion.failed, 0);
    assert_ne!(partial.execution_audit_id, completed.execution_audit_id);
    assert_eq!(replay.execution_audit_id, completed.execution_audit_id);
    let store = artifacts.store_ref();
    assert!(pending_artifact_deletions(store).is_empty());
    assert_eq!(
        audit_action_for(
            store,
            completed
                .execution_audit_id
                .as_deref()
                .expect("retry cleanup audit ID"),
        ),
        "log_cleanup_execute"
    );
    assert!(
        store
            .query_request(failed_request)
            .expect("failed request retry query")
            .is_none()
    );
    assert!(
        store
            .query_artifact(unrelated_artifact)
            .expect("unrelated artifact query")
            .is_some()
    );
    assert_cleanup_execute_audits(artifacts, 2);
}

#[test]
fn cleanup_retains_only_failed_selected_owners_then_retries_exact_targets() {
    let (_root, mut artifacts) = fixture();
    let first_request = "00000000-0000-4000-8000-000000000121";
    let failed_request = "00000000-0000-4000-8000-000000000122";
    let unrelated_request = "00000000-0000-4000-8000-000000000123";
    let first_artifact = "00000000-0000-4000-8000-000000000221";
    let failed_artifact = "00000000-0000-4000-8000-000000000222";
    let unrelated_artifact = "00000000-0000-4000-8000-000000000223";
    seed_terminal(artifacts.store_ref(), first_request, "2025-01-01T00:00:00Z");
    seed_terminal(
        artifacts.store_ref(),
        failed_request,
        "2025-01-02T00:00:00Z",
    );
    seed_terminal(
        artifacts.store_ref(),
        unrelated_request,
        "2025-03-03T00:00:00Z",
    );
    write_delete_artifact(&artifacts, first_artifact, first_request);
    write_delete_artifact(&artifacts, failed_artifact, failed_request);
    write_delete_artifact(&artifacts, unrelated_artifact, unrelated_request);
    let request = request_with_limit(16, "2025-02-01T00:00:00Z", 2);
    artifacts
        .store_ref()
        .preview_cleanup(&request, &NeverCancelled)
        .expect("preview");
    fail_artifact_removal(&mut artifacts, failed_artifact);

    let partial = artifacts
        .execute_cleanup(request.operation_id, &request.reason, &NeverCancelled)
        .expect("partial cleanup");
    assert_partial_cleanup_retry_state(
        &artifacts,
        &partial,
        first_request,
        failed_request,
        unrelated_request,
        failed_artifact,
    );

    restore_artifact_removal(&mut artifacts);
    let completed = artifacts
        .execute_cleanup(request.operation_id, &request.reason, &NeverCancelled)
        .expect("retry cleanup");
    let replay = artifacts
        .execute_cleanup(request.operation_id, &request.reason, &NeverCancelled)
        .expect("completed replay");
    assert_completed_cleanup_retry_state(
        &artifacts,
        &partial,
        &completed,
        &replay,
        failed_request,
        unrelated_artifact,
    );
}

#[test]
fn cleanup_request_limit_partial_replays_without_retrying_later_targets() {
    let (_root, artifacts) = fixture();
    let first_request = "00000000-0000-4000-8000-000000000124";
    let later_request = "00000000-0000-4000-8000-000000000125";
    seed_terminal(artifacts.store_ref(), first_request, "2025-01-01T00:00:00Z");
    seed_terminal(artifacts.store_ref(), later_request, "2025-01-02T00:00:00Z");
    let request = request_with_limit(17, "2025-02-01T00:00:00Z", 1);
    let preview = artifacts
        .store_ref()
        .preview_cleanup(&request, &NeverCancelled)
        .expect("preview");
    assert!(preview.has_more);
    let partial = artifacts
        .execute_cleanup(request.operation_id, &request.reason, &NeverCancelled)
        .expect("limited cleanup");
    let replay = artifacts
        .execute_cleanup(request.operation_id, &request.reason, &NeverCancelled)
        .expect("limited cleanup replay");
    assert_eq!(partial, replay);
    assert_eq!(partial.state, MaintenanceReceiptState::Partial);
    assert_eq!(partial.artifact_deletion.failed, 0);
    assert_eq!(partial.executed, partial.planned);
    assert!(
        artifacts
            .store_ref()
            .query_request(first_request)
            .unwrap()
            .is_none()
    );
    assert!(
        artifacts
            .store_ref()
            .query_request(later_request)
            .unwrap()
            .is_some()
    );
    assert_cleanup_execute_audits(&artifacts, 1);
}
#[test]
fn cleanup_reconciles_missing_and_corrupt_pointers_without_file_paths() {
    let (root, artifacts) = fixture();
    let request_id = "00000000-0000-4000-8000-000000000126";
    let missing_artifact = "00000000-0000-4000-8000-000000000224";
    let corrupt_artifact = "00000000-0000-4000-8000-000000000225";
    seed_terminal(artifacts.store_ref(), request_id, "2025-01-01T00:00:00Z");
    write_delete_artifact(&artifacts, missing_artifact, request_id);
    write_delete_artifact(&artifacts, corrupt_artifact, request_id);
    std::fs::remove_file(
        root.path()
            .join("artifacts")
            .join(request_id)
            .join(missing_artifact),
    )
    .expect("remove backing file");
    std::fs::write(
        root.path()
            .join("artifacts")
            .join(request_id)
            .join(corrupt_artifact),
        b"checksum changed",
    )
    .expect("corrupt backing file");
    let request = request_with_limit(18, "2025-02-01T00:00:00Z", 1);
    artifacts
        .store_ref()
        .preview_cleanup(&request, &NeverCancelled)
        .expect("preview");
    let receipt = artifacts
        .execute_cleanup(request.operation_id, &request.reason, &NeverCancelled)
        .expect("reconcile cleanup");
    assert_eq!(receipt.state, MaintenanceReceiptState::Completed);
    assert_eq!(receipt.executed, receipt.planned);
    assert_eq!(receipt.artifact_deletion.removed, 2);
    assert_eq!(receipt.artifact_deletion.failed, 0);
    assert!(!format!("{receipt:?}").contains(&*root.path().to_string_lossy()));
}

#[test]
fn cancelled_execute_leaves_preview_targets_untouched() {
    let (_root, artifacts) = fixture();
    let store = artifacts.store_ref();
    seed_terminal(store, "old-cancelled", "2025-01-01T00:00:00Z");
    let cancelled_preview = request(4, "2025-02-01T00:00:00Z");
    assert!(matches!(
        store.preview_cleanup(&cancelled_preview, &Cancelled),
        Err(LogStoreError::MaintenanceExecutionCancelled)
    ));
    assert_eq!(
        store
            .count_table("maintenance_operations")
            .expect("operations"),
        0
    );
    let request = request(3, "2025-02-01T00:00:00Z");
    let preview = store
        .preview_cleanup(&request, &NeverCancelled)
        .expect("preview");
    assert!(matches!(
        artifacts.execute_cleanup(request.operation_id, &request.reason, &Cancelled),
        Err(LogStoreError::MaintenanceExecutionCancelled)
    ));
    assert!(store.query_request("old-cancelled").unwrap().is_some());
    let replayed_preview = store
        .preview_cleanup(&request, &NeverCancelled)
        .expect("preview replay");
    assert_eq!(replayed_preview, preview);
    assert_eq!(replayed_preview.state, MaintenanceReceiptState::Previewed);
    let execute_audits: i64 = store
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM audit_entries WHERE action = 'log_cleanup_execute'",
            [],
            |row| row.get(0),
        )
        .expect("execute audit count");
    assert_eq!(
        execute_audits, 0,
        "cancelled execution must not orphan an audit"
    );
}

#[test]
fn cancellation_during_cleanup_removal_keeps_the_preview_retryable() {
    let (root, mut artifacts) = fixture();
    let request_id = "00000000-0000-4000-8000-000000000127";
    let first_artifact = "00000000-0000-4000-8000-000000000226";
    let second_artifact = "00000000-0000-4000-8000-000000000227";
    seed_terminal(artifacts.store_ref(), request_id, "2025-01-01T00:00:00Z");
    write_delete_artifact(&artifacts, first_artifact, request_id);
    write_delete_artifact(&artifacts, second_artifact, request_id);
    let request = request(19, "2025-02-01T00:00:00Z");
    artifacts
        .store_ref()
        .preview_cleanup(&request, &NeverCancelled)
        .expect("preview cleanup");
    let (started, release, calls) = block_first_artifact_removal(&mut artifacts);
    let control = SwitchableCancellation::default();

    let (cancelled, queued_before_removal) = std::thread::scope(|scope| {
        let worker = scope
            .spawn(|| artifacts.execute_cleanup(request.operation_id, &request.reason, &control));
        started.recv().expect("first removal started");
        let queued_before_removal = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pending_artifact_deletions(artifacts.store_ref())
        }));
        control.cancel();
        release.send(()).expect("release first removal");
        (
            worker.join().expect("cleanup worker does not panic"),
            queued_before_removal,
        )
    });

    let queued_before_removal = match queued_before_removal {
        Ok(queued) => queued,
        Err(_) => panic!("queue inspection before removal panicked"),
    };
    assert_eq!(
        queued_before_removal,
        vec![
            (first_artifact.to_owned(), request_id.to_owned()),
            (second_artifact.to_owned(), request_id.to_owned()),
        ],
        "cleanup queues the exact bounded target set before file I/O"
    );
    assert!(matches!(
        cancelled,
        Err(LogStoreError::MaintenanceExecutionCancelled)
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(
        !root
            .path()
            .join("artifacts")
            .join(request_id)
            .join(first_artifact)
            .exists()
    );
    assert!(
        root.path()
            .join("artifacts")
            .join(request_id)
            .join(second_artifact)
            .exists()
    );
    assert!(
        artifacts
            .store_ref()
            .query_request(request_id)
            .expect("request remains")
            .is_some()
    );
    assert!(
        artifacts
            .store_ref()
            .query_artifact(first_artifact)
            .expect("first pointer remains")
            .is_some()
    );
    assert!(
        artifacts
            .store_ref()
            .query_artifact(second_artifact)
            .expect("second pointer remains")
            .is_some()
    );
    let pending = artifacts
        .store_ref()
        .preview_cleanup(&request, &NeverCancelled)
        .expect("preview remains replayable");
    assert_eq!(pending.state, MaintenanceReceiptState::Previewed);
    assert_eq!(pending.execution_audit_id, None);
    assert_cleanup_execute_audits(&artifacts, 0);

    let completed = artifacts
        .execute_cleanup(request.operation_id, &request.reason, &NeverCancelled)
        .expect("retry cleanup");
    assert_eq!(completed.state, MaintenanceReceiptState::Completed);
    assert_eq!(completed.executed, completed.planned);
    assert!(pending_artifact_deletions(artifacts.store_ref()).is_empty());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(
        artifacts
            .store_ref()
            .query_request(request_id)
            .expect("request reconciled")
            .is_none()
    );
    assert_cleanup_execute_audits(&artifacts, 1);
}
