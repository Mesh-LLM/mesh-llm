use super::*;

#[test]
fn metadata_only_delete_completes_terminal_request_without_artifact_pointers() {
    let (_root, artifacts) = fixture();
    let store = artifacts.store_ref();
    let request_id = "00000000-0000-4000-8000-000000000140";
    seed_terminal(store, request_id, "2025-01-01T00:00:00Z");
    let request = delete_request(30, request_id);

    let prepared = store
        .prepare_metadata_only_delete_request(&request, &NeverCancelled)
        .expect("metadata-only preparation");
    assert_eq!(prepared.state, MaintenanceReceiptState::Previewed);
    let completed = store
        .execute_prepared_metadata_only_delete_request(&request, &NeverCancelled)
        .expect("metadata-only execution");

    assert_eq!(completed.state, MaintenanceReceiptState::Completed);
    assert_eq!(completed.executed, completed.planned);
    assert!(store.query_request(request_id).unwrap().is_none());
}

#[test]
fn delete_one_cascades_owned_artifact_and_replays_its_completed_receipt() {
    let (root, artifacts) = fixture();
    let store = artifacts.store_ref();
    let request_id = "00000000-0000-4000-8000-000000000101";
    let artifact_id = "00000000-0000-4000-8000-000000000201";
    seed_terminal(store, request_id, "2025-01-01T00:00:00Z");
    artifacts
        .write_artifact(
            artifact_id,
            request_id,
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
    let request = delete_request(10, request_id);
    let first = artifacts
        .delete_request_cascade(&request, &NeverCancelled)
        .expect("delete");
    let replay = artifacts
        .delete_request_cascade(&request, &NeverCancelled)
        .expect("replay");

    assert_eq!(first, replay);
    assert_eq!(first.action, MaintenanceAction::DeleteOne);
    assert_eq!(first.state, MaintenanceReceiptState::Completed);
    assert_eq!(first.planned.requests, 1);
    assert_eq!(first.executed, first.planned);
    assert!(store.query_request(request_id).unwrap().is_none());
    assert!(artifacts.read_artifact(artifact_id).is_err());
    assert!(
        !root
            .path()
            .join("artifacts")
            .join(request_id)
            .join(artifact_id)
            .exists()
    );
    let audit_id = first
        .execution_audit_id
        .as_deref()
        .expect("delete receipt audit ID");
    assert_eq!(audit_action_for(store, audit_id), "log_delete_request");
    let audit_count: i64 = store
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM audit_entries WHERE action = 'log_delete_request'",
            [],
            |row| row.get(0),
        )
        .expect("audit count");
    assert_eq!(audit_count, 1);
    let detail: String = store
        .conn()
        .query_row(
            "SELECT detail_json FROM audit_entries WHERE action = 'log_delete_request'",
            [],
            |row| row.get(0),
        )
        .expect("audit detail");
    assert!(detail.contains("trusted_local_operator") && detail.contains("logs_api"));
    assert!(detail.contains(&request.operation_id.to_string()));
}

#[test]
fn prepared_delete_one_persists_retryable_intent_before_artifact_io() {
    let (root, artifacts) = fixture();
    let request_id = "00000000-0000-4000-8000-000000000141";
    let artifact_id = "00000000-0000-4000-8000-000000000241";
    seed_terminal(artifacts.store_ref(), request_id, "2025-01-01T00:00:00Z");
    write_delete_artifact(&artifacts, artifact_id, request_id);
    let request = delete_request(31, request_id);

    let prepared = artifacts
        .prepare_delete_request(&request, &NeverCancelled)
        .expect("persist delete intent");
    assert_eq!(prepared.state, MaintenanceReceiptState::Previewed);
    assert!(prepared.execution_audit_id.is_none());
    assert!(
        artifacts
            .store_ref()
            .query_request(request_id)
            .unwrap()
            .is_some()
    );
    assert!(
        artifacts
            .store_ref()
            .query_artifact(artifact_id)
            .unwrap()
            .is_some()
    );
    assert!(
        root.path()
            .join("artifacts")
            .join(request_id)
            .join(artifact_id)
            .exists(),
        "preparation must not touch artifact files"
    );
    assert_eq!(
        artifacts
            .store_ref()
            .delete_one_receipt(&request)
            .expect("receipt lookup"),
        Some(prepared)
    );

    let completed = artifacts
        .execute_prepared_delete_request(&request, &NeverCancelled)
        .expect("execute prepared delete");
    assert_eq!(completed.state, MaintenanceReceiptState::Completed);
    assert!(
        artifacts
            .store_ref()
            .query_request(request_id)
            .unwrap()
            .is_none()
    );
    assert!(
        artifacts
            .store_ref()
            .query_artifact(artifact_id)
            .unwrap()
            .is_none()
    );
}

#[test]
fn concurrent_preparation_replays_one_immutable_delete_target() {
    let (_root, artifacts) = fixture();
    let request_id = "00000000-0000-4000-8000-000000000142";
    seed_terminal(artifacts.store_ref(), request_id, "2025-01-01T00:00:00Z");
    let request = delete_request(32, request_id);

    let (first, second) = std::thread::scope(|scope| {
        let first = scope.spawn(|| artifacts.prepare_delete_request(&request, &NeverCancelled));
        let second = scope.spawn(|| artifacts.prepare_delete_request(&request, &NeverCancelled));
        (
            first.join().expect("first preparation does not panic"),
            second.join().expect("second preparation does not panic"),
        )
    });
    assert_eq!(
        first.expect("first receipt"),
        second.expect("second receipt")
    );
    assert_eq!(
        artifacts
            .store_ref()
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM maintenance_operation_targets WHERE operation_id = ?1",
                [request.operation_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .expect("one immutable target"),
        1
    );
}

#[test]
fn delete_one_retains_failed_artifacts_and_same_operation_retry_reconciles_them() {
    let (root, mut artifacts) = fixture();
    let request_id = "00000000-0000-4000-8000-000000000111";
    let successful_artifact = "00000000-0000-4000-8000-000000000211";
    let failed_artifact = "00000000-0000-4000-8000-000000000212";
    let unrelated_request = "00000000-0000-4000-8000-000000000112";
    let unrelated_artifact = "00000000-0000-4000-8000-000000000213";
    seed_terminal(artifacts.store_ref(), request_id, "2025-01-01T00:00:00Z");
    seed_terminal(
        artifacts.store_ref(),
        unrelated_request,
        "2025-01-02T00:00:00Z",
    );
    write_delete_artifact(&artifacts, successful_artifact, request_id);
    write_delete_artifact(&artifacts, failed_artifact, request_id);
    write_delete_artifact(&artifacts, unrelated_artifact, unrelated_request);
    fail_artifact_removal(&mut artifacts, failed_artifact);

    let request = delete_request(14, request_id);
    let partial = artifacts
        .delete_request_cascade(&request, &NeverCancelled)
        .expect("partial delete");
    assert_partial_delete_state(
        &root,
        &artifacts,
        request_id,
        successful_artifact,
        failed_artifact,
        unrelated_artifact,
        &partial,
    );
    assert_eq!(
        pending_artifact_deletions(artifacts.store_ref()),
        vec![(failed_artifact.to_owned(), request_id.to_owned())],
        "only the failed delete-one artifact remains queued"
    );
    assert_eq!(
        artifacts
            .store_ref()
            .cleanup_maintenance_receipts("2999-01-01T00:00:00Z", 0)
            .expect("partial retry receipt retention"),
        0,
        "failed artifact retry state must never be retained as terminal history"
    );
    assert!(
        artifacts
            .store_ref()
            .delete_one_receipt(&request)
            .expect("partial receipt after retention")
            .is_some()
    );

    restore_artifact_removal(&mut artifacts);
    let completed = artifacts
        .delete_request_cascade(&request, &NeverCancelled)
        .expect("retry succeeds");
    let replay = artifacts
        .delete_request_cascade(&request, &NeverCancelled)
        .expect("completed replay");
    assert_completed_delete_state(
        &artifacts,
        request_id,
        failed_artifact,
        unrelated_request,
        unrelated_artifact,
        &completed,
        &replay,
    );
    assert!(pending_artifact_deletions(artifacts.store_ref()).is_empty());
    assert_ne!(partial.execution_audit_id, completed.execution_audit_id);
    assert_eq!(replay.execution_audit_id, completed.execution_audit_id);
    assert_eq!(
        audit_action_for(
            artifacts.store_ref(),
            completed
                .execution_audit_id
                .as_deref()
                .expect("retry delete audit ID"),
        ),
        "log_delete_request"
    );
}

#[test]
fn delete_one_reconciles_missing_and_corrupt_artifact_pointers_without_path_leakage() {
    let (root, artifacts) = fixture();
    let request_id = "00000000-0000-4000-8000-000000000113";
    let missing_artifact = "00000000-0000-4000-8000-000000000214";
    let corrupt_artifact = "00000000-0000-4000-8000-000000000215";
    seed_terminal(artifacts.store_ref(), request_id, "2025-01-01T00:00:00Z");
    for artifact_id in [missing_artifact, corrupt_artifact] {
        artifacts
            .write_artifact(
                artifact_id,
                request_id,
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
    }
    let missing_path = root
        .path()
        .join("artifacts")
        .join(request_id)
        .join(missing_artifact);
    std::fs::remove_file(&missing_path).expect("remove backing file");
    let corrupt_path = root
        .path()
        .join("artifacts")
        .join(request_id)
        .join(corrupt_artifact);
    std::fs::write(&corrupt_path, b"changed-after-checksum").expect("corrupt backing file");

    let receipt = artifacts
        .delete_request_cascade(&delete_request(15, request_id), &NeverCancelled)
        .expect("reconcile missing and corrupt pointers");
    assert_eq!(receipt.state, MaintenanceReceiptState::Completed);
    assert_eq!(receipt.executed, receipt.planned);
    assert_eq!(receipt.artifact_deletion.removed, 2);
    assert_eq!(receipt.artifact_deletion.failed, 0);
    assert_eq!(receipt.artifact_deletion.failure_class, None);
    assert!(
        artifacts
            .store_ref()
            .query_request(request_id)
            .unwrap()
            .is_none()
    );
    assert!(
        artifacts
            .store_ref()
            .query_artifact(missing_artifact)
            .unwrap()
            .is_none()
    );
    assert!(
        artifacts
            .store_ref()
            .query_artifact(corrupt_artifact)
            .unwrap()
            .is_none()
    );
    assert!(!format!("{receipt:?}").contains(&*root.path().to_string_lossy()));
}

#[test]
fn delete_one_missing_request_is_a_stable_completed_noop() {
    let (_root, artifacts) = fixture();
    let missing_id = "00000000-0000-4000-8000-000000000102";
    let request = delete_request(11, missing_id);
    let first = artifacts
        .delete_request_cascade(&request, &NeverCancelled)
        .expect("missing delete");
    let replay = artifacts
        .delete_request_cascade(&request, &NeverCancelled)
        .expect("missing replay");
    assert_eq!(first, replay);
    assert_eq!(first.action, MaintenanceAction::DeleteOne);
    assert_eq!(first.planned, MaintenanceCounts::default());
    assert_eq!(first.executed, MaintenanceCounts::default());
}

#[test]
fn terminal_receipt_retention_is_bounded_and_cascades_its_immutable_targets() {
    let (_root, artifacts) = fixture();
    let store = artifacts.store_ref();
    let request_id = "00000000-0000-4000-8000-000000000124";
    seed_terminal(store, request_id, "2025-01-01T00:00:00Z");
    artifacts
        .delete_request_cascade(&delete_request(24, request_id), &NeverCancelled)
        .expect("completed delete receipt");
    assert_eq!(store.count_table("maintenance_operations").unwrap(), 1);
    assert_eq!(
        store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM maintenance_operation_targets",
                [],
                |row| row.get::<_, i64>(0)
            )
            .expect("target count"),
        1
    );

    assert_eq!(
        store
            .cleanup_maintenance_receipts("2999-01-01T00:00:00Z", 0)
            .expect("bounded receipt cleanup"),
        1
    );
    assert_eq!(store.count_table("maintenance_operations").unwrap(), 0);
    assert_eq!(
        store
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM maintenance_operation_targets",
                [],
                |row| row.get::<_, i64>(0)
            )
            .expect("cascaded target count"),
        0
    );
}

#[test]
fn delete_one_never_removes_an_active_request() {
    let (_root, artifacts) = fixture();
    let store = artifacts.store_ref();
    let request_id = "00000000-0000-4000-8000-000000000104";
    store
        .insert_summary(
            request_id,
            Some("model"),
            Some("route"),
            None,
            None,
            "2025-01-01T00:00:00Z",
            None,
            None,
            None,
        )
        .expect("active summary");
    let request = delete_request(13, request_id);
    let receipt = artifacts
        .delete_request_cascade(&request, &NeverCancelled)
        .expect("active no-op");
    assert_eq!(receipt.planned, MaintenanceCounts::default());
    assert_eq!(receipt.executed, MaintenanceCounts::default());
    assert!(store.query_request(request_id).unwrap().is_some());
}

#[test]
fn cancelled_delete_one_leaves_no_receipt_and_can_be_retried() {
    let (_root, artifacts) = fixture();
    let store = artifacts.store_ref();
    let request_id = "00000000-0000-4000-8000-000000000103";
    seed_terminal(store, request_id, "2025-01-01T00:00:00Z");
    let request = delete_request(12, request_id);
    assert!(matches!(
        artifacts.delete_request_cascade(&request, &Cancelled),
        Err(LogStoreError::MaintenanceExecutionCancelled)
    ));
    assert!(store.query_request(request_id).unwrap().is_some());
    assert_eq!(
        store
            .count_table("maintenance_operations")
            .expect("operations"),
        0
    );
    assert_eq!(
        artifacts
            .delete_request_cascade(&request, &NeverCancelled)
            .expect("retry")
            .state,
        MaintenanceReceiptState::Completed
    );
}

fn cancel_during_first_delete_removal(
    artifacts: &ArtifactFileStore,
    request: &DeleteOneRequest,
    control: &SwitchableCancellation,
    started: mpsc::Receiver<()>,
    release: mpsc::SyncSender<()>,
) -> (
    Result<MaintenanceReceipt, LogStoreError>,
    Vec<(String, String)>,
) {
    let (cancelled, queued_before_removal) = std::thread::scope(|scope| {
        let worker = scope.spawn(|| artifacts.execute_prepared_delete_request(request, control));
        started.recv().expect("first removal started");
        let queued_before_removal = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pending_artifact_deletions(artifacts.store_ref())
        }));
        control.cancel();
        release.send(()).expect("release first removal");
        (
            worker.join().expect("delete worker does not panic"),
            queued_before_removal,
        )
    });
    (
        cancelled,
        queued_before_removal.expect("queue inspection before removal must not panic"),
    )
}

fn assert_cancelled_delete_removal_state(
    root: &tempfile::TempDir,
    artifacts: &ArtifactFileStore,
    request_id: &str,
    first_artifact: &str,
    second_artifact: &str,
) {
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
            .unwrap()
            .is_some()
    );
    for artifact_id in [first_artifact, second_artifact] {
        assert!(
            artifacts
                .store_ref()
                .query_artifact(artifact_id)
                .unwrap()
                .is_some()
        );
    }
}

fn assert_pending_delete_receipt_is_durable(
    artifacts: &ArtifactFileStore,
    request: &DeleteOneRequest,
) {
    let pending = artifacts
        .store_ref()
        .delete_one_receipt(request)
        .expect("receipt lookup")
        .expect("receipt persisted before filesystem work");
    assert_eq!(pending.state, MaintenanceReceiptState::Previewed);
    assert_eq!(pending.planned.requests, 1);
    assert_eq!(
        artifacts
            .store_ref()
            .cleanup_maintenance_receipts("2999-01-01T00:00:00Z", 0)
            .expect("pending delete retention"),
        0,
        "interrupted delete intent must survive receipt retention"
    );
    assert!(
        artifacts
            .store_ref()
            .delete_one_receipt(request)
            .unwrap()
            .is_some()
    );
    assert_eq!(durable_delete_target_count(artifacts, request), 1);
    assert_eq!(delete_execute_audit_count(artifacts), 0);
}

fn durable_delete_target_count(artifacts: &ArtifactFileStore, request: &DeleteOneRequest) -> i64 {
    artifacts
        .store_ref()
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM maintenance_operation_targets WHERE operation_id = ?1",
            [request.operation_id.to_string()],
            |row| row.get(0),
        )
        .expect("durable delete target count")
}

fn delete_execute_audit_count(artifacts: &ArtifactFileStore) -> i64 {
    artifacts
        .store_ref()
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM audit_entries WHERE action = 'log_delete_request'",
            [],
            |row| row.get(0),
        )
        .expect("delete audit count")
}

#[test]
fn cancellation_during_delete_removal_leaves_a_durable_pending_receipt_and_can_be_retried() {
    let (root, mut artifacts) = fixture();
    let request_id = "00000000-0000-4000-8000-000000000128";
    let first_artifact = "00000000-0000-4000-8000-000000000228";
    let second_artifact = "00000000-0000-4000-8000-000000000229";
    seed_terminal(artifacts.store_ref(), request_id, "2025-01-01T00:00:00Z");
    write_delete_artifact(&artifacts, first_artifact, request_id);
    write_delete_artifact(&artifacts, second_artifact, request_id);
    let request = delete_request(20, request_id);
    let (started, release, calls) = block_first_artifact_removal(&mut artifacts);
    let control = SwitchableCancellation::default();
    let prepared = artifacts
        .prepare_delete_request(&request, &NeverCancelled)
        .expect("prepare before bounded execution");
    assert_eq!(prepared.state, MaintenanceReceiptState::Previewed);

    let (cancelled, queued_before_removal) =
        cancel_during_first_delete_removal(&artifacts, &request, &control, started, release);
    assert_eq!(
        queued_before_removal,
        vec![
            (first_artifact.to_owned(), request_id.to_owned()),
            (second_artifact.to_owned(), request_id.to_owned()),
        ],
        "delete-one queues every owned artifact before file I/O"
    );
    assert!(matches!(
        cancelled,
        Err(LogStoreError::MaintenanceExecutionCancelled)
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_cancelled_delete_removal_state(
        &root,
        &artifacts,
        request_id,
        first_artifact,
        second_artifact,
    );
    assert_pending_delete_receipt_is_durable(&artifacts, &request);

    let completed = artifacts
        .execute_prepared_delete_request(&request, &NeverCancelled)
        .expect("retry delete");
    assert_eq!(completed.state, MaintenanceReceiptState::Completed);
    assert_eq!(completed.executed, completed.planned);
    assert_eq!(durable_delete_target_count(&artifacts, &request), 1);
    assert!(pending_artifact_deletions(artifacts.store_ref()).is_empty());
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(
        artifacts
            .store_ref()
            .query_request(request_id)
            .expect("request reconciled")
            .is_none()
    );
    assert_eq!(delete_execute_audit_count(&artifacts), 1);
}
