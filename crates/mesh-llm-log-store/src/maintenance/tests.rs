use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicUsize, Ordering},
    mpsc,
};

use super::*;
use crate::RealClock;

struct NeverCancelled;

impl MaintenanceExecutionControl for NeverCancelled {
    fn is_cancelled(&self) -> bool {
        false
    }
}

struct Cancelled;

impl MaintenanceExecutionControl for Cancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[derive(Default)]
struct SwitchableCancellation(AtomicBool);

impl SwitchableCancellation {
    fn cancel(&self) {
        self.0.store(true, Ordering::Release);
    }
}

impl MaintenanceExecutionControl for SwitchableCancellation {
    fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }
}

fn block_first_artifact_removal(
    artifacts: &mut ArtifactFileStore,
) -> (mpsc::Receiver<()>, mpsc::SyncSender<()>, Arc<AtomicUsize>) {
    let (started_tx, started_rx) = mpsc::sync_channel(1);
    let (release_tx, release_rx) = mpsc::sync_channel(1);
    let release_rx = Arc::new(Mutex::new(release_rx));
    let calls = Arc::new(AtomicUsize::new(0));
    let removal_calls = Arc::clone(&calls);

    artifacts.set_remove_file_for_test(Arc::new(move |path| {
        if removal_calls.fetch_add(1, Ordering::SeqCst) == 0 {
            started_tx
                .send(())
                .map_err(|_| std::io::Error::other("blocked remover start receiver dropped"))?;
            release_rx
                .lock()
                .expect("blocked remover release mutex poisoned")
                .recv()
                .map_err(|_| std::io::Error::other("blocked remover release sender dropped"))?;
        }
        std::fs::remove_file(path)
    }));

    (started_rx, release_tx, calls)
}

fn fixture() -> (tempfile::TempDir, ArtifactFileStore) {
    let root = tempfile::tempdir().expect("temporary root");
    let store = LogStore::open(root.path().join("db"), Arc::new(RealClock)).expect("store");
    let artifacts =
        ArtifactFileStore::open(root.path().join("artifacts"), Arc::new(RealClock), store)
            .expect("artifacts");
    (root, artifacts)
}

fn seed_terminal(store: &LogStore, request_id: &str, created_at: &str) {
    seed_terminal_with_metadata(
        store,
        request_id,
        created_at,
        "route",
        "model",
        "provider",
        "engine",
        "completed",
    );
}

#[allow(clippy::too_many_arguments)]
fn seed_terminal_with_metadata(
    store: &LogStore,
    request_id: &str,
    created_at: &str,
    route: &str,
    model: &str,
    provider: &str,
    engine: &str,
    outcome: &str,
) {
    store
        .insert_summary(
            request_id,
            Some(model),
            Some(route),
            Some(provider),
            Some(engine),
            created_at,
            None,
            None,
            None,
        )
        .expect("summary");
    store
        .conn()
        .execute(
            "UPDATE summaries SET state = ?2 WHERE request_id = ?1",
            rusqlite::params![request_id, outcome],
        )
        .expect("terminal");
}

fn audit_action_for(store: &LogStore, audit_id: &str) -> String {
    store
        .conn()
        .query_row(
            "SELECT action FROM audit_entries WHERE entry_id = ?1",
            [audit_id],
            |row| row.get(0),
        )
        .expect("durable audit entry")
}

fn request(id: u128, cutoff: &str) -> CleanupPreviewRequest {
    request_with_limit(id, cutoff, 1)
}

fn request_with_limit(id: u128, cutoff: &str, request_limit: usize) -> CleanupPreviewRequest {
    CleanupPreviewRequest {
        operation_id: MaintenanceOperationId::new(Uuid::from_u128(id)),
        scope: CleanupScope::new(
            MaintenanceTimestamp::try_from(cutoff).expect("cutoff"),
            request_limit,
        )
        .expect("scope"),
        reason: MaintenanceReason::try_from("operator cleanup").expect("reason"),
    }
}

fn delete_request(id: u128, request_id: &str) -> DeleteOneRequest {
    DeleteOneRequest::new(
        MaintenanceOperationId::new(Uuid::from_u128(id)),
        request_id,
        MaintenanceReason::try_from("operator delete").expect("reason"),
    )
    .expect("delete request")
}

fn write_delete_artifact(artifacts: &ArtifactFileStore, artifact_id: &str, request_id: &str) {
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

fn pending_artifact_deletions(store: &LogStore) -> Vec<(String, String)> {
    let connection = store.conn();
    let mut statement = connection
        .prepare(
            "SELECT artifact_id, request_id FROM pending_artifact_deletions \
             ORDER BY artifact_id, request_id",
        )
        .expect("pending artifact deletion table");
    statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .expect("pending artifact deletion query")
        .collect::<Result<Vec<_>, _>>()
        .expect("pending artifact deletion rows")
}

fn fail_artifact_removal(artifacts: &mut ArtifactFileStore, artifact_id: &str) {
    let fail_id = artifact_id.to_owned();
    artifacts.set_remove_file_for_test(Arc::new(move |path| {
        if path.file_name().and_then(|name| name.to_str()) == Some(fail_id.as_str()) {
            Err(std::io::Error::other("injected artifact removal failure"))
        } else {
            std::fs::remove_file(path)
        }
    }));
}

fn restore_artifact_removal(artifacts: &mut ArtifactFileStore) {
    artifacts.set_remove_file_for_test(Arc::new(|path: &std::path::Path| {
        std::fs::remove_file(path)
    }));
}

fn assert_cleanup_execute_audits(artifacts: &ArtifactFileStore, expected: i64) {
    let audit_count: i64 = artifacts
        .store_ref()
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM audit_entries WHERE action = 'log_cleanup_execute'",
            [],
            |row| row.get(0),
        )
        .expect("cleanup audit count");
    assert_eq!(audit_count, expected);
}

fn assert_partial_delete_state(
    root: &tempfile::TempDir,
    artifacts: &ArtifactFileStore,
    request_id: &str,
    successful_artifact: &str,
    failed_artifact: &str,
    unrelated_artifact: &str,
    receipt: &MaintenanceReceipt,
) {
    assert_eq!(receipt.state, MaintenanceReceiptState::Partial);
    assert_eq!(receipt.planned.artifacts, 2);
    assert_eq!(receipt.executed.artifacts, 1);
    assert_eq!(receipt.artifact_deletion.removed, 1);
    assert_eq!(receipt.artifact_deletion.failed, 1);
    assert_eq!(
        receipt.artifact_deletion.failure_class,
        Some(ArtifactDeletionFailureClass::Io)
    );
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
            .query_artifact(successful_artifact)
            .unwrap()
            .is_none()
    );
    assert!(
        artifacts
            .store_ref()
            .query_artifact(failed_artifact)
            .unwrap()
            .is_some()
    );
    assert!(
        artifacts
            .store_ref()
            .query_artifact(unrelated_artifact)
            .unwrap()
            .is_some()
    );
    let artifact_root = root.path().join("artifacts");
    assert!(
        !artifact_root
            .join(request_id)
            .join(successful_artifact)
            .exists()
    );
    assert!(
        artifact_root
            .join(request_id)
            .join(failed_artifact)
            .exists()
    );
    assert!(!format!("{receipt:?}").contains(&*root.path().to_string_lossy()));
}

fn assert_completed_delete_state(
    artifacts: &ArtifactFileStore,
    request_id: &str,
    failed_artifact: &str,
    unrelated_request: &str,
    unrelated_artifact: &str,
    completed: &MaintenanceReceipt,
    replay: &MaintenanceReceipt,
) {
    assert_eq!(completed, replay);
    assert_eq!(completed.state, MaintenanceReceiptState::Completed);
    assert_eq!(completed.executed, completed.planned);
    assert_eq!(completed.artifact_deletion.removed, 2);
    assert_eq!(completed.artifact_deletion.failed, 0);
    assert_eq!(completed.artifact_deletion.failure_class, None);
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
            .query_artifact(failed_artifact)
            .unwrap()
            .is_none()
    );
    assert!(
        artifacts
            .store_ref()
            .query_request(unrelated_request)
            .unwrap()
            .is_some()
    );
    assert!(
        artifacts
            .store_ref()
            .query_artifact(unrelated_artifact)
            .unwrap()
            .is_some()
    );
    let audit_count: i64 = artifacts
        .store_ref()
        .conn()
        .query_row(
            "SELECT COUNT(*) FROM audit_entries WHERE action = 'log_delete_request'",
            [],
            |row| row.get(0),
        )
        .expect("audit count");
    assert_eq!(audit_count, 2, "completed replay must not add an audit");
}

mod cleanup;
mod delete_one;
