use std::sync::Arc;

use crate::{ArtifactFileStore, Clock, LogStore, QuerySort, RealClock, UnavailableArtifactPointer};

#[test]
fn startup_reconciliation_preserves_intentionally_unavailable_artifacts() {
    let database_root = tempfile::tempdir().expect("database root");
    let artifact_root = tempfile::tempdir().expect("artifact root");
    let clock: Arc<dyn Clock> = Arc::new(RealClock);
    let store = LogStore::open(database_root.path(), Arc::clone(&clock)).expect("open store");
    let request_id = "00000000-0000-4000-8000-000000000301";
    let artifact_id = "00000000-0000-4000-8000-000000000302";
    let occurred_at = "2025-01-01T00:00:00.000000000Z";
    store
        .upsert_summary_metadata(request_id, None, None, None, None, occurred_at)
        .expect("summary");
    store
        .insert_unavailable_artifact_pointer(UnavailableArtifactPointer {
            artifact_id,
            request_id,
            occurred_at,
            kind: "response",
            media_kind: Some("application/octet-stream"),
            version: 1,
            reason: "streaming_response_not_assembled",
        })
        .expect("unavailable pointer");

    let artifacts = ArtifactFileStore::open(
        artifact_root.path().to_path_buf(),
        Arc::clone(&clock),
        store,
    )
    .expect("open artifact store and reconcile");
    let page = artifacts
        .store_ref()
        .query_artifacts(
            request_id,
            &crate::PageQuery {
                limit: 10,
                cursor: None,
                sort: QuerySort::Ascending,
            },
        )
        .expect("query unavailable pointer");
    assert_eq!(page.items.len(), 1);
    assert_eq!(
        page.items[0].unavailable_reason.as_deref(),
        Some("streaming_response_not_assembled")
    );
    assert!(!page.items[0].missing);
    assert!(!page.items[0].corrupt);

    assert_eq!(
        artifacts
            .store_ref()
            .update_artifact_pointer_missing(artifact_id)
            .expect("guarded missing update"),
        0,
        "metadata-only artifacts can never be reclassified as missing files"
    );
    assert_eq!(
        artifacts
            .store_ref()
            .update_artifact_pointer_corrupt(artifact_id)
            .expect("guarded corrupt update"),
        0,
        "metadata-only artifacts can never be reclassified as corrupt files"
    );
    assert_eq!(
        artifacts
            .store_ref()
            .get_artifact_pointer(artifact_id)
            .expect("pointer query")
            .expect("pointer remains")
            .unavailable_reason
            .as_deref(),
        Some("streaming_response_not_assembled")
    );
}
