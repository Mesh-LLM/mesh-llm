//! Acceptance tests for mesh-llm-log-store.
//! All tests use real temp SQLite files (no in-memory shortcut).

use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use super::cursor::{decode_cursor, encode_cursor};
use super::error::LogStoreError;
use super::migrations::CURRENT_VERSION;
use super::repositories::{AuditEntryFilters, AuditEntrySeverity, AuditEntrySource};
use super::repositories::{
    WebhookDeliveryErrorCode, WebhookDeliveryInsertOutcome, WebhookDeliveryRecord,
    WebhookDeliveryState, WebhookManualRetryOutcome, WebhookRetryOutcome, WebhookTerminalOutcome,
};
use super::store::{Clock as ClockTrait, LogStore, SystemClock};
use crate::repositories;

/// Fixed clock returning deterministic ISO timestamps.
#[derive(Debug)]
struct TestClock {
    instant: AtomicU64,
}

impl Default for TestClock {
    fn default() -> Self {
        Self {
            instant: AtomicU64::new(0),
        }
    }
}

impl ClockTrait for TestClock {
    fn now(&self) -> String {
        let n = self.instant.fetch_add(1, Ordering::Relaxed);
        format!(
            "2025-01-01T{:02}:{:02}:{:02}Z",
            (n / 3600) % 24,
            (n / 60) % 60,
            n % 60
        )
    }
}

#[test]
fn test_clock_remains_monotonic_after_sixty_reads() {
    let clock = TestClock::default();
    let first = clock.now();
    for _ in 0..59 {
        clock.now();
    }
    let sixty_first = clock.now();

    assert_eq!(first, "2025-01-01T00:00:00Z");
    assert_eq!(sixty_first, "2025-01-01T00:01:00Z");
    assert!(first < sixty_first);
}

/// Open a fresh store backed by a temp directory. Directory is cleaned up on drop.
fn open_store() -> (LogStore, Arc<dyn ClockTrait>, tempfile::TempDir) {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).expect("open log store");
    (store, clock, tmp)
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

mod cursor_pagination;
mod retention_cleanup;
mod retention_policy;
mod schema_lifecycle;
mod summary_audit;
mod summary_events;
mod webhook;
