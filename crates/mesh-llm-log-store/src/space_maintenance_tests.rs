//! SQLite-space-maintenance tests independent of higher-layer table ownership.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use crate::{Clock, LogStore};

#[derive(Default)]
struct TestClock(AtomicU64);

impl Clock for TestClock {
    fn now(&self) -> String {
        let instant = self.0.fetch_add(1, Ordering::Relaxed);
        format!("2025-01-01T00:00:{instant:02}Z")
    }
}

fn open_store() -> (LogStore, tempfile::TempDir) {
    let directory = tempfile::tempdir().expect("temporary log store");
    let clock: Arc<dyn Clock> = Arc::new(TestClock::default());
    let store = LogStore::open(directory.path(), clock).expect("open log store");
    (store, directory)
}

fn insert_expired_audit_entry(store: &LogStore) {
    store
        .conn()
        .execute(
            "INSERT INTO audit_entries (entry_id, occurred_at, actor, action) VALUES (?, ?, ?, ?)",
            rusqlite::params!["expired-audit", "2024-01-01T00:00:00Z", "system", "cleanup"],
        )
        .expect("insert expired audit entry");
}

fn page_count(store: &LogStore) -> i64 {
    store
        .conn()
        .query_row("PRAGMA page_count", [], |row| row.get(0))
        .expect("SQLite page count")
}

#[test]
fn new_store_reports_zero_reclaimed_pages_when_nothing_can_be_reclaimed() {
    let (store, _directory) = open_store();
    let auto_vacuum: i64 = store
        .conn()
        .query_row("PRAGMA auto_vacuum", [], |row| row.get(0))
        .expect("new store auto-vacuum mode");
    assert_eq!(auto_vacuum, 2);

    store
        .conn()
        .execute_batch("PRAGMA analysis_limit = 400; PRAGMA optimize;")
        .expect("prepare optimizer metadata");
    let before = page_count(&store);
    let maintenance = store
        .maintain_space_after_cleanup()
        .expect("bounded maintenance");
    let after = page_count(&store);

    assert_eq!(maintenance.incremental_vacuum_pages, 0);
    assert_eq!(maintenance.incremental_vacuum_pages as i64, before - after);
}

#[test]
fn new_store_reports_actual_nonzero_page_count_reduction() {
    let (store, _directory) = open_store();
    let detail = "x".repeat(32 * 1024);
    {
        let mut connection = store.conn();
        let transaction = connection.transaction().expect("bulk insert transaction");
        for index in 0..128 {
            transaction
                .execute(
                    "INSERT INTO audit_entries \
                     (entry_id, occurred_at, actor, action, detail_json) \
                     VALUES (?, ?, ?, ?, ?)",
                    rusqlite::params![
                        format!("reclaim-{index}"),
                        "2024-01-01T00:00:00Z",
                        "system",
                        "cleanup",
                        detail,
                    ],
                )
                .expect("insert reclaimable audit entry");
        }
        transaction.commit().expect("commit reclaimable pages");
        connection
            .execute("DELETE FROM audit_entries", [])
            .expect("delete reclaimable audit entries");
    }

    let before = page_count(&store);
    let maintenance = store
        .maintain_space_after_cleanup()
        .expect("bounded maintenance");
    let after = page_count(&store);

    assert!(maintenance.incremental_vacuum_pages > 0);
    assert_eq!(
        maintenance.incremental_vacuum_pages as i64,
        before - after,
        "outcome must report measured physical page reclamation"
    );
}

#[test]
fn legacy_auto_vacuum_none_cleanup_remains_correct_without_full_vacuum() {
    let directory = tempfile::tempdir().expect("temporary legacy database root");
    let database = directory.path().join("log_store.db");
    let connection = rusqlite::Connection::open(&database).expect("create legacy database");
    connection
        .execute_batch("PRAGMA auto_vacuum = NONE;")
        .expect("set legacy auto-vacuum mode");
    drop(connection);

    let clock: Arc<dyn Clock> = Arc::new(TestClock::default());
    let store = LogStore::open(directory.path(), clock).expect("open legacy store");
    let auto_vacuum: i64 = store
        .conn()
        .query_row("PRAGMA auto_vacuum", [], |row| row.get(0))
        .expect("legacy auto-vacuum mode");
    assert_eq!(auto_vacuum, 0);

    insert_expired_audit_entry(&store);
    store
        .cascade_cleanup_before("2025-01-01T00:00:00Z")
        .expect("legacy cleanup");
    assert_eq!(
        store
            .maintain_space_after_cleanup()
            .expect("safe legacy maintenance")
            .incremental_vacuum_pages,
        0
    );
    assert_eq!(
        store
            .conn()
            .query_row("SELECT COUNT(*) FROM audit_entries", [], |row| row
                .get::<_, i64>(0))
            .expect("expired legacy audit entry removed"),
        0
    );
}
