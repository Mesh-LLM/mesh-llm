use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use crate::{Clock as ClockTrait, LogStore};

/// Fixed clock returning deterministic ISO timestamps.
#[derive(Debug)]
pub(super) struct TestClock {
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
        format!("2025-01-01T00:00:{:02}Z", n % 60)
    }
}

/// Open a fresh store backed by a temp directory. Directory is cleaned up on drop.
pub(super) fn open_store() -> (LogStore, Arc<dyn ClockTrait>, tempfile::TempDir) {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).expect("open log store");
    (store, clock, tmp)
}
