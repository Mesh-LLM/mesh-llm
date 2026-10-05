use crate::artifact_privacy::ArtifactPrivacy;
use crate::capture::{
    ARTIFACT_CAPTURE_DISABLED_PRIVACY_UNAVAILABLE, ArtifactCaptureOutcome, FailOpenArtifactCapture,
};
use crate::error::LogStoreError;
use crate::store::{Clock as ClockTrait, LogStore};
use std::fs;
use std::path::Path;
use std::sync::{
    Arc, Barrier,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};

#[derive(Debug, Default)]
struct TestClock(AtomicU64);

impl ClockTrait for TestClock {
    fn now(&self) -> String {
        let instant = self.0.fetch_add(1, Ordering::Relaxed);
        format!("2025-01-01T00:00:{instant:02}Z")
    }
}

#[derive(Clone, Copy)]
enum RejectAt {
    Never,
    Directory(usize),
    File(usize),
}

struct CountingPrivacy {
    reject_at: RejectAt,
    directory_calls: AtomicUsize,
    file_calls: AtomicUsize,
}

impl CountingPrivacy {
    fn rejecting_directory() -> Self {
        Self::rejecting_directory_at(1)
    }

    fn rejecting_directory_at(call: usize) -> Self {
        Self {
            reject_at: RejectAt::Directory(call),
            directory_calls: AtomicUsize::new(0),
            file_calls: AtomicUsize::new(0),
        }
    }

    fn rejecting_file(call: usize) -> Self {
        Self {
            reject_at: RejectAt::File(call),
            directory_calls: AtomicUsize::new(0),
            file_calls: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.directory_calls.load(Ordering::SeqCst) + self.file_calls.load(Ordering::SeqCst)
    }
}

impl ArtifactPrivacy for CountingPrivacy {
    fn prepare_directory(&self, _path: &Path) -> Result<(), LogStoreError> {
        let call = self.directory_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if matches!(self.reject_at, RejectAt::Directory(expected) if expected == call) {
            Err(LogStoreError::PrivacyNotGuaranteed)
        } else {
            Ok(())
        }
    }

    fn prepare_file(&self, _path: &Path) -> Result<(), LogStoreError> {
        let call = self.file_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if matches!(self.reject_at, RejectAt::File(expected) if expected == call) {
            Err(LogStoreError::PrivacyNotGuaranteed)
        } else {
            Ok(())
        }
    }
}

fn setup_store() -> (Arc<LogStore>, Arc<dyn ClockTrait>, tempfile::TempDir) {
    let database_root = tempfile::tempdir().expect("database root");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = Arc::new(LogStore::open(database_root.path(), clock.clone()).expect("open store"));
    (store, clock, database_root)
}

fn insert_summary(store: &LogStore, clock: &dyn ClockTrait, request_id: &str) {
    store
        .insert_summary(
            request_id,
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .expect("insert summary");
}

fn write(
    capture: &FailOpenArtifactCapture,
    clock: &dyn ClockTrait,
    artifact_id: &str,
    request_id: &str,
) -> Result<ArtifactCaptureOutcome, LogStoreError> {
    capture.write_artifact(
        artifact_id,
        request_id,
        "request_body",
        &clock.now(),
        b"redacted artifact body",
        Some("text/plain"),
        1,
        true,
        false,
        4096,
        8192,
    )
}

fn test_redactor() -> crate::artifacts::ArtifactRedactor {
    Arc::new(|content| content.to_vec())
}

fn has_content_files(root: &Path) -> bool {
    let entries = fs::read_dir(root).expect("artifact root readable");
    entries.flatten().any(|entry| {
        let path = entry.path();
        path.is_file()
            || (path.is_dir()
                && entry.file_name() != "tmp"
                && fs::read_dir(path)
                    .expect("request directory readable")
                    .flatten()
                    .any(|child| child.path().is_file()))
    })
}

#[cfg(unix)]
#[test]
fn symlinked_artifact_root_disables_capture_with_one_sanitized_marker() {
    use std::os::unix::fs::symlink;

    let (store, clock, _database_root) = setup_store();
    let parent = tempfile::tempdir().expect("artifact parent");
    let actual_root = parent.path().join("actual-root");
    std::fs::create_dir(&actual_root).expect("actual artifact root");
    let symlink_root = parent.path().join("symlink-root");
    symlink(&actual_root, &symlink_root).expect("artifact root symlink");

    let capture = FailOpenArtifactCapture::open(symlink_root, clock, store, test_redactor())
        .expect("unsafe root disables capture instead of failing startup");

    assert!(capture.is_disabled());
    let marker = capture.take_health_marker().expect("one health marker");
    assert_eq!(
        marker.reason().code(),
        ARTIFACT_CAPTURE_DISABLED_PRIVACY_UNAVAILABLE
    );
    assert!(capture.take_health_marker().is_none());
}

#[test]
fn root_privacy_rejection_disables_capture_and_keeps_metadata_usable() {
    let (store, clock, _database_root) = setup_store();
    let artifact_root = tempfile::tempdir().expect("artifact root");
    let privacy = Arc::new(CountingPrivacy::rejecting_directory());
    let capture = FailOpenArtifactCapture::open_with_privacy(
        artifact_root.path().to_path_buf(),
        clock.clone(),
        store.clone(),
        test_redactor(),
        privacy,
    )
    .expect("privacy rejection disables rather than fails opening");

    assert!(capture.is_disabled());
    assert!(matches!(
        write(&capture, clock.as_ref(), "art-open", "req-open"),
        Ok(ArtifactCaptureOutcome::Disabled(reason))
            if reason.code() == ARTIFACT_CAPTURE_DISABLED_PRIVACY_UNAVAILABLE
    ));
    assert!(!has_content_files(artifact_root.path()));

    insert_summary(store.as_ref(), clock.as_ref(), "req-metadata");
    store
        .insert_lifecycle_event(
            "req-metadata",
            "event-metadata",
            r#"{"type":"admitted"}"#,
            &clock.now(),
        )
        .expect("metadata lifecycle remains usable");
    store
        .insert_audit_entry(
            "audit-metadata",
            Some("req-metadata"),
            &clock.now(),
            "system",
            "artifact_capture_disabled",
            None,
        )
        .expect("metadata audit remains usable");
    assert!(store.get_summary("req-metadata").unwrap().is_some());
    assert_eq!(store.count_table("lifecycle_events").unwrap(), 1);
    assert_eq!(store.count_table("audit_entries").unwrap(), 1);
}

#[test]
fn disabled_capture_exposes_the_sanitised_health_marker_once() {
    let (store, clock, _database_root) = setup_store();
    let artifact_root = tempfile::tempdir().expect("artifact root");
    let capture = FailOpenArtifactCapture::open_with_privacy(
        artifact_root.path().to_path_buf(),
        clock,
        store,
        test_redactor(),
        Arc::new(CountingPrivacy::rejecting_directory()),
    )
    .expect("open disabled capture");

    let marker = capture.take_health_marker().expect("first marker");
    assert_eq!(
        marker.reason().code(),
        ARTIFACT_CAPTURE_DISABLED_PRIVACY_UNAVAILABLE
    );
    assert!(capture.take_health_marker().is_none());
}

#[test]
fn write_privacy_rejection_cleans_content_latches_and_skips_future_privacy_work() {
    let (store, clock, _database_root) = setup_store();
    insert_summary(store.as_ref(), clock.as_ref(), "req-write");
    let artifact_root = tempfile::tempdir().expect("artifact root");
    let privacy = Arc::new(CountingPrivacy::rejecting_file(1));
    let capture = FailOpenArtifactCapture::open_with_privacy(
        artifact_root.path().to_path_buf(),
        clock.clone(),
        store,
        test_redactor(),
        privacy.clone(),
    )
    .expect("open active capture");

    assert!(matches!(
        write(&capture, clock.as_ref(), "art-write", "req-write"),
        Ok(ArtifactCaptureOutcome::Disabled(_))
    ));
    assert!(capture.is_disabled());
    assert!(
        !artifact_root
            .path()
            .join("tmp")
            .join("art-write.part")
            .exists()
    );
    assert!(
        !artifact_root
            .path()
            .join("req-write")
            .join("art-write")
            .exists()
    );

    let calls_after_latch = privacy.calls();
    assert!(matches!(
        write(&capture, clock.as_ref(), "art-after-latch", "req-write"),
        Ok(ArtifactCaptureOutcome::Disabled(_))
    ));
    assert_eq!(privacy.calls(), calls_after_latch);
    assert!(!has_content_files(artifact_root.path()));
}

#[test]
fn every_write_time_privacy_failure_disables_without_content_files() {
    // Opening prepares root and tmp directories. The next two directory calls are request and
    // tmp preparation; file calls are temp then final artifact preparation.
    let failures = [
        ("request-directory", RejectAt::Directory(3)),
        ("temporary-directory", RejectAt::Directory(4)),
        ("temporary-file", RejectAt::File(1)),
        ("final-file", RejectAt::File(2)),
    ];

    for (name, reject_at) in failures {
        let (store, clock, _database_root) = setup_store();
        let request_id = format!("req-{name}");
        insert_summary(store.as_ref(), clock.as_ref(), &request_id);
        let artifact_root = tempfile::tempdir().expect("artifact root");
        let capture = FailOpenArtifactCapture::open_with_privacy(
            artifact_root.path().to_path_buf(),
            clock.clone(),
            store,
            test_redactor(),
            Arc::new(CountingPrivacy {
                reject_at,
                directory_calls: AtomicUsize::new(0),
                file_calls: AtomicUsize::new(0),
            }),
        )
        .expect("open active capture");

        assert!(matches!(
            write(
                &capture,
                clock.as_ref(),
                &format!("art-{name}"),
                &request_id
            ),
            Ok(ArtifactCaptureOutcome::Disabled(_))
        ));
        assert!(capture.is_disabled());
        assert!(!has_content_files(artifact_root.path()));
    }
}

#[test]
fn concurrent_marker_take_returns_exactly_one_marker() {
    let (store, clock, _database_root) = setup_store();
    let artifact_root = tempfile::tempdir().expect("artifact root");
    let capture = Arc::new(
        FailOpenArtifactCapture::open_with_privacy(
            artifact_root.path().to_path_buf(),
            clock,
            store,
            test_redactor(),
            Arc::new(CountingPrivacy::rejecting_directory()),
        )
        .expect("open disabled capture"),
    );
    let barrier = Arc::new(Barrier::new(3));
    let mut workers = Vec::new();
    for _ in 0..2 {
        let capture = capture.clone();
        let barrier = barrier.clone();
        workers.push(std::thread::spawn(move || {
            barrier.wait();
            capture.take_health_marker().is_some()
        }));
    }
    barrier.wait();

    let marker_count = workers
        .into_iter()
        .map(|worker| worker.join().expect("marker worker"))
        .filter(|received| *received)
        .count();
    assert_eq!(marker_count, 1);
}

#[test]
fn non_privacy_errors_remain_typed_and_do_not_disable_capture() {
    let (store, clock, _database_root) = setup_store();
    let artifact_root = tempfile::tempdir().expect("artifact root");
    let capture = FailOpenArtifactCapture::open_with_privacy(
        artifact_root.path().to_path_buf(),
        clock.clone(),
        store,
        test_redactor(),
        Arc::new(CountingPrivacy {
            reject_at: RejectAt::Never,
            directory_calls: AtomicUsize::new(0),
            file_calls: AtomicUsize::new(0),
        }),
    )
    .expect("open active capture");

    assert!(matches!(
        write(&capture, clock.as_ref(), "../unsafe", "req-unsafe"),
        Err(LogStoreError::PathUnsafe { .. })
    ));
    assert!(!capture.is_disabled());
    assert!(capture.take_health_marker().is_none());
}
