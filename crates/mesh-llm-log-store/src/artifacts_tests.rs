//! Artifact storage tests — real tempdir filesystem, no in-memory shortcuts.

use crate::artifact_privacy::ArtifactPrivacy;
use crate::artifacts::{ArtifactFileStore, ArtifactStatus};
use crate::error::LogStoreError;
use crate::store::{Clock as ClockTrait, LogStore};
use sha2::{Digest, Sha256};
use std::fs;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

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
        format!("2025-01-01T00:00:{:02}Z", n % 60)
    }
}

fn expected_checksum(content: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content);
    hex::encode(hasher.finalize())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PrivacyPathKind {
    Directory,
    File,
}

#[derive(Default)]
struct RecordingPrivacy {
    paths: Mutex<Vec<(std::path::PathBuf, PrivacyPathKind)>>,
    reject_files: bool,
}

impl RecordingPrivacy {
    fn rejecting_files() -> Self {
        Self {
            paths: Mutex::new(Vec::new()),
            reject_files: true,
        }
    }

    fn paths(&self) -> Vec<(std::path::PathBuf, PrivacyPathKind)> {
        self.paths.lock().expect("privacy calls lock").clone()
    }

    fn record(&self, path: &std::path::Path, kind: PrivacyPathKind) {
        self.paths
            .lock()
            .expect("privacy calls lock")
            .push((path.to_path_buf(), kind));
    }
}

impl ArtifactPrivacy for RecordingPrivacy {
    fn prepare_directory(&self, path: &std::path::Path) -> Result<(), LogStoreError> {
        self.record(path, PrivacyPathKind::Directory);
        Ok(())
    }

    fn prepare_file(&self, path: &std::path::Path) -> Result<(), LogStoreError> {
        self.record(path, PrivacyPathKind::File);
        if self.reject_files {
            return Err(LogStoreError::PrivacyNotGuaranteed);
        }
        Ok(())
    }
}

#[cfg(unix)]
#[test]
fn artifact_root_rejects_intermediate_symlinks_without_creating_content() {
    use std::os::unix::fs::symlink;

    let database_root = tempfile::tempdir().expect("database root");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(database_root.path(), clock.clone()).expect("open store");
    let parent = tempfile::tempdir().expect("parent root");
    let target = tempfile::tempdir().expect("target root");
    let link = parent.path().join("redirect");
    symlink(target.path(), &link).expect("intermediate artifact link");
    let configured = link.join("nested-artifacts");

    let error = match ArtifactFileStore::open(configured, clock, store) {
        Ok(_) => panic!("intermediate symlink must fail closed"),
        Err(error) => error,
    };
    assert!(matches!(error, LogStoreError::PathUnsafe { .. }));
    assert!(!target.path().join("nested-artifacts").exists());
}

// ════════════════════════════════
// 1. Write-then-read roundtrip
// ════════════════════════════════

#[test]
fn artifact_write_then_read_roundtrip() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    let content = b"hello artifact world";
    let receipt = afs
        .write_artifact(
            "art-1",
            "req-1",
            "log",
            &clock.now(),
            content,
            None::<&str>,
            1,
            false,
            false,
            4096,
            8192,
        )
        .unwrap();

    // Receipt has no filesystem paths.
    assert_eq!(receipt.artifact_id, "art-1");
    assert_eq!(receipt.bytes, content.len());
    assert_eq!(receipt.checksum, expected_checksum(content));
    assert!(!receipt.checksum.contains('/')); // no path chars in checksum

    let art_content = afs.read_artifact("art-1").unwrap();
    assert_eq!(art_content.bytes, content);
    assert_eq!(art_content.checksum, receipt.checksum);
}

#[test]
fn artifact_without_stored_content_has_consistent_read_and_status_results() {
    let database_root = tempfile::tempdir().expect("database root");
    let artifact_root = tempfile::tempdir().expect("artifact root");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(database_root.path(), clock.clone()).expect("open store");
    store
        .insert_summary(
            "req-no-content",
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
    store
        .insert_artifact_pointer(
            "art-no-content",
            "req-no-content",
            &clock.now(),
            "log",
            None,
        )
        .expect("insert artifact pointer");
    store
        .conn()
        .execute(
            "UPDATE artifact_pointers SET media_kind = 'text/plain' WHERE artifact_id = 'art-no-content'",
            [],
        )
        .expect("set media kind without stored content");
    let artifact_path = artifact_root
        .path()
        .join("req-no-content")
        .join("art-no-content");
    fs::create_dir_all(artifact_path.parent().expect("artifact parent"))
        .expect("create artifact parent");
    fs::write(&artifact_path, b"untracked content").expect("write artifact file");

    let artifact_store = ArtifactFileStore::open(artifact_root.path().to_path_buf(), clock, store)
        .expect("open artifact store");

    assert_eq!(
        artifact_store
            .status("art-no-content")
            .expect("artifact status"),
        ArtifactStatus::Missing
    );
    assert!(matches!(
        artifact_store.read_artifact("art-no-content"),
        Err(LogStoreError::ArtifactMissing { .. })
    ));
}

// ════════════════════════════════
// 2. Atomic write — no partial final file on failure
// ════════════════════════════════

#[test]
fn artifact_atomic_write_no_partial_final_file() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    // Pre-create a .part file in tmp/ and an incomplete final file.
    let old_part = afs_root.path().join("tmp").join("art-1.part");
    fs::create_dir_all(old_part.parent().unwrap()).unwrap();
    fs::write(&old_part, b"incomplete data").unwrap();

    let fake_final = afs_root.path().join("req-1").join("art-1");
    fs::create_dir_all(fake_final.parent().unwrap()).unwrap();
    fs::write(&fake_final, b"stale partial content").unwrap();

    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // Recovery should have removed the orphan .part file.
    assert!(!old_part.exists());

    // Write new artifact — final path only has complete data after rename.
    let content = b"complete new content";
    afs.write_artifact(
        "art-1",
        "req-1",
        "log",
        &clock.now(),
        content,
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .unwrap();

    // Final file has only the complete new content.
    let final_data = fs::read(&fake_final).unwrap();
    assert_eq!(final_data.as_slice(), content);

    // No .part left behind.
    assert!(!old_part.exists());
}

// ════════════════════════════════
// 3. Redaction applied before write
// ════════════════════════════════

#[test]
fn artifact_redaction_applied_before_write() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    // The production constructor requires a storage-bound redactor. Claiming
    // `redacted = false` below cannot bypass this function.
    let afs = ArtifactFileStore::open_with_redactor(
        afs_root.path().to_path_buf(),
        clock.clone(),
        store,
        Arc::new(|data: &[u8]| -> Vec<u8> {
            String::from_utf8_lossy(data)
                .replace("supersecret123", "[REDACTED]")
                .into_bytes()
        }),
    )
    .unwrap();

    let secret = b"password=supersecret123";
    let receipt = afs
        .write_artifact(
            "art-secret",
            "req-1",
            "log",
            &clock.now(),
            secret,
            None::<&str>,
            1,
            false,
            false,
            4096,
            8192,
        )
        .unwrap();

    // Stored content must be redacted even though the caller claimed false.
    let art_content = afs.read_artifact("art-secret").unwrap();
    assert_eq!(art_content.bytes.as_slice(), b"password=[REDACTED]");
    assert!(receipt.redacted);

    // Raw secret is NOT in the stored bytes.
    assert!(!String::from_utf8_lossy(&art_content.bytes).contains("supersecret"));
}

// ════════════════════════════════
// 4. Truncation respects byte_limit (UTF-8 safe)
// ════════════════════════════════

#[test]
fn artifact_individual_limit_is_rejected_without_pointer_or_files() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // Content larger than the individual limit is rejected rather than truncated.
    let content = "hello é world this is a long message that exceeds the limit".as_bytes();
    let byte_limit = 20;

    let result = afs.write_artifact(
        "art-trunc",
        "req-1",
        "log",
        &clock.now(),
        content,
        None::<&str>,
        1,
        false,
        false,
        byte_limit,
        8192,
    );

    assert!(matches!(
        result,
        Err(LogStoreError::ArtifactLimitExceeded { ref kind, limit_bytes, .. })
            if kind == "byte" && limit_bytes == byte_limit
    ));
    assert!(
        afs.store_ref()
            .get_artifact_pointer("art-trunc")
            .unwrap()
            .is_none()
    );
    assert!(!afs_root.path().join("req-1").join("art-trunc").exists());
    assert!(!afs_root.path().join("tmp").join("art-trunc.part").exists());
}

#[test]
fn artifact_binary_content_at_limit_is_preserved_without_utf8_loss() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-binary",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();
    let content = [0xff; 64];

    let receipt = afs
        .write_artifact(
            "art-binary",
            "req-binary",
            "log",
            &clock.now(),
            &content,
            None::<&str>,
            1,
            false,
            false,
            content.len(),
            8192,
        )
        .unwrap();

    assert!(!receipt.truncated);
    assert_eq!(receipt.bytes, content.len());
    assert_eq!(receipt.checksum, expected_checksum(&content));
    assert_eq!(afs.read_artifact("art-binary").unwrap().bytes, content);
}

// ════════════════════════════════
// 5. Individual and aggregate limits rejected without partial files
// ════════════════════════════════

#[test]
fn artifact_individual_and_aggregate_limits_rejected_without_partial_files() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // byte_limit exceeded → rejected with no durable pointer or artifact file.
    let big_content = vec![0u8; 1024];
    {
        let result = afs.write_artifact(
            "art-big",
            "req-1",
            "log",
            &clock.now(),
            &big_content,
            None::<&str>,
            1,
            false,
            false,
            64,
            8192,
        );
        assert!(matches!(
            result,
            Err(LogStoreError::ArtifactLimitExceeded { ref kind, limit_bytes: 64, .. }) if kind == "byte"
        ));
        assert!(
            afs.store_ref()
                .get_artifact_pointer("art-big")
                .unwrap()
                .is_none()
        );
        assert!(!afs_root.path().join("req-1").join("art-big").exists());
    }

    // Aggregate limit exceeded on second write (two writes exceed total).
    let content_a = b"first artifact data";
    afs.write_artifact(
        "art-a",
        "req-1",
        "log",
        &clock.now(),
        content_a,
        None::<&str>,
        1,
        false,
        false,
        4096,
        32, // aggregate_limit=32
    )
    .unwrap();

    let content_b = b"second artifact data";
    let result = afs.write_artifact(
        "art-b",
        "req-1",
        "log",
        &clock.now(),
        content_b,
        None::<&str>,
        1,
        false,
        false,
        4096,
        32, // aggregate_limit=32 already exceeded by art-a (17 bytes) + art-b (20 bytes)
    );

    match result {
        Err(LogStoreError::ArtifactLimitExceeded { kind, .. }) => assert_eq!(kind, "aggregate"),
        other => panic!(
            "expected ArtifactLimitExceeded(aggregate), got: {:?}",
            other
        ),
    }

    // art-b file should not exist.
    let art_b_path = afs_root.path().join("req-1").join("art-b");
    assert!(!art_b_path.exists());
}

// ════════════════════════════════
// 6. Path confinement rejects unsafe segments
// ════════════════════════════════

#[test]
fn artifact_path_confinement_rejects_unsafe_segments() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // ../ traversal in artifact_id.
    let result = afs.write_artifact(
        "../etc/passwd",
        "req-1",
        "log",
        &clock.now(),
        b"nope",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    );
    assert!(matches!(result, Err(LogStoreError::PathUnsafe { .. })));

    // / in artifact_id.
    let result = afs.write_artifact(
        "foo/bar",
        "req-1",
        "log",
        &clock.now(),
        b"nope",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    );
    assert!(matches!(result, Err(LogStoreError::PathUnsafe { .. })));

    // \ in artifact_id.
    let result = afs.write_artifact(
        "foo\\bar",
        "req-1",
        "log",
        &clock.now(),
        b"nope",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    );
    assert!(matches!(result, Err(LogStoreError::PathUnsafe { .. })));

    // NUL in artifact_id.
    let result = afs.write_artifact(
        "foo\x00bar",
        "req-1",
        "log",
        &clock.now(),
        b"nope",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    );
    assert!(matches!(result, Err(LogStoreError::PathUnsafe { .. })));

    // Absolute path in request_id.
    let result = afs.write_artifact(
        "art-ok",
        "/etc/passwd",
        "log",
        &clock.now(),
        b"nope",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    );
    assert!(matches!(result, Err(LogStoreError::PathUnsafe { .. })));

    // ".." as artifact_id.
    let result = afs.write_artifact(
        "..",
        "req-1",
        "log",
        &clock.now(),
        b"nope",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    );
    assert!(matches!(result, Err(LogStoreError::PathUnsafe { .. })));

    // "." as artifact_id.
    let result = afs.write_artifact(
        ".",
        "req-1",
        "log",
        &clock.now(),
        b"nope",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    );
    assert!(matches!(result, Err(LogStoreError::PathUnsafe { .. })));

    // Verify no file was created outside root.
    let etc = afs_root.path().join("..").join("etc");
    assert!(!etc.exists());
}

// ════════════════════════════════
// 7. Symlink rejected
// ════════════════════════════════

#[cfg(unix)]
#[test]
fn artifact_symlink_rejected() {
    use std::os::unix::fs::symlink;

    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();

    // Insert a summary for the real request.
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");

    // Create a symlink inside artifact root pointing outside.
    let target_dir = tempfile::tempdir().expect("symlink target dir");
    let link_path = afs_root.path().join("req-1");
    symlink(target_dir.path(), &link_path).unwrap();

    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // Attempt to write through the symlinked dir.
    let result = afs.write_artifact(
        "art-sym",
        "req-1",
        "log",
        &clock.now(),
        b"nope",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    );

    // Should be rejected because the parent dir is a symlink.
    assert!(matches!(result, Err(LogStoreError::PathUnsafe { .. })));

    // No file written to target_dir.
    assert_eq!(fs::read_dir(target_dir.path()).unwrap().count(), 0);
}

#[cfg(unix)]
#[test]
fn artifact_reads_statuses_and_deletes_reject_a_symlinked_request_parent() {
    use std::os::unix::fs::symlink;

    let db_root = tempfile::tempdir().expect("database root");
    let artifact_root = tempfile::tempdir().expect("artifact root");
    let outside = tempfile::tempdir().expect("outside root");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(db_root.path(), clock.clone()).expect("open store");
    store
        .insert_summary(
            "req-link",
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
    let afs = ArtifactFileStore::open(artifact_root.path().to_path_buf(), clock.clone(), store)
        .expect("open artifact store");
    afs.write_artifact(
        "art-link",
        "req-link",
        "log",
        &clock.now(),
        b"owned bytes",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .expect("write artifact");

    let owned_request_dir = artifact_root.path().join("req-link");
    fs::remove_file(owned_request_dir.join("art-link")).expect("remove owned file");
    fs::remove_dir(&owned_request_dir).expect("remove owned request dir");
    let sentinel = outside.path().join("art-link");
    fs::write(&sentinel, b"outside sentinel").expect("write sentinel");
    symlink(outside.path(), &owned_request_dir).expect("link request parent outside");

    assert!(matches!(
        afs.read_artifact("art-link"),
        Err(LogStoreError::PathUnsafe { .. })
    ));
    assert!(matches!(
        afs.status("art-link"),
        Err(LogStoreError::PathUnsafe { .. })
    ));
    assert!(matches!(
        afs.delete_artifact("art-link"),
        Err(LogStoreError::PathUnsafe { .. })
    ));
    assert_eq!(
        fs::read(&sentinel).expect("outside sentinel survives"),
        b"outside sentinel"
    );
    assert!(
        afs.store_ref()
            .get_artifact_pointer("art-link")
            .unwrap()
            .is_some()
    );
}

#[cfg(unix)]
#[test]
fn artifact_operations_reject_a_symlinked_target_without_touching_the_target() {
    use std::os::unix::fs::symlink;

    let db_root = tempfile::tempdir().expect("database root");
    let artifact_root = tempfile::tempdir().expect("artifact root");
    let outside = tempfile::tempdir().expect("outside root");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(db_root.path(), clock.clone()).expect("open store");
    store
        .insert_summary(
            "req-target",
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
    let afs = ArtifactFileStore::open(artifact_root.path().to_path_buf(), clock.clone(), store)
        .expect("open artifact store");
    afs.write_artifact(
        "art-target",
        "req-target",
        "log",
        &clock.now(),
        b"owned bytes",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .expect("write artifact");

    let owned_target = artifact_root.path().join("req-target").join("art-target");
    fs::remove_file(&owned_target).expect("remove owned target");
    let sentinel = outside.path().join("sentinel");
    fs::write(&sentinel, b"outside sentinel").expect("write sentinel");
    symlink(&sentinel, &owned_target).expect("link target outside");

    assert!(matches!(
        afs.read_artifact("art-target"),
        Err(LogStoreError::PathUnsafe { .. })
    ));
    assert!(matches!(
        afs.status("art-target"),
        Err(LogStoreError::PathUnsafe { .. })
    ));
    assert!(matches!(
        afs.delete_artifact("art-target"),
        Err(LogStoreError::PathUnsafe { .. })
    ));
    assert_eq!(
        fs::read(&sentinel).expect("outside sentinel survives"),
        b"outside sentinel"
    );
    assert!(
        afs.store_ref()
            .get_artifact_pointer("art-target")
            .unwrap()
            .is_some()
    );
}

#[cfg(unix)]
#[test]
fn startup_recovery_skips_symlinked_request_dirs_and_never_traverses_outside_root() {
    use std::os::unix::fs::symlink;

    let db_root = tempfile::tempdir().expect("database root");
    let artifact_root = tempfile::tempdir().expect("artifact root");
    let outside = tempfile::tempdir().expect("outside root");
    let sentinel = outside.path().join("orphan-artifact");
    fs::write(&sentinel, b"outside sentinel").expect("write sentinel");
    symlink(outside.path(), artifact_root.path().join("req-outside"))
        .expect("link request parent outside");

    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(db_root.path(), clock.clone()).expect("open store");
    let _afs = ArtifactFileStore::open(artifact_root.path().to_path_buf(), clock, store)
        .expect("open artifact store");

    assert_eq!(
        fs::read(&sentinel).expect("outside sentinel survives"),
        b"outside sentinel"
    );
}

// ════════════════════════════════
// 8. Checksum verification: corrupt and missing
// ════════════════════════════════

#[test]
fn artifact_checksum_verification_corrupt_and_missing() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // Write a valid artifact.
    let content = b"valid content here";
    afs.write_artifact(
        "art-ok",
        "req-1",
        "log",
        &clock.now(),
        content,
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .unwrap();

    // Status should be Ok.
    let status = afs.status("art-ok").unwrap();
    assert!(matches!(status, ArtifactStatus::Ok { .. }));

    // Corrupt the file on disk (modify one byte).
    let art_path = afs_root.path().join("req-1").join("art-ok");
    let mut data = fs::read(&art_path).unwrap();
    data[0] ^= 0xFF; // flip first byte
    fs::write(&art_path, &data).unwrap();

    // Read should return ArtifactCorrupt.
    match afs.read_artifact("art-ok") {
        Err(LogStoreError::ArtifactCorrupt { .. }) => {} // expected
        other => panic!("expected ArtifactCorrupt, got: {:?}", other),
    }

    // Status reports Corrupt.
    assert_eq!(afs.status("art-ok").unwrap(), ArtifactStatus::Corrupt);

    // Now delete the file entirely (simulate missing).
    fs::remove_file(&art_path).unwrap();

    match afs.read_artifact("art-ok") {
        Err(LogStoreError::ArtifactMissing { .. }) => {} // expected
        other => panic!("expected ArtifactMissing, got: {:?}", other),
    }

    assert_eq!(afs.status("art-ok").unwrap(), ArtifactStatus::Missing);
}

// ════════════════════════════════
// 9. Delete removes file and row
// ════════════════════════════════

#[test]
fn artifact_delete_removes_file_and_row() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // Write two artifacts for req-1.
    afs.write_artifact(
        "art-d1",
        "req-1",
        "log",
        &clock.now(),
        b"delete me 1",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .unwrap();
    afs.write_artifact(
        "art-d2",
        "req-1",
        "log",
        &clock.now(),
        b"delete me 2",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .unwrap();

    // Delete single artifact.
    afs.delete_artifact("art-d1").unwrap();

    let art_path = afs_root.path().join("req-1").join("art-d1");
    assert!(!art_path.exists());

    match afs.read_artifact("art-d1") {
        Err(LogStoreError::ArtifactMissing { .. }) => {} // expected — row is gone
        other => panic!("expected ArtifactMissing after delete, got: {:?}", other),
    }

    // art-d2 still exists.
    assert!(afs_root.path().join("req-1").join("art-d2").exists());

    // Delete all artifacts for req-1.
    let count = afs.delete_artifacts_for_request("req-1").unwrap();
    assert_eq!(count, 1); // only art-d2 remains to be deleted

    assert!(!afs_root.path().join("req-1").exists());
}

// ════════════════════════════════
// 10. Cascade cleanup deletes artifact files
// ════════════════════════════════

#[test]
fn artifact_cascade_cleanup_deletes_artifact_files() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();

    // Insert summaries + artifacts for Jan and Mar.
    store
        .insert_summary(
            "req-jan",
            None,
            None,
            None,
            None,
            "2025-01-15T00:00:00Z",
            None,
            None,
            None,
        )
        .unwrap();
    store.conn().execute(
        "INSERT INTO lifecycle_events (event_id, request_id, occurred_at, payload_json) VALUES (?, ?, ?, ?)",
        rusqlite::params!["ev-jan-1", "req-jan", "2025-01-15T00:00:00Z", r#"{"type":"admitted"}"#],
    ).unwrap();

    store
        .insert_summary(
            "req-mar",
            None,
            None,
            None,
            None,
            "2025-03-15T00:00:00Z",
            None,
            None,
            None,
        )
        .unwrap();
    store.conn().execute(
        "INSERT INTO lifecycle_events (event_id, request_id, occurred_at, payload_json) VALUES (?, ?, ?, ?)",
        rusqlite::params!["ev-mar-1", "req-mar", "2025-03-15T00:00:00Z", r#"{"type":"admitted"}"#],
    ).unwrap();
    store
        .write_terminal_event(
            "req-jan",
            "ev-jan-terminal",
            r#"{"type":"completed"}"#,
            "completed",
            None,
            "2025-01-15T00:00:00Z",
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // Write artifacts for both requests.
    afs.write_artifact(
        "art-jan",
        "req-jan",
        "log",
        "2025-01-15T00:00:00Z",
        b"jan data",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .unwrap();
    // Need a new store reference to write for req-mar (same store).
    afs.write_artifact(
        "art-mar",
        "req-mar",
        "log",
        "2025-03-15T00:00:00Z",
        b"mar data",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .unwrap();

    // Verify both files exist.
    assert!(afs_root.path().join("req-jan").join("art-jan").exists());
    assert!(afs_root.path().join("req-mar").join("art-mar").exists());

    // An unowned file with the same artifact filename in another request
    // directory must never be selected by cascade cleanup.
    let duplicate_unowned = afs_root.path().join("req-unowned").join("art-jan");
    fs::create_dir(duplicate_unowned.parent().unwrap()).unwrap();
    fs::write(&duplicate_unowned, b"must survive").unwrap();

    // Cascade cleanup before Feb (removes Jan entries).
    let store_ref = afs.store_ref();
    let (_, pointers) = store_ref
        .cascade_cleanup_before("2025-02-01T00:00:00Z")
        .unwrap();
    assert_eq!(pointers.len(), 1);
    assert_eq!(pointers[0].request_id, "req-jan");
    assert_eq!(pointers[0].artifact_id, "art-jan");

    // Delete only the file path retained from the removed pointer row.
    afs.delete_artifact_files(&pointers)
        .expect("delete queued artifact files");

    // Jan file should be gone; both a referenced newer file and the unowned
    // duplicate filename survive.
    assert!(!afs_root.path().join("req-jan").join("art-jan").exists());
    assert!(afs_root.path().join("req-mar").join("art-mar").exists());
    assert!(duplicate_unowned.exists());

    // DB row for art-jan is gone too (CASCADE).
    match afs.read_artifact("art-jan") {
        Err(LogStoreError::ArtifactMissing { .. }) => {} // expected
        other => panic!("expected ArtifactMissing after cascade, got: {:?}", other),
    }
}

#[cfg(unix)]
#[test]
fn cascade_cleanup_retries_durable_artifact_deletions_after_filesystem_failure() {
    use std::os::unix::fs::symlink;

    let database_root = tempfile::tempdir().expect("database root");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(database_root.path(), clock.clone()).expect("open store");
    store
        .insert_summary(
            "req-retry",
            None,
            None,
            None,
            None,
            "2025-01-15T00:00:00Z",
            None,
            None,
            None,
        )
        .expect("insert summary");
    store
        .write_terminal_event(
            "req-retry",
            "ev-retry-terminal",
            r#"{"type":"completed"}"#,
            "completed",
            None,
            "2025-01-15T00:00:01Z",
        )
        .expect("insert terminal event");

    let artifact_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(artifact_root.path().to_path_buf(), clock.clone(), store)
        .expect("open artifact store");
    afs.write_artifact(
        "art-retry",
        "req-retry",
        "log",
        "2025-01-15T00:00:00Z",
        b"retained until deleted",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .expect("write artifact");

    let (_, first_attempt) = afs
        .store_ref()
        .cascade_cleanup_before("2025-02-01T00:00:00Z")
        .expect("select cleanup work");
    assert_eq!(first_attempt.len(), 1);

    let artifact_path = artifact_root.path().join("req-retry").join("art-retry");
    fs::remove_file(&artifact_path).expect("replace artifact with link");
    let outside = artifact_root.path().join("outside");
    fs::write(&outside, b"must survive").expect("outside target");
    symlink(&outside, &artifact_path).expect("unsafe replacement");
    assert!(afs.delete_artifact_files(&first_attempt).is_err());
    assert_eq!(fs::read(&outside).unwrap(), b"must survive");

    fs::remove_file(&artifact_path).expect("remove unsafe replacement");
    fs::create_dir(&artifact_path).expect("replace artifact with directory");
    assert!(afs.delete_artifact_files(&first_attempt).is_err());
    fs::remove_dir(&artifact_path).expect("remove unsafe directory replacement");

    let (_, retry) = afs
        .store_ref()
        .cascade_cleanup_before("2025-02-01T00:00:00Z")
        .expect("reload durable cleanup work");
    assert_eq!(retry, first_attempt);
    afs.delete_artifact_files(&retry)
        .expect("acknowledge retry");

    let (_, completed) = afs
        .store_ref()
        .cascade_cleanup_before("2025-02-01T00:00:00Z")
        .expect("verify cleanup completion");
    assert!(completed.is_empty());
}

#[test]
fn committed_cascade_survives_crash_reopen_and_retries_pending_file_deletion() {
    let root = tempfile::tempdir().expect("temporary root");
    let database_root = root.path().join("database");
    let artifact_root = root.path().join("artifacts");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(&database_root, Arc::clone(&clock)).expect("store");
    store
        .insert_summary(
            "request-crash",
            None,
            None,
            None,
            None,
            "2025-01-01T00:00:00Z",
            None,
            None,
            None,
        )
        .expect("summary");
    store
        .write_terminal_event(
            "request-crash",
            "terminal-crash",
            r#"{"type":"completed"}"#,
            "completed",
            None,
            "2025-01-01T00:00:01Z",
        )
        .expect("terminal");
    let artifacts = ArtifactFileStore::open(artifact_root.clone(), Arc::clone(&clock), store)
        .expect("artifacts");
    artifacts
        .write_artifact(
            "artifact-crash",
            "request-crash",
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

    artifacts
        .store_ref()
        .cascade_cleanup_before("2025-02-01T00:00:00Z")
        .expect("commit owner cascade");
    let unsafe_leaf = artifact_root.join("request-crash").join("artifact-crash");
    fs::remove_file(&unsafe_leaf).expect("replace artifact with unsafe directory");
    fs::create_dir(&unsafe_leaf).expect("unsafe directory");
    drop(artifacts);

    let reopened_store =
        LogStore::reopen_at(&database_root, Arc::clone(&clock)).expect("reopen store");
    let pending = |store: &LogStore| {
        let connection = store.conn();
        let mut statement = connection
            .prepare(
                "SELECT artifact_id, request_id FROM pending_artifact_deletions \
                 ORDER BY artifact_id",
            )
            .expect("pending deletion table");
        statement
            .query_map([], |row| {
                Ok(crate::repositories::CascadeArtifactPointer {
                    artifact_id: row.get(0)?,
                    request_id: row.get(1)?,
                })
            })
            .expect("pending query")
            .collect::<Result<Vec<_>, _>>()
            .expect("pending rows")
    };
    assert_eq!(
        pending(&reopened_store),
        vec![crate::repositories::CascadeArtifactPointer {
            artifact_id: "artifact-crash".to_owned(),
            request_id: "request-crash".to_owned(),
        }],
        "the committed cascade must survive a process restart"
    );

    let reopened =
        ArtifactFileStore::open(artifact_root, Arc::clone(&clock), reopened_store).expect("reopen");
    let first_attempt = pending(reopened.store_ref());
    reopened
        .delete_artifact_files(&first_attempt)
        .expect_err("unsafe replacement must fail closed");
    assert_eq!(
        pending(reopened.store_ref()),
        first_attempt,
        "unsafe-path failures remain durable and retryable"
    );

    fs::remove_dir(&unsafe_leaf).expect("repair unsafe leaf");
    let retry = pending(reopened.store_ref());
    reopened
        .delete_artifact_files(&retry)
        .expect("missing file reconciliation succeeds");
    assert!(
        pending(reopened.store_ref()).is_empty(),
        "missing-file reconciliation acknowledges"
    );
}

// ════════════════════════════════
// 11. Startup recovery removes orphans
// ════════════════════════════════

#[test]
fn artifact_startup_recovery_removes_orphans() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());

    // Create orphan .part file in tmp/ and an unreferenced artifact file.
    let afs_root = tempfile::tempdir().expect("artifact root");
    let tmp_dir = afs_root.path().join("tmp");
    fs::create_dir_all(&tmp_dir).unwrap();
    fs::write(tmp_dir.join("orphan.part"), b"stale temp data").unwrap();

    // Create an unreferenced artifact file (no DB pointer row).
    let req_dir = afs_root.path().join("req-ghost");
    fs::create_dir_all(&req_dir).unwrap();
    fs::write(req_dir.join("art-ghost"), b"unreferenced data").unwrap();

    // Open store — recovery should clean up.
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let _afs =
        ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // Orphan .part removed.
    assert!(!tmp_dir.join("orphan.part").exists());

    // Unreferenced file removed.
    assert!(!req_dir.join("art-ghost").exists());
}

// ════════════════════════════════
// 12. Reopen preserves artifacts
// ════════════════════════════════

#[test]
fn artifact_reopen_preserves_artifacts() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());

    // Open store, write artifact.
    let store1 = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store1
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs1 =
        ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store1).unwrap();

    let content = b"persistent artifact data";
    afs1.write_artifact(
        "art-persist",
        "req-1",
        "log",
        &clock.now(),
        content,
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .unwrap();

    // Drop artifact store (which owns the LogStore via Arc).
    drop(afs1);

    // Reopen at same paths.
    let store2 = LogStore::reopen_at(tmp.path(), clock.clone()).unwrap();
    let afs2 = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock, store2).unwrap();

    // Read succeeds with correct content.
    let art_content = afs2.read_artifact("art-persist").unwrap();
    assert_eq!(art_content.bytes.as_slice(), content);
}

// ════════════════════════════════
// 13. Unix modes 0700/0600
// ════════════════════════════════

#[cfg(unix)]
#[test]
fn artifact_unix_modes_0700_0600() {
    use std::os::unix::fs::PermissionsExt;

    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // Write an artifact.
    afs.write_artifact(
        "art-perm",
        "req-1",
        "log",
        &clock.now(),
        b"permission test",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .unwrap();

    // Root dir should be mode 0700.
    let root_meta = fs::metadata(afs_root.path()).unwrap();
    assert_eq!(root_meta.permissions().mode() & 0o777, 0o700);

    // Request subdirectory should be mode 0700.
    let req_dir = afs_root.path().join("req-1");
    let dir_meta = fs::metadata(&req_dir).unwrap();
    assert_eq!(dir_meta.permissions().mode() & 0o777, 0o700);

    // Artifact file should be mode 0600.
    let art_path = req_dir.join("art-perm");
    let file_meta = fs::metadata(&art_path).unwrap();
    assert_eq!(file_meta.permissions().mode() & 0o777, 0o600);
}

// ════════════════════════════════
// 14. Duplicate write rejected (AlreadyExists)
// ════════════════════════════════

#[test]
fn artifact_duplicate_write_rejected() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // First write succeeds.
    afs.write_artifact(
        "art-dup",
        "req-1",
        "log",
        &clock.now(),
        b"first",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .unwrap();

    // Second write with same artifact_id → AlreadyExists.
    let result = afs.write_artifact(
        "art-dup",
        "req-1",
        "log",
        &clock.now(),
        b"second attempt",
        None::<&str>,
        2,
        false,
        false,
        4096,
        8192,
    );

    match result {
        Err(LogStoreError::AlreadyExists { .. }) => {} // expected
        other => panic!(
            "expected AlreadyExists on duplicate write, got: {:?}",
            other
        ),
    }

    // Original file still intact (no orphan).
    let art_path = afs_root.path().join("req-1").join("art-dup");
    assert!(art_path.exists());
    assert_eq!(fs::read(&art_path).unwrap(), b"first");
}

// ════════════════════════════════
// 15. Transactional rollback removes file on FK failure
// ════════════════════════════════

#[test]
fn artifact_transactional_rollback_removes_file() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(tmp.path(), clock.clone()).unwrap();

    // Insert a summary for req-1 but NOT for nonexistent-request.
    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();

    let afs_root = tempfile::tempdir().expect("artifact root");
    let afs = ArtifactFileStore::open(afs_root.path().to_path_buf(), clock.clone(), store).unwrap();

    // Attempt to write artifact for nonexistent request_id → FK violation.
    let result = afs.write_artifact(
        "art-fk",
        "nonexistent-request",
        "log",
        &clock.now(),
        b"fk fail",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    );

    // Should fail with a SQLite error (FK constraint).
    assert!(result.is_err());

    // No orphan file should exist.
    let req_dir = afs_root.path().join("nonexistent-request");
    if req_dir.exists() {
        let entries: Vec<_> = fs::read_dir(&req_dir).unwrap().collect();
        assert_eq!(
            entries.len(),
            0,
            "no artifact files should exist after FK failure"
        );
    }

    // No .part left in tmp/.
    let tmp_path = afs_root.path().join("tmp").join("art-fk.part");
    assert!(!tmp_path.exists());
}

// ════════════════════════════════
// 16. Privacy seam: every content path is prepared before use
// ════════════════════════════════

#[test]
fn artifact_privacy_prepares_root_tmp_request_and_files() {
    let db_root = tempfile::tempdir().expect("database root");
    let artifact_root = tempfile::tempdir().expect("artifact root");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(db_root.path(), clock.clone()).expect("open store");
    store
        .insert_summary(
            "req-privacy",
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

    let privacy = Arc::new(RecordingPrivacy::default());
    let afs = ArtifactFileStore::open_with_privacy_for_test(
        artifact_root.path().to_path_buf(),
        clock.clone(),
        store,
        privacy.clone(),
    )
    .expect("open artifact store");
    afs.write_artifact(
        "art-privacy",
        "req-privacy",
        "log",
        &clock.now(),
        b"privacy prepared content",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .expect("write artifact");

    let paths = privacy.paths();
    let canonical_root = artifact_root.path().canonicalize().expect("canonical root");
    let tmp = canonical_root.join("tmp");
    let request = canonical_root.join("req-privacy");
    let temp_file = tmp.join("art-privacy.part");
    let final_file = request.join("art-privacy");
    assert!(paths.iter().any(|(path, kind)| {
        *kind == PrivacyPathKind::Directory
            && (path == artifact_root.path() || path == &canonical_root)
    }));
    assert!(paths.contains(&(tmp, PrivacyPathKind::Directory)));
    assert!(paths.contains(&(request, PrivacyPathKind::Directory)));
    assert!(paths.contains(&(temp_file, PrivacyPathKind::File)));
    assert!(paths.contains(&(final_file, PrivacyPathKind::File)));
}

#[test]
fn artifact_privacy_failure_prevents_content_and_cleans_temp_file() {
    let db_root = tempfile::tempdir().expect("database root");
    let artifact_root = tempfile::tempdir().expect("artifact root");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(db_root.path(), clock.clone()).expect("open store");
    store
        .insert_summary(
            "req-privacy-failure",
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

    let privacy = Arc::new(RecordingPrivacy::rejecting_files());
    let afs = ArtifactFileStore::open_with_privacy_for_test(
        artifact_root.path().to_path_buf(),
        clock.clone(),
        store,
        privacy,
    )
    .expect("open artifact store");
    let result = afs.write_artifact(
        "art-privacy-failure",
        "req-privacy-failure",
        "log",
        &clock.now(),
        b"must never be written",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    );

    assert!(matches!(result, Err(LogStoreError::PrivacyNotGuaranteed)));
    assert!(
        !artifact_root
            .path()
            .join("tmp")
            .join("art-privacy-failure.part")
            .exists()
    );
    assert!(
        !artifact_root
            .path()
            .join("req-privacy-failure")
            .join("art-privacy-failure")
            .exists()
    );
}

#[cfg(windows)]
#[test]
fn windows_artifact_paths_have_current_owner_and_exact_user_only_dacl() {
    use std::ffi::c_void;
    use std::mem::align_of;
    use std::os::windows::ffi::OsStrExt;
    use std::ptr::null_mut;
    use windows_sys::Win32::Foundation::{CloseHandle, LocalFree};
    use windows_sys::Win32::Security::Authorization::{GetNamedSecurityInfoW, SE_FILE_OBJECT};
    use windows_sys::Win32::Security::{
        ACCESS_ALLOWED_ACE, DACL_SECURITY_INFORMATION, EqualSid, GetAce,
        GetSecurityDescriptorControl, GetTokenInformation, IsWellKnownSid,
        OWNER_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR, PSID, SE_DACL_PROTECTED, TOKEN_QUERY,
        TOKEN_USER, TokenUser, WinBuiltinUsersSid, WinWorldSid,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    struct Token(*mut c_void);

    impl Drop for Token {
        fn drop(&mut self) {
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }

    struct Descriptor(PSECURITY_DESCRIPTOR);

    impl Drop for Descriptor {
        fn drop(&mut self) {
            unsafe {
                let _ = LocalFree(self.0);
            }
        }
    }

    fn wide(path: &std::path::Path) -> Vec<u16> {
        path.as_os_str().encode_wide().chain(Some(0)).collect()
    }

    fn with_current_user_sid<T>(f: impl FnOnce(PSID) -> T) -> T {
        let mut token = null_mut();
        assert_ne!(
            unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) },
            0
        );
        let _token = Token(token);

        let mut bytes = 0_u32;
        unsafe {
            let _ = GetTokenInformation(token, TokenUser, null_mut(), 0, &mut bytes);
        }
        assert_ne!(bytes, 0);
        let mut buffer = vec![0_usize; (bytes as usize).div_ceil(align_of::<usize>())];
        assert_ne!(
            unsafe {
                GetTokenInformation(
                    token,
                    TokenUser,
                    buffer.as_mut_ptr().cast::<c_void>(),
                    bytes,
                    &mut bytes,
                )
            },
            0
        );
        let token_user = buffer.as_ptr().cast::<TOKEN_USER>();
        f(unsafe { (*token_user).User.Sid })
    }

    fn assert_current_owner_and_exact_dacl(path: &std::path::Path, expected_flags: u32) {
        with_current_user_sid(|current_user| {
            let path_wide = wide(path);
            let mut owner = null_mut();
            let mut dacl = null_mut();
            let mut descriptor: PSECURITY_DESCRIPTOR = null_mut();
            assert_eq!(
                unsafe {
                    GetNamedSecurityInfoW(
                        path_wide.as_ptr(),
                        SE_FILE_OBJECT,
                        OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
                        &mut owner,
                        null_mut(),
                        &mut dacl,
                        null_mut(),
                        &mut descriptor,
                    )
                },
                0
            );
            let _descriptor = Descriptor(descriptor);
            assert!(!owner.is_null());
            assert!(!dacl.is_null());
            assert_eq!(unsafe { EqualSid(owner, current_user) }, 1);

            let mut control = 0_u16;
            let mut revision = 0_u32;
            assert_ne!(
                unsafe { GetSecurityDescriptorControl(descriptor, &mut control, &mut revision) },
                0
            );
            assert_ne!(control & SE_DACL_PROTECTED, 0);
            assert_eq!(unsafe { (*dacl).AceCount }, 1);

            let mut ace = null_mut();
            assert_ne!(unsafe { GetAce(dacl, 0, &mut ace) }, 0);
            let allowed = ace.cast::<ACCESS_ALLOWED_ACE>();
            let ace_sid = unsafe {
                (&(*allowed).SidStart as *const u32)
                    .cast_mut()
                    .cast::<c_void>()
            };
            assert_eq!(unsafe { (*allowed).Header.AceType }, 0);
            assert_eq!(unsafe { (*allowed).Header.AceFlags as u32 }, expected_flags);
            assert_ne!(unsafe { EqualSid(ace_sid, current_user) }, 0);
            assert_eq!(unsafe { IsWellKnownSid(ace_sid, WinWorldSid) }, 0);
            assert_eq!(unsafe { IsWellKnownSid(ace_sid, WinBuiltinUsersSid) }, 0);
        });
    }

    let db_root = tempfile::tempdir().expect("database root");
    let artifact_root = tempfile::tempdir().expect("artifact root");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(db_root.path(), clock.clone()).expect("open store");
    store
        .insert_summary(
            "req-windows-privacy",
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
    let afs = ArtifactFileStore::open(artifact_root.path().to_path_buf(), clock.clone(), store)
        .expect("open artifact store");
    afs.write_artifact(
        "art-windows-privacy",
        "req-windows-privacy",
        "log",
        &clock.now(),
        b"windows privacy artifact",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .expect("write artifact");

    let canonical_root = artifact_root.path().canonicalize().expect("canonical root");
    assert_current_owner_and_exact_dacl(
        &canonical_root,
        windows_sys::Win32::Security::OBJECT_INHERIT_ACE
            | windows_sys::Win32::Security::CONTAINER_INHERIT_ACE,
    );
    assert_current_owner_and_exact_dacl(
        &canonical_root.join("tmp"),
        windows_sys::Win32::Security::OBJECT_INHERIT_ACE
            | windows_sys::Win32::Security::CONTAINER_INHERIT_ACE,
    );
    assert_current_owner_and_exact_dacl(
        &canonical_root.join("req-windows-privacy"),
        windows_sys::Win32::Security::OBJECT_INHERIT_ACE
            | windows_sys::Win32::Security::CONTAINER_INHERIT_ACE,
    );
    assert_current_owner_and_exact_dacl(
        &canonical_root
            .join("req-windows-privacy")
            .join("art-windows-privacy"),
        0,
    );
}
