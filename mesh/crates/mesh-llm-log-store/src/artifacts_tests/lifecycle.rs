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
