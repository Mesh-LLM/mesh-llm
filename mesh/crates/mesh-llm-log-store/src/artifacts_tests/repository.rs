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
