use super::*;

#[test]
fn fresh_db_initializes_to_current_version() {
    let (store, _, _tmp) = open_store();
    assert_eq!(store.schema_version(), CURRENT_VERSION);
}

#[test]
fn system_clock_uses_fixed_width_nanoseconds() {
    let timestamp = SystemClock.now();

    assert_eq!(timestamp.len(), 30);
    assert_eq!(&timestamp[19..20], ".");
    assert_eq!(&timestamp[29..30], "Z");
    assert!(timestamp[20..29].bytes().all(|byte| byte.is_ascii_digit()));
}

#[cfg(unix)]
#[test]
fn sqlite_root_database_and_sidecars_are_owner_private() {
    use std::os::unix::fs::PermissionsExt;

    let root = tempfile::tempdir().expect("database root");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(root.path(), clock.clone()).expect("open store");
    store
        .insert_summary(
            "private-db",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .expect("write transaction");

    let database = store.db_path().to_path_buf();
    assert_eq!(
        std::fs::metadata(root.path()).unwrap().permissions().mode() & 0o777,
        0o700
    );
    assert_eq!(
        std::fs::metadata(&database).unwrap().permissions().mode() & 0o777,
        0o600
    );
    for suffix in ["-wal", "-shm"] {
        let sidecar = std::path::PathBuf::from(format!("{}{}", database.display(), suffix));
        if sidecar.exists() {
            assert_eq!(
                std::fs::metadata(sidecar).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }
}

#[cfg(unix)]
#[test]
fn sqlite_root_symlink_is_rejected_before_canonicalization() {
    use std::os::unix::fs::symlink;

    let parent = tempfile::tempdir().expect("parent root");
    let target = tempfile::tempdir().expect("target root");
    let link = parent.path().join("configured-log-root");
    symlink(target.path(), &link).expect("database root link");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());

    let error = match LogStore::open(&link, clock) {
        Ok(_) => panic!("symlinked root must fail closed"),
        Err(error) => error,
    };
    assert!(matches!(error, LogStoreError::PathUnsafe { .. }));
    assert!(!target.path().join("log_store.db").exists());
    assert!(!error.to_string().contains(&link.display().to_string()));
}

#[cfg(unix)]
#[test]
fn sqlite_root_rejects_intermediate_symlinks_without_creating_content() {
    use std::os::unix::fs::symlink;

    let parent = tempfile::tempdir().expect("parent root");
    let target = tempfile::tempdir().expect("target root");
    let link = parent.path().join("redirect");
    symlink(target.path(), &link).expect("intermediate root link");
    let configured = link.join("nested-log-root");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());

    let error = match LogStore::open(&configured, clock) {
        Ok(_) => panic!("intermediate symlink must fail closed"),
        Err(error) => error,
    };
    assert!(matches!(error, LogStoreError::PathUnsafe { .. }));
    assert!(!target.path().join("nested-log-root").exists());
}

#[cfg(windows)]
#[test]
fn sqlite_root_database_and_sidecars_have_only_current_user_acl() {
    let root = tempfile::tempdir().expect("database root");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(root.path(), clock.clone()).expect("open store");
    store
        .insert_summary(
            "private-db",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .expect("write transaction");

    crate::artifact_privacy::verify_current_user_only_storage_path(root.path(), true)
        .expect("private root DACL");
    let database = store.db_path();
    crate::artifact_privacy::verify_current_user_only_storage_path(database, false)
        .expect("private database DACL");
    for suffix in ["-wal", "-shm"] {
        let sidecar = std::path::PathBuf::from(format!("{}{}", database.display(), suffix));
        if sidecar.exists() {
            crate::artifact_privacy::verify_current_user_only_storage_path(&sidecar, false)
                .expect("private SQLite sidecar DACL");
        }
    }
}

#[test]
fn current_schema_reopen_preserves_data_without_schema_work() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());

    // Insert data.
    let store1 = LogStore::open(tmp.path(), clock.clone()).expect("open current schema");
    store1
        .insert_summary(
            "s-001",
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
    drop(store1);

    // Reopen at same path — migrations should be skipped, data preserved.
    let (store2, _, _tmp) = {
        let s = LogStore::reopen_at(tmp.path(), clock.clone()).expect("reopen");
        (s, clock.clone(), tmp)
    };

    assert_eq!(store2.schema_version(), CURRENT_VERSION);
    let row = store2
        .get_summary("s-001")
        .unwrap()
        .expect("summary exists after reopen");
    assert_eq!(row.request_id, "s-001");
}

#[test]
fn current_schema_is_a_noop_on_reopen() {
    let tmp = tempfile::tempdir().expect("create temp dir");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());

    for _ in 0..3 {
        let s = LogStore::open(tmp.path(), clock.clone()).expect("open");
        assert_eq!(s.schema_version(), CURRENT_VERSION);
        drop(s);
    }
}

#[test]
fn unknown_nonempty_version_zero_is_rejected_before_database_mutation() {
    assert_unknown_schema_rejected_before_mutation(0);
}

#[test]
fn unknown_future_version_is_rejected_before_database_mutation() {
    assert_unknown_schema_rejected_before_mutation(2);
}

#[test]
fn markerless_identified_version_one_is_rejected_before_wal_or_schema_mutation() {
    let root = tempfile::tempdir().expect("database root");
    let database = root.path().join("log_store.db");
    let connection = rusqlite::Connection::open(&database).expect("open fixture database");
    connection
        .execute_batch(
            "CREATE TABLE sentinel (value TEXT NOT NULL); INSERT INTO sentinel VALUES ('kept');
             PRAGMA application_id = 0x4D4C4F47; PRAGMA user_version = 1;",
        )
        .expect("seed forged schema");
    drop(connection);

    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let error = match LogStore::open(root.path(), clock) {
        Ok(_) => panic!("markerless schema must fail closed"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        LogStoreError::SchemaIncompatible {
            found: 1,
            supported: 1
        }
    ));
    assert!(!root.path().join("log_store.db-wal").exists());
    assert!(!root.path().join("log_store.db-shm").exists());
    let connection = rusqlite::Connection::open(database).expect("reopen fixture database");
    assert_eq!(
        connection
            .pragma_query_value(None, "journal_mode", |row| row.get::<_, String>(0))
            .expect("journal mode"),
        "delete"
    );
    assert_eq!(
        connection
            .query_row("SELECT value FROM sentinel", [], |row| row
                .get::<_, String>(0))
            .expect("sentinel row"),
        "kept"
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("schema object count"),
        1
    );
}

fn assert_unknown_schema_rejected_before_mutation(version: u32) {
    let root = tempfile::tempdir().expect("database root");
    let database = root.path().join("log_store.db");
    let connection = rusqlite::Connection::open(&database).expect("open fixture database");
    connection
        .execute_batch(&format!(
            "CREATE TABLE sentinel (value TEXT NOT NULL); INSERT INTO sentinel VALUES ('kept'); PRAGMA user_version = {version};"
        ))
        .expect("seed unknown schema");
    drop(connection);

    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let error = match LogStore::open(root.path(), clock) {
        Ok(_) => panic!("unknown schema must fail closed"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        LogStoreError::SchemaIncompatible { found, supported: 1 } if found == version
    ));
    let connection = rusqlite::Connection::open(database).expect("reopen fixture database");
    assert_eq!(
        connection
            .pragma_query_value(None, "user_version", |row| row.get::<_, u32>(0))
            .expect("schema version"),
        version
    );
    assert_eq!(
        connection
            .pragma_query_value(None, "journal_mode", |row| row.get::<_, String>(0))
            .expect("journal mode"),
        "delete"
    );
    assert_eq!(
        connection
            .query_row("SELECT value FROM sentinel", [], |row| row
                .get::<_, String>(0))
            .expect("sentinel row"),
        "kept"
    );
    assert_eq!(
        connection
            .query_row(
                "SELECT COUNT(*) FROM sqlite_schema WHERE type = 'table' AND name IN ('summaries', 'audit_entries')",
                [],
                |row| row.get::<_, i64>(0),
            )
            .expect("log table count"),
        0
    );
}

// ════════════════════════════════════
//  TERMINAL EVENT TRANSACTION TESTS
// ════════════════════════════════════

#[test]
fn terminal_event_and_summary_are_one_transaction() {
    let (store, clock, _tmp) = open_store();

    store
        .insert_summary(
            "txn-s1",
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

    let payload = r#"{"type":"completed","status_code":200}"#;
    store
        .write_terminal_event(
            "txn-s1",
            "evt-1",
            payload,
            "completed",
            Some(200),
            &clock.now(),
        )
        .expect("write terminal succeeds");

    let row = store
        .get_summary("txn-s1")
        .unwrap()
        .expect("summary exists");
    assert_eq!(row.state, "completed");
    assert_eq!(row.status_code, Some(200));
}

#[test]
fn duplicate_terminal_write_returns_typed_conflict() {
    let (store, clock, _tmp) = open_store();

    store
        .insert_summary(
            "dup-s1",
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

    let payload1 = r#"{"type":"completed","status_code":200}"#;
    store
        .write_terminal_event(
            "dup-s1",
            "evt-1",
            payload1,
            "completed",
            Some(200),
            &clock.now(),
        )
        .expect("first write succeeds");

    let err = store
        .write_terminal_event(
            "dup-s1",
            "evt-2",
            r#"{"type":"failed","error":"boom"}"#,
            "failed",
            None,
            &clock.now(),
        )
        .unwrap_err();
    assert!(matches!(err, LogStoreError::DuplicateTerminalEvent { .. }));
}

#[test]
fn duplicate_terminal_error_keeps_payload_out_of_the_error() {
    let (store, clock, _tmp) = open_store();
    let secret = "supersecret-terminal-payload";
    store
        .insert_summary(
            "duplicate-safe",
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
        .write_terminal_event(
            "duplicate-safe",
            "first-terminal",
            r#"{"type":"completed"}"#,
            "completed",
            None,
            &clock.now(),
        )
        .expect("first terminal");

    let error = store
        .write_terminal_event(
            "duplicate-safe",
            "second-terminal",
            &format!(r#"{{"type":"failed","error":"{secret}"}}"#),
            "failed",
            None,
            &clock.now(),
        )
        .expect_err("duplicate terminal should fail");

    assert!(matches!(
        error,
        LogStoreError::DuplicateTerminalEvent { ref event_type, .. } if event_type == "failed"
    ));
    assert!(!error.to_string().contains(secret));
}

#[test]
fn terminal_detection_uses_the_typed_top_level_event_type() {
    let (store, clock, _tmp) = open_store();
    store
        .insert_summary(
            "typed-terminal",
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
        .insert_lifecycle_event(
            "typed-terminal",
            "nested-terminal-text",
            r#"{"type":"admitted","context":{"type":"completed"}}"#,
            &clock.now(),
        )
        .expect("nested terminal text is not terminal");
    assert!(
        !store
            .has_terminal_event("typed-terminal")
            .expect("terminal state")
    );

    store
        .insert_lifecycle_event(
            "typed-terminal",
            "actual-terminal",
            r#"{"type":"completed"}"#,
            &clock.now(),
        )
        .expect("actual terminal event");
    assert!(
        store
            .has_terminal_event("typed-terminal")
            .expect("terminal state")
    );
}

#[test]
fn duplicate_non_terminal_same_type_events_allowed() {
    let (store, clock, _tmp) = open_store();

    store
        .insert_summary(
            "multi-s1",
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

    // Insert two non-terminal events with different event_ids — both should succeed.
    let payload = r#"{"type":"admitted","model":"llama3"}"#;
    store
        .insert_lifecycle_event("multi-s1", "evt-started-1", payload, &clock.now())
        .expect("first started succeeds");
    store
        .insert_lifecycle_event("multi-s1", "evt-stream-2", payload, &clock.now())
        .expect("second admitted also succeeds — no unique(summary,event_type) constraint");

    // Verify both events exist.
    let count = store.count_table("lifecycle_events").unwrap();
    assert_eq!(count, 2);
}
