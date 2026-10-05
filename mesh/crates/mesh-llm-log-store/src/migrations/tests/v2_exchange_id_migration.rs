use super::*;

/// Builds a version-1 database: no `exchange_id` column and no
/// `idx_summaries_exchange_id` index. This is what every log_store.db created
/// before schema version 2 looks like on disk.
fn build_v1_database(connection: &Connection) {
    run_migrations(
        connection,
        MigrationPlan {
            target: 1,
            initialize: crate::schema::initialize,
            migrations: &[],
        },
    )
    .expect("bootstrap v1 schema");
}

#[test]
fn v1_database_has_no_exchange_id_column_or_index() {
    let connection = Connection::open_in_memory().expect("open database");
    build_v1_database(&connection);

    assert_eq!(schema_version(&connection).expect("schema version"), 1);
    assert!(!table_columns(&connection, "summaries").contains(&"exchange_id".to_string()));
    assert!(
        !schema_object_names(&connection, "index")
            .contains(&"idx_summaries_exchange_id".to_string())
    );
}

#[test]
fn v1_database_migrates_forward_to_v2_and_accepts_exchange_id_inserts() {
    let connection = Connection::open_in_memory().expect("open database");
    build_v1_database(&connection);

    apply_migrations(&connection).expect("migrate v1 database forward to current version");

    assert_eq!(schema_version(&connection).expect("schema version"), 2);
    assert_eq!(
        table_columns(&connection, "summaries")
            .last()
            .map(String::as_str),
        Some("exchange_id")
    );
    assert!(
        schema_object_names(&connection, "index")
            .contains(&"idx_summaries_exchange_id".to_string())
    );

    connection
        .execute(
            "INSERT INTO summaries (request_id, created_at, exchange_id) \
             VALUES ('request-1', '2026-09-25T00:00:00Z', 'exch-abc123')",
            [],
        )
        .expect("summary insert with exchange_id succeeds after migration");

    let error = connection
        .execute(
            "INSERT INTO summaries (request_id, created_at, exchange_id) \
             VALUES ('request-2', '2026-09-25T00:00:01Z', 'exch-abc123')",
            [],
        )
        .expect_err("unique partial index rejects a second claim of the same exchange_id");
    assert!(matches!(
        error,
        rusqlite::Error::SqliteFailure(_, Some(message))
            if message.contains("UNIQUE constraint failed") && message.contains("exchange_id")
    ));
}

#[test]
fn v1_database_reopen_after_migration_is_idempotent() {
    let connection = Connection::open_in_memory().expect("open database");
    build_v1_database(&connection);

    apply_migrations(&connection).expect("migrate forward once");
    apply_migrations(&connection).expect("reopening an already-migrated database is a no-op");

    assert_eq!(schema_version(&connection).expect("schema version"), 2);
}
