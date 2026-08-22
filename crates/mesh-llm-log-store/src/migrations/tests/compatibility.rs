use super::*;
use crate::{LogStore, RealClock};
use std::sync::Arc;

fn initialized_schema(connection: &Connection) -> Result<(), rusqlite::Error> {
    connection.execute_batch(
        "CREATE TABLE sentinel (value TEXT NOT NULL); INSERT INTO sentinel VALUES ('kept');",
    )
}

fn no_op_migration(_: &Connection) -> Result<(), rusqlite::Error> {
    Ok(())
}

fn initialize_to_version(connection: &Connection, version: u32) {
    let migrations = [
        Migration {
            version: 2,
            apply: no_op_migration,
        },
        Migration {
            version: 3,
            apply: no_op_migration,
        },
    ];
    let migration_count = usize::try_from(version.saturating_sub(1)).expect("migration count");
    run_migrations(
        connection,
        MigrationPlan {
            target: version,
            initialize: initialized_schema,
            migrations: &migrations[..migration_count],
        },
    )
    .expect("initialize synthetic schema");
}

fn assert_inspection_preserves_database(
    connection: &Connection,
    target: u32,
    expected: Option<(u32, u32)>,
) {
    let identity = application_id(connection).expect("application identity");
    let found = schema_version(connection).expect("schema version");
    let object_count = connection
        .query_row(
            "SELECT COUNT(*) FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .expect("schema object count");

    assert_eq!(
        incompatible_schema_for_target(connection, target).expect("inspect schema"),
        expected
    );
    assert_eq!(
        application_id(connection).expect("application identity"),
        identity
    );
    assert_eq!(schema_version(connection).expect("schema version"), found);
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
        object_count
    );
}

#[test]
fn initialized_version_one_is_compatible_with_target_two_without_mutation() {
    let connection = Connection::open_in_memory().expect("open database");
    initialize_to_version(&connection, 1);
    assert_inspection_preserves_database(&connection, 2, None);
}

#[test]
fn initialized_version_two_is_compatible_with_target_two_without_mutation() {
    let connection = Connection::open_in_memory().expect("open database");
    initialize_to_version(&connection, 2);
    assert_inspection_preserves_database(&connection, 2, None);
}

#[test]
fn initialized_version_three_is_incompatible_with_target_two_without_mutation() {
    let connection = Connection::open_in_memory().expect("open database");
    initialize_to_version(&connection, 3);
    assert_inspection_preserves_database(&connection, 2, Some((3, 2)));
}

#[test]
fn nonempty_version_zero_is_incompatible_with_target_two_without_mutation() {
    let connection = Connection::open_in_memory().expect("open database");
    initialized_schema(&connection).expect("seed unknown schema");
    assert_inspection_preserves_database(&connection, 2, Some((0, 2)));
}

#[test]
fn missing_identity_is_incompatible_without_mutation() {
    let connection = Connection::open_in_memory().expect("open database");
    initialize_to_version(&connection, 1);
    connection
        .pragma_update(None, "application_id", 0_u32)
        .expect("clear application identity");
    assert_inspection_preserves_database(&connection, 2, Some((1, 2)));
}

#[test]
fn wrong_identity_is_incompatible_without_mutation() {
    let connection = Connection::open_in_memory().expect("open database");
    initialize_to_version(&connection, 1);
    connection
        .pragma_update(None, "application_id", 0x4E4F5045_u32)
        .expect("replace application identity");
    assert_inspection_preserves_database(&connection, 2, Some((1, 2)));
}

#[test]
fn markerless_identified_version_one_is_incompatible_without_mutation() {
    let connection = Connection::open_in_memory().expect("open database");
    initialized_schema(&connection).expect("seed unknown schema");
    connection
        .pragma_update(None, "application_id", APPLICATION_ID)
        .expect("seed application identity");
    connection
        .pragma_update(None, "user_version", 1_u32)
        .expect("seed schema version");
    assert_inspection_preserves_database(&connection, 2, Some((1, 2)));
}

#[test]
fn high_bit_header_values_are_classified_as_incompatible() {
    let connection = Connection::open_in_memory().expect("open database");
    connection
        .pragma_update(None, "user_version", -2_147_483_647_i32)
        .expect("seed high-bit schema version");
    connection
        .pragma_update(None, "application_id", -2_147_483_646_i32)
        .expect("seed high-bit application identity");

    assert_eq!(
        incompatible_schema_for_target(&connection, 2).expect("inspect high-bit schema"),
        Some((0x8000_0001, 2))
    );
}

#[test]
fn released_schema_marker_eleven_store_upgrades_without_losing_rows() {
    let root = tempfile::tempdir().expect("temporary store root");
    let database_path = root.path().join("log_store.db");
    {
        let connection = Connection::open(&database_path).expect("open released database");
        super::released_schema_fixture::install(&connection);
    }

    let store = LogStore::open(root.path(), Arc::new(RealClock))
        .expect("upgrade released version eleven store");

    assert_eq!(store.schema_version(), CURRENT_VERSION);
    super::released_schema_fixture::assert_seeded_history(&store.conn());
    drop(store);

    let reopened = LogStore::open(root.path(), Arc::new(RealClock))
        .expect("reopen upgraded version eleven store");
    assert_eq!(reopened.schema_version(), CURRENT_VERSION);
    assert_eq!(
        table_columns(&reopened.conn(), "summaries"),
        [
            "request_id",
            "state",
            "created_at",
            "terminal_at",
            "route",
            "model",
            "provider",
            "engine",
            "status_code",
            "error_msg",
            "tenant_id",
            "account_id",
            "user_id",
            "caller_endpoint_id",
            "caller_addr",
            "caller_path_type",
        ]
    );
    super::released_schema_fixture::assert_seeded_history(&reopened.conn());
}
