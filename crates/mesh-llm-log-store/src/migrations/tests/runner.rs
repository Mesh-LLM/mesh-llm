use super::*;

mod identity;

fn synthetic_initial_schema(connection: &Connection) -> Result<(), rusqlite::Error> {
    connection.execute_batch("CREATE TABLE baseline (value TEXT NOT NULL);")
}

fn failing_initial_schema(connection: &Connection) -> Result<(), rusqlite::Error> {
    connection.execute_batch(
        "CREATE TABLE initialization_partial (value TEXT); INSERT INTO missing_table VALUES (1);",
    )
}

fn conflicting_initial_schema(connection: &Connection) -> Result<(), rusqlite::Error> {
    connection.execute_batch(
        "CREATE TABLE baseline (value TEXT NOT NULL);
         CREATE TABLE _mesh_llm_log_store_lineage (
             id INTEGER NOT NULL PRIMARY KEY,
             epoch INTEGER NOT NULL
         ) WITHOUT ROWID;",
    )
}

fn migration_v2(connection: &Connection) -> Result<(), rusqlite::Error> {
    connection.execute_batch(
        "CREATE TABLE future_v2 (value TEXT NOT NULL); INSERT INTO future_v2 VALUES ('v2');",
    )
}

fn failing_migration_v2(connection: &Connection) -> Result<(), rusqlite::Error> {
    connection.execute_batch(
        "CREATE TABLE future_v2_partial (value TEXT); INSERT INTO missing_table VALUES (2);",
    )
}

fn failing_migration_v3(connection: &Connection) -> Result<(), rusqlite::Error> {
    connection.execute_batch(
        "CREATE TABLE future_v3_partial (value TEXT); INSERT INTO missing_table VALUES (3);",
    )
}

fn migration_v3(connection: &Connection) -> Result<(), rusqlite::Error> {
    connection.execute_batch(
        "CREATE TABLE future_v3 (value TEXT NOT NULL); INSERT INTO future_v3 VALUES ('v3');",
    )
}

fn version(connection: &Connection) -> u32 {
    connection
        .pragma_query_value(None, "user_version", |row| row.get(0))
        .expect("schema version")
}

fn application_id(connection: &Connection) -> u32 {
    connection
        .pragma_query_value(None, "application_id", |row| row.get(0))
        .expect("application identity")
}

fn user_objects(connection: &Connection) -> i64 {
    connection
        .query_row(
            "SELECT COUNT(*) FROM sqlite_schema WHERE type IN ('table', 'index', 'view', 'trigger') AND name NOT LIKE 'sqlite_%'",
            [],
            |row| row.get(0),
        )
        .expect("user object count")
}

fn plan<'a>(
    target: u32,
    initialize: MigrationFn,
    migrations: &'a [Migration],
) -> MigrationPlan<'a> {
    MigrationPlan {
        target,
        initialize,
        migrations,
    }
}

#[test]
fn empty_version_zero_bootstraps_atomically() {
    let connection = Connection::open_in_memory().expect("open database");

    run_migrations(&connection, plan(1, synthetic_initial_schema, &[])).expect("initialize schema");

    assert_eq!(version(&connection), 1);
    assert_eq!(application_id(&connection), APPLICATION_ID);
    assert_eq!(user_objects(&connection), 2);
    assert!(lineage::is_valid(&connection).expect("inspect lineage"));
}

#[test]
fn nonempty_version_zero_is_rejected_without_mutation() {
    let connection = Connection::open_in_memory().expect("open database");
    connection
        .execute_batch("CREATE TABLE sentinel (value TEXT); INSERT INTO sentinel VALUES ('kept');")
        .expect("seed unknown schema");

    assert!(run_migrations(&connection, plan(1, synthetic_initial_schema, &[])).is_err());

    assert_eq!(version(&connection), 0);
    assert_eq!(user_objects(&connection), 1);
    assert_eq!(
        connection
            .query_row("SELECT value FROM sentinel", [], |row| row
                .get::<_, String>(0))
            .expect("sentinel row"),
        "kept"
    );
}

#[test]
fn preexisting_lineage_marker_makes_version_zero_nonfresh() {
    let connection = Connection::open_in_memory().expect("open database");
    lineage::install(&connection).expect("seed lineage marker");

    assert!(run_migrations(&connection, plan(1, synthetic_initial_schema, &[])).is_err());

    assert_eq!(version(&connection), 0);
    assert_eq!(application_id(&connection), 0);
    assert_eq!(user_objects(&connection), 1);
    assert!(lineage::is_valid(&connection).expect("inspect lineage"));
}

#[test]
fn production_rejects_version_two_without_mutation() {
    let connection = Connection::open_in_memory().expect("open database");
    lineage::install(&connection).expect("seed lineage marker");
    connection
        .execute_batch(
            "CREATE TABLE sentinel (value TEXT); PRAGMA application_id = 0x4D4C4F47; \
             PRAGMA user_version = 2;",
        )
        .expect("seed future schema");

    assert!(apply_migrations(&connection).is_err());

    assert_eq!(version(&connection), 2);
    assert_eq!(user_objects(&connection), 2);
    assert!(lineage::is_valid(&connection).expect("inspect lineage"));
}

#[test]
fn invalid_registries_are_rejected_before_initialization() {
    let invalid_registries: &[(u32, &[Migration])] = &[
        (2, &[]),
        (
            3,
            &[Migration {
                version: 2,
                apply: migration_v2,
            }],
        ),
        (
            3,
            &[
                Migration {
                    version: 3,
                    apply: migration_v3,
                },
                Migration {
                    version: 2,
                    apply: migration_v2,
                },
            ],
        ),
        (
            3,
            &[
                Migration {
                    version: 2,
                    apply: migration_v2,
                },
                Migration {
                    version: 2,
                    apply: migration_v2,
                },
            ],
        ),
        (
            2,
            &[Migration {
                version: 1,
                apply: migration_v2,
            }],
        ),
        (
            3,
            &[
                Migration {
                    version: 2,
                    apply: migration_v2,
                },
                Migration {
                    version: 3,
                    apply: migration_v3,
                },
                Migration {
                    version: 4,
                    apply: migration_v3,
                },
            ],
        ),
    ];

    for (target, registry) in invalid_registries {
        let connection = Connection::open_in_memory().expect("open database");
        assert!(
            run_migrations(
                &connection,
                plan(*target, synthetic_initial_schema, registry),
            )
            .is_err()
        );
        assert_eq!(version(&connection), 0);
        assert_eq!(user_objects(&connection), 0);
    }
}

#[test]
fn synthetic_version_two_step_commits_schema_data_and_version() {
    let connection = Connection::open_in_memory().expect("open database");
    let registry = [Migration {
        version: 2,
        apply: migration_v2,
    }];

    run_migrations(&connection, plan(2, synthetic_initial_schema, &registry))
        .expect("migrate to v2");

    assert_eq!(version(&connection), 2);
    assert_eq!(application_id(&connection), APPLICATION_ID);
    assert_eq!(user_objects(&connection), 3);
    assert!(lineage::is_valid(&connection).expect("inspect lineage"));
    assert_eq!(
        connection
            .query_row("SELECT value FROM future_v2", [], |row| row
                .get::<_, String>(0))
            .expect("v2 row"),
        "v2"
    );
}

#[test]
fn failed_version_two_step_rolls_back_its_schema_and_version() {
    let connection = Connection::open_in_memory().expect("open database");
    let registry = [Migration {
        version: 2,
        apply: failing_migration_v2,
    }];

    assert!(run_migrations(&connection, plan(2, synthetic_initial_schema, &registry)).is_err());

    assert_eq!(version(&connection), 1);
    assert_eq!(application_id(&connection), APPLICATION_ID);
    assert_eq!(user_objects(&connection), 2);
    assert!(lineage::is_valid(&connection).expect("inspect lineage"));
}

#[test]
fn committed_version_two_resumes_with_corrected_version_three() {
    let connection = Connection::open_in_memory().expect("open database");
    let failing = [
        Migration {
            version: 2,
            apply: migration_v2,
        },
        Migration {
            version: 3,
            apply: failing_migration_v3,
        },
    ];
    assert!(run_migrations(&connection, plan(3, synthetic_initial_schema, &failing)).is_err());
    assert_eq!(version(&connection), 2);
    assert_eq!(user_objects(&connection), 3);
    assert!(lineage::is_valid(&connection).expect("inspect lineage"));

    let corrected = [
        Migration {
            version: 2,
            apply: migration_v2,
        },
        Migration {
            version: 3,
            apply: migration_v3,
        },
    ];
    run_migrations(&connection, plan(3, synthetic_initial_schema, &corrected)).expect("resume v3");

    assert_eq!(version(&connection), 3);
    assert_eq!(user_objects(&connection), 4);
    assert!(lineage::is_valid(&connection).expect("inspect lineage"));
    assert_eq!(
        connection
            .query_row("SELECT value FROM future_v3", [], |row| row
                .get::<_, String>(0))
            .expect("v3 row"),
        "v3"
    );
}

#[test]
fn failing_initialization_rolls_back_schema_and_version() {
    let connection = Connection::open_in_memory().expect("open database");

    assert!(run_migrations(&connection, plan(1, failing_initial_schema, &[])).is_err());

    assert_eq!(version(&connection), 0);
    assert_eq!(application_id(&connection), 0);
    assert_eq!(user_objects(&connection), 0);
}

#[test]
fn initialization_lineage_conflict_rolls_back_every_object_and_pragma() {
    let connection = Connection::open_in_memory().expect("open database");

    assert!(run_migrations(&connection, plan(1, conflicting_initial_schema, &[])).is_err());

    assert_eq!(version(&connection), 0);
    assert_eq!(application_id(&connection), 0);
    assert_eq!(user_objects(&connection), 0);
}
