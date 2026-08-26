use super::*;

#[test]
fn initialized_version_one_is_a_data_preserving_noop() {
    let connection = Connection::open_in_memory().expect("open database");
    run_migrations(&connection, plan(1, synthetic_initial_schema, &[])).expect("initialize schema");
    connection
        .execute("INSERT INTO baseline VALUES ('kept')", [])
        .expect("seed current schema data");

    run_migrations(&connection, plan(1, synthetic_initial_schema, &[])).expect("reopen schema");

    assert!(lineage::is_valid(&connection).expect("inspect lineage"));
    assert_eq!(
        connection
            .query_row("SELECT value FROM baseline", [], |row| row
                .get::<_, String>(0))
            .expect("preserved row"),
        "kept"
    );
}

#[test]
fn markerless_identified_version_one_is_rejected_without_mutation() {
    let connection = Connection::open_in_memory().expect("open database");
    connection
        .execute_batch(
            "CREATE TABLE sentinel (value TEXT NOT NULL); INSERT INTO sentinel VALUES ('kept');
             PRAGMA application_id = 0x4D4C4F47; PRAGMA user_version = 1;",
        )
        .expect("seed forged schema");

    assert!(run_migrations(&connection, plan(1, synthetic_initial_schema, &[])).is_err());

    assert_eq!(version(&connection), 1);
    assert_eq!(application_id(&connection), APPLICATION_ID);
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
fn nonzero_version_without_identity_is_rejected_without_mutation() {
    let connection = Connection::open_in_memory().expect("open database");
    let registry = [Migration {
        version: 2,
        apply: migration_v2,
    }];
    connection
        .execute_batch(
            "CREATE TABLE sentinel (value TEXT); INSERT INTO sentinel VALUES ('kept'); \
             PRAGMA user_version = 1;",
        )
        .expect("seed foreign schema");

    assert!(run_migrations(&connection, plan(2, synthetic_initial_schema, &registry)).is_err());

    assert_eq!(version(&connection), 1);
    assert_eq!(application_id(&connection), 0);
    assert!(!schema_object_names(&connection, "table").contains(&"future_v2".to_owned()));
    assert_eq!(
        connection
            .query_row("SELECT value FROM sentinel", [], |row| row
                .get::<_, String>(0))
            .expect("sentinel row"),
        "kept"
    );
}

#[test]
fn nonzero_version_with_wrong_identity_is_rejected_without_mutation() {
    let connection = Connection::open_in_memory().expect("open database");
    let registry = [Migration {
        version: 2,
        apply: migration_v2,
    }];
    connection
        .execute_batch(
            "CREATE TABLE sentinel (value TEXT); PRAGMA application_id = 0x4E4F5045; \
             PRAGMA user_version = 1;",
        )
        .expect("seed foreign schema");

    assert!(run_migrations(&connection, plan(2, synthetic_initial_schema, &registry)).is_err());

    assert_eq!(version(&connection), 1);
    assert_eq!(application_id(&connection), 0x4E4F5045);
    assert_eq!(user_objects(&connection), 1);
    assert!(!schema_object_names(&connection, "table").contains(&"future_v2".to_owned()));
}

#[test]
fn empty_version_zero_with_identity_is_rejected_without_mutation() {
    let connection = Connection::open_in_memory().expect("open database");
    connection
        .pragma_update(None, "application_id", APPLICATION_ID)
        .expect("seed identity without schema");

    assert!(run_migrations(&connection, plan(1, synthetic_initial_schema, &[])).is_err());

    assert_eq!(version(&connection), 0);
    assert_eq!(application_id(&connection), APPLICATION_ID);
    assert_eq!(user_objects(&connection), 0);
}

#[test]
fn identified_version_one_advances_to_future_target() {
    let connection = Connection::open_in_memory().expect("open database");
    run_migrations(&connection, plan(1, synthetic_initial_schema, &[])).expect("initialize schema");
    let registry = [Migration {
        version: 2,
        apply: migration_v2,
    }];

    run_migrations(&connection, plan(2, synthetic_initial_schema, &registry))
        .expect("migrate identified schema");

    assert_eq!(version(&connection), 2);
    assert_eq!(application_id(&connection), APPLICATION_ID);
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
fn malformed_lineage_states_are_rejected_without_migration() {
    const CASES: &[&str] = &[
        "CREATE TABLE _mesh_llm_log_store_lineage (id INTEGER NOT NULL PRIMARY KEY, epoch INTEGER NOT NULL) WITHOUT ROWID;
         INSERT INTO _mesh_llm_log_store_lineage VALUES (1, 2);",
        "CREATE TABLE _mesh_llm_log_store_lineage (id INTEGER NOT NULL PRIMARY KEY, epoch INTEGER NOT NULL, extra INTEGER) WITHOUT ROWID;
         INSERT INTO _mesh_llm_log_store_lineage VALUES (1, 1, 0);",
        "CREATE TABLE _mesh_llm_log_store_lineage (id INTEGER NOT NULL PRIMARY KEY, epoch INTEGER NOT NULL) WITHOUT ROWID;
         INSERT INTO _mesh_llm_log_store_lineage VALUES (1, 1);
         INSERT INTO _mesh_llm_log_store_lineage VALUES (2, 1);",
        "CREATE VIEW _mesh_llm_log_store_lineage AS SELECT 1 AS id, 1 AS epoch;",
    ];
    for marker_sql in CASES {
        let connection = Connection::open_in_memory().expect("open database");
        connection
            .execute_batch(&format!(
                "CREATE TABLE sentinel (value TEXT NOT NULL); INSERT INTO sentinel VALUES ('kept');
                 {marker_sql}
                 PRAGMA application_id = 0x4D4C4F47; PRAGMA user_version = 1;"
            ))
            .expect("seed malformed lineage");
        let objects_before = user_objects(&connection);

        let error = run_migrations(&connection, plan(1, synthetic_initial_schema, &[]))
            .expect_err("reject malformed lineage");

        assert!(matches!(error, rusqlite::Error::InvalidQuery));
        assert_eq!(version(&connection), 1);
        assert_eq!(application_id(&connection), APPLICATION_ID);
        assert_eq!(user_objects(&connection), objects_before);
        assert_eq!(
            connection
                .query_row("SELECT value FROM sentinel", [], |row| row
                    .get::<_, String>(0))
                .expect("sentinel row"),
            "kept"
        );
    }
}

#[test]
fn extra_private_object_invalidates_an_initialized_lineage() {
    let connection = Connection::open_in_memory().expect("open database");
    run_migrations(&connection, plan(1, synthetic_initial_schema, &[])).expect("initialize schema");
    connection
        .execute_batch("CREATE VIEW _mesh_llm_log_store_extra AS SELECT 1 AS value;")
        .expect("seed extra private object");

    let error = run_migrations(&connection, plan(1, synthetic_initial_schema, &[]))
        .expect_err("reject extra private object");

    assert!(matches!(error, rusqlite::Error::InvalidQuery));
    assert_eq!(version(&connection), 1);
    assert_eq!(application_id(&connection), APPLICATION_ID);
    assert!(
        schema_object_names(&connection, "view").contains(&"_mesh_llm_log_store_extra".to_owned())
    );
}
