use super::released_schema_fixture::{
    self, SOURCE_VERSION_ELEVEN, SOURCE_VERSION_THREE, SOURCE_VERSIONS,
};
use super::*;

fn released_connection() -> Connection {
    released_schema_fixture::connection()
}

fn assert_released_state_is_unchanged(connection: &Connection, source_version: u32) {
    assert_eq!(
        schema_version(connection).expect("schema version"),
        source_version
    );
    assert_eq!(application_id(connection).expect("application identity"), 0);
    released_schema_fixture::assert_seeded_history(connection);
    assert!(!lineage::is_valid(connection).expect("inspect lineage"));
}

fn assert_rejected_without_mutation(connection: &Connection) {
    let objects_before = schema_object_names(connection, "table");
    let source_version = schema_version(connection).expect("source version");

    // When
    let compatibility = incompatible_schema(connection).expect("inspect compatibility");

    // Then
    assert_eq!(compatibility, Some((source_version, CURRENT_VERSION)));
    assert_eq!(schema_object_names(connection, "table"), objects_before);
    assert_released_state_is_unchanged(connection, source_version);
}

fn assert_imported_identity_and_caller_columns(connection: &Connection) {
    assert_eq!(
        schema_version(connection).expect("schema version"),
        CURRENT_VERSION
    );
    assert_eq!(
        application_id(connection).expect("application identity"),
        APPLICATION_ID
    );
    assert!(lineage::is_valid(connection).expect("inspect lineage"));
    let caller_values = connection
        .query_row(
            "SELECT caller_endpoint_id, caller_addr, caller_path_type
             FROM summaries WHERE request_id = 'released-request'",
            [],
            |row| {
                Ok((
                    row.get::<_, Option<String>>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            },
        )
        .expect("read imported caller columns");
    assert_eq!(caller_values, (None, None, None));
}

#[test]
fn released_schema_source_versions_are_classified_for_import_without_mutation() {
    for source_version in SOURCE_VERSIONS {
        // Given
        let connection = Connection::open_in_memory().expect("open released database");
        released_schema_fixture::install_at_marker(&connection, source_version);

        // When
        let classification = classify_schema(&connection).expect("classify released schema");
        let compatibility = incompatible_schema(&connection).expect("inspect compatibility");

        // Then
        assert_eq!(
            classification,
            SchemaClassification::ReleasedSchema { source_version }
        );
        assert_eq!(compatibility, None);
        assert_released_state_is_unchanged(&connection, source_version);
    }
}

#[test]
fn released_schema_is_accepted_only_for_the_current_production_target() {
    for source_version in SOURCE_VERSIONS {
        // Given
        let connection = Connection::open_in_memory().expect("open released database");
        released_schema_fixture::install_at_marker(&connection, source_version);

        // When
        let compatibility = incompatible_schema_for_target(&connection, CURRENT_VERSION + 1)
            .expect("inspect non-production target compatibility");

        // Then
        assert_eq!(compatibility, Some((source_version, CURRENT_VERSION + 1)));
        assert_released_state_is_unchanged(&connection, source_version);
    }
}

#[test]
fn released_schema_without_if_not_exists_is_classified_for_import_without_mutation() {
    // Given
    let connection = released_connection();
    connection
        .execute_batch(
            "PRAGMA writable_schema = ON;
             UPDATE sqlite_schema SET sql = replace(sql, ' IF NOT EXISTS', '')
             WHERE sql IS NOT NULL;
             PRAGMA writable_schema = OFF;
             PRAGMA schema_version = 2;",
        )
        .expect("remove optional creation clause");

    // When
    let classification = classify_schema(&connection).expect("classify released schema");

    // Then
    assert_eq!(
        classification,
        SchemaClassification::ReleasedSchema {
            source_version: SOURCE_VERSION_ELEVEN
        }
    );
    assert_released_state_is_unchanged(&connection, SOURCE_VERSION_ELEVEN);
}

#[test]
fn released_schema_predicates_match_independently_of_sql_whitespace() {
    // Given
    let connection = released_connection();
    connection
        .execute_batch(
            "PRAGMA writable_schema = ON;
             UPDATE sqlite_schema SET sql = replace(sql, 'bytes >= 0', 'bytes    >=\n0')
             WHERE type = 'table' AND name = 'artifact_pointers';
             PRAGMA writable_schema = OFF;",
        )
        .expect("reformat released predicate");

    // When
    let classification = classify_schema(&connection).expect("classify reformatted schema");

    // Then
    assert_eq!(
        classification,
        SchemaClassification::ReleasedSchema {
            source_version: SOURCE_VERSION_ELEVEN
        }
    );
    assert_released_state_is_unchanged(&connection, SOURCE_VERSION_ELEVEN);
}

#[test]
fn released_schema_with_extra_missing_or_altered_objects_is_rejected() {
    const MUTATIONS: &[&str] = &[
        "CREATE TABLE unrelated (value TEXT);",
        "DROP TABLE cleanup_runs;",
        "DROP INDEX idx_summaries_state;
         CREATE INDEX idx_summaries_state ON summaries (created_at);",
        "CREATE VIEW released_view AS SELECT request_id FROM summaries;",
        "CREATE TRIGGER released_trigger AFTER INSERT ON summaries BEGIN SELECT 1; END;",
        "ALTER TABLE cleanup_runs ADD COLUMN lookalike TEXT;",
        "CREATE TABLE _mesh_llm_log_store_lookalike (value TEXT);",
    ];

    for mutation in MUTATIONS {
        // Given
        let connection = released_connection();
        connection
            .execute_batch(mutation)
            .expect("mutate released schema");

        // When / Then
        assert_rejected_without_mutation(&connection);
    }
}

#[test]
fn released_schema_without_audit_autoincrement_is_rejected_without_mutation() {
    // Given
    let connection = Connection::open_in_memory().expect("open released database");
    released_schema_fixture::install_without_audit_autoincrement(&connection);
    let tables_before = schema_object_names(&connection, "table");
    let summary_columns_before = table_columns(&connection, "summaries");
    let history_before = released_schema_fixture::seeded_history(&connection);

    // When
    let compatibility = incompatible_schema(&connection).expect("inspect compatibility");

    // Then
    assert_eq!(
        compatibility,
        Some((SOURCE_VERSION_ELEVEN, CURRENT_VERSION))
    );
    assert_eq!(schema_object_names(&connection, "table"), tables_before);
    assert_eq!(
        table_columns(&connection, "summaries"),
        summary_columns_before
    );
    assert_eq!(
        schema_version(&connection).expect("schema version"),
        SOURCE_VERSION_ELEVEN
    );
    assert_eq!(
        application_id(&connection).expect("application identity"),
        0
    );
    assert!(!lineage::is_valid(&connection).expect("inspect lineage"));
    assert_eq!(
        released_schema_fixture::seeded_history(&connection),
        history_before
    );
}

#[test]
fn released_schema_with_altered_foreign_key_metadata_is_rejected() {
    // Given
    let connection = released_connection();
    connection
        .execute_batch(
            "PRAGMA writable_schema = ON;
             UPDATE sqlite_schema
             SET sql = replace(sql, 'ON DELETE CASCADE', 'ON DELETE SET NULL')
             WHERE type = 'table' AND name = 'lifecycle_events';
             PRAGMA writable_schema = OFF;
             PRAGMA schema_version = 2;",
        )
        .expect("alter released foreign key");

    // When / Then
    assert_rejected_without_mutation(&connection);
}

#[test]
fn released_schema_with_changed_quoted_check_literal_case_is_rejected_without_mutation() {
    // Given
    let connection = released_connection();
    connection
        .execute_batch(
            "PRAGMA writable_schema = ON;
             UPDATE sqlite_schema SET sql = replace(
                 sql,
                 'CHECK (state IN (''active'',',
                 'CHECK (state IN (''ACTIVE'','
             )
             WHERE type = 'table' AND name = 'summaries';
             PRAGMA writable_schema = OFF;
             PRAGMA schema_version = 2;",
        )
        .expect("change released check literal case");

    // When / Then
    assert_rejected_without_mutation(&connection);
}

#[test]
fn released_schema_with_deferrable_foreign_key_is_rejected_without_mutation() {
    // Given
    let connection = released_connection();
    connection
        .execute_batch(
            "PRAGMA writable_schema = ON;
             UPDATE sqlite_schema SET sql = replace(
                 sql,
                 'ON DELETE CASCADE',
                 'ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED'
             )
             WHERE type = 'table' AND name = 'lifecycle_events';
             PRAGMA writable_schema = OFF;
             PRAGMA schema_version = 2;",
        )
        .expect("defer released foreign key");

    // When / Then
    assert_rejected_without_mutation(&connection);
}

#[test]
fn released_schema_with_collated_nonindexed_column_is_rejected_without_mutation() {
    // Given
    let connection = released_connection();
    connection
        .execute_batch(
            "PRAGMA writable_schema = ON;
             UPDATE sqlite_schema SET sql = replace(
                 sql,
                 'error_msg    TEXT,',
                 'error_msg    TEXT COLLATE NOCASE,'
             )
             WHERE type = 'table' AND name = 'summaries';
             PRAGMA writable_schema = OFF;
             PRAGMA schema_version = 2;",
        )
        .expect("collate released nonindexed column");

    // When / Then
    assert_rejected_without_mutation(&connection);
}

#[test]
fn released_schema_with_explicit_binary_nonindexed_column_collation_is_accepted() {
    // Given
    let connection = released_connection();
    connection
        .execute_batch(
            "PRAGMA writable_schema = ON;
             UPDATE sqlite_schema SET sql = replace(
                 sql,
                 'error_msg    TEXT,',
                 'error_msg    TEXT COLLATE BINARY,'
             )
             WHERE type = 'table' AND name = 'summaries';
             PRAGMA writable_schema = OFF;
             PRAGMA schema_version = 2;",
        )
        .expect("make released binary collation explicit");

    // When
    let classification = classify_schema(&connection).expect("classify released schema");

    // Then
    assert_eq!(
        classification,
        SchemaClassification::ReleasedSchema {
            source_version: SOURCE_VERSION_ELEVEN
        }
    );
    assert_released_state_is_unchanged(&connection, SOURCE_VERSION_ELEVEN);
}

#[test]
fn released_schema_with_near_match_implicit_unique_index_is_rejected() {
    // Given
    let connection = released_connection();
    connection
        .execute_batch(
            "PRAGMA writable_schema = ON;
             UPDATE sqlite_schema SET sql = replace(
                 sql,
                 'UNIQUE(request_id, event_id)',
                 'UNIQUE(request_id, event_id, occurred_at) /* UNIQUE(request_id, event_id) */'
             )
             WHERE type = 'table' AND name = 'lifecycle_events';
             PRAGMA writable_schema = OFF;
             PRAGMA schema_version = 2;",
        )
        .expect("alter released implicit unique index");

    // When / Then
    assert_rejected_without_mutation(&connection);
}

#[test]
fn released_schema_with_altered_check_or_partial_index_predicate_is_rejected() {
    const MUTATIONS: &[&str] = &[
        "PRAGMA writable_schema = ON;
         UPDATE sqlite_schema SET sql = replace(sql, 'bytes >= 0', 'bytes >= -1')
         WHERE type = 'table' AND name = 'artifact_pointers';
         PRAGMA writable_schema = OFF;",
        "PRAGMA writable_schema = ON;
         UPDATE sqlite_schema SET sql = replace(sql, 'bytes >= 0', 'bytes >= 0 OR bytes = -1')
         WHERE type = 'table' AND name = 'artifact_pointers';
         PRAGMA writable_schema = OFF;",
        "PRAGMA writable_schema = ON;
         UPDATE sqlite_schema SET sql = replace(sql, 'WHERE is_terminal = 1', 'WHERE is_terminal >= 0')
         WHERE type = 'index' AND name = 'idx_terminal_event_one_per_request';
         PRAGMA writable_schema = OFF;",
        "PRAGMA writable_schema = ON;
         UPDATE sqlite_schema SET sql = replace(
             sql,
             'WHERE is_terminal = 1',
             'WHERE is_terminal = 1 OR request_id IS NULL'
         )
         WHERE type = 'index' AND name = 'idx_terminal_event_one_per_request';
         PRAGMA writable_schema = OFF;",
    ];

    for mutation in MUTATIONS {
        // Given
        let connection = released_connection();
        connection
            .execute_batch(mutation)
            .expect("alter released predicate");

        // When / Then
        assert_rejected_without_mutation(&connection);
    }
}

#[test]
fn released_schema_with_nonzero_application_identity_is_rejected() {
    // Given
    let connection = released_connection();
    connection
        .pragma_update(None, "application_id", APPLICATION_ID)
        .expect("set private identity");

    // When
    let compatibility = incompatible_schema(&connection).expect("inspect compatibility");

    // Then
    assert_eq!(
        compatibility,
        Some((SOURCE_VERSION_ELEVEN, CURRENT_VERSION))
    );
    assert_eq!(
        application_id(&connection).expect("application identity"),
        APPLICATION_ID
    );
    assert_eq!(
        schema_version(&connection).expect("schema version"),
        SOURCE_VERSION_ELEVEN
    );
}

#[test]
fn released_schema_with_any_partial_caller_column_state_is_rejected() {
    const CALLER_COLUMNS: &[&str] = &[
        "caller_endpoint_id TEXT",
        "caller_addr TEXT",
        "caller_path_type TEXT",
    ];

    for count in 1..=CALLER_COLUMNS.len() {
        // Given
        let connection = released_connection();
        for column in &CALLER_COLUMNS[..count] {
            connection
                .execute_batch(&format!("ALTER TABLE summaries ADD COLUMN {column};"))
                .expect("add caller column");
        }

        // When / Then
        assert_rejected_without_mutation(&connection);
    }
}

#[test]
fn all_other_source_versions_are_rejected_without_mutation() {
    for version in [0_u32, 1, 2, 4, 10, 12, 99] {
        // Given
        let connection = released_connection();
        connection
            .pragma_update(None, "user_version", version)
            .expect("set unsupported version");

        // When
        let compatibility = incompatible_schema(&connection).expect("inspect compatibility");

        // Then
        assert_eq!(compatibility, Some((version, CURRENT_VERSION)));
        assert_eq!(
            schema_version(&connection).expect("schema version"),
            version
        );
        assert_eq!(
            application_id(&connection).expect("application identity"),
            0
        );
        assert_eq!(
            connection
                .query_row("SELECT COUNT(*) FROM summaries", [], |row| row
                    .get::<_, u32>(0))
                .expect("summary count"),
            1
        );
    }
}

#[test]
fn forced_released_schema_import_failure_rolls_back_both_source_versions() {
    for source_version in SOURCE_VERSIONS {
        // Given
        let connection = Connection::open_in_memory().expect("open released database");
        released_schema_fixture::install_at_marker(&connection, source_version);
        let tables_before = schema_object_names(&connection, "table");

        // When
        let error = released_schema::import_with_before_commit_failure(&connection, source_version)
            .expect_err("force failure before import commit");

        // Then
        assert!(matches!(error, rusqlite::Error::InvalidQuery));
        assert_eq!(schema_object_names(&connection, "table"), tables_before);
        assert_eq!(
            table_columns(&connection, "summaries"),
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
            ]
        );
        assert_released_state_is_unchanged(&connection, source_version);
    }
}

#[test]
fn importer_revalidates_the_classified_source_version_before_writing() {
    // Given
    let connection = released_connection();
    let tables_before = schema_object_names(&connection, "table");

    // When
    let error = released_schema::import(&connection, SOURCE_VERSION_THREE)
        .expect_err("reject mismatched classified source version");

    // Then
    assert!(matches!(error, rusqlite::Error::InvalidQuery));
    assert_eq!(schema_object_names(&connection, "table"), tables_before);
    assert_released_state_is_unchanged(&connection, SOURCE_VERSION_ELEVEN);
}

#[test]
fn production_import_installs_private_identity_lineage_and_nullable_columns() {
    // Given
    let root = tempfile::tempdir().expect("temporary store root");
    let database_path = root.path().join("log_store.db");
    let connection = Connection::open(&database_path).expect("open released database");
    released_schema_fixture::install(&connection);
    drop(connection);

    // When
    let store = crate::LogStore::open(root.path(), std::sync::Arc::new(crate::RealClock))
        .expect("import released store");

    // Then
    let connection = store.conn();
    assert_imported_identity_and_caller_columns(&connection);
    released_schema_fixture::assert_seeded_history(&connection);
}

#[test]
fn production_import_accepts_published_marker_three_and_preserves_rows_after_reopen() {
    // Given
    let root = tempfile::tempdir().expect("temporary store root");
    let database_path = root.path().join("log_store.db");
    let released_history = {
        let connection = Connection::open(&database_path).expect("open released database");
        released_schema_fixture::install_at_marker(&connection, SOURCE_VERSION_THREE);
        released_schema_fixture::seeded_history(&connection)
    };

    // When
    let store = crate::LogStore::open(root.path(), std::sync::Arc::new(crate::RealClock))
        .expect("import published marker-three store");

    // Then
    {
        let connection = store.conn();
        assert_imported_identity_and_caller_columns(&connection);
        assert_eq!(
            released_schema_fixture::seeded_history(&connection),
            released_history
        );
    }
    drop(store);

    let reopened = crate::LogStore::open(root.path(), std::sync::Arc::new(crate::RealClock))
        .expect("reopen imported marker-three store");
    let connection = reopened.conn();
    assert_imported_identity_and_caller_columns(&connection);
    assert_eq!(
        released_schema_fixture::seeded_history(&connection),
        released_history
    );
}
