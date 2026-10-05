use super::*;

#[test]
fn parser_treats_omitted_and_explicit_binary_column_collation_equally() {
    // Given
    let omitted = "CREATE TABLE sample (value TEXT)";
    let explicit = "CREATE TABLE sample (value TEXT COLLATE binary)";

    // When
    let omitted = parse(omitted).expect("parse omitted collation");
    let explicit = parse(explicit).expect("parse explicit collation");

    // Then
    assert_eq!(omitted.columns, explicit.columns);
    assert_eq!(omitted.columns[0].collation, Collation::Binary);
}

#[test]
fn parser_reads_inline_and_table_foreign_key_timing() {
    // Given
    let sql = "CREATE TABLE sample (
        parent_id TEXT REFERENCES parents(id) NOT DEFERRABLE INITIALLY IMMEDIATE,
        backup_id TEXT,
        FOREIGN KEY (backup_id) REFERENCES parents(id) DEFERRABLE INITIALLY DEFERRED
    )";

    // When
    let table = parse(sql).expect("parse foreign keys");

    // Then
    assert_eq!(
        table.foreign_keys,
        [
            ForeignKey {
                columns: vec!["PARENT_ID".to_owned()],
                deferrability: Deferrability::NotDeferrable,
                initial_timing: InitialTiming::Immediate,
            },
            ForeignKey {
                columns: vec!["BACKUP_ID".to_owned()],
                deferrability: Deferrability::Deferrable,
                initial_timing: InitialTiming::Deferred,
            },
        ]
    );
}

#[test]
fn parser_ignores_commas_and_parentheses_inside_nested_or_quoted_regions() {
    // Given
    let sql = "CREATE TABLE sample (
        value TEXT DEFAULT 'kept, with (punctuation)',
        state TEXT CHECK (state IN ('one', 'two'))
    )";

    // When
    let table = parse(sql).expect("parse nested and quoted regions");

    // Then
    assert_eq!(table.columns.len(), 2);
}

#[test]
fn parser_records_integer_primary_key_autoincrement_semantics() {
    // Given
    let autoincrement =
        "CREATE TABLE sample (sequence INTEGER PRIMARY KEY AUTOINCREMENT, value TEXT)";
    let rowid_reuse = "CREATE TABLE sample (sequence INTEGER PRIMARY KEY, value TEXT)";

    // When
    let autoincrement = parse(autoincrement).expect("parse autoincrement primary key");
    let rowid_reuse = parse(rowid_reuse).expect("parse reusable rowid primary key");

    // Then
    assert_eq!(
        autoincrement.autoincrement_column.as_deref(),
        Some("SEQUENCE")
    );
    assert_eq!(rowid_reuse.autoincrement_column, None);
}

#[test]
fn parser_rejects_impossible_or_misplaced_autoincrement_semantics() {
    const INVALID_TABLES: &[&str] = &[
        "CREATE TABLE sample (sequence TEXT PRIMARY KEY AUTOINCREMENT)",
        "CREATE TABLE sample (sequence INTEGER AUTOINCREMENT PRIMARY KEY)",
        "CREATE TABLE sample (sequence INTEGER, PRIMARY KEY (sequence) AUTOINCREMENT)",
        "CREATE TABLE sample (sequence INTEGER PRIMARY KEY AUTOINCREMENT AUTOINCREMENT)",
        "CREATE TABLE sample (first INTEGER PRIMARY KEY AUTOINCREMENT, second INTEGER PRIMARY KEY AUTOINCREMENT)",
    ];

    for sql in INVALID_TABLES {
        // When
        let parsed = parse(sql);

        // Then
        assert_eq!(parsed, None, "accepted invalid AUTOINCREMENT clause: {sql}");
    }
}
