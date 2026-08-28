use rusqlite::Connection;

#[test]
fn deleting_maximum_rowid_reuses_it_without_autoincrement_but_not_with_autoincrement() {
    // Given
    let connection = Connection::open_in_memory().expect("open database");
    connection
        .execute_batch(
            "CREATE TABLE reusable (sequence INTEGER PRIMARY KEY);
             CREATE TABLE monotonic (sequence INTEGER PRIMARY KEY AUTOINCREMENT);
             INSERT INTO reusable DEFAULT VALUES;
             INSERT INTO reusable DEFAULT VALUES;
             INSERT INTO reusable DEFAULT VALUES;
             INSERT INTO monotonic DEFAULT VALUES;
             INSERT INTO monotonic DEFAULT VALUES;
             INSERT INTO monotonic DEFAULT VALUES;
             DELETE FROM reusable WHERE sequence = 3;
             DELETE FROM monotonic WHERE sequence = 3;",
        )
        .expect("seed and delete maximum rowids");

    // When
    connection
        .execute_batch(
            "INSERT INTO reusable DEFAULT VALUES;
             INSERT INTO monotonic DEFAULT VALUES;",
        )
        .expect("insert replacement rows");

    // Then
    let reusable = connection
        .query_row("SELECT MAX(sequence) FROM reusable", [], |row| {
            row.get::<_, u64>(0)
        })
        .expect("read reusable sequence");
    let monotonic = connection
        .query_row("SELECT MAX(sequence) FROM monotonic", [], |row| {
            row.get::<_, u64>(0)
        })
        .expect("read monotonic sequence");
    assert_eq!((reusable, monotonic), (3, 4));
}
