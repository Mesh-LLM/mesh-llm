use rusqlite::Connection;
use rusqlite::types::ValueRef;

const TABLE_NAME: &str = "_mesh_llm_log_store_lineage";

pub(super) fn install(connection: &Connection) -> Result<(), rusqlite::Error> {
    connection.execute_batch(
        "CREATE TABLE _mesh_llm_log_store_lineage (
            id INTEGER NOT NULL PRIMARY KEY CHECK (id = 1),
            epoch INTEGER NOT NULL CHECK (epoch = 1)
        ) WITHOUT ROWID;
        INSERT INTO _mesh_llm_log_store_lineage (id, epoch) VALUES (1, 1);",
    )
}

pub(super) fn is_valid(connection: &Connection) -> Result<bool, rusqlite::Error> {
    if !has_exact_private_object(connection)?
        || !has_expected_table_shape(connection)?
        || !has_expected_columns(connection)?
    {
        return Ok(false);
    }
    has_expected_row(connection)
}

fn has_exact_private_object(connection: &Connection) -> Result<bool, rusqlite::Error> {
    connection.query_row(
        "SELECT COUNT(*) = 1
             AND COUNT(*) FILTER (
                 WHERE type = 'table' AND name = ?1 AND tbl_name = ?1 AND sql IS NOT NULL
             ) = 1
         FROM sqlite_schema
         WHERE name GLOB '_mesh_llm_log_store_*'",
        [TABLE_NAME],
        |row| row.get(0),
    )
}

fn has_expected_table_shape(connection: &Connection) -> Result<bool, rusqlite::Error> {
    connection.query_row(
        "SELECT COUNT(*) = 1
         FROM pragma_table_list
         WHERE schema = 'main' AND name = ?1 AND type = 'table'
           AND ncol = 2 AND wr = 1 AND strict = 0",
        [TABLE_NAME],
        |row| row.get(0),
    )
}

fn has_expected_columns(connection: &Connection) -> Result<bool, rusqlite::Error> {
    connection.query_row(
        "SELECT COUNT(*) = 2
             AND COUNT(*) FILTER (
                 WHERE cid = 0 AND name = 'id' AND type = 'INTEGER'
                   AND \"notnull\" = 1 AND dflt_value IS NULL AND pk = 1 AND hidden = 0
             ) = 1
             AND COUNT(*) FILTER (
                 WHERE cid = 1 AND name = 'epoch' AND type = 'INTEGER'
                   AND \"notnull\" = 1 AND dflt_value IS NULL AND pk = 0 AND hidden = 0
             ) = 1
         FROM pragma_table_xinfo(?1)",
        [TABLE_NAME],
        |row| row.get(0),
    )
}

fn has_expected_row(connection: &Connection) -> Result<bool, rusqlite::Error> {
    let mut statement = connection.prepare("SELECT * FROM _mesh_llm_log_store_lineage")?;
    if statement.column_count() != 2 {
        return Ok(false);
    }
    let mut rows = statement.query([])?;
    let Some(row) = rows.next()? else {
        return Ok(false);
    };
    let row_is_valid = matches!(
        (row.get_ref(0)?, row.get_ref(1)?),
        (ValueRef::Integer(1), ValueRef::Integer(1))
    );
    Ok(row_is_valid && rows.next()?.is_none())
}
