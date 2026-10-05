//! Explicit schema ownership, including adoption of pre-versioned PoC ledgers.
use anyhow::{Result, ensure};
use rusqlite::Connection;

/// 1: first versioned schema. 2: `serving_accounting.forgiven`, set by the
/// operator `unblock` command for delivered output that will not be invoiced.
const VERSION: u32 = 2;

pub(super) fn initialize(connection: &mut Connection) -> Result<()> {
    let version: u32 = connection.pragma_query_value(None, "user_version", |row| row.get(0))?;
    ensure!(
        version <= VERSION,
        "payment ledger schema is newer than this binary"
    );
    // Connection settings must precede the schema transaction (WAL cannot be
    // enabled inside one). Never lower a newer schema version.
    connection.execute_batch(
        "PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;",
    )?;
    if version < VERSION {
        let transaction = connection.transaction()?;
        if version == 0 {
            transaction.execute_batch(include_str!("schema.sql"))?;
        }
        // Version 2. Pre-versioned and version-1 ledgers created the table
        // without this column; `CREATE TABLE IF NOT EXISTS` does not add it.
        ensure_column(
            &transaction,
            "serving_accounting",
            "forgiven",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        transaction.pragma_update(None, "user_version", VERSION)?;
        transaction.commit()?;
    }
    Ok(())
}

fn ensure_column(
    connection: &Connection,
    table: &str,
    column: &str,
    definition: &str,
) -> Result<()> {
    let columns: Vec<String> = connection
        .prepare(&format!("PRAGMA table_info({table})"))?
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<rusqlite::Result<_>>()?;
    if !columns.iter().any(|name| name == column) {
        connection.execute_batch(&format!(
            "ALTER TABLE {table} ADD COLUMN {column} {definition}"
        ))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn has_forgiven_column(connection: &Connection) -> bool {
        connection
            .prepare("PRAGMA table_info(serving_accounting)")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .any(|name| name.unwrap() == "forgiven")
    }

    #[test]
    fn adopts_legacy_data_and_rejects_future_schema_without_downgrade() -> Result<()> {
        let mut connection = Connection::open_in_memory()?;
        connection.execute_batch(include_str!("schema.sql"))?;
        connection.execute("INSERT INTO settings VALUES ('sentinel','preserved')", [])?;
        initialize(&mut connection)?;
        initialize(&mut connection)?;
        let value: String = connection.query_row(
            "SELECT value FROM settings WHERE key='sentinel'",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(value, "preserved");
        connection.pragma_update(None, "user_version", VERSION + 1)?;
        assert!(initialize(&mut connection).is_err());
        let version: u32 = connection.pragma_query_value(None, "user_version", |row| row.get(0))?;
        assert_eq!(version, VERSION + 1);
        Ok(())
    }

    #[test]
    fn version_one_and_pre_versioned_ledgers_gain_the_forgiven_column() -> Result<()> {
        for legacy_version in [0, 1] {
            let mut connection = Connection::open_in_memory()?;
            connection.execute_batch(include_str!("schema.sql"))?;
            connection.execute_batch("ALTER TABLE serving_accounting DROP COLUMN forgiven")?;
            connection.execute(
                "INSERT INTO serving_requests(id,peer) VALUES ('kept','peer')",
                [],
            )?;
            connection.execute(
                "INSERT INTO serving_accounting(id,pricing,max_output,tokens,finished) VALUES ('kept','{}',10,3,1)",
                [],
            )?;
            connection.pragma_update(None, "user_version", legacy_version)?;
            assert!(!has_forgiven_column(&connection));
            initialize(&mut connection)?;
            assert!(has_forgiven_column(&connection));
            let (forgiven, tokens): (u32, i64) = connection.query_row(
                "SELECT forgiven,tokens FROM serving_accounting WHERE id='kept'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?;
            assert_eq!((forgiven, tokens), (0, 3));
            let version: u32 =
                connection.pragma_query_value(None, "user_version", |row| row.get(0))?;
            assert_eq!(version, VERSION);
        }
        Ok(())
    }
}
