//! Explicit schema ownership, including adoption of pre-versioned PoC ledgers.
use anyhow::{Result, ensure};
use rusqlite::Connection;

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
    if version == 0 {
        let transaction = connection.transaction()?;
        transaction.execute_batch(include_str!("schema.sql"))?;
        transaction.pragma_update(None, "user_version", 1)?;
        transaction.commit()?;
    }
    if version < 2 {
        let transaction = connection.transaction()?;
        transaction.execute_batch("CREATE TABLE IF NOT EXISTS serving_terms(id TEXT PRIMARY KEY REFERENCES serving_requests(id), terms TEXT NOT NULL);")?;
        transaction.pragma_update(None, "user_version", VERSION)?;
        transaction.commit()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
