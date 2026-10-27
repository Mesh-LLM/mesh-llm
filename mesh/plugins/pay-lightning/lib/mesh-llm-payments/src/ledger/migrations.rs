//! Explicit schema ownership, including adoption of pre-versioned PoC ledgers.
use anyhow::{Result, ensure};
use rusqlite::Connection;

/// 1: first versioned schema. 2: `serving_accounting.forgiven`.
/// 3/4: unsupported, unreleased development schemas.
/// 5: token-metered billing with cached-input discounts and the remainder floor.
/// Released buyer records are retained; retired seller accounting is dropped.
/// 6: `metered_requests.model`, so a reused request ID cannot change model;
/// `metered_requests.min_payment_msat`, the minimum recorded at admission; and
/// per-model admission deposits in `model_deposits`.
const VERSION: u32 = 6;

/// First schema with the metered tables; later versions extend it in place.
const METERED_VERSION: u32 = 5;

/// Requests admitted under schema 5 carry no model. An empty model matches no
/// replay, which is safe: a request still admitted at upgrade is abandoned at
/// startup, and a replay of a finished request never reaches the identity check.
/// Their minimum payment is taken as the 1-sat default, which nothing reads
/// again for a request that is already finalized or abandoned.
const V5_TO_V6: &str = "ALTER TABLE metered_requests ADD COLUMN model TEXT NOT NULL DEFAULT '';
     ALTER TABLE metered_requests ADD COLUMN min_payment_msat INTEGER NOT NULL DEFAULT 1000;";

/// Seller-side tables of the retired fixed-allowance protocol. Their debt was
/// recorded under a protocol no peer can settle any more, so it is dropped
/// rather than carried into the strike rule.
const LEGACY_TABLES: &str = "DROP TABLE IF EXISTS receivables; DROP TABLE IF EXISTS serving_accounting; DROP TABLE IF EXISTS serving_requests;";

pub(super) fn initialize(connection: &mut Connection) -> Result<()> {
    let version: u32 = connection.pragma_query_value(None, "user_version", |row| row.get(0))?;
    ensure!(
        version <= VERSION,
        "payment ledger schema is newer than this binary"
    );
    if version < METERED_VERSION {
        reject_draft(connection, version)?;
    }
    // Connection settings must precede the schema transaction (WAL cannot be
    // enabled inside one). Never lower a newer schema version.
    connection.execute_batch(
        "PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA foreign_keys=ON;",
    )?;
    if version < VERSION {
        let transaction = connection.transaction()?;
        if version == METERED_VERSION {
            transaction.execute_batch(V5_TO_V6)?;
        }
        transaction.execute_batch(include_str!("schema.sql"))?;
        transaction.execute_batch(LEGACY_TABLES)?;
        transaction.pragma_update(None, "user_version", VERSION)?;
        transaction.commit()?;
    }
    Ok(())
}

// Development schemas have no supported conversion. Check before changing even
// connection settings, and leave all accounting intact for explicit recovery.
fn reject_draft(connection: &Connection, version: u32) -> Result<()> {
    let draft_tables: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name IN ('metered_requests','metered_invoices','peer_credit'))",
        [],
        |row| row.get(0),
    )?;
    let settings: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='settings')",
        [],
        |row| row.get(0),
    )?;
    let buyer_draft = settings && connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM settings WHERE key GLOB 'metered-payer/*' OR key GLOB 'metered-payee/*')",
        [],
        |row| row.get::<_, bool>(0),
    )?;
    ensure!(
        version < 3 && !draft_tables && !buyer_draft,
        "unsupported draft payment ledger; data retained, back up payments.sqlite3 and use a compatible binary or explicit migration"
    );
    Ok(())
}

#[cfg(test)]
mod upgrade_tests;

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn has_table(connection: &Connection, table: &str) -> bool {
        connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?)",
                [table],
                |row| row.get(0),
            )
            .unwrap()
    }

    #[test]
    fn adopts_legacy_data_and_rejects_future_schema_without_downgrade() -> Result<()> {
        let mut connection = Connection::open_in_memory()?;
        connection.execute_batch(include_str!("migrations/fixtures/main_v2.sql"))?;
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
    fn older_ledgers_gain_metered_tables_and_drop_fixed_allowance_accounting() -> Result<()> {
        for legacy_version in [0, 1, 2] {
            let mut connection = Connection::open_in_memory()?;
            connection.execute_batch(
                "CREATE TABLE settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                 CREATE TABLE pricing(model TEXT PRIMARY KEY,value TEXT NOT NULL);
                 CREATE TABLE requests(id TEXT PRIMARY KEY, terms TEXT NOT NULL, state TEXT NOT NULL, cap INTEGER NOT NULL CHECK(cap>0), spent INTEGER NOT NULL CHECK(spent>=0));
                 CREATE TABLE charges(hash TEXT PRIMARY KEY, request_id TEXT NOT NULL REFERENCES requests(id), segment INTEGER NOT NULL, invoice TEXT NOT NULL, amount INTEGER NOT NULL, max_total INTEGER NOT NULL, state TEXT NOT NULL, total INTEGER NOT NULL, settled_day INTEGER, UNIQUE(request_id,segment));
                 CREATE TABLE receivables(hash TEXT PRIMARY KEY, request_id TEXT NOT NULL, peer TEXT NOT NULL, segment INTEGER NOT NULL, invoice TEXT NOT NULL, tokens INTEGER NOT NULL, state TEXT NOT NULL, UNIQUE(request_id,segment));
                 CREATE TABLE serving_requests(id TEXT PRIMARY KEY,peer TEXT NOT NULL);
                 CREATE TABLE serving_accounting(id TEXT PRIMARY KEY REFERENCES serving_requests(id), pricing TEXT NOT NULL, max_output INTEGER NOT NULL, tokens INTEGER NOT NULL DEFAULT 0, finished INTEGER NOT NULL DEFAULT 0);
                 INSERT INTO pricing VALUES ('m','{}');
                 INSERT INTO serving_requests(id,peer) VALUES ('kept','peer');",
            )?;
            connection.pragma_update(None, "user_version", legacy_version)?;
            assert!(!has_table(&connection, "metered_requests"));
            initialize(&mut connection)?;
            assert!(has_table(&connection, "metered_requests"));
            assert!(has_table(&connection, "metered_invoices"));
            for legacy in ["receivables", "serving_requests", "serving_accounting"] {
                assert!(!has_table(&connection, legacy), "{legacy}");
            }
            let price: String =
                connection.query_row("SELECT value FROM pricing WHERE model='m'", [], |row| {
                    row.get(0)
                })?;
            assert_eq!(price, "{}");
            let version: u32 =
                connection.pragma_query_value(None, "user_version", |row| row.get(0))?;
            assert_eq!(version, VERSION);
        }
        Ok(())
    }
}
