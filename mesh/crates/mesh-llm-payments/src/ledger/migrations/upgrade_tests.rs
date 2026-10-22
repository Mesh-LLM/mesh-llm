//! Released schema fixture from main 8b848ead5. No wallet or network access.
use super::*;

const MAIN: &str = include_str!("fixtures/main_v2.sql");

fn version(connection: &Connection) -> u32 {
    connection
        .pragma_query_value(None, "user_version", |r| r.get(0))
        .unwrap()
}

fn admit(connection: &Connection) -> Result<()> {
    connection.execute("INSERT INTO metered_requests(id,peer,pricing,require_deposit,deposit_msat,state) VALUES ('new','peer','{}',1,1000,'admitted')", [])?;
    connection.execute(
        "INSERT INTO metered_invoices VALUES ('hash','new','peer','deposit','{}','unpaid')",
        [],
    )?;
    Ok(())
}

#[test]
fn actual_main_schema_preserves_buyer_records_and_policy() -> Result<()> {
    for old_version in [0, 1, 2] {
        let mut c = Connection::open_in_memory()?;
        c.execute_batch(MAIN)?;
        c.execute_batch("INSERT INTO settings VALUES ('policy','{\"mode\":\"automatic\",\"daily_budget_msat\":20000}');
            INSERT INTO settings VALUES ('wallet-pin','preserved');
            INSERT INTO requests VALUES ('old','{}','approved',10000,2000);
            INSERT INTO charges VALUES ('paid','old',0,'{}',1990,3000,'succeeded',2000,1);
            INSERT INTO charges VALUES ('uncertain','old',1,'{}',1000,4000,'pending',0,NULL);
            INSERT INTO serving_requests VALUES ('seller','peer');
            INSERT INTO receivables VALUES ('debt','seller','peer',0,'{}',100,'unpaid');")?;
        c.pragma_update(None, "user_version", old_version)?;
        initialize(&mut c)?;
        initialize(&mut c)?;
        assert_eq!(version(&c), VERSION);
        assert_eq!(super::super::settled_today(&c, 86_400_000)?, 2000);
        assert_eq!(
            c.query_row(
                "SELECT value FROM settings WHERE key='wallet-pin'",
                [],
                |r| r.get::<_, String>(0)
            )?,
            "preserved"
        );
        assert_eq!(
            c.query_row("SELECT COUNT(*) FROM charges", [], |r| r.get::<_, i64>(0))?,
            2
        );
        assert!(!tests::has_table(&c, "receivables"));
        admit(&c)?;
    }
    Ok(())
}

#[test]
fn unsupported_drafts_are_rejected_without_changing_the_file() -> Result<()> {
    for old_version in [0, 1, 2, 3, 4] {
        for marker in [
            "CREATE TABLE metered_requests(id TEXT); INSERT INTO metered_requests VALUES ('debt');",
            "CREATE TABLE metered_invoices(id TEXT); INSERT INTO metered_invoices VALUES ('receipt');",
            "CREATE TABLE peer_credit(msat INTEGER); INSERT INTO peer_credit VALUES (1500);",
            "INSERT INTO settings VALUES ('metered-payer/old','authorization');",
            "INSERT INTO settings VALUES ('metered-payee/old','pin');",
        ] {
            let dir = tempfile::tempdir()?;
            let path = dir.path().join("payments.sqlite3");
            {
                let c = Connection::open(&path)?;
                c.execute_batch(MAIN)?;
                c.execute_batch(marker)?;
                c.pragma_update(None, "user_version", old_version)?;
            }
            let before = std::fs::read(&path)?;
            let mut c = Connection::open(&path)?;
            let error = initialize(&mut c).unwrap_err();
            assert!(error.to_string().contains("unsupported draft"), "{error:#}");
            assert_eq!(version(&c), old_version);
            drop(c);
            assert_eq!(std::fs::read(&path)?, before);
        }
    }
    Ok(())
}

#[test]
fn empty_versioned_drafts_are_not_adopted() -> Result<()> {
    for old_version in [3, 4] {
        let mut c = Connection::open_in_memory()?;
        c.pragma_update(None, "user_version", old_version)?;
        assert!(initialize(&mut c).is_err());
        assert_eq!(version(&c), old_version);
        assert!(!tests::has_table(&c, "settings"));
    }
    Ok(())
}

#[test]
fn current_metered_records_survive_reopen() -> Result<()> {
    let mut c = Connection::open_in_memory()?;
    initialize(&mut c)?;
    admit(&c)?;
    c.execute(
        "INSERT INTO settings VALUES ('metered-payer/new','authorization')",
        [],
    )?;
    initialize(&mut c)?;
    assert_eq!(version(&c), VERSION);
    let count: u32 = c.query_row("SELECT COUNT(*) FROM metered_invoices", [], |r| r.get(0))?;
    assert_eq!(count, 1);
    let value: String = c.query_row(
        "SELECT value FROM settings WHERE key='metered-payer/new'",
        [],
        |r| r.get(0),
    )?;
    assert_eq!(value, "authorization");
    Ok(())
}

/// `metered_requests` as schema 5 created it, before `model`.
const V5_METERED_REQUESTS: &str = "CREATE TABLE metered_requests(
    id TEXT PRIMARY KEY, peer TEXT NOT NULL,
    pricing TEXT NOT NULL,
    require_deposit INTEGER NOT NULL, deposit_msat INTEGER NOT NULL,
    dispatched INTEGER NOT NULL DEFAULT 0,
    state TEXT NOT NULL, outcome TEXT, bill_msat INTEGER,
    deposit_paid_msat INTEGER NOT NULL DEFAULT 0, remaining_msat INTEGER NOT NULL DEFAULT 0
);";

#[test]
fn schema_5_metered_records_gain_an_empty_model() -> Result<()> {
    let mut c = Connection::open_in_memory()?;
    // The old table first; the rest of the schema is unchanged since 5.
    c.execute_batch(V5_METERED_REQUESTS)?;
    c.execute_batch(include_str!("../schema.sql"))?;
    admit(&c)?;
    c.pragma_update(None, "user_version", 5)?;
    initialize(&mut c)?;
    initialize(&mut c)?;
    assert_eq!(version(&c), VERSION);
    let (model, deposit): (String, i64) = c.query_row(
        "SELECT model,deposit_msat FROM metered_requests WHERE id='new'",
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    )?;
    assert_eq!((model.as_str(), deposit), ("", 1000));
    assert!(tests::has_table(&c, "model_deposits"));
    let count: u32 = c.query_row("SELECT COUNT(*) FROM metered_invoices", [], |r| r.get(0))?;
    assert_eq!(count, 1);
    Ok(())
}

#[test]
fn corrupt_file_is_not_replaced() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("payments.sqlite3");
    let bytes = b"not a SQLite database";
    std::fs::write(&path, bytes)?;
    assert!(super::super::Ledger::open(dir.path()).is_err());
    assert_eq!(std::fs::read(path)?, bytes);
    Ok(())
}
