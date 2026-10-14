//! Historical SQL fixtures copied from main 8b848ead5, credit draft 1b9d88794,
//! estimate draft dd02d797d, and early estimate 95ddf720b. No wallet or network access.
use super::*;

const MAIN: &str = include_str!("fixtures/main_v2.sql");
const CREDIT: &str = include_str!("fixtures/credit_v3.sql");
const ESTIMATE: &str = include_str!("fixtures/estimate_v3.sql");

fn version(connection: &Connection) -> u32 {
    connection
        .pragma_query_value(None, "user_version", |r| r.get(0))
        .unwrap()
}

fn assert_canonical(connection: &Connection) -> Result<()> {
    let fresh = Connection::open_in_memory()?;
    fresh.execute_batch(include_str!("../schema.sql"))?;
    for table in ["metered_requests", "metered_invoices"] {
        assert_eq!(
            draft_schema::ddl(connection, table)?,
            draft_schema::ddl(&fresh, table)?,
            "{table}"
        );
    }
    Ok(())
}

fn admit(connection: &Connection) -> Result<()> {
    assert_canonical(connection)?;

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
        assert_canonical(&c)?;
        assert_eq!(version(&c), VERSION);
        assert_eq!(super::super::committed_today(&c, 86_400_000)?, 6000);
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
fn empty_estimate_draft_upgrades() -> Result<()> {
    let mut c = Connection::open_in_memory()?;
    c.execute_batch(ESTIMATE)?;
    c.pragma_update(None, "user_version", 3)?;
    initialize(&mut c)?;
    admit(&c)?;
    Ok(())
}

#[test]
fn empty_credit_draft_can_be_adopted() -> Result<()> {
    let mut c = Connection::open_in_memory()?;
    c.execute_batch(MAIN)?;
    c.execute_batch(CREDIT)?;
    c.pragma_update(None, "user_version", 3)?;
    initialize(&mut c)?;
    admit(&c)?;
    Ok(())
}

#[test]
fn populated_credit_draft_is_rejected_without_losing_money_or_version() -> Result<()> {
    let mut c = Connection::open_in_memory()?;
    c.execute_batch(MAIN)?;
    c.execute_batch(CREDIT)?;
    c.execute("INSERT INTO peer_credit VALUES ('peer',1500)", [])?;
    c.pragma_update(None, "user_version", 3)?;
    let error = initialize(&mut c).unwrap_err();
    assert!(error.to_string().contains("unsupported draft"), "{error:#}");
    assert_eq!(version(&c), 3);
    assert_eq!(
        c.query_row("SELECT msat FROM peer_credit", [], |r| r.get::<_, i64>(0))?,
        1500
    );
    assert!(tests::has_table(&c, "receivables"));
    Ok(())
}

#[test]
fn populated_draft_metered_state_is_never_reinterpreted() -> Result<()> {
    for old_version in [0, 3, 4] {
        for marker in ["seller", "buyer", "payee"] {
            let mut c = Connection::open_in_memory()?;
            c.execute_batch(include_str!("../schema.sql"))?;
            c.execute_batch(
                "INSERT INTO requests VALUES ('old','{}','approved',10000,0);
                INSERT INTO charges VALUES ('uncertain','old',1,'{}',1000,4000,'pending',0,NULL);",
            )?;
            match marker {
                "seller" => c.execute_batch("INSERT INTO metered_requests(id,peer,pricing,require_deposit,deposit_msat,state) VALUES ('old','peer','{}',0,0,'billed');
                    INSERT INTO metered_invoices VALUES ('debt','old','peer','final','{}','unpaid');")?,
                "buyer" => { c.execute("INSERT INTO settings VALUES ('metered-payer/old','{}')", [])?; }
                _ => { c.execute("INSERT INTO settings VALUES ('metered-payee/old','pin')", [])?; }
            }
            c.pragma_update(None, "user_version", old_version)?;
            let before = snapshot(&c)?;
            let error = initialize(&mut c).unwrap_err();
            assert!(error.to_string().contains("unsupported draft"), "{error:#}");
            assert_eq!(version(&c), old_version);
            assert_eq!(snapshot(&c)?, before);
            assert_eq!(super::super::committed_today(&c, 86_400_000)?, 4000);
        }
    }
    Ok(())
}

#[test]
fn empty_canonical_drafts_upgrade() -> Result<()> {
    for old_version in [3, 4] {
        let mut c = Connection::open_in_memory()?;
        c.execute_batch(include_str!("../schema.sql"))?;
        c.pragma_update(None, "user_version", old_version)?;
        initialize(&mut c)?;
        assert_eq!(version(&c), VERSION);
        admit(&c)?;
    }
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

#[test]
fn populated_partial_drafts_are_not_dropped() -> Result<()> {
    for schema in [
        "CREATE TABLE metered_requests(id TEXT PRIMARY KEY, old_debt INTEGER); INSERT INTO metered_requests VALUES ('old',1000);",
        "CREATE TABLE metered_invoices(hash TEXT PRIMARY KEY, invoice TEXT); INSERT INTO metered_invoices VALUES ('old','receipt');",
    ] {
        let mut c = Connection::open_in_memory()?;
        c.execute_batch(MAIN)?;
        c.execute_batch(schema)?;
        c.pragma_update(None, "user_version", 3)?;
        let before: String =
            c.query_row("SELECT group_concat(sql) FROM sqlite_master", [], |r| {
                r.get(0)
            })?;
        assert!(
            initialize(&mut c)
                .unwrap_err()
                .to_string()
                .contains("unsupported draft")
        );
        let after: String =
            c.query_row("SELECT group_concat(sql) FROM sqlite_master", [], |r| {
                r.get(0)
            })?;
        assert_eq!(before, after);
        assert_eq!(version(&c), 3);
    }
    Ok(())
}

#[test]
fn early_constraints_and_dangling_foreign_keys_are_not_adopted_as_canonical() -> Result<()> {
    for dangling in [false, true] {
        for populated in [false, true] {
            let mut c = Connection::open_in_memory()?;
            if dangling {
                c.execute_batch(&ESTIMATE.replace(
                    "id TEXT PRIMARY KEY, peer TEXT NOT NULL",
                    "id TEXT PRIMARY KEY REFERENCES serving_requests(id), peer TEXT NOT NULL",
                ))?;
            } else {
                c.execute_batch(include_str!("fixtures/early_estimate.sql"))?;
            }
            c.execute_batch("PRAGMA foreign_keys=OFF;")?;
            if populated {
                c.execute_batch("INSERT INTO metered_requests(id,peer,pricing,estimated_input_tokens,require_deposit,deposit_msat,state) VALUES ('old','peer','{}',1,0,0,'admitted');")?;
            }
            let before = draft_schema::ddl(&c, "metered_requests")?;
            let result = initialize(&mut c);
            if populated {
                assert!(result.is_err());
                assert_eq!(version(&c), 0);
                assert_eq!(before, draft_schema::ddl(&c, "metered_requests")?);
            } else {
                result?;
                admit(&c)?;
                c.execute("UPDATE metered_invoices SET state='forgiven'", [])?;
            }
        }
    }
    Ok(())
}

fn snapshot(c: &Connection) -> Result<Vec<Vec<rusqlite::types::Value>>> {
    let mut result = Vec::new();
    for table in [
        "sqlite_master",
        "settings",
        "requests",
        "charges",
        "metered_requests",
        "metered_invoices",
    ] {
        let mut statement = c.prepare(&format!("SELECT * FROM {table} ORDER BY rowid"))?;
        let columns = statement.column_count();
        for row in statement.query_map([], |row| {
            (0..columns)
                .map(|i| row.get(i))
                .collect::<rusqlite::Result<Vec<_>>>()
        })? {
            result.push(row?);
        }
    }
    Ok(result)
}
