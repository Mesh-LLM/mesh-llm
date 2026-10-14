//! Version 3 was used by several unreleased, incompatible seller drafts.
//! Never discard populated draft accounting merely because it was unreleased.
use anyhow::{Result, ensure};
use rusqlite::Connection;

pub(super) fn adopt(connection: &Connection) -> Result<()> {
    let requests = columns(connection, "metered_requests")?;
    let invoices = columns(connection, "metered_invoices")?;
    let credit = columns(connection, "peer_credit")?;
    // Table shape does not identify the billing semantics of unreleased drafts.
    // Check before any DDL or startup recovery can reinterpret financial state.
    ensure_empty_if_present(connection, "metered_requests", &requests)?;
    ensure_empty_if_present(connection, "metered_invoices", &invoices)?;
    if !columns(connection, "settings")?.is_empty() {
        let buyer_draft: bool = connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM settings WHERE key GLOB 'metered-payer/*' OR key GLOB 'metered-payee/*')",
            [],
            |row| row.get(0),
        )?;
        ensure!(
            !buyer_draft,
            "unsupported draft payment ledger: metered buyer records; data retained, back up payments.sqlite3 and use a compatible binary or explicit migration"
        );
    }
    // DDL text is pinned: cosmetic schema changes need a versioned migration too.
    let canonical = Connection::open_in_memory()?;
    canonical.execute_batch(include_str!("../schema.sql"))?;
    let estimate = Connection::open_in_memory()?;
    estimate.execute_batch(include_str!("fixtures/estimate_v3.sql"))?;
    let has_estimate = ddl(connection, "metered_requests")? == ddl(&estimate, "metered_requests")?;
    let compatible = (has_estimate
        || ddl(connection, "metered_requests")? == ddl(&canonical, "metered_requests")?)
        && ddl(connection, "metered_invoices")? == ddl(&canonical, "metered_invoices")?;

    // Even an otherwise canonical schema must not hide old peer credit.
    ensure_empty_if_present(connection, "peer_credit", &credit)?;
    if compatible {
        if has_estimate {
            // This draft differs only by an obsolete admission estimate. DROP
            // COLUMN preserves receipts, debt, and all other request fields.
            connection.execute_batch(
                "ALTER TABLE metered_requests DROP COLUMN estimated_input_tokens;",
            )?;
        }
    } else {
        // Empty drafts can be recreated. Anything populated and unrecognized
        // needs an explicit conversion, not a financial-state reset.
        ensure_empty_if_present(connection, "metered_requests", &requests)?;
        ensure_empty_if_present(connection, "metered_invoices", &invoices)?;
        connection.execute_batch(
            "DROP TABLE IF EXISTS metered_invoices; DROP TABLE IF EXISTS metered_requests;",
        )?;
    }
    connection.execute_batch("DROP TABLE IF EXISTS peer_credit;")?;
    Ok(())
}

fn columns(connection: &Connection, table: &str) -> Result<Vec<String>> {
    let mut statement =
        connection.prepare("SELECT name FROM pragma_table_info(?1) ORDER BY name")?;
    Ok(statement
        .query_map([table], |row| row.get(0))?
        .collect::<rusqlite::Result<_>>()?)
}

fn ensure_empty_if_present(connection: &Connection, table: &str, columns: &[String]) -> Result<()> {
    if !columns.is_empty() {
        // Only the fixed internal table names above reach this function.
        let populated: bool = connection.query_row(
            &format!("SELECT EXISTS(SELECT 1 FROM {table})"),
            [],
            |row| row.get(0),
        )?;
        ensure!(
            !populated,
            "unsupported draft payment ledger: populated {table}; data retained, back up payments.sqlite3 and use a compatible binary or explicit migration"
        );
    }
    Ok(())
}

pub(super) fn ddl(connection: &Connection, table: &str) -> Result<Option<String>> {
    use rusqlite::OptionalExtension;
    Ok(connection
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?1",
            [table],
            |row| row.get::<_, String>(0),
        )
        .optional()?
        .map(|sql| sql.split_whitespace().collect::<Vec<_>>().join(" ")))
}
