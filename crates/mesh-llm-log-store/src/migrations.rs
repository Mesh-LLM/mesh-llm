//! Initial schema installation and future forward migrations for log_store.

use rusqlite::{Connection, Transaction, TransactionBehavior};

mod lineage;
mod released_schema;

type MigrationFn = fn(&Connection) -> Result<(), rusqlite::Error>;

#[derive(Clone, Copy)]
struct Migration {
    version: u32,
    apply: MigrationFn,
}

struct MigrationPlan<'a> {
    target: u32,
    initialize: MigrationFn,
    migrations: &'a [Migration],
}

const MIGRATIONS: &[Migration] = &[];
const APPLICATION_ID: u32 = 0x4D4C4F47;

/// Current schema version for the integrated local logging feature.
pub const CURRENT_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SchemaClassification {
    Fresh,
    Private { version: u32 },
    ReleasedSchema { source_version: u32 },
    Incompatible { found: u32 },
}

pub fn apply_migrations(connection: &Connection) -> Result<(), rusqlite::Error> {
    run_migrations(
        connection,
        MigrationPlan {
            target: CURRENT_VERSION,
            initialize: crate::schema::initialize,
            migrations: MIGRATIONS,
        },
    )
}

pub(crate) fn incompatible_schema(
    connection: &Connection,
) -> Result<Option<(u32, u32)>, rusqlite::Error> {
    incompatible_schema_for_target(connection, CURRENT_VERSION)
}

fn incompatible_schema_for_target(
    connection: &Connection,
    target: u32,
) -> Result<Option<(u32, u32)>, rusqlite::Error> {
    let classification = classify_schema(connection)?;
    let compatible = match classification {
        SchemaClassification::Fresh => true,
        SchemaClassification::Private { version } => version <= target,
        SchemaClassification::ReleasedSchema { .. } => target == CURRENT_VERSION,
        SchemaClassification::Incompatible { .. } => false,
    };
    Ok((!compatible).then_some((schema_version(connection)?, target)))
}

fn classify_schema(connection: &Connection) -> Result<SchemaClassification, rusqlite::Error> {
    if let Some(source_version) = released_schema::source_version(connection)? {
        return Ok(SchemaClassification::ReleasedSchema { source_version });
    }
    let found = schema_version(connection)?;
    let identity = application_id(connection)?;
    if found == 0 && identity == 0 && !has_user_objects(connection)? {
        return Ok(SchemaClassification::Fresh);
    }
    if found > 0 && identity == APPLICATION_ID && lineage::is_valid(connection)? {
        return Ok(SchemaClassification::Private { version: found });
    }
    Ok(SchemaClassification::Incompatible { found })
}

fn run_migrations(connection: &Connection, plan: MigrationPlan<'_>) -> Result<(), rusqlite::Error> {
    validate_registry(plan.target, plan.migrations)?;
    let classification = classify_schema(connection)?;
    let mut current = match classification {
        SchemaClassification::Fresh => 0,
        SchemaClassification::Private { version } if version <= plan.target => version,
        SchemaClassification::ReleasedSchema { source_version }
            if plan.target == CURRENT_VERSION =>
        {
            released_schema::import(connection, source_version)?;
            CURRENT_VERSION
        }
        SchemaClassification::Private { .. }
        | SchemaClassification::ReleasedSchema { .. }
        | SchemaClassification::Incompatible { .. } => return Err(rusqlite::Error::InvalidQuery),
    };
    if matches!(classification, SchemaClassification::Fresh) {
        let transaction = immediate_transaction(connection)?;
        if schema_version(&transaction)? != 0
            || application_id(&transaction)? != 0
            || has_user_objects(&transaction)?
        {
            return Err(rusqlite::Error::InvalidQuery);
        }
        set_application_id(&transaction, APPLICATION_ID)?;
        (plan.initialize)(&transaction)?;
        lineage::install(&transaction)?;
        set_schema_version(&transaction, 1)?;
        transaction.commit()?;
        current = 1;
    }
    for migration in plan
        .migrations
        .iter()
        .filter(|migration| migration.version > current)
    {
        let transaction = immediate_transaction(connection)?;
        if schema_version(&transaction)? != migration.version - 1
            || application_id(&transaction)? != APPLICATION_ID
            || !lineage::is_valid(&transaction)?
        {
            return Err(rusqlite::Error::InvalidQuery);
        }
        (migration.apply)(&transaction)?;
        set_schema_version(&transaction, migration.version)?;
        transaction.commit()?;
    }
    Ok(())
}

fn validate_registry(target: u32, migrations: &[Migration]) -> Result<(), rusqlite::Error> {
    let terminal = target.checked_add(1).ok_or(rusqlite::Error::InvalidQuery)?;
    let mut expected = 2_u32;
    for migration in migrations {
        if migration.version != expected {
            return Err(rusqlite::Error::InvalidQuery);
        }
        expected = expected
            .checked_add(1)
            .ok_or(rusqlite::Error::InvalidQuery)?;
    }
    if expected != terminal {
        return Err(rusqlite::Error::InvalidQuery);
    }
    Ok(())
}

fn immediate_transaction(connection: &Connection) -> Result<Transaction<'_>, rusqlite::Error> {
    Transaction::new_unchecked(connection, TransactionBehavior::Immediate)
}

fn schema_version(connection: &Connection) -> Result<u32, rusqlite::Error> {
    let value: i32 = connection.pragma_query_value(None, "user_version", |row| row.get(0))?;
    Ok(u32::from_ne_bytes(value.to_ne_bytes()))
}

fn set_schema_version(connection: &Connection, version: u32) -> Result<(), rusqlite::Error> {
    connection.pragma_update(None, "user_version", version)
}

fn application_id(connection: &Connection) -> Result<u32, rusqlite::Error> {
    let value: i32 = connection.pragma_query_value(None, "application_id", |row| row.get(0))?;
    Ok(u32::from_ne_bytes(value.to_ne_bytes()))
}

fn set_application_id(connection: &Connection, application_id: u32) -> Result<(), rusqlite::Error> {
    connection.pragma_update(None, "application_id", application_id)
}

fn has_user_objects(connection: &Connection) -> Result<bool, rusqlite::Error> {
    connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_schema WHERE type IN ('table', 'index', 'view', 'trigger') AND name NOT LIKE 'sqlite_%')",
        [],
        |row| row.get(0),
    )
}

#[cfg(test)]
mod tests;
