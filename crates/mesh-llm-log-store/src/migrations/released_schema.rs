use rusqlite::Connection;

mod fingerprint;

const SOURCE_VERSIONS: [u32; 2] = [3, 11];

pub(super) fn source_version(connection: &Connection) -> Result<Option<u32>, rusqlite::Error> {
    let source_version = super::schema_version(connection)?;
    Ok(matches_source(connection, source_version)?.then_some(source_version))
}

fn matches_source(connection: &Connection, source_version: u32) -> Result<bool, rusqlite::Error> {
    Ok(SOURCE_VERSIONS.contains(&source_version)
        && super::schema_version(connection)? == source_version
        && super::application_id(connection)? == 0
        && !fingerprint::has_private_objects(connection)?
        && fingerprint::matches(connection)?)
}

pub(super) fn import(connection: &Connection, source_version: u32) -> Result<(), rusqlite::Error> {
    import_with_hook(connection, source_version, |_| Ok(()))
}

fn import_with_hook(
    connection: &Connection,
    source_version: u32,
    before_commit: impl FnOnce(&Connection) -> Result<(), rusqlite::Error>,
) -> Result<(), rusqlite::Error> {
    let transaction = super::immediate_transaction(connection)?;
    if !matches_source(&transaction, source_version)? {
        return Err(rusqlite::Error::InvalidQuery);
    }
    transaction.execute_batch(
        "ALTER TABLE summaries ADD COLUMN caller_endpoint_id TEXT;
         ALTER TABLE summaries ADD COLUMN caller_addr TEXT;
         ALTER TABLE summaries ADD COLUMN caller_path_type TEXT;",
    )?;
    super::lineage::install(&transaction)?;
    super::set_application_id(&transaction, super::APPLICATION_ID)?;
    super::set_schema_version(&transaction, super::CURRENT_VERSION)?;
    before_commit(&transaction)?;
    transaction.commit()
}

#[cfg(test)]
pub(super) fn import_with_before_commit_failure(
    connection: &Connection,
    source_version: u32,
) -> Result<(), rusqlite::Error> {
    import_with_hook(connection, source_version, |_| {
        Err(rusqlite::Error::InvalidQuery)
    })
}
