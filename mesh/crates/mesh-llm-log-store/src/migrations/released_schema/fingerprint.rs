use rusqlite::Connection;

mod contract;
mod create_table;
mod predicate;
mod semantic_contract;

pub(super) fn has_private_objects(connection: &Connection) -> Result<bool, rusqlite::Error> {
    connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_schema WHERE name GLOB '_mesh_llm_log_store_*')",
        [],
        |row| row.get(0),
    )
}

pub(super) fn matches(connection: &Connection) -> Result<bool, rusqlite::Error> {
    Ok(has_exact_objects(connection)?
        && has_exact_tables(connection)?
        && has_exact_indexes(connection)?
        && has_exact_foreign_keys(connection)?
        && has_exact_create_table_semantics(connection)?
        && predicate::matches(connection)?)
}

fn has_exact_objects(connection: &Connection) -> Result<bool, rusqlite::Error> {
    let unexpected: bool = connection.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_schema
         WHERE type IN ('view', 'trigger') AND name NOT LIKE 'sqlite_%')",
        [],
        |row| row.get(0),
    )?;
    if unexpected {
        return Ok(false);
    }
    let tables = object_names(connection, "table")?;
    let indexes = object_names(connection, "index")?;
    Ok(
        names_match(&tables, contract::TABLES.iter().map(|table| table.name))
            && names_match(&indexes, contract::INDEXES.iter().map(|index| index.name)),
    )
}

fn object_names(
    connection: &Connection,
    object_type: &str,
) -> Result<Vec<String>, rusqlite::Error> {
    let mut statement = connection.prepare(
        "SELECT name FROM sqlite_schema
         WHERE type = ?1 AND name NOT LIKE 'sqlite_%' ORDER BY name",
    )?;
    statement
        .query_map([object_type], |row| row.get(0))?
        .collect()
}

fn names_match<'a>(actual: &[String], expected: impl Iterator<Item = &'a str>) -> bool {
    actual.iter().map(String::as_str).eq(expected)
}

fn has_exact_tables(connection: &Connection) -> Result<bool, rusqlite::Error> {
    for table in contract::TABLES {
        let shape = connection.query_row(
            "SELECT ncol, wr, strict FROM pragma_table_list
             WHERE schema = 'main' AND type = 'table' AND name = ?1",
            [table.name],
            |row| {
                Ok((
                    row.get::<_, usize>(0)?,
                    row.get::<_, bool>(1)?,
                    row.get::<_, bool>(2)?,
                ))
            },
        );
        let Ok(shape) = shape else {
            return Ok(false);
        };
        let columns = column_signature(connection, table.name)?;
        if shape.1
            || shape.2
            || columns.split('|').count() != shape.0
            || columns != table.columns
            || implicit_index_signature(connection, table.name)? != table.implicit_indexes
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn column_signature(connection: &Connection, table: &str) -> Result<String, rusqlite::Error> {
    connection.query_row(
        "SELECT COALESCE(group_concat(signature, '|'), '') FROM (
             SELECT printf('%s:%s:%d:%s:%d:%d', name, type, \"notnull\",
                           COALESCE(dflt_value, '-'), pk, hidden) AS signature
             FROM pragma_table_xinfo(?1) ORDER BY cid
         )",
        [table],
        |row| row.get(0),
    )
}

fn implicit_index_signature(
    connection: &Connection,
    table: &str,
) -> Result<String, rusqlite::Error> {
    let mut statement = connection.prepare(
        "SELECT name, \"unique\", origin, partial FROM pragma_index_list(?1)
         WHERE origin != 'c' ORDER BY name",
    )?;
    let indexes = statement
        .query_map([table], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, bool>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, bool>(3)?,
            ))
        })?
        .collect::<Result<Vec<_>, _>>()?;
    indexes
        .into_iter()
        .map(|(name, unique, origin, partial)| {
            Ok(format!(
                "{table}:{name}:{}:{origin}:{}:{}",
                u8::from(unique),
                u8::from(partial),
                index_column_signature(connection, &name)?
            ))
        })
        .collect::<Result<Vec<_>, rusqlite::Error>>()
        .map(|signatures| signatures.join("\n"))
}

fn has_exact_indexes(connection: &Connection) -> Result<bool, rusqlite::Error> {
    for index in contract::INDEXES {
        let metadata = connection.query_row(
            "SELECT \"unique\", origin, partial FROM pragma_index_list(?1) WHERE name = ?2",
            [index.table, index.name],
            |row| {
                Ok((
                    row.get::<_, bool>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, bool>(2)?,
                ))
            },
        );
        let Ok(metadata) = metadata else {
            return Ok(false);
        };
        if metadata != (index.unique, "c".to_owned(), index.partial)
            || index_column_signature(connection, index.name)? != index.columns
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn index_column_signature(connection: &Connection, index: &str) -> Result<String, rusqlite::Error> {
    connection.query_row(
        "SELECT COALESCE(group_concat(signature, '|'), '') FROM (
             SELECT printf('%d:%s:%d:%s', cid, COALESCE(name, '-'), \"desc\", coll) AS signature
             FROM pragma_index_xinfo(?1) WHERE key = 1 ORDER BY seqno
         )",
        [index],
        |row| row.get(0),
    )
}

fn has_exact_foreign_keys(connection: &Connection) -> Result<bool, rusqlite::Error> {
    for table in contract::TABLES {
        if foreign_key_signature(connection, table.name)? != table.foreign_keys {
            return Ok(false);
        }
    }
    Ok(true)
}

fn foreign_key_signature(connection: &Connection, table: &str) -> Result<String, rusqlite::Error> {
    connection.query_row(
        "SELECT COALESCE(group_concat(signature, '|'), '') FROM (
             SELECT printf('%s:%s:%s:%s:%s:%s', \"table\", \"from\", \"to\",
                           on_update, on_delete, match) AS signature
             FROM pragma_foreign_key_list(?1) ORDER BY id, seq
         )",
        [table],
        |row| row.get(0),
    )
}

fn has_exact_create_table_semantics(connection: &Connection) -> Result<bool, rusqlite::Error> {
    for expected in semantic_contract::TABLES {
        let sql = connection.query_row(
            "SELECT sql FROM sqlite_schema WHERE type = 'table' AND name = ?1",
            [expected.name],
            |row| row.get::<_, String>(0),
        )?;
        let Some(actual) = create_table::parse(&sql) else {
            return Ok(false);
        };
        if !expected.matches(&actual) {
            return Ok(false);
        }
    }
    Ok(true)
}
