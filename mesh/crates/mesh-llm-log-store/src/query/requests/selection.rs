use mesh_llm_events::logging::timestamp::canonical_logging_timestamp;
use rusqlite::types::Value;

use super::super::{QueryPage, RequestQuery};
use crate::LogStoreError;
use crate::cursor::encode_cursor;

pub(crate) struct Conditions {
    pub(crate) sql: String,
}

impl Conditions {
    pub(crate) fn append_to_sql(&self, sql: &mut String) {
        sql.push_str(&self.sql);
    }
}

pub(super) fn request_conditions(query: &RequestQuery) -> (Conditions, Vec<Value>) {
    let mut sql = String::new();
    let mut values = Vec::new();
    if let Some(from) = &query.from {
        sql.push_str(" WHERE created_at >= ?");
        values.push(Value::Text(normalize_timestamp(from)));
    } else {
        sql.push_str(" WHERE 1 = 1");
    }
    if let Some(to) = &query.to {
        sql.push_str(" AND created_at <= ?");
        values.push(Value::Text(normalize_timestamp(to)));
    }
    for (column, value) in [
        ("route", &query.route),
        ("model", &query.model),
        ("provider", &query.provider),
        ("engine", &query.engine),
    ] {
        if let Some(value) = value {
            sql.push_str(&format!(" AND {column} = ?"));
            values.push(Value::Text(value.clone()));
        }
    }
    if let Some(route) = &query.exclude_route {
        sql.push_str(" AND (route IS NULL OR route != ?)");
        values.push(Value::Text(route.clone()));
    }
    if let Some(prefix) = &query.exclude_route_prefix {
        sql.push_str(" AND (route IS NULL OR substr(route, 1, length(?)) != ?)");
        values.push(Value::Text(prefix.clone()));
        values.push(Value::Text(prefix.clone()));
    }
    if let Some(status_code) = query.status_code {
        sql.push_str(" AND status_code = ?");
        values.push(Value::Integer(i64::from(status_code)));
    }
    if let Some(outcome) = query.outcome {
        sql.push_str(" AND state = ?");
        values.push(Value::Text(outcome.as_str().to_string()));
    }
    (Conditions { sql }, values)
}

pub(crate) struct CursorPosition<'a> {
    pub(crate) table: &'static str,
    pub(crate) timestamp_column: &'static str,
    pub(crate) id_column: &'static str,
    pub(crate) timestamp: &'a str,
    pub(crate) id: &'a str,
    pub(crate) conditions: &'a Conditions,
    pub(crate) values: &'a [Value],
}

pub(crate) fn validate_cursor_position(
    connection: &rusqlite::Connection,
    position: CursorPosition<'_>,
) -> Result<(), LogStoreError> {
    let mut sql = format!(
        "SELECT 1 FROM {} WHERE {} = ? AND {} = ?",
        position.table, position.timestamp_column, position.id_column
    );
    let mut parameters = vec![
        Value::Text(position.timestamp.to_string()),
        Value::Text(position.id.to_string()),
    ];
    if position.conditions.sql.starts_with(" WHERE ") {
        sql.push_str(" AND ");
        sql.push_str(&position.conditions.sql[7..]);
    }
    parameters.extend(position.values.iter().cloned());
    match connection.query_row(&sql, rusqlite::params_from_iter(parameters.iter()), |_| {
        Ok(())
    }) {
        Ok(()) => Ok(()),
        Err(rusqlite::Error::QueryReturnedNoRows) => Err(LogStoreError::CursorInvalid),
        Err(error) => Err(LogStoreError::QueryFailed(error.to_string())),
    }
}

pub(crate) fn query_page<T>(
    connection: &rusqlite::Connection,
    sql: String,
    values: Vec<Value>,
    limit: usize,
    map: fn(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
    cursor_fields: impl Fn(&T) -> (&str, &str),
) -> Result<QueryPage<T>, LogStoreError> {
    let mut statement = connection.prepare(&sql).map_err(LogStoreError::Sqlite)?;
    let rows = statement
        .query_map(rusqlite::params_from_iter(values.iter()), map)
        .map_err(LogStoreError::Sqlite)?;
    let mut items = rows
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| LogStoreError::QueryFailed(error.to_string()))?;
    let has_more = items.len() > limit;
    items.truncate(limit);
    let next_cursor = if has_more {
        items.last().map(|item| {
            let (timestamp, id) = cursor_fields(item);
            encode_cursor(timestamp, id)
        })
    } else {
        None
    };
    Ok(QueryPage { items, next_cursor })
}

fn normalize_timestamp(value: &str) -> String {
    canonical_logging_timestamp(value).expect("RequestQuery::validate parses time bounds")
}
