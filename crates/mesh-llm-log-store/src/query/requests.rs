use rusqlite::types::Value;

use mesh_llm_events::logging::timestamp::canonical_logging_timestamp;

use super::{QueryPage, RequestQuery, RequestRecord};
use crate::cursor::{decode_ordering_cursor, encode_cursor};
use crate::{LogStore, LogStoreError};

const REQUEST_COLUMNS: &str =
    "request_id, state, created_at, terminal_at, route, model, provider, engine, status_code";
const REQUEST_ID_BATCH_SIZE: usize = 100;

impl LogStore {
    /// Return one durable request summary without exposing internal identity or
    /// error columns. Active-state merging is intentionally owned by the host.
    pub fn query_request(&self, request_id: &str) -> Result<Option<RequestRecord>, LogStoreError> {
        let connection = self.conn();
        let sql = format!("SELECT {REQUEST_COLUMNS} FROM summaries WHERE request_id = ?");
        match connection.query_row(&sql, [request_id], request_record) {
            Ok(record) => Ok(Some(record)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(error) => Err(LogStoreError::QueryFailed(error.to_string())),
        }
    }

    /// Return durable summaries for a bounded set of request IDs.
    ///
    /// The caller may supply more IDs than SQLite's conservative parameter
    /// budget because this method splits them into fixed-size statements while
    /// holding the connection only once. Missing IDs are omitted.
    pub fn query_requests_by_ids(
        &self,
        request_ids: &[String],
    ) -> Result<Vec<RequestRecord>, LogStoreError> {
        if request_ids.is_empty() {
            return Ok(Vec::new());
        }

        let connection = self.conn();
        let mut records = Vec::with_capacity(request_ids.len());
        for request_ids in request_ids.chunks(REQUEST_ID_BATCH_SIZE) {
            let placeholders = std::iter::repeat_n("?", request_ids.len())
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "SELECT {REQUEST_COLUMNS} FROM summaries WHERE request_id IN ({placeholders})"
            );
            let mut statement = connection.prepare(&sql).map_err(LogStoreError::Sqlite)?;
            let rows = statement
                .query_map(rusqlite::params_from_iter(request_ids), request_record)
                .map_err(LogStoreError::Sqlite)?;
            records.extend(
                rows.collect::<Result<Vec<_>, _>>()
                    .map_err(|error| LogStoreError::QueryFailed(error.to_string()))?,
            );
        }
        Ok(records)
    }

    /// List durable summaries by a stable `(created_at, request_id)` keyset.
    /// A valid cursor must name a row in the exact current filter scope, so a
    /// forged pair and a cursor retained away both fail deterministically.
    pub fn query_requests(
        &self,
        query: &RequestQuery,
    ) -> Result<QueryPage<RequestRecord>, LogStoreError> {
        query.validate()?;
        let (conditions, mut values) = request_conditions(query);
        let connection = self.conn();

        if let Some(cursor) = &query.cursor {
            let (timestamp, request_id) = decode_ordering_cursor(cursor)?;
            validate_cursor_position(
                &connection,
                CursorPosition {
                    table: "summaries",
                    timestamp_column: "created_at",
                    id_column: "request_id",
                    timestamp: &timestamp,
                    id: &request_id,
                    conditions: &conditions,
                    values: &values,
                },
            )?;
            values.push(Value::Text(timestamp));
            values.push(Value::Text(request_id));
        }

        let mut sql = format!("SELECT {REQUEST_COLUMNS} FROM summaries");
        conditions.append_to_sql(&mut sql);
        if query.cursor.is_some() {
            sql.push_str(&format!(
                " AND (created_at, request_id) {} (?, ?)",
                query.sort.cursor_operator()
            ));
        }
        sql.push_str(&format!(
            " ORDER BY created_at {}, request_id {} LIMIT ?",
            query.sort.sql_order(),
            query.sort.sql_order()
        ));
        values.push(Value::Integer(i64::try_from(query.limit + 1).map_err(
            |_| LogStoreError::InvalidQuery("limit is out of range".to_string()),
        )?));
        query_page(
            &connection,
            sql,
            values,
            query.limit,
            request_record,
            |record| (&record.created_at, &record.request_id),
        )
    }
}

pub(super) struct Conditions {
    pub(super) sql: String,
}

impl Conditions {
    pub(super) fn append_to_sql(&self, sql: &mut String) {
        sql.push_str(&self.sql);
    }
}

fn request_conditions(query: &RequestQuery) -> (Conditions, Vec<Value>) {
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
    if let Some(status_code) = query.status_code {
        sql.push_str(" AND status_code = ?");
        values.push(Value::Integer(i64::from(status_code)));
    }
    if let Some(outcome) = query.outcome {
        sql.push_str(" AND state = ?");
        values.push(Value::Text(outcome.as_str().to_string()));
    }
    let conditions = Conditions { sql };
    (conditions, values)
}

pub(super) struct CursorPosition<'a> {
    pub(super) table: &'static str,
    pub(super) timestamp_column: &'static str,
    pub(super) id_column: &'static str,
    pub(super) timestamp: &'a str,
    pub(super) id: &'a str,
    pub(super) conditions: &'a Conditions,
    pub(super) values: &'a [Value],
}

pub(super) fn validate_cursor_position(
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
    // `values` is supplied separately so callers cannot accidentally validate
    // the cursor with a different set of filters than their page query.
    parameters.extend(position.values.iter().cloned());
    match connection.query_row(&sql, rusqlite::params_from_iter(parameters.iter()), |_| {
        Ok(())
    }) {
        Ok(()) => Ok(()),
        Err(rusqlite::Error::QueryReturnedNoRows) => Err(LogStoreError::CursorInvalid),
        Err(error) => Err(LogStoreError::QueryFailed(error.to_string())),
    }
}

pub(super) fn query_page<T>(
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

fn request_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<RequestRecord> {
    Ok(RequestRecord {
        request_id: row.get(0)?,
        outcome: row.get(1)?,
        created_at: row.get(2)?,
        terminal_at: row.get(3)?,
        route: row.get(4)?,
        model: row.get(5)?,
        provider: row.get(6)?,
        engine: row.get(7)?,
        status_code: row.get(8)?,
    })
}
