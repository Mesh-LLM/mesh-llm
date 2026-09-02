use std::collections::BTreeMap;

use rusqlite::types::Value;

use super::{RelatedBatchQuery, RelatedQuery};
use crate::cursor::decode_ordering_cursor;
use crate::query::requests::selection::{
    Conditions, CursorPosition, query_page, validate_cursor_position,
};
use crate::query::{QueryPage, validate_identifier};
use crate::{LogStore, LogStoreError};

impl LogStore {
    pub(super) fn query_related_page<T>(
        &self,
        related: RelatedQuery<'_>,
        map: fn(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
        cursor_fields: impl Fn(&T) -> (&str, &str),
    ) -> Result<QueryPage<T>, LogStoreError> {
        validate_identifier(related.request_id)?;
        related.page.validate()?;
        let conditions = Conditions {
            sql: " WHERE request_id = ?".to_string(),
        };
        let mut values = vec![Value::Text(related.request_id.to_string())];
        let connection = self.conn();
        if let Some(cursor) = &related.page.cursor {
            let (timestamp, id) = decode_ordering_cursor(cursor)?;
            validate_cursor_position(
                &connection,
                CursorPosition {
                    table: related.table,
                    timestamp_column: "occurred_at",
                    id_column: related.id_column,
                    timestamp: &timestamp,
                    id: &id,
                    conditions: &conditions,
                    values: &values,
                },
            )?;
            values.push(Value::Text(timestamp));
            values.push(Value::Text(id));
        }
        let mut sql = format!("SELECT {} FROM {}", related.columns, related.table);
        conditions.append_to_sql(&mut sql);
        if related.page.cursor.is_some() {
            sql.push_str(&format!(
                " AND (occurred_at, {}) {} (?, ?)",
                related.id_column,
                related.page.sort.cursor_operator()
            ));
        }
        sql.push_str(&format!(
            " ORDER BY occurred_at {}, {} {} LIMIT ?",
            related.page.sort.sql_order(),
            related.id_column,
            related.page.sort.sql_order()
        ));
        values.push(Value::Integer(
            i64::try_from(related.page.limit + 1)
                .map_err(|_| LogStoreError::InvalidQuery("limit is out of range".to_string()))?,
        ));
        query_page(
            &connection,
            sql,
            values,
            related.page.limit,
            map,
            cursor_fields,
        )
    }

    pub(super) fn query_related_for_requests<T>(
        &self,
        request_ids: &[String],
        per_request_limit: usize,
        related: RelatedBatchQuery,
        map: fn(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
        request_id: impl Fn(&T) -> &str,
    ) -> Result<BTreeMap<String, Vec<T>>, LogStoreError> {
        if request_ids.is_empty() {
            return Ok(BTreeMap::new());
        }
        if request_ids.len() > super::super::MAX_QUERY_LIMIT {
            return Err(LogStoreError::InvalidQuery(format!(
                "request owner count must be at most {}",
                super::super::MAX_QUERY_LIMIT
            )));
        }
        super::super::validate_limit(per_request_limit)?;
        for owner in request_ids {
            validate_identifier(owner)?;
        }
        let placeholders = std::iter::repeat_n("?", request_ids.len())
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT {columns} FROM (\
                SELECT {columns}, ROW_NUMBER() OVER (\
                    PARTITION BY request_id ORDER BY occurred_at ASC, {id_column} ASC\
                ) AS owner_row_number \
                FROM {table} WHERE request_id IN ({placeholders})\
            ) WHERE owner_row_number <= ? \
            ORDER BY request_id ASC, occurred_at ASC, {id_column} ASC",
            columns = related.columns,
            id_column = related.id_column,
            table = related.table,
        );
        let mut values = request_ids
            .iter()
            .cloned()
            .map(Value::Text)
            .collect::<Vec<_>>();
        values.push(Value::Integer(i64::try_from(per_request_limit).map_err(
            |_| LogStoreError::InvalidQuery("limit is out of range".to_string()),
        )?));
        let connection = self.conn();
        let mut statement = connection.prepare(&sql).map_err(LogStoreError::Sqlite)?;
        let rows = statement
            .query_map(rusqlite::params_from_iter(values), map)
            .map_err(LogStoreError::Sqlite)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| LogStoreError::QueryFailed(error.to_string()))?;
        let mut grouped = BTreeMap::new();
        for row in rows {
            grouped
                .entry(request_id(&row).to_owned())
                .or_insert_with(Vec::new)
                .push(row);
        }
        Ok(grouped)
    }
}
