mod projection;
pub(super) mod selection;

use rusqlite::types::Value;

use super::{QueryPage, RequestQuery, RequestRecord, RequestRecordWithCaller};
use crate::cursor::decode_ordering_cursor;
use crate::{LogStore, LogStoreError};

use projection::{REQUEST_COLUMNS, request_record_with_caller};
use selection::{CursorPosition, query_page, validate_cursor_position};

const REQUEST_ID_BATCH_SIZE: usize = 100;

impl LogStore {
    pub fn query_request(&self, request_id: &str) -> Result<Option<RequestRecord>, LogStoreError> {
        self.query_request_with_caller(request_id)
            .map(|record| record.map(Into::into))
    }

    pub fn query_request_with_caller(
        &self,
        request_id: &str,
    ) -> Result<Option<RequestRecordWithCaller>, LogStoreError> {
        let connection = self.conn();
        let sql = format!("SELECT {REQUEST_COLUMNS} FROM summaries WHERE request_id = ?");
        match connection.query_row(&sql, [request_id], request_record_with_caller) {
            Ok(record) => Ok(Some(record)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(error) => Err(LogStoreError::QueryFailed(error.to_string())),
        }
    }

    pub fn query_requests_by_ids(
        &self,
        request_ids: &[String],
    ) -> Result<Vec<RequestRecord>, LogStoreError> {
        self.query_requests_by_ids_with_caller(request_ids)
            .map(|records| records.into_iter().map(Into::into).collect())
    }

    pub fn query_requests_by_ids_with_caller(
        &self,
        request_ids: &[String],
    ) -> Result<Vec<RequestRecordWithCaller>, LogStoreError> {
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
                .query_map(
                    rusqlite::params_from_iter(request_ids),
                    request_record_with_caller,
                )
                .map_err(LogStoreError::Sqlite)?;
            records.extend(
                rows.collect::<Result<Vec<_>, _>>()
                    .map_err(|error| LogStoreError::QueryFailed(error.to_string()))?,
            );
        }
        Ok(records)
    }

    pub fn query_requests(
        &self,
        query: &RequestQuery,
    ) -> Result<QueryPage<RequestRecord>, LogStoreError> {
        self.query_requests_with_caller(query)
            .map(|page| QueryPage {
                items: page.items.into_iter().map(Into::into).collect(),
                next_cursor: page.next_cursor,
            })
    }

    pub fn query_requests_with_caller(
        &self,
        query: &RequestQuery,
    ) -> Result<QueryPage<RequestRecordWithCaller>, LogStoreError> {
        query.validate()?;
        let (conditions, mut values) = selection::request_conditions(query);
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
            request_record_with_caller,
            |record| (&record.request.created_at, &record.request.request_id),
        )
    }
}
