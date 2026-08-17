use std::collections::BTreeMap;

use rusqlite::types::Value;

use super::requests::{Conditions, CursorPosition, query_page, validate_cursor_position};
use super::{
    ArtifactRecord, EventRecord, PageQuery, ProxyQuery, ProxyRecord, QueryPage, validate_identifier,
};
use crate::cursor::decode_ordering_cursor;
use crate::{LogStore, LogStoreError};

impl LogStore {
    /// Return lifecycle records only for the named durable request.
    pub fn query_events(
        &self,
        request_id: &str,
        query: &PageQuery,
    ) -> Result<QueryPage<EventRecord>, LogStoreError> {
        self.query_related_page(
            RelatedQuery {
                table: "lifecycle_events",
                columns: "event_id, request_id, occurred_at, payload_json",
                id_column: "event_id",
                request_id,
                page: query,
            },
            event_record,
            |record| (&record.occurred_at, &record.event_id),
        )
    }

    /// Return artifact metadata, never filesystem paths or file contents.
    pub fn query_artifacts(
        &self,
        request_id: &str,
        query: &PageQuery,
    ) -> Result<QueryPage<ArtifactRecord>, LogStoreError> {
        self.query_related_page(
            RelatedQuery {
                table: "artifact_pointers",
                columns: "artifact_id, request_id, occurred_at, kind, media_kind, checksum, bytes, version, redacted, truncated, missing, corrupt, unavailable_reason",
                id_column: "artifact_id",
                request_id,
                page: query,
            },
            artifact_record,
            |record| (&record.occurred_at, &record.artifact_id),
        )
    }

    /// Load the first bounded child-event window for a set of request owners
    /// in one SQLite query. Callers can use the extra row per owner to report
    /// a partial export without issuing an N+1 query fan-out.
    pub fn query_events_for_requests(
        &self,
        request_ids: &[String],
        per_request_limit: usize,
    ) -> Result<BTreeMap<String, Vec<EventRecord>>, LogStoreError> {
        self.query_related_for_requests(
            request_ids,
            per_request_limit,
            RelatedBatchQuery {
                table: "lifecycle_events",
                columns: "event_id, request_id, occurred_at, payload_json",
                id_column: "event_id",
            },
            event_record,
            |record| &record.request_id,
        )
    }

    /// Load the first bounded artifact window for a set of request owners in
    /// one SQLite query. Results contain metadata only, never file paths or
    /// captured content.
    pub fn query_artifacts_for_requests(
        &self,
        request_ids: &[String],
        per_request_limit: usize,
    ) -> Result<BTreeMap<String, Vec<ArtifactRecord>>, LogStoreError> {
        self.query_related_for_requests(
            request_ids,
            per_request_limit,
            RelatedBatchQuery {
                table: "artifact_pointers",
                columns: "artifact_id, request_id, occurred_at, kind, media_kind, checksum, bytes, version, redacted, truncated, missing, corrupt, unavailable_reason",
                id_column: "artifact_id",
            },
            artifact_record,
            |record| &record.request_id,
        )
    }

    /// Return exactly one artifact metadata record without path-bearing storage
    /// details. Content reading and authorization stay outside this substrate.
    pub fn query_artifact(
        &self,
        artifact_id: &str,
    ) -> Result<Option<ArtifactRecord>, LogStoreError> {
        validate_identifier(artifact_id)?;
        let columns = "artifact_id, request_id, occurred_at, kind, media_kind, checksum, bytes, version, redacted, truncated, missing, corrupt, unavailable_reason";
        let sql = format!("SELECT {columns} FROM artifact_pointers WHERE artifact_id = ?");
        match self.conn().query_row(&sql, [artifact_id], artifact_record) {
            Ok(record) => Ok(Some(record)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(error) => Err(LogStoreError::QueryFailed(error.to_string())),
        }
    }

    /// List proxy attempts separately from request summaries. Targets are
    /// routing metadata; request/response artifacts are not joined here.
    pub fn query_proxy_records(
        &self,
        query: &ProxyQuery,
    ) -> Result<QueryPage<ProxyRecord>, LogStoreError> {
        query.validate()?;
        let mut filter_sql = String::from(" WHERE 1 = 1");
        let mut values = Vec::new();
        for (column, value) in [
            ("request_id", &query.request_id),
            ("provider", &query.provider),
            ("engine", &query.engine),
        ] {
            if let Some(value) = value {
                filter_sql.push_str(&format!(" AND {column} = ?"));
                values.push(Value::Text(value.clone()));
            }
        }
        if let Some(status_code) = query.status_code {
            filter_sql.push_str(" AND status_code = ?");
            values.push(Value::Integer(i64::from(status_code)));
        }
        let conditions = Conditions { sql: filter_sql };
        let connection = self.conn();
        if let Some(cursor) = &query.page.cursor {
            let (timestamp, attempt_id) = decode_ordering_cursor(cursor)?;
            validate_cursor_position(
                &connection,
                CursorPosition {
                    table: "proxy_records",
                    timestamp_column: "occurred_at",
                    id_column: "attempt_id",
                    timestamp: &timestamp,
                    id: &attempt_id,
                    conditions: &conditions,
                    values: &values,
                },
            )?;
            values.push(Value::Text(timestamp));
            values.push(Value::Text(attempt_id));
        }

        let columns = "attempt_id, request_id, occurred_at, target, provider, engine, started_at, completed_at, status_code";
        let mut sql = format!("SELECT {columns} FROM proxy_records");
        conditions.append_to_sql(&mut sql);
        if query.page.cursor.is_some() {
            sql.push_str(&format!(
                " AND (occurred_at, attempt_id) {} (?, ?)",
                query.page.sort.cursor_operator()
            ));
        }
        sql.push_str(&format!(
            " ORDER BY occurred_at {}, attempt_id {} LIMIT ?",
            query.page.sort.sql_order(),
            query.page.sort.sql_order()
        ));
        values.push(Value::Integer(
            i64::try_from(query.page.limit + 1)
                .map_err(|_| LogStoreError::InvalidQuery("limit is out of range".to_string()))?,
        ));
        query_page(
            &connection,
            sql,
            values,
            query.page.limit,
            proxy_record,
            |record| (&record.occurred_at, &record.attempt_id),
        )
    }

    fn query_related_page<T>(
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

    fn query_related_for_requests<T>(
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
        if request_ids.len() > super::MAX_QUERY_LIMIT {
            return Err(LogStoreError::InvalidQuery(format!(
                "request owner count must be at most {}",
                super::MAX_QUERY_LIMIT
            )));
        }
        super::validate_limit(per_request_limit)?;
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
        let mut grouped: BTreeMap<String, Vec<T>> = BTreeMap::new();
        for row in rows {
            grouped
                .entry(request_id(&row).to_owned())
                .or_default()
                .push(row);
        }
        Ok(grouped)
    }
}

struct RelatedQuery<'a> {
    table: &'static str,
    columns: &'static str,
    id_column: &'static str,
    request_id: &'a str,
    page: &'a PageQuery,
}

struct RelatedBatchQuery {
    table: &'static str,
    columns: &'static str,
    id_column: &'static str,
}

fn event_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<EventRecord> {
    Ok(EventRecord {
        event_id: row.get(0)?,
        request_id: row.get(1)?,
        occurred_at: row.get(2)?,
        payload_json: row.get(3)?,
    })
}

fn artifact_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<ArtifactRecord> {
    Ok(ArtifactRecord {
        artifact_id: row.get(0)?,
        request_id: row.get(1)?,
        occurred_at: row.get(2)?,
        kind: row.get(3)?,
        media_kind: row.get(4)?,
        checksum: row.get(5)?,
        bytes: row.get(6)?,
        version: row.get(7)?,
        redacted: row.get::<_, i32>(8)? != 0,
        truncated: row.get::<_, i32>(9)? != 0,
        missing: row.get::<_, i32>(10)? != 0,
        corrupt: row.get::<_, i32>(11)? != 0,
        unavailable_reason: row.get(12)?,
    })
}

fn proxy_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProxyRecord> {
    Ok(ProxyRecord {
        attempt_id: row.get(0)?,
        request_id: row.get(1)?,
        occurred_at: row.get(2)?,
        target: row.get(3)?,
        provider: row.get(4)?,
        engine: row.get(5)?,
        started_at: row.get(6)?,
        completed_at: row.get(7)?,
        status_code: row.get(8)?,
    })
}
