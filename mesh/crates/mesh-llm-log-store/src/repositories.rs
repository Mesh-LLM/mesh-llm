//! Typed repositories for log-store persistence operations.

use crate::cursor::{decode_ordering_cursor, encode_cursor};
use crate::error::LogStoreError;
use crate::store::LogStore;
use crate::timestamps::{
    canonical_optional_persisted_timestamp, canonical_persisted_timestamp,
    canonical_timestamp_metadata,
};
use rusqlite::{OptionalExtension, Row, Transaction};
use serde::Serialize;

mod audit;
mod caller_metadata;
mod cleanup;

pub use audit::{
    AuditEntryFilters, AuditEntryRow, AuditEntrySeverity, AuditEntrySource,
    DEFAULT_AUDIT_ENTRY_LIMIT, MAX_AUDIT_ENTRY_LIMIT,
};
pub use cleanup::{
    RetentionCleanupResult, RetentionPolicy, RetentionTable, RetentionTablePolicy,
    RetentionTableResult,
};

const MAX_WEBHOOK_ATTEMPTS: u32 = 20;
const MAX_WEBHOOK_IDENTIFIER_BYTES: usize = 128;
const MAX_WEBHOOK_TIMESTAMP_BYTES: usize = 64;
const MIN_HTTP_STATUS_CODE: u16 = 100;
const MAX_HTTP_STATUS_CODE: u16 = 599;
const CONFIGURED_WEBHOOK_TARGET: &str = "configured_webhook";

// ─── Row types returned by queries ──────────────────────

#[derive(Debug, Clone)]
pub struct SummaryRow {
    pub request_id: String,
    pub state: String,
    pub created_at: String,
    #[allow(dead_code)]
    pub terminal_at: Option<String>,
    #[allow(dead_code)]
    pub route: Option<String>,
    #[allow(dead_code)]
    pub model: Option<String>,
    #[allow(dead_code)]
    pub provider: Option<String>,
    #[allow(dead_code)]
    pub engine: Option<String>,
    #[allow(dead_code)]
    pub status_code: Option<i64>,
    #[allow(dead_code)]
    pub error_msg: Option<String>,
    #[allow(dead_code)]
    pub tenant_id: Option<String>,
    #[allow(dead_code)]
    pub account_id: Option<String>,
    #[allow(dead_code)]
    pub user_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct LifecycleEventRow {
    pub event_id: String,
    pub request_id: String,
    pub occurred_at: String,
}

/// Closed terminal request classification safe to retain for webhook delivery.
///
/// The fixed vocabulary excludes raw errors, endpoints, paths, prompts,
/// completions, and other request content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WebhookTerminalOutcome {
    Completed,
    Failed,
    Rejected,
    Cancelled,
    Dropped,
}

impl WebhookTerminalOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Rejected => "rejected",
            Self::Cancelled => "cancelled",
            Self::Dropped => "dropped",
        }
    }

    fn parse(value: &str) -> Result<Self, LogStoreError> {
        match value {
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            "rejected" => Ok(Self::Rejected),
            "cancelled" => Ok(Self::Cancelled),
            "dropped" => Ok(Self::Dropped),
            _ => Err(LogStoreError::QueryFailed(
                "webhook terminal outcome is invalid".to_string(),
            )),
        }
    }
}

/// Durable, privacy-safe state for one scoped terminal webhook delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebhookDeliveryState {
    Pending,
    InFlight,
    Succeeded,
    Retry,
    DeadLetter,
    ManualRetry,
}

impl WebhookDeliveryState {
    const fn code(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InFlight => "in_flight",
            Self::Succeeded => "succeeded",
            Self::Retry => "retry",
            Self::DeadLetter => "dead_letter",
            Self::ManualRetry => "manual_retry",
        }
    }

    fn parse(value: &str) -> Result<Self, LogStoreError> {
        match value {
            "pending" => Ok(Self::Pending),
            "in_flight" => Ok(Self::InFlight),
            "succeeded" => Ok(Self::Succeeded),
            "retry" => Ok(Self::Retry),
            "dead_letter" => Ok(Self::DeadLetter),
            "manual_retry" => Ok(Self::ManualRetry),
            _ => Err(LogStoreError::QueryFailed(
                "webhook delivery state is invalid".to_string(),
            )),
        }
    }
}

/// Sanitized, bounded classification of a failed webhook attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebhookDeliveryErrorCode {
    Timeout,
    Transport,
    Http4xx,
    Http5xx,
    Configuration,
}

impl WebhookDeliveryErrorCode {
    const fn code(self) -> &'static str {
        match self {
            Self::Timeout => "timeout",
            Self::Transport => "transport",
            Self::Http4xx => "http_4xx",
            Self::Http5xx => "http_5xx",
            Self::Configuration => "configuration",
        }
    }

    fn parse(value: Option<String>) -> Result<Option<Self>, LogStoreError> {
        match value.as_deref() {
            None => Ok(None),
            Some("timeout") => Ok(Some(Self::Timeout)),
            Some("transport") => Ok(Some(Self::Transport)),
            Some("http_4xx") => Ok(Some(Self::Http4xx)),
            Some("http_5xx") => Ok(Some(Self::Http5xx)),
            Some("configuration") => Ok(Some(Self::Configuration)),
            Some(_) => Err(LogStoreError::QueryFailed(
                "webhook delivery error code is invalid".to_string(),
            )),
        }
    }
}

/// Persisted record returned to the asynchronous webhook worker. The claim
/// generation is a fencing value: only the worker that atomically incremented
/// it can complete or retry that in-flight attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebhookDeliveryRecord {
    pub delivery_id: String,
    pub request_id: Option<String>,
    pub terminal_outcome: WebhookTerminalOutcome,
    pub created_at: String,
    pub updated_at: String,
    pub state: WebhookDeliveryState,
    pub attempt_number: u32,
    pub max_attempts: u32,
    pub next_attempt_at: Option<String>,
    pub lease_expires_at: Option<String>,
    pub claim_generation: u64,
    /// Immutable HTTP status belonging to the terminal request result.
    pub terminal_status_code: Option<u16>,
    /// HTTP status most recently returned by the webhook receiver.
    pub response_status_code: Option<u16>,
    pub last_error_code: Option<WebhookDeliveryErrorCode>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebhookDeliveryInsertOutcome {
    Created(WebhookDeliveryRecord),
    Existing(WebhookDeliveryRecord),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebhookRetryOutcome {
    RetryScheduled,
    DeadLettered,
}

/// Atomic outcome of an operator request to retry a webhook delivery.
///
/// `AlreadyScheduled` covers a delivery that a concurrent scheduler has
/// already moved from `manual_retry` to `in_flight`, keeping retries
/// idempotent at the API boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebhookManualRetryOutcome {
    Scheduled,
    AlreadyScheduled,
    NotRetryable,
    NotFound,
}

/// A file artifact whose durable pointer was removed by cascade cleanup.
///
/// Keeping this ownership tuple in the transaction result means post-commit
/// file cleanup never has to rediscover a path by artifact filename.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CascadeArtifactPointer {
    pub artifact_id: String,
    pub request_id: String,
}

/// Paginated query result with an optional cursor for the next page.
#[derive(Debug)]
pub struct Page<T> {
    pub items: Vec<T>,
    pub next_cursor: Option<String>,
}

/// Static SQL shape for a descending opaque `(occurred_at, id)` keyset page.
///
/// The table and column names are private constants supplied by repository
/// methods, while the cursor values are always bound parameters.
struct OpaqueKeysetPage {
    table: &'static str,
    columns: &'static str,
    timestamp_column: &'static str,
    id_column: &'static str,
}

fn list_opaque_keyset_page<T>(
    connection: &rusqlite::Connection,
    page: OpaqueKeysetPage,
    limit: usize,
    after_cursor: Option<&str>,
    map: impl Fn(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
    cursor_fields: impl Fn(&T) -> (&str, &str),
) -> Result<Page<T>, LogStoreError> {
    let (cursor_predicate, values) = if let Some(cursor) = after_cursor {
        let (timestamp, id) = decode_ordering_cursor(cursor)?;
        (
            format!(
                " WHERE ({}, {}) < (?, ?)",
                page.timestamp_column, page.id_column
            ),
            vec![timestamp, id],
        )
    } else {
        (String::new(), Vec::new())
    };
    let sql = format!(
        "SELECT {} FROM {}{} ORDER BY {} DESC, {} DESC LIMIT {}",
        page.columns, page.table, cursor_predicate, page.timestamp_column, page.id_column, limit,
    );
    let mut statement = connection.prepare(&sql).map_err(LogStoreError::Sqlite)?;
    let items = statement
        .query_map(rusqlite::params_from_iter(values.iter()), map)
        .map_err(LogStoreError::Sqlite)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| LogStoreError::QueryFailed(error.to_string()))?;
    let Some(last) = items.last() else {
        return Ok(Page {
            items,
            next_cursor: None,
        });
    };
    let (timestamp, id) = cursor_fields(last);
    let probe_sql = format!(
        "SELECT EXISTS(SELECT 1 FROM {} WHERE ({}, {}) < (?, ?) LIMIT 1)",
        page.table, page.timestamp_column, page.id_column,
    );
    let has_more = connection
        .query_row(&probe_sql, rusqlite::params![timestamp, id], |row| {
            row.get::<_, i32>(0)
        })
        .map(|value| value != 0)
        .map_err(LogStoreError::Sqlite)?;
    let next_cursor = has_more.then(|| encode_cursor(timestamp, id));

    Ok(Page { items, next_cursor })
}

// ─── Internal helpers ──────────────

pub(crate) fn is_unique_constraint_error(e: &rusqlite::Error) -> bool {
    if let rusqlite::Error::SqliteFailure(err, _) = e {
        matches!(
            err.extended_code,
            rusqlite::ffi::SQLITE_CONSTRAINT_PRIMARYKEY | rusqlite::ffi::SQLITE_CONSTRAINT_UNIQUE
        )
    } else {
        false
    }
}

fn is_foreign_key_constraint_error(e: &rusqlite::Error) -> bool {
    matches!(
        e,
        rusqlite::Error::SqliteFailure(err, _)
            if err.extended_code == rusqlite::ffi::SQLITE_CONSTRAINT_FOREIGNKEY
    )
}

pub(crate) fn map_insert_constraint_error(
    e: &rusqlite::Error,
    entity: String,
) -> Option<LogStoreError> {
    if is_foreign_key_constraint_error(e) {
        Some(LogStoreError::ForeignKeyViolation { entity })
    } else if is_unique_constraint_error(e) {
        Some(LogStoreError::AlreadyExists { entity })
    } else {
        None
    }
}

fn validate_webhook_identifier(value: &str, field: &'static str) -> Result<(), LogStoreError> {
    if value.is_empty() || value.len() > MAX_WEBHOOK_IDENTIFIER_BYTES {
        return Err(LogStoreError::InvalidQuery(format!(
            "webhook {field} must be between 1 and {MAX_WEBHOOK_IDENTIFIER_BYTES} bytes"
        )));
    }
    Ok(())
}

fn validate_webhook_timestamp(value: &str, field: &'static str) -> Result<(), LogStoreError> {
    if value.is_empty() || value.len() > MAX_WEBHOOK_TIMESTAMP_BYTES {
        return Err(LogStoreError::InvalidQuery(format!(
            "webhook {field} must be between 1 and {MAX_WEBHOOK_TIMESTAMP_BYTES} bytes"
        )));
    }
    Ok(())
}

fn canonical_webhook_timestamp(value: &str, field: &'static str) -> Result<String, LogStoreError> {
    validate_webhook_timestamp(value, field)?;
    mesh_llm_events::logging::timestamp::canonical_logging_timestamp(value).map_err(|_| {
        LogStoreError::InvalidQuery(format!("webhook {field} must be an RFC 3339 timestamp"))
    })
}

fn validate_webhook_max_attempts(max_attempts: u32) -> Result<(), LogStoreError> {
    if !(1..=MAX_WEBHOOK_ATTEMPTS).contains(&max_attempts) {
        return Err(LogStoreError::InvalidQuery(format!(
            "webhook max_attempts must be between 1 and {MAX_WEBHOOK_ATTEMPTS}"
        )));
    }
    Ok(())
}

fn validate_optional_http_status(
    status_code: Option<u16>,
    field: &str,
) -> Result<(), LogStoreError> {
    if status_code
        .is_some_and(|status| !(MIN_HTTP_STATUS_CODE..=MAX_HTTP_STATUS_CODE).contains(&status))
    {
        return Err(LogStoreError::InvalidQuery(format!(
            "{field} must be between {MIN_HTTP_STATUS_CODE} and {MAX_HTTP_STATUS_CODE}"
        )));
    }
    Ok(())
}

fn optional_http_status_from_row(value: Option<i64>, field: &str) -> rusqlite::Result<Option<u16>> {
    value
        .map(|status_code| {
            let status_code = u16::try_from(status_code).map_err(|_| {
                to_sqlite_conversion_error(LogStoreError::QueryFailed(format!(
                    "webhook {field} is invalid"
                )))
            })?;
            validate_optional_http_status(Some(status_code), field)
                .map_err(to_sqlite_conversion_error)?;
            Ok(status_code)
        })
        .transpose()
}

fn webhook_record_from_row(row: &Row<'_>) -> rusqlite::Result<WebhookDeliveryRecord> {
    let terminal_outcome: String = row.get("terminal_outcome")?;
    let state: String = row.get("state")?;
    let last_error_code: Option<String> = row.get("last_error_code")?;
    let terminal_status_code: Option<i64> = row.get("terminal_status_code")?;
    let response_status_code: Option<i64> = row.get("status_code")?;
    let attempt_number: i64 = row.get("attempt_number")?;
    let max_attempts: i64 = row.get("max_attempts")?;
    let claim_generation: i64 = row.get("claim_generation")?;
    let terminal_outcome =
        WebhookTerminalOutcome::parse(&terminal_outcome).map_err(to_sqlite_conversion_error)?;
    let state = WebhookDeliveryState::parse(&state).map_err(to_sqlite_conversion_error)?;
    let last_error_code =
        WebhookDeliveryErrorCode::parse(last_error_code).map_err(to_sqlite_conversion_error)?;
    let terminal_status_code =
        optional_http_status_from_row(terminal_status_code, "terminal status code")?;
    let response_status_code =
        optional_http_status_from_row(response_status_code, "receiver response status code")?;
    let attempt_number = u32::try_from(attempt_number).map_err(|_| {
        to_sqlite_conversion_error(LogStoreError::QueryFailed(
            "webhook attempt number is invalid".to_string(),
        ))
    })?;
    let max_attempts = u32::try_from(max_attempts).map_err(|_| {
        to_sqlite_conversion_error(LogStoreError::QueryFailed(
            "webhook max attempts is invalid".to_string(),
        ))
    })?;
    let claim_generation = u64::try_from(claim_generation).map_err(|_| {
        to_sqlite_conversion_error(LogStoreError::QueryFailed(
            "webhook claim generation is invalid".to_string(),
        ))
    })?;

    Ok(WebhookDeliveryRecord {
        delivery_id: row.get("delivery_id")?,
        request_id: row.get("request_id")?,
        terminal_outcome,
        created_at: row.get("created_at")?,
        updated_at: row.get("updated_at")?,
        state,
        attempt_number,
        max_attempts,
        next_attempt_at: row.get("next_attempt_at")?,
        lease_expires_at: row.get("lease_expires_at")?,
        claim_generation,
        terminal_status_code,
        response_status_code,
        last_error_code,
    })
}

fn to_sqlite_conversion_error(error: LogStoreError) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(0, rusqlite::types::Type::Text, Box::new(error))
}

fn select_webhook_delivery(
    connection: &rusqlite::Connection,
    delivery_id: &str,
) -> Result<Option<WebhookDeliveryRecord>, LogStoreError> {
    connection
        .query_row(
            "SELECT delivery_id, request_id, terminal_outcome, created_at, updated_at, state, attempt_number, \
                max_attempts, next_attempt_at, lease_expires_at, claim_generation, terminal_status_code, status_code, last_error_code \
             FROM webhook_deliveries WHERE delivery_id = ?",
            [delivery_id],
            webhook_record_from_row,
        )
        .optional()
        .map_err(|error| LogStoreError::QueryFailed(error.to_string()))
}

#[derive(Clone, Copy)]
struct TerminalEventWrite<'a> {
    request_id: &'a str,
    event_id: &'a str,
    payload_json: &'a str,
    terminal_status: &'a str,
    terminal_status_code: Option<u16>,
    occurred_at: &'a str,
    event_type: LifecycleEventType,
}

fn write_terminal_event_in_transaction(
    transaction: &Transaction<'_>,
    event: TerminalEventWrite<'_>,
) -> Result<(), LogStoreError> {
    validate_optional_http_status(event.terminal_status_code, "terminal status code")?;
    let has_terminal = transaction
        .query_row(
            "SELECT COUNT(*) FROM lifecycle_events WHERE request_id = ? AND is_terminal = 1",
            rusqlite::params![event.request_id],
            |row| row.get::<_, i64>(0),
        )
        .map(|count| count > 0)
        .map_err(LogStoreError::Sqlite)?;
    if has_terminal {
        return Err(LogStoreError::DuplicateTerminalEvent {
            summary_id: event.request_id.to_string(),
            event_type: event.event_type.code().to_string(),
        });
    }

    transaction
        .execute(
            "INSERT INTO lifecycle_events (event_id, request_id, occurred_at, payload_json, event_type, is_terminal) \
             VALUES (?, ?, ?, ?, ?, 1)",
            rusqlite::params![
                event.event_id,
                event.request_id,
                event.occurred_at,
                event.payload_json,
                event.event_type.code(),
            ],
        )
        .map_err(LogStoreError::Sqlite)?;
    transaction
        .execute(
            "UPDATE summaries SET state = ?, terminal_at = ?, status_code = ? WHERE request_id = ?",
            rusqlite::params![
                event.terminal_status,
                event.occurred_at,
                event.terminal_status_code,
                event.request_id
            ],
        )
        .map_err(LogStoreError::Sqlite)?;
    Ok(())
}

fn enqueue_webhook_delivery_in_transaction(
    transaction: &Transaction<'_>,
    delivery_id: &str,
    request_id: &str,
    terminal_status_code: Option<u16>,
    created_at: &str,
    max_attempts: u32,
) -> Result<WebhookDeliveryInsertOutcome, LogStoreError> {
    validate_optional_http_status(terminal_status_code, "terminal status code")?;
    let terminal_outcome = transaction
        .query_row(
            "SELECT event_type FROM lifecycle_events WHERE request_id = ? AND is_terminal = 1",
            rusqlite::params![request_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(LogStoreError::Sqlite)?
        .ok_or_else(|| {
            LogStoreError::InvalidQuery(
                "webhook delivery requires a durable terminal event".to_string(),
            )
        })?;
    let terminal_outcome = WebhookTerminalOutcome::parse(&terminal_outcome)?;
    let inserted = transaction
        .execute(
            "INSERT INTO webhook_deliveries \
                (delivery_id, request_id, terminal_outcome, terminal_status_code, occurred_at, target_url, attempt_number, response_body, error_msg, \
                 state, created_at, updated_at, next_attempt_at, lease_expires_at, claim_generation, max_attempts, last_error_code) \
             VALUES (?, ?, ?, ?, ?, ?, 0, NULL, NULL, ?, ?, ?, ?, NULL, 0, ?, NULL) \
             ON CONFLICT(delivery_id) DO NOTHING",
            rusqlite::params![
                delivery_id,
                request_id,
                terminal_outcome.as_str(),
                terminal_status_code,
                created_at,
                CONFIGURED_WEBHOOK_TARGET,
                WebhookDeliveryState::Pending.code(),
                created_at,
                created_at,
                created_at,
                max_attempts,
            ],
        )
        .map_err(|error| LogStoreError::InsertFailed(error.to_string()))?;
    let record = select_webhook_delivery(transaction, delivery_id)?
        .ok_or_else(|| LogStoreError::QueryFailed("webhook delivery insert disappeared".into()))?;
    if record.request_id.as_deref() != Some(request_id)
        || record.terminal_outcome != terminal_outcome
        || record.terminal_status_code != terminal_status_code
        || record.max_attempts != max_attempts
    {
        return Err(LogStoreError::InvalidQuery(
            "webhook delivery_id conflicts with immutable delivery intent".to_string(),
        ));
    }
    if inserted == 1 {
        Ok(WebhookDeliveryInsertOutcome::Created(record))
    } else {
        Ok(WebhookDeliveryInsertOutcome::Existing(record))
    }
}

fn terminal_status_code_for_request(
    transaction: &Transaction<'_>,
    request_id: &str,
) -> Result<Option<u16>, LogStoreError> {
    let status_code = transaction
        .query_row(
            "SELECT status_code FROM summaries WHERE request_id = ?",
            [request_id],
            |row| row.get::<_, Option<i64>>(0),
        )
        .map_err(LogStoreError::Sqlite)?;
    let status_code = status_code
        .map(u16::try_from)
        .transpose()
        .map_err(|_| LogStoreError::QueryFailed("terminal status code is invalid".to_string()))?;
    validate_optional_http_status(status_code, "terminal status code")?;
    Ok(status_code)
}

/// Closed event classification stored alongside the serialized payload.
///
/// The payload remains the compatibility surface for event readers, but
/// terminal semantics are deliberately derived once and stored as typed data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LifecycleEventType {
    Admitted,
    RouteSelected,
    AttemptStarted,
    AttemptCompleted,
    AttemptFailed,
    BackendStreamFirstItem,
    StreamStarted,
    StreamChunk,
    StreamCompleted,
    UsageRecorded,
    StreamError,
    AuditError,
    Completed,
    Failed,
    Rejected,
    Cancelled,
    Dropped,
    Unknown,
}

impl LifecycleEventType {
    fn from_payload(payload_json: &str) -> Self {
        serde_json::from_str::<serde_json::Value>(payload_json)
            .ok()
            .and_then(|value| {
                value
                    .get("type")
                    .and_then(serde_json::Value::as_str)
                    .map(Self::from_code)
            })
            .unwrap_or(Self::Unknown)
    }

    fn from_code(code: &str) -> Self {
        match code {
            "admitted" => Self::Admitted,
            "route_selected" => Self::RouteSelected,
            "attempt_started" => Self::AttemptStarted,
            "attempt_completed" => Self::AttemptCompleted,
            "attempt_failed" => Self::AttemptFailed,
            "backend_stream_first_item" => Self::BackendStreamFirstItem,
            "stream_started" => Self::StreamStarted,
            "stream_chunk" => Self::StreamChunk,
            "stream_completed" => Self::StreamCompleted,
            "usage_recorded" => Self::UsageRecorded,
            "stream_error" => Self::StreamError,
            "audit_error" => Self::AuditError,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "rejected" => Self::Rejected,
            "cancelled" => Self::Cancelled,
            "dropped" => Self::Dropped,
            _ => Self::Unknown,
        }
    }

    const fn code(self) -> &'static str {
        match self {
            Self::Admitted => "admitted",
            Self::RouteSelected => "route_selected",
            Self::AttemptStarted => "attempt_started",
            Self::AttemptCompleted => "attempt_completed",
            Self::AttemptFailed => "attempt_failed",
            Self::BackendStreamFirstItem => "backend_stream_first_item",
            Self::StreamStarted => "stream_started",
            Self::StreamChunk => "stream_chunk",
            Self::StreamCompleted => "stream_completed",
            Self::UsageRecorded => "usage_recorded",
            Self::StreamError => "stream_error",
            Self::AuditError => "audit_error",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Rejected => "rejected",
            Self::Cancelled => "cancelled",
            Self::Dropped => "dropped",
            Self::Unknown => "unknown",
        }
    }

    const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Rejected | Self::Cancelled | Self::Dropped
        )
    }

    fn terminal_from_status(status: &str) -> Option<Self> {
        let event_type = Self::from_code(status);
        event_type.is_terminal().then_some(event_type)
    }
}

/// Check if a request already has any typed terminal event.
fn check_existing_terminal(
    cxn: &rusqlite::Connection,
    request_id: &str,
) -> Result<bool, LogStoreError> {
    let count: i64 = cxn
        .query_row(
            "SELECT COUNT(*) FROM lifecycle_events WHERE request_id = ? AND is_terminal = 1",
            rusqlite::params![request_id],
            |row| row.get(0),
        )
        .map_err(LogStoreError::Sqlite)?;
    Ok(count > 0)
}

// ─── LogStore repository methods ──────────────

impl LogStore {
    // ════════════════════════════
    //  Summaries
    // ════════════════════════════

    /// Insert a new summary. Returns AlreadyExists on duplicate PK.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_summary(
        &self,
        request_id: &str,
        model: Option<&str>,
        route: Option<&str>,
        provider: Option<&str>,
        engine: Option<&str>,
        occurred_at: &str,
        tenant_id: Option<&str>,
        account_id: Option<&str>,
        user_id: Option<&str>,
    ) -> Result<(), LogStoreError> {
        let occurred_at = canonical_persisted_timestamp(occurred_at)?;
        let conn = self.conn();
        match conn.execute(
            "INSERT INTO summaries (request_id, created_at, model, route, provider, engine, tenant_id, account_id, user_id) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rusqlite::params![request_id, occurred_at, model, route, provider, engine, tenant_id, account_id, user_id],
        ) {
            Ok(_) => Ok(()),
            Err(ref e) => match map_insert_constraint_error(e, format!("summary {request_id}")) {
                Some(error) => Err(error),
                None => Err(LogStoreError::InsertFailed(e.to_string())),
            },
        }
    }

    /// Get a summary by request_id. Returns None if not found (no-op style).
    pub fn get_summary(&self, request_id: &str) -> Result<Option<SummaryRow>, LogStoreError> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT request_id, state, created_at, terminal_at, route, model, provider, engine, \
             status_code, error_msg, tenant_id, account_id, user_id \
             FROM summaries WHERE request_id = ?",
        ).map_err(LogStoreError::Sqlite)?;

        let row_fn = |row: &rusqlite::Row<'_>| -> rusqlite::Result<SummaryRow> {
            Ok(SummaryRow {
                request_id: row.get(0)?,
                state: row.get(1)?,
                created_at: row.get(2)?,
                terminal_at: row.get(3)?,
                route: row.get(4)?,
                model: row.get(5)?,
                provider: row.get(6)?,
                engine: row.get(7)?,
                status_code: row.get(8)?,
                error_msg: row.get(9)?,
                tenant_id: row.get(10)?,
                account_id: row.get(11)?,
                user_id: row.get(12)?,
            })
        };

        match stmt.query_row(rusqlite::params![request_id], row_fn) {
            Ok(row) => Ok(Some(row)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(LogStoreError::QueryFailed(e.to_string())),
        }
    }

    /// Paginated summary listing keyed on (created_at, request_id).
    pub fn list_summaries(
        &self,
        limit: usize,
        after_cursor: Option<&str>,
    ) -> Result<Page<SummaryRow>, LogStoreError> {
        let conn = self.conn();

        let row_fn = |row: &rusqlite::Row<'_>| -> rusqlite::Result<SummaryRow> {
            Ok(SummaryRow {
                request_id: row.get(0)?,
                state: row.get(1)?,
                created_at: row.get(2)?,
                terminal_at: row.get(3)?,
                route: row.get(4)?,
                model: row.get(5)?,
                provider: row.get(6)?,
                engine: row.get(7)?,
                status_code: row.get(8)?,
                error_msg: row.get(9)?,
                tenant_id: row.get(10)?,
                account_id: row.get(11)?,
                user_id: row.get(12)?,
            })
        };

        list_opaque_keyset_page(
            &conn,
            OpaqueKeysetPage {
                table: "summaries",
                columns: "request_id, state, created_at, terminal_at, route, model, provider, engine, \
                          status_code, error_msg, tenant_id, account_id, user_id",
                timestamp_column: "created_at",
                id_column: "request_id",
            },
            limit,
            after_cursor,
            row_fn,
            |row| (&row.created_at, &row.request_id),
        )
    }

    /// Update summary terminal state. No-op if request_id not found (returns 0 rows affected).
    pub fn update_summary_terminal(
        &self,
        request_id: &str,
        terminal_status: &str,
        terminal_at: &str,
    ) -> Result<usize, LogStoreError> {
        let terminal_at = canonical_persisted_timestamp(terminal_at)?;
        let conn = self.conn();
        conn.execute(
            "UPDATE summaries SET state = ?, terminal_at = ? WHERE request_id = ?",
            rusqlite::params![terminal_status, terminal_at, request_id],
        )
        .map_err(LogStoreError::Sqlite)
    }

    // ════════════════════════════
    //  Lifecycle Events
    // ════════════════════════════

    /// Insert a lifecycle event. Caller serializes the payload to JSON before calling.
    pub fn insert_lifecycle_event(
        &self,
        request_id: &str,
        event_id: &str,
        payload_json: &str,
        occurred_at: &str,
    ) -> Result<(), LogStoreError> {
        let occurred_at = canonical_persisted_timestamp(occurred_at)?;
        let conn = self.conn();
        let event_type = LifecycleEventType::from_payload(payload_json);

        // Pre-check for terminal duplicates.
        if event_type.is_terminal() && check_existing_terminal(&conn, request_id)? {
            return Err(LogStoreError::DuplicateTerminalEvent {
                summary_id: request_id.to_string(),
                event_type: event_type.code().to_string(),
            });
        }

        match conn.execute(
            "INSERT INTO lifecycle_events (event_id, request_id, occurred_at, payload_json, event_type, is_terminal) \
             VALUES (?, ?, ?, ?, ?, ?)",
            rusqlite::params![
                event_id,
                request_id,
                occurred_at,
                payload_json,
                event_type.code(),
                i64::from(event_type.is_terminal()),
            ],
        ) {
            Ok(_) => Ok(()),
            Err(ref e) if is_foreign_key_constraint_error(e) => {
                Err(LogStoreError::ForeignKeyViolation {
                    entity: "lifecycle_event".to_string(),
                })
            }
            Err(ref e) if is_unique_constraint_error(e) => {
                // Could be UNIQUE(request_id, event_id) or the partial terminal index.
                if event_type.is_terminal() {
                    return Err(LogStoreError::DuplicateTerminalEvent {
                        summary_id: request_id.to_string(),
                        event_type: event_type.code().to_string(),
                    });
                }
                Err(LogStoreError::AlreadyExists {
                    entity: "lifecycle_event".to_string(),
                })
            }
            Err(e) => Err(LogStoreError::InsertFailed(e.to_string())),
        }
    }

    /// Atomic write of a terminal event + summary state update. Both succeed or neither does.
    pub fn write_terminal_event(
        &self,
        request_id: &str,
        event_id: &str,
        payload_json: &str,
        terminal_status: &str,
        terminal_status_code: Option<u16>,
        occurred_at: &str,
    ) -> Result<(), LogStoreError> {
        let event_type =
            LifecycleEventType::terminal_from_status(terminal_status).ok_or_else(|| {
                LogStoreError::InvalidQuery(
                    "terminal status must be a terminal lifecycle type".to_string(),
                )
            })?;
        let occurred_at = canonical_persisted_timestamp(occurred_at)?;
        self.txn(|transaction| {
            write_terminal_event_in_transaction(
                transaction,
                TerminalEventWrite {
                    request_id,
                    event_id,
                    payload_json,
                    terminal_status,
                    terminal_status_code,
                    occurred_at: &occurred_at,
                    event_type,
                },
            )
        })
    }

    /// Atomically write a terminal event, update its summary, and create the
    /// deterministic durable webhook outbox record. All three writes succeed
    /// or all three roll back.
    #[allow(clippy::too_many_arguments)]
    pub fn write_terminal_event_with_webhook(
        &self,
        request_id: &str,
        event_id: &str,
        payload_json: &str,
        terminal_status: &str,
        terminal_status_code: Option<u16>,
        occurred_at: &str,
        delivery_id: &str,
        max_attempts: u32,
    ) -> Result<WebhookDeliveryInsertOutcome, LogStoreError> {
        let event_type =
            LifecycleEventType::terminal_from_status(terminal_status).ok_or_else(|| {
                LogStoreError::InvalidQuery(
                    "terminal status must be a terminal lifecycle type".to_string(),
                )
            })?;
        let occurred_at = canonical_webhook_timestamp(occurred_at, "created_at")?;
        validate_webhook_identifier(delivery_id, "delivery_id")?;
        validate_webhook_identifier(request_id, "request_id")?;
        validate_webhook_max_attempts(max_attempts)?;

        self.txn(|transaction| {
            write_terminal_event_in_transaction(
                transaction,
                TerminalEventWrite {
                    request_id,
                    event_id,
                    payload_json,
                    terminal_status,
                    terminal_status_code,
                    occurred_at: &occurred_at,
                    event_type,
                },
            )?;
            enqueue_webhook_delivery_in_transaction(
                transaction,
                delivery_id,
                request_id,
                terminal_status_code,
                &occurred_at,
                max_attempts,
            )
        })
    }

    /// Check if a summary already has any terminal event.
    pub fn has_terminal_event(&self, request_id: &str) -> Result<bool, LogStoreError> {
        let conn = self.conn();
        check_existing_terminal(&conn, request_id)
    }

    /// Paginated lifecycle event listing keyed on (occurred_at, event_id).
    pub fn list_lifecycle_events(
        &self,
        limit: usize,
        after_cursor: Option<&str>,
    ) -> Result<Page<LifecycleEventRow>, LogStoreError> {
        let conn = self.conn();

        let row_fn = |row: &rusqlite::Row<'_>| -> rusqlite::Result<LifecycleEventRow> {
            Ok(LifecycleEventRow {
                event_id: row.get(0)?,
                request_id: row.get(1)?,
                occurred_at: row.get(2)?,
            })
        };

        list_opaque_keyset_page(
            &conn,
            OpaqueKeysetPage {
                table: "lifecycle_events",
                columns: "event_id, request_id, occurred_at",
                timestamp_column: "occurred_at",
                id_column: "event_id",
            },
            limit,
            after_cursor,
            row_fn,
            |row| (&row.occurred_at, &row.event_id),
        )
    }

    /// List all lifecycle events for a specific summary, ordered chronologically.
    pub fn list_events_for_summary(
        &self,
        request_id: &str,
    ) -> Result<Vec<LifecycleEventRow>, LogStoreError> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare(
                "SELECT event_id, request_id, occurred_at FROM lifecycle_events \
             WHERE request_id = ? ORDER BY occurred_at ASC",
            )
            .map_err(LogStoreError::Sqlite)?;

        let rows_iter = stmt
            .query_map(rusqlite::params![request_id], |row: &rusqlite::Row<'_>| {
                Ok(LifecycleEventRow {
                    event_id: row.get(0)?,
                    request_id: row.get(1)?,
                    occurred_at: row.get(2)?,
                })
            })
            .map_err(LogStoreError::Sqlite)?;

        let mut items = Vec::new();
        for result in rows_iter {
            match result {
                Ok(item) => items.push(item),
                Err(e) => return Err(LogStoreError::QueryFailed(e.to_string())),
            }
        }
        Ok(items)
    }

    // ════════════════════════════
    //  Proxy Records
    // ════════════════════════════

    #[allow(clippy::too_many_arguments)]
    pub fn insert_proxy_record(
        &self,
        attempt_id: &str,
        request_id: &str,
        occurred_at: &str,
        target: &str,
        provider: Option<&str>,
        engine: Option<&str>,
        started_at: Option<&str>,
        completed_at: Option<&str>,
        status_code: Option<i64>,
        error_msg: Option<&str>,
    ) -> Result<(), LogStoreError> {
        let occurred_at = canonical_persisted_timestamp(occurred_at)?;
        let started_at = canonical_optional_persisted_timestamp(started_at)?;
        let completed_at = canonical_optional_persisted_timestamp(completed_at)?;
        let conn = self.conn();
        match conn.execute(
            "INSERT INTO proxy_records (attempt_id, request_id, occurred_at, target, provider, engine, started_at, completed_at, status_code, error_msg) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rusqlite::params![attempt_id, request_id, occurred_at, target, provider, engine, started_at, completed_at, status_code, error_msg],
        ) {
            Ok(_) => Ok(()),
            Err(ref e) => {
                match map_insert_constraint_error(e, format!("proxy_record {attempt_id}")) {
                    Some(error) => Err(error),
                    None => Err(LogStoreError::InsertFailed(e.to_string())),
                }
            }
        }
    }

    // ════════════════════════════
    //  Webhook Deliveries
    // ════════════════════════════

    /// Create one idempotent, scoped terminal webhook record in the durable
    /// pending state. The caller owns deterministic delivery-id derivation;
    /// re-enqueueing that same ID after a restart returns the existing record
    /// without creating a duplicate terminal delivery.
    pub fn enqueue_webhook_delivery(
        &self,
        delivery_id: &str,
        request_id: &str,
        created_at: &str,
        max_attempts: u32,
    ) -> Result<WebhookDeliveryInsertOutcome, LogStoreError> {
        validate_webhook_identifier(delivery_id, "delivery_id")?;
        validate_webhook_identifier(request_id, "request_id")?;
        let created_at = canonical_webhook_timestamp(created_at, "created_at")?;
        validate_webhook_max_attempts(max_attempts)?;

        self.txn(|transaction| {
            let terminal_status_code = terminal_status_code_for_request(transaction, request_id)?;
            enqueue_webhook_delivery_in_transaction(
                transaction,
                delivery_id,
                request_id,
                terminal_status_code,
                &created_at,
                max_attempts,
            )
        })
    }

    /// Atomically claim the oldest eligible pending/retry/manual retry record.
    /// A stale in-flight lease is reclaimable after restart. The incremented
    /// claim generation fences completion/retry writes from displaced workers.
    pub fn claim_next_webhook_delivery(
        &self,
        now: &str,
        lease_expires_at: &str,
    ) -> Result<Option<WebhookDeliveryRecord>, LogStoreError> {
        let now = canonical_webhook_timestamp(now, "claim timestamp")?;
        let lease_expires_at = canonical_webhook_timestamp(lease_expires_at, "lease expiration")?;

        self.txn(|tx| loop {
            let candidate: Option<(String, String, i64, i64)> = tx
                .query_row(
                    "SELECT delivery_id, state, attempt_number, max_attempts FROM webhook_deliveries \
                     WHERE (state IN ('pending', 'retry', 'manual_retry') \
                            AND (next_attempt_at IS NULL OR next_attempt_at <= ?)) \
                        OR (state = 'in_flight' AND lease_expires_at <= ?) \
                     ORDER BY COALESCE(next_attempt_at, created_at), created_at, delivery_id LIMIT 1",
                    rusqlite::params![now, now],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
                )
                .optional()
                .map_err(|error| LogStoreError::QueryFailed(error.to_string()))?;
            let Some((delivery_id, state, attempt_number, max_attempts)) = candidate else {
                return Ok(None);
            };
            if state == WebhookDeliveryState::InFlight.code() && attempt_number >= max_attempts {
                tx.execute(
                    "UPDATE webhook_deliveries \
                     SET state = 'dead_letter', updated_at = ?, lease_expires_at = NULL, \
                         next_attempt_at = NULL, last_error_code = 'transport' \
                     WHERE delivery_id = ? AND state = 'in_flight' \
                       AND lease_expires_at <= ? AND attempt_number >= max_attempts",
                    rusqlite::params![now, delivery_id, now],
                )
                .map_err(|error| LogStoreError::InsertFailed(error.to_string()))?;
                // A stale exhausted lease is a durable state transition, not
                // an idle scheduler result. Keep scanning in this transaction
                // so another eligible record can be claimed immediately and
                // shutdown drains do not stop after only this cleanup.
                continue;
            }
            let changed = tx
                .execute(
                    "UPDATE webhook_deliveries \
                     SET state = 'in_flight', attempt_number = attempt_number + 1, \
                         updated_at = ?, lease_expires_at = ?, next_attempt_at = NULL, \
                         claim_generation = claim_generation + 1, last_error_code = NULL \
                     WHERE delivery_id = ? AND (\
                        (state IN ('pending', 'retry', 'manual_retry') \
                         AND (next_attempt_at IS NULL OR next_attempt_at <= ?)) \
                        OR (state = 'in_flight' AND lease_expires_at <= ? \
                            AND attempt_number < max_attempts))",
                    rusqlite::params![now, lease_expires_at, delivery_id, now, now],
                )
                .map_err(|error| LogStoreError::InsertFailed(error.to_string()))?;
            if changed == 0 {
                continue;
            }
            return select_webhook_delivery(tx, &delivery_id);
        })
    }

    /// Complete an in-flight attempt only when its fencing generation still
    /// belongs to this worker. A duplicate/stale completion is a harmless
    /// false result, never a second delivery state transition.
    pub fn complete_webhook_delivery(
        &self,
        delivery_id: &str,
        claim_generation: u64,
        completed_at: &str,
        response_status_code: u16,
    ) -> Result<bool, LogStoreError> {
        validate_webhook_identifier(delivery_id, "delivery_id")?;
        let completed_at = canonical_webhook_timestamp(completed_at, "completion timestamp")?;
        if !(200..=299).contains(&response_status_code) {
            return Err(LogStoreError::InvalidQuery(
                "webhook success status must be between 200 and 299".to_string(),
            ));
        }
        let conn = self.conn();
        conn.execute(
            "UPDATE webhook_deliveries \
             SET state = 'succeeded', updated_at = ?, lease_expires_at = NULL, \
                 next_attempt_at = NULL, status_code = ?, response_body = NULL, \
                 error_msg = NULL, last_error_code = NULL \
             WHERE delivery_id = ? AND state = 'in_flight' AND claim_generation = ?",
            rusqlite::params![
                completed_at,
                response_status_code,
                delivery_id,
                claim_generation
            ],
        )
        .map(|changed| changed == 1)
        .map_err(|error| LogStoreError::InsertFailed(error.to_string()))
    }

    /// Record a bounded failure code and either schedule the next attempt or
    /// atomically enter dead-letter after the configured maximum.
    pub fn retry_or_dead_letter_webhook_delivery(
        &self,
        delivery_id: &str,
        claim_generation: u64,
        updated_at: &str,
        next_attempt_at: &str,
        error_code: WebhookDeliveryErrorCode,
        response_status_code: Option<u16>,
    ) -> Result<Option<WebhookRetryOutcome>, LogStoreError> {
        validate_webhook_identifier(delivery_id, "delivery_id")?;
        let updated_at = canonical_webhook_timestamp(updated_at, "update timestamp")?;
        let next_attempt_at =
            canonical_webhook_timestamp(next_attempt_at, "next attempt timestamp")?;
        validate_optional_http_status(response_status_code, "receiver response status code")?;

        self.txn(|tx| {
            let Some(record) = select_webhook_delivery(tx, delivery_id)? else {
                return Ok(None);
            };
            if record.state != WebhookDeliveryState::InFlight
                || record.claim_generation != claim_generation
            {
                return Ok(None);
            }
            let (state, retry_at, outcome) = if record.attempt_number >= record.max_attempts {
                (
                    WebhookDeliveryState::DeadLetter,
                    None,
                    WebhookRetryOutcome::DeadLettered,
                )
            } else {
                (
                    WebhookDeliveryState::Retry,
                    Some(next_attempt_at.as_str()),
                    WebhookRetryOutcome::RetryScheduled,
                )
            };
            tx.execute(
                "UPDATE webhook_deliveries \
                 SET state = ?, updated_at = ?, lease_expires_at = NULL, next_attempt_at = ?, \
                     status_code = ?, response_body = NULL, error_msg = NULL, last_error_code = ? \
                 WHERE delivery_id = ? AND state = 'in_flight' AND claim_generation = ?",
                rusqlite::params![
                    state.code(),
                    updated_at,
                    retry_at,
                    response_status_code,
                    error_code.code(),
                    delivery_id,
                    claim_generation,
                ],
            )
            .map_err(|error| LogStoreError::InsertFailed(error.to_string()))?;
            Ok(Some(outcome))
        })
    }

    /// Explicit operator-driven retry opens a new bounded attempt cycle from
    /// a dead-letter record. It leaves the terminal request untouched and
    /// records the distinct manual-retry state for a later audit worker.
    pub fn manually_retry_webhook_delivery(
        &self,
        delivery_id: &str,
        requested_at: &str,
    ) -> Result<WebhookManualRetryOutcome, LogStoreError> {
        validate_webhook_identifier(delivery_id, "delivery_id")?;
        let requested_at = canonical_webhook_timestamp(requested_at, "manual retry timestamp")?;
        self.txn(|tx| {
            let state = tx
                .query_row(
                    "SELECT state FROM webhook_deliveries WHERE delivery_id = ?1",
                    [delivery_id],
                    |row| row.get::<_, String>(0),
                )
                .optional()
                .map_err(LogStoreError::Sqlite)?;
            let Some(state) = state else {
                return Ok(WebhookManualRetryOutcome::NotFound);
            };
            if state == WebhookDeliveryState::DeadLetter.code() {
                tx.execute(
                    "UPDATE webhook_deliveries \
                     SET state = 'manual_retry', attempt_number = 0, updated_at = ?, \
                         next_attempt_at = ?, lease_expires_at = NULL, response_body = NULL, \
                         error_msg = NULL, last_error_code = NULL \
                     WHERE delivery_id = ? AND state = 'dead_letter'",
                    rusqlite::params![requested_at, requested_at, delivery_id],
                )
                .map_err(|error| LogStoreError::InsertFailed(error.to_string()))?;
                return Ok(WebhookManualRetryOutcome::Scheduled);
            }

            let outcome = match state.as_str() {
                "pending" | "retry" | "manual_retry" | "in_flight" => {
                    WebhookManualRetryOutcome::AlreadyScheduled
                }
                "succeeded" => WebhookManualRetryOutcome::NotRetryable,
                _ => WebhookManualRetryOutcome::NotRetryable,
            };
            Ok(outcome)
        })
    }

    /// Load one delivery record for restart/resumption and focused tests.
    pub fn webhook_delivery(
        &self,
        delivery_id: &str,
    ) -> Result<Option<WebhookDeliveryRecord>, LogStoreError> {
        validate_webhook_identifier(delivery_id, "delivery_id")?;
        select_webhook_delivery(&self.conn(), delivery_id)
    }

    /// Compatibility helper for retention tests. It deliberately discards
    /// the old URL/body/error inputs at the durable boundary; new code must
    /// use the typed state-machine API above.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_webhook_delivery(
        &self,
        delivery_id: &str,
        request_id: Option<&str>,
        occurred_at: &str,
        attempt_number: i64,
        status_code: Option<i64>,
    ) -> Result<(), LogStoreError> {
        validate_webhook_identifier(delivery_id, "delivery_id")?;
        let occurred_at = canonical_webhook_timestamp(occurred_at, "occurred_at")?;
        let attempt_number = attempt_number.max(1);
        let state = if status_code.is_some_and(|status| (200..=299).contains(&status)) {
            WebhookDeliveryState::Succeeded
        } else {
            WebhookDeliveryState::DeadLetter
        };
        let conn = self.conn();
        match conn.execute(
            "INSERT INTO webhook_deliveries \
                (delivery_id, request_id, terminal_outcome, occurred_at, target_url, attempt_number, status_code, response_body, error_msg, \
                 state, created_at, updated_at, max_attempts) \
             VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)",
            rusqlite::params![
                delivery_id,
                request_id,
                WebhookTerminalOutcome::Completed.as_str(),
                occurred_at,
                CONFIGURED_WEBHOOK_TARGET,
                attempt_number,
                status_code,
                state.code(),
                occurred_at,
                occurred_at,
                attempt_number,
            ],
        ) {
            Ok(_) => Ok(()),
            Err(ref e) => {
                match map_insert_constraint_error(e, format!("webhook_delivery {delivery_id}")) {
                    Some(error) => Err(error),
                    None => Err(LogStoreError::InsertFailed(e.to_string())),
                }
            }
        }
    }

    // ════════════════════════════
    //  Cleanup Runs
    // ════════════════════════════

    pub fn insert_cleanup_run(
        &self,
        run_id: &str,
        occurred_at: &str,
        policy_name: &str,
        cutoff_before: &str,
        deleted_count: i64,
        duration_ms: Option<i64>,
    ) -> Result<(), LogStoreError> {
        let occurred_at = canonical_persisted_timestamp(occurred_at)?;
        let cutoff_before = canonical_timestamp_metadata(cutoff_before);
        let conn = self.conn();
        match conn.execute(
            "INSERT INTO cleanup_runs (run_id, occurred_at, policy_name, cutoff_before, deleted_count, duration_ms) VALUES (?, ?, ?, ?, ?, ?)",
            rusqlite::params![run_id, occurred_at, policy_name, cutoff_before, deleted_count, duration_ms],
        ) {
            Ok(_) => Ok(()),
            Err(ref e) => match map_insert_constraint_error(e, format!("cleanup_run {run_id}")) {
                Some(error) => Err(error),
                None => Err(LogStoreError::InsertFailed(e.to_string())),
            },
        }
    }

    // ════════════════════════════
    //  Aggregation Queries
    // ════════════════════════════

    /// Count summaries grouped by state.
    pub fn count_summaries_by_status(&self) -> Result<Vec<(String, i64)>, LogStoreError> {
        let conn = self.conn();
        let mut stmt = conn
            .prepare("SELECT state, COUNT(*) FROM summaries GROUP BY state")
            .map_err(LogStoreError::Sqlite)?;

        let rows_iter = stmt
            .query_map([], |row: &rusqlite::Row<'_>| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .map_err(LogStoreError::Sqlite)?;

        let mut items = Vec::new();
        for result in rows_iter {
            match result {
                Ok(item) => items.push(item),
                Err(e) => return Err(LogStoreError::QueryFailed(e.to_string())),
            }
        }
        Ok(items)
    }

    // ════════════════════════════
    //  Test Helpers
    // ════════════════════════════

    #[cfg(test)]
    pub fn count_table(&self, table_name: &str) -> Result<i64, LogStoreError> {
        let conn = self.conn();
        let sql = format!("SELECT COUNT(*) FROM {}", table_name);
        conn.query_row(&sql, [], |row| row.get::<_, i64>(0))
            .map_err(LogStoreError::Sqlite)
    }

    #[cfg(test)]
    pub fn clear_table(&self, table_name: &str) -> Result<usize, LogStoreError> {
        let conn = self.conn();
        let sql = format!("DELETE FROM {}", table_name);
        conn.execute(&sql, []).map_err(LogStoreError::Sqlite)
    }
}
