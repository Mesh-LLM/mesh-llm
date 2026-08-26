//! Durable, typed, bounded queries for the local logging ledger.
//!
//! This module deliberately stops at store-level records. Trusted-local HTTP
//! parsing, active-registry merging, and artifact-content authorization belong
//! to later host/API layers.

mod records;
mod related;
mod requests;

pub use records::{RequestRecord, RequestRecordWithCaller};

use chrono::DateTime;

use crate::LogStoreError;

/// Maximum records a single durable query may ask SQLite to materialize.
pub const MAX_QUERY_LIMIT: usize = 100;
const MAX_FILTER_BYTES: usize = 512;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuerySort {
    Ascending,
    Descending,
}

impl QuerySort {
    pub(crate) const fn sql_order(self) -> &'static str {
        match self {
            Self::Ascending => "ASC",
            Self::Descending => "DESC",
        }
    }

    pub(crate) const fn cursor_operator(self) -> &'static str {
        match self {
            Self::Ascending => ">",
            Self::Descending => "<",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestOutcome {
    Active,
    Completed,
    Failed,
    Rejected,
    Cancelled,
    Dropped,
}

impl RequestOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Rejected => "rejected",
            Self::Cancelled => "cancelled",
            Self::Dropped => "dropped",
        }
    }
}

#[derive(Clone, Debug)]
pub struct RequestQuery {
    pub limit: usize,
    pub cursor: Option<String>,
    pub from: Option<String>,
    pub to: Option<String>,
    pub route: Option<String>,
    pub exclude_route: Option<String>,
    pub exclude_route_prefix: Option<String>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub engine: Option<String>,
    pub status_code: Option<u16>,
    pub outcome: Option<RequestOutcome>,
    pub sort: QuerySort,
}

impl RequestQuery {
    pub fn validate(&self) -> Result<(), LogStoreError> {
        validate_limit(self.limit)?;
        validate_time_range(self.from.as_deref(), self.to.as_deref())?;
        for value in [
            &self.route,
            &self.exclude_route,
            &self.exclude_route_prefix,
            &self.model,
            &self.provider,
            &self.engine,
        ] {
            validate_filter(value.as_deref())?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct PageQuery {
    pub limit: usize,
    pub cursor: Option<String>,
    pub sort: QuerySort,
}

impl PageQuery {
    pub fn validate(&self) -> Result<(), LogStoreError> {
        validate_limit(self.limit)
    }
}

#[derive(Clone, Debug)]
pub struct ProxyQuery {
    pub page: PageQuery,
    pub request_id: Option<String>,
    pub provider: Option<String>,
    pub engine: Option<String>,
    pub status_code: Option<u16>,
}

impl ProxyQuery {
    pub fn validate(&self) -> Result<(), LogStoreError> {
        self.page.validate()?;
        for value in [&self.request_id, &self.provider, &self.engine] {
            validate_filter(value.as_deref())?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EventRecord {
    pub event_id: String,
    pub request_id: String,
    pub occurred_at: String,
    pub payload_json: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactRecord {
    pub artifact_id: String,
    pub request_id: String,
    pub occurred_at: String,
    pub kind: String,
    pub media_kind: Option<String>,
    pub checksum: Option<String>,
    pub bytes: i64,
    pub version: i32,
    pub redacted: bool,
    pub truncated: bool,
    pub missing: bool,
    pub corrupt: bool,
    /// Closed reason for intentionally omitted content. This is independent
    /// of missing/corrupt file health.
    pub unavailable_reason: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProxyRecord {
    pub attempt_id: String,
    pub request_id: String,
    pub occurred_at: String,
    pub target: String,
    pub provider: Option<String>,
    pub engine: Option<String>,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
    pub status_code: Option<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QueryPage<T> {
    pub items: Vec<T>,
    pub next_cursor: Option<String>,
}

pub(crate) fn validate_limit(limit: usize) -> Result<(), LogStoreError> {
    if !(1..=MAX_QUERY_LIMIT).contains(&limit) {
        return Err(LogStoreError::InvalidQuery(format!(
            "limit must be between 1 and {MAX_QUERY_LIMIT}"
        )));
    }
    Ok(())
}

fn validate_filter(value: Option<&str>) -> Result<(), LogStoreError> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.is_empty() || value.len() > MAX_FILTER_BYTES || value.contains('\0') {
        return Err(LogStoreError::InvalidQuery(
            "filter must be non-empty and at most 512 bytes".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_identifier(value: &str) -> Result<(), LogStoreError> {
    validate_filter(Some(value))
}

fn validate_time_range(from: Option<&str>, to: Option<&str>) -> Result<(), LogStoreError> {
    let from = from.map(parse_timestamp).transpose()?;
    let to = to.map(parse_timestamp).transpose()?;
    if let (Some(from), Some(to)) = (from, to)
        && from > to
    {
        return Err(LogStoreError::InvalidQuery(
            "from must not be later than to".to_string(),
        ));
    }
    Ok(())
}

fn parse_timestamp(value: &str) -> Result<DateTime<chrono::FixedOffset>, LogStoreError> {
    DateTime::parse_from_rfc3339(value).map_err(|_| {
        LogStoreError::InvalidQuery("time bounds must be RFC3339 timestamps".to_string())
    })
}
