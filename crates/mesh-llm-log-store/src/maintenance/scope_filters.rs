//! Durable, safe-to-display cleanup selection predicates.

use serde::{Deserialize, Serialize};

use super::MaintenanceTimestamp;
use crate::LogStoreError;

const MAX_SCOPE_FILTER_BYTES: usize = 128;

/// Durable-only terminal-summary predicates for a bounded cleanup snapshot.
/// Values are normalized before persistence and safe to return in an operator
/// receipt; active request state is deliberately not representable.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CleanupFilters {
    from: Option<String>,
    to: Option<String>,
    route: Option<String>,
    model: Option<String>,
    provider: Option<String>,
    engine: Option<String>,
    outcome: Option<CleanupOutcome>,
}

impl CleanupFilters {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        from: Option<MaintenanceTimestamp>,
        to: Option<MaintenanceTimestamp>,
        route: Option<String>,
        model: Option<String>,
        provider: Option<String>,
        engine: Option<String>,
        outcome: Option<CleanupOutcome>,
    ) -> Result<Self, LogStoreError> {
        let filters = Self {
            from: from.map(|value| value.0),
            to: to.map(|value| value.0),
            route: normalize_scope_filter(route, "route")?,
            model: normalize_scope_filter(model, "model")?,
            provider: normalize_scope_filter(provider, "provider")?,
            engine: normalize_scope_filter(engine, "engine")?,
            outcome,
        };
        if filters.from > filters.to && filters.to.is_some() {
            return Err(LogStoreError::MaintenanceScopeInvalid { field: "from" });
        }
        Ok(filters)
    }

    pub fn from(&self) -> Option<&str> {
        self.from.as_deref()
    }
    pub fn to(&self) -> Option<&str> {
        self.to.as_deref()
    }
    pub fn route(&self) -> Option<&str> {
        self.route.as_deref()
    }
    pub fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }
    pub fn provider(&self) -> Option<&str> {
        self.provider.as_deref()
    }
    pub fn engine(&self) -> Option<&str> {
        self.engine.as_deref()
    }
    pub const fn outcome(&self) -> Option<CleanupOutcome> {
        self.outcome
    }
}

/// Terminal request states eligible for durable cleanup.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CleanupOutcome {
    Completed,
    Failed,
    Rejected,
    Cancelled,
    Dropped,
}

impl CleanupOutcome {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Rejected => "rejected",
            Self::Cancelled => "cancelled",
            Self::Dropped => "dropped",
        }
    }
}

impl TryFrom<&str> for CleanupOutcome {
    type Error = LogStoreError;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            "rejected" => Ok(Self::Rejected),
            "cancelled" => Ok(Self::Cancelled),
            "dropped" => Ok(Self::Dropped),
            _ => Err(LogStoreError::MaintenanceScopeInvalid { field: "outcome" }),
        }
    }
}

fn normalize_scope_filter(
    value: Option<String>,
    field: &'static str,
) -> Result<Option<String>, LogStoreError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = value.trim();
    let path_or_secret_shaped = value.starts_with('/')
        || value.starts_with("~/")
        || value.as_bytes().get(1) == Some(&b':')
        || value
            .chars()
            .any(|character| matches!(character, '\\' | '?' | '#' | '=' | '&'))
        || value.contains("://");
    if value.is_empty()
        || value.len() > MAX_SCOPE_FILTER_BYTES
        || value.chars().any(char::is_control)
        || path_or_secret_shaped
    {
        return Err(LogStoreError::MaintenanceScopeInvalid { field });
    }
    Ok(Some(value.to_owned()))
}
