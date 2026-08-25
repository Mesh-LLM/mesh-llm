use std::collections::BTreeMap;

pub const DEFAULT_AUDIT_ENTRY_LIMIT: usize = 50;
pub const MAX_AUDIT_ENTRY_LIMIT: usize = 100;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditEntryRow {
    pub sequence: i64,
    pub entry_id: String,
    pub request_id: Option<String>,
    pub occurred_at: String,
    pub source: String,
    pub code: String,
    pub severity: Option<AuditEntrySeverity>,
    pub context_version: Option<u8>,
    pub subject_kind: Option<String>,
    pub subject_id: Option<String>,
    pub operation_id: Option<String>,
    pub correlation_request_id: Option<String>,
    pub reason_code: Option<String>,
    pub outcome: Option<String>,
    pub duration_ms: Option<u64>,
    pub numeric_summaries: BTreeMap<String, u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditEntrySource {
    LoggingService,
    Runtime,
    Mesh,
    Cli,
    LogsApi,
}

impl AuditEntrySource {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::LoggingService => "logging_service",
            Self::Runtime => "runtime",
            Self::Mesh => "mesh",
            Self::Cli => "cli",
            Self::LogsApi => "logs_api",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditEntrySeverity {
    Info,
    Warning,
    Error,
}

impl AuditEntrySeverity {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }

    pub(super) fn parse(value: Option<String>) -> Option<Self> {
        match value.as_deref() {
            Some("info") => Some(Self::Info),
            Some("warning") => Some(Self::Warning),
            Some("error") => Some(Self::Error),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AuditEntryFilters {
    pub source: Option<AuditEntrySource>,
    pub severity: Option<AuditEntrySeverity>,
    pub from: Option<String>,
    pub to: Option<String>,
}
