//! Bounded operational-audit records admitted by the logging service.

mod context;

pub use context::{OperationalAuditContext, OperationalAuditPathType, OperationalAuditSubjectKind};

/// Static, bounded operational audit data admitted by the logging service.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OperationalAuditRecord {
    source: &'static str,
    code: &'static str,
    severity: Option<OperationalAuditSeverity>,
    context: Option<OperationalAuditContext>,
    detail_json: Option<String>,
    entry_id: Option<String>,
    occurred_at: Option<String>,
}

impl OperationalAuditRecord {
    pub const fn builder(
        source: &'static str,
        code: &'static str,
    ) -> OperationalAuditRecordBuilder {
        OperationalAuditRecordBuilder {
            source,
            code,
            severity: None,
        }
    }

    pub const fn source(&self) -> &'static str {
        self.source
    }

    pub const fn code(&self) -> &'static str {
        self.code
    }

    pub const fn severity(&self) -> Option<OperationalAuditSeverity> {
        self.severity
    }

    pub fn with_context(mut self, context: OperationalAuditContext) -> Self {
        self.context = Some(context);
        self
    }

    pub(crate) fn context(&self) -> Option<&OperationalAuditContext> {
        self.context.as_ref()
    }

    pub(super) fn with_internal_detail(mut self, detail_json: String) -> Self {
        self.detail_json = Some(detail_json);
        self
    }

    pub(crate) fn detail_json(&self) -> Option<&str> {
        self.detail_json.as_deref()
    }

    /// Attach a shared entry identity so the live bus frame and the durable
    /// persistence row carry the same `entry_id` and `occurred_at`.
    pub(crate) fn with_identity(mut self, entry_id: String, occurred_at: String) -> Self {
        self.entry_id = Some(entry_id);
        self.occurred_at = Some(occurred_at);
        self
    }

    /// Shared entry identifier, present when the record was produced through
    /// the live enqueue path. Absent for fallback audit records.
    pub(crate) fn entry_id(&self) -> Option<&str> {
        self.entry_id.as_deref()
    }

    /// Shared occurrence timestamp, present when the record was produced
    /// through the live enqueue path. Absent for fallback audit records.
    pub(crate) fn occurred_at(&self) -> Option<&str> {
        self.occurred_at.as_deref()
    }
}

/// Builder for static operational audit records.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OperationalAuditRecordBuilder {
    source: &'static str,
    code: &'static str,
    severity: Option<OperationalAuditSeverity>,
}

impl OperationalAuditRecordBuilder {
    pub const fn severity(mut self, severity: OperationalAuditSeverity) -> Self {
        self.severity = Some(severity);
        self
    }

    pub const fn build(self) -> OperationalAuditRecord {
        OperationalAuditRecord {
            source: self.source,
            code: self.code,
            severity: self.severity,
            context: None,
            detail_json: None,
            entry_id: None,
            occurred_at: None,
        }
    }
}

/// Bounded severity vocabulary for operational audit records.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperationalAuditSeverity {
    Info,
    Warning,
    Error,
}

impl OperationalAuditSeverity {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}
