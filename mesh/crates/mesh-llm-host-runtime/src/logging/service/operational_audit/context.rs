use std::{collections::BTreeMap, net::SocketAddr};

use mesh_llm_events::CliCommandSummary;

const OPERATIONAL_AUDIT_CONTEXT_VERSION: u8 = 1;
const MAX_CONTEXT_VALUE_CHARS: usize = 256;
const MAX_NUMERIC_SUMMARIES: usize = 8;

/// Closed subject vocabulary for diagnostic operational-audit context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperationalAuditSubjectKind {
    Runtime,
    Model,
    RuntimeInstance,
    CliCommand,
}

impl OperationalAuditSubjectKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Runtime => "runtime",
            Self::Model => "model",
            Self::RuntimeInstance => "runtime_instance",
            Self::CliCommand => "cli_command",
        }
    }

    pub(crate) fn parse(value: &str) -> Option<OperationalAuditContextSubjectKind> {
        match value {
            "runtime" => Some(OperationalAuditContextSubjectKind::Public(Self::Runtime)),
            "model" => Some(OperationalAuditContextSubjectKind::Public(Self::Model)),
            "runtime_instance" => Some(OperationalAuditContextSubjectKind::Public(
                Self::RuntimeInstance,
            )),
            "cli_command" => Some(OperationalAuditContextSubjectKind::Public(Self::CliCommand)),
            "mesh_peer" => Some(OperationalAuditContextSubjectKind::MeshPeer),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OperationalAuditContextSubjectKind {
    Public(OperationalAuditSubjectKind),
    MeshPeer,
}

impl OperationalAuditContextSubjectKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Public(kind) => kind.as_str(),
            Self::MeshPeer => "mesh_peer",
        }
    }
}

/// Closed connection-path vocabulary for mesh peer audit context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OperationalAuditPathType {
    Direct,
    Relay,
}

impl OperationalAuditPathType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Relay => "relay",
        }
    }
}

/// Versioned, bounded context shared by audit replay and durable storage.
///
/// Values are sanitized before this type accepts them. Arbitrary detail JSON,
/// command arguments, error text, and payload bodies cannot enter this shape.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OperationalAuditContext {
    subject_kind: Option<OperationalAuditContextSubjectKind>,
    subject_id: Option<String>,
    remote_addr: Option<SocketAddr>,
    path_type: Option<OperationalAuditPathType>,
    operation_id: Option<String>,
    request_id: Option<String>,
    reason_code: Option<&'static str>,
    outcome: Option<&'static str>,
    duration_ms: Option<u64>,
    numeric_summaries: BTreeMap<&'static str, u64>,
    command_summary: Option<String>,
}

impl OperationalAuditContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn subject(mut self, kind: OperationalAuditSubjectKind, id: &str) -> Self {
        self.subject_kind = Some(OperationalAuditContextSubjectKind::Public(kind));
        self.subject_id = bounded_context_value(id);
        self
    }

    pub(crate) fn mesh_peer_subject(mut self, id: &str) -> Self {
        self.subject_kind = Some(OperationalAuditContextSubjectKind::MeshPeer);
        self.subject_id = bounded_context_value(id);
        self
    }

    pub fn operation_id(mut self, operation_id: &str) -> Self {
        self.operation_id = bounded_context_value(operation_id);
        self
    }

    pub const fn network_path(
        mut self,
        path_type: OperationalAuditPathType,
        remote_addr: Option<SocketAddr>,
    ) -> Self {
        self.path_type = Some(path_type);
        self.remote_addr = match path_type {
            OperationalAuditPathType::Direct => remote_addr,
            OperationalAuditPathType::Relay => None,
        };
        self
    }

    pub fn request_id(mut self, request_id: &str) -> Self {
        self.request_id = bounded_context_value(request_id);
        self
    }

    pub const fn reason_code(mut self, reason_code: &'static str) -> Self {
        if Self::valid_static_code(reason_code) {
            self.reason_code = Some(reason_code);
        }
        self
    }

    pub const fn outcome(mut self, outcome: &'static str) -> Self {
        if Self::valid_static_code(outcome) {
            self.outcome = Some(outcome);
        }
        self
    }

    pub const fn duration_ms(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    pub fn numeric_summary(mut self, key: &'static str, value: u64) -> Self {
        if self.numeric_summaries.len() < MAX_NUMERIC_SUMMARIES && Self::valid_static_code(key) {
            self.numeric_summaries.insert(key, value);
        }
        self
    }

    pub fn command_summary(mut self, summary: &str) -> Self {
        self.command_summary =
            CliCommandSummary::sanitize(summary).map(|summary| summary.as_str().to_owned());
        self
    }

    pub(crate) const fn valid_static_code(value: &str) -> bool {
        valid_static_code(value)
    }

    pub(crate) fn fields(&self) -> serde_json::Map<String, serde_json::Value> {
        let mut fields = serde_json::Map::new();
        fields.insert(
            "context_version".into(),
            serde_json::json!(OPERATIONAL_AUDIT_CONTEXT_VERSION),
        );
        if let Some(kind) = self.subject_kind {
            fields.insert("subject_kind".into(), serde_json::json!(kind.as_str()));
        }
        insert_optional_string(&mut fields, "subject_id", self.subject_id.as_deref());
        if let Some(remote_addr) = self.remote_addr {
            fields.insert(
                "remote_addr".into(),
                serde_json::json!(remote_addr.to_string()),
            );
        }
        if let Some(path_type) = self.path_type {
            fields.insert("path_type".into(), serde_json::json!(path_type.as_str()));
        }
        insert_optional_string(&mut fields, "operation_id", self.operation_id.as_deref());
        insert_optional_string(&mut fields, "request_id", self.request_id.as_deref());
        insert_optional_string(&mut fields, "reason_code", self.reason_code);
        insert_optional_string(&mut fields, "outcome", self.outcome);
        if let Some(duration_ms) = self.duration_ms {
            fields.insert("duration_ms".into(), serde_json::json!(duration_ms));
        }
        if !self.numeric_summaries.is_empty() {
            fields.insert(
                "numeric_summaries".into(),
                serde_json::json!(self.numeric_summaries),
            );
        }
        insert_optional_string(
            &mut fields,
            "command_summary",
            self.command_summary.as_deref(),
        );
        fields
    }
}

fn insert_optional_string(
    fields: &mut serde_json::Map<String, serde_json::Value>,
    key: &'static str,
    value: Option<&str>,
) {
    if let Some(value) = value {
        fields.insert(key.into(), serde_json::json!(value));
    }
}

const fn valid_static_code(value: &str) -> bool {
    if value.is_empty() || value.len() > 64 {
        return false;
    }
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let byte = bytes[index];
        if !(byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_') {
            return false;
        }
        index += 1;
    }
    true
}

fn bounded_context_value(value: &str) -> Option<String> {
    let sanitized = crate::logging::policy::sanitize_paths_in_text(value);
    let sanitized = crate::logging::policy::redact_urls_in_text(&sanitized);
    let sanitized = crate::logging::policy::apply_redaction(&sanitized).0;
    let trimmed = sanitized.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(trimmed.chars().take(MAX_CONTEXT_VALUE_CHARS).collect())
}

#[cfg(test)]
mod tests;
