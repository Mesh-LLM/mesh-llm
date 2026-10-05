use std::{collections::BTreeMap, net::SocketAddr};

use mesh_llm_events::CliCommandSummary;
use serde::Deserialize;

#[derive(Default, Deserialize)]
pub(super) struct StoredAuditDetail {
    pub(super) severity: Option<String>,
    pub(super) context_version: Option<u8>,
    pub(super) subject_kind: Option<String>,
    pub(super) subject_id: Option<String>,
    pub(super) remote_addr: Option<String>,
    pub(super) path_type: Option<String>,
    pub(super) operation_id: Option<String>,
    pub(super) request_id: Option<String>,
    pub(super) reason_code: Option<String>,
    pub(super) outcome: Option<String>,
    pub(super) command_summary: Option<String>,
    pub(super) duration_ms: Option<u64>,
    #[serde(default)]
    pub(super) numeric_summaries: BTreeMap<String, u64>,
}

impl StoredAuditDetail {
    pub(super) fn parse(raw: Option<String>) -> Self {
        raw.and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or_default()
    }

    pub(super) fn bounded(mut self) -> Self {
        if self.context_version != Some(1) {
            return Self {
                severity: self.severity,
                ..Self::default()
            };
        }
        self.subject_kind = self.subject_kind.filter(|value| {
            matches!(
                value.as_str(),
                "runtime" | "model" | "runtime_instance" | "cli_command" | "mesh_peer"
            )
        });
        self.subject_id = bounded_audit_value(self.subject_id);
        self.path_type = self
            .path_type
            .filter(|value| matches!(value.as_str(), "direct" | "relay"));
        self.remote_addr = match self.path_type.as_deref() {
            Some("direct") => self
                .remote_addr
                .and_then(|value| value.parse::<SocketAddr>().ok())
                .map(|value| value.to_string()),
            Some("relay") | None => None,
            Some(_) => None,
        };
        self.operation_id = bounded_audit_value(self.operation_id);
        self.request_id = bounded_audit_value(self.request_id);
        self.reason_code = bounded_audit_code(self.reason_code);
        self.outcome = bounded_audit_code(self.outcome);
        self.command_summary = bounded_command_summary(self.command_summary);
        self.numeric_summaries = self
            .numeric_summaries
            .into_iter()
            .filter(|(key, _)| bounded_code(key))
            .take(8)
            .collect();
        self
    }
}

fn bounded_audit_value(value: Option<String>) -> Option<String> {
    value.filter(|value| !value.is_empty() && value.chars().count() <= 256)
}

fn bounded_command_summary(value: Option<String>) -> Option<String> {
    value
        .and_then(|value| CliCommandSummary::sanitize(&value))
        .map(|summary| summary.as_str().to_owned())
}

fn bounded_audit_code(value: Option<String>) -> Option<String> {
    value.filter(|value| bounded_code(value))
}

fn bounded_code(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}
