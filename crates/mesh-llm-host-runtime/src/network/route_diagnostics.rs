//! Bounded local routing decision diagnostics for operator-facing status.

use crate::network::metrics::AttemptTarget;
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;

const MAX_ROUTE_DECISIONS: usize = 16;
const MAX_TARGETS_PER_DECISION: usize = 16;
const MAX_REASON_CODES: usize = 2;
const MAX_LABEL_BYTES: usize = 128;
const REDACTED_SUFFIX: &str = "...";

#[derive(Clone, Debug, Default, Serialize, PartialEq, Eq)]
pub struct RouteDecisionSnapshot {
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_target: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected_kind: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub reason_codes: Vec<String>,
    pub targets: Vec<RouteDecisionTargetSnapshot>,
    pub age_secs: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct RouteDecisionRecord {
    pub(crate) model: String,
    pub(crate) required_tokens: Option<u32>,
    pub(crate) selected_target: Option<AttemptTarget>,
    pub(crate) reason_codes: Vec<RouteDecisionReason>,
    pub(crate) targets: Vec<RouteDecisionTargetRecord>,
}

#[derive(Clone, Debug)]
pub(crate) struct RouteDecisionTargetRecord {
    pub(crate) target: AttemptTarget,
    pub(crate) context_length: Option<u32>,
    pub(crate) required_tokens: Option<u32>,
    pub(crate) avg_tokens_per_second_milli: Option<u64>,
    pub(crate) throughput_samples: u64,
    pub(crate) reason_codes: Vec<RouteDecisionReason>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RouteDecisionReason {
    Selected,
    NotSelected,
    ContextTooSmall,
    ContextUnknown,
    HealthFiltered,
    NoEligibleTarget,
    ThroughputPreferred,
    AffinitySelected,
}

impl RouteDecisionReason {
    fn code(self) -> &'static str {
        match self {
            Self::Selected => "selected",
            Self::NotSelected => "not_selected",
            Self::ContextTooSmall => "context_too_small",
            Self::ContextUnknown => "context_unknown",
            Self::HealthFiltered => "health_filtered",
            Self::NoEligibleTarget => "no_eligible_target",
            Self::ThroughputPreferred => "throughput_preferred",
            Self::AffinitySelected => "affinity_selected",
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, PartialEq, Eq)]
pub struct RouteDecisionTargetSnapshot {
    pub target: String,
    pub kind: String,
    pub selected: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_tokens_per_second_milli: Option<u64>,
    pub throughput_samples: u64,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub reason_codes: Vec<String>,
}

#[derive(Clone)]
pub(crate) struct RouteDiagnostics {
    inner: Arc<Mutex<VecDeque<RouteDecisionEntry>>>,
    capacity: usize,
    target_capacity: usize,
}

impl Default for RouteDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

impl RouteDiagnostics {
    pub(crate) fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::new())),
            capacity: MAX_ROUTE_DECISIONS,
            target_capacity: MAX_TARGETS_PER_DECISION,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_capacity_for_test(capacity: usize, target_capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(VecDeque::new())),
            capacity: capacity.max(1),
            target_capacity: target_capacity.max(1),
        }
    }

    pub(crate) fn record(&self, record: RouteDecisionRecord) {
        let mut entries = self.inner.lock().unwrap();
        entries.push_front(RouteDecisionEntry::from_record(
            record,
            self.target_capacity,
        ));
        while entries.len() > self.capacity {
            entries.pop_back();
        }
    }

    pub(crate) fn snapshot(&self) -> Vec<RouteDecisionSnapshot> {
        self.inner
            .lock()
            .unwrap()
            .iter()
            .map(RouteDecisionEntry::snapshot)
            .collect()
    }
}

#[derive(Clone, Debug)]
struct RouteDecisionEntry {
    model: String,
    required_tokens: Option<u32>,
    selected_target: Option<TargetLabel>,
    reason_codes: Vec<RouteDecisionReason>,
    targets: Vec<RouteDecisionTargetEntry>,
    recorded_at: Instant,
}

impl RouteDecisionEntry {
    fn from_record(record: RouteDecisionRecord, target_capacity: usize) -> Self {
        Self {
            model: sanitize_label(&record.model),
            required_tokens: record.required_tokens,
            selected_target: record
                .selected_target
                .as_ref()
                .map(TargetLabel::from_attempt),
            reason_codes: bounded_reason_codes(record.reason_codes),
            targets: record
                .targets
                .into_iter()
                .take(target_capacity)
                .map(RouteDecisionTargetEntry::from_record)
                .collect(),
            recorded_at: Instant::now(),
        }
    }

    fn snapshot(&self) -> RouteDecisionSnapshot {
        RouteDecisionSnapshot {
            model: self.model.clone(),
            required_tokens: self.required_tokens,
            selected_target: self
                .selected_target
                .as_ref()
                .map(|target| target.label.clone()),
            selected_kind: self
                .selected_target
                .as_ref()
                .map(|target| target.kind.clone()),
            reason_codes: reason_strings(&self.reason_codes),
            targets: self
                .targets
                .iter()
                .map(RouteDecisionTargetEntry::snapshot)
                .collect(),
            age_secs: self.recorded_at.elapsed().as_secs(),
        }
    }
}

#[derive(Clone, Debug)]
struct RouteDecisionTargetEntry {
    target: TargetLabel,
    selected: bool,
    context_length: Option<u32>,
    required_tokens: Option<u32>,
    avg_tokens_per_second_milli: Option<u64>,
    throughput_samples: u64,
    reason_codes: Vec<RouteDecisionReason>,
}

impl RouteDecisionTargetEntry {
    fn from_record(record: RouteDecisionTargetRecord) -> Self {
        let selected = record.reason_codes.contains(&RouteDecisionReason::Selected);
        Self {
            target: TargetLabel::from_attempt(&record.target),
            selected,
            context_length: record.context_length,
            required_tokens: record.required_tokens,
            avg_tokens_per_second_milli: record.avg_tokens_per_second_milli,
            throughput_samples: record.throughput_samples,
            reason_codes: bounded_reason_codes(record.reason_codes),
        }
    }

    fn snapshot(&self) -> RouteDecisionTargetSnapshot {
        RouteDecisionTargetSnapshot {
            target: self.target.label.clone(),
            kind: self.target.kind.clone(),
            selected: self.selected,
            context_length: self.context_length,
            required_tokens: self.required_tokens,
            avg_tokens_per_second_milli: self.avg_tokens_per_second_milli,
            throughput_samples: self.throughput_samples,
            reason_codes: reason_strings(&self.reason_codes),
        }
    }
}

#[derive(Clone, Debug)]
struct TargetLabel {
    label: String,
    kind: String,
}

impl TargetLabel {
    fn from_attempt(target: &AttemptTarget) -> Self {
        match target {
            AttemptTarget::Local(label) => Self {
                label: sanitize_label(label),
                kind: "local".to_string(),
            },
            AttemptTarget::Remote(label) => Self {
                label: sanitize_label(label),
                kind: "remote".to_string(),
            },
            AttemptTarget::Endpoint(label) => Self {
                label: sanitize_label(&endpoint_label_without_secrets(label)),
                kind: "endpoint".to_string(),
            },
        }
    }
}

fn bounded_reason_codes(reasons: Vec<RouteDecisionReason>) -> Vec<RouteDecisionReason> {
    let mut bounded = Vec::new();
    for reason in reasons {
        if !bounded.contains(&reason) {
            bounded.push(reason);
        }
        if bounded.len() >= MAX_REASON_CODES {
            break;
        }
    }
    bounded
}

fn reason_strings(reasons: &[RouteDecisionReason]) -> Vec<String> {
    reasons
        .iter()
        .map(|reason| reason.code().to_string())
        .collect()
}

fn strip_query(label: &str) -> &str {
    label
        .split_once('?')
        .map(|(prefix, _)| prefix)
        .unwrap_or(label)
}

fn endpoint_label_without_secrets(label: &str) -> String {
    let stripped = strip_query(label);
    let Some((scheme, rest)) = stripped.split_once("://") else {
        return stripped.to_string();
    };
    let authority_end = rest.find('/').unwrap_or(rest.len());
    let (authority, suffix) = rest.split_at(authority_end);
    let host_authority = authority
        .rsplit_once('@')
        .map(|(_, authority)| authority)
        .unwrap_or(authority);
    format!("{scheme}://{host_authority}{suffix}")
}

fn sanitize_label(label: &str) -> String {
    let trimmed = label.trim();
    if trimmed.len() <= MAX_LABEL_BYTES {
        return trimmed.to_string();
    }
    let keep = MAX_LABEL_BYTES.saturating_sub(REDACTED_SUFFIX.len());
    let boundary = trimmed
        .char_indices()
        .map(|(idx, _)| idx)
        .take_while(|idx| *idx <= keep)
        .last()
        .unwrap_or(0);
    format!("{}{}", &trimmed[..boundary], REDACTED_SUFFIX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::metrics::AttemptTarget;

    #[test]
    fn route_diagnostics_snapshot_bounds_reasons_and_redacts_target_labels() {
        let diagnostics = RouteDiagnostics::with_capacity_for_test(2, 2);
        diagnostics.record(RouteDecisionRecord {
            model: "qwen".into(),
            required_tokens: Some(8192),
            selected_target: Some(AttemptTarget::Remote("peer-a".into())),
            reason_codes: vec![RouteDecisionReason::Selected],
            targets: vec![
                RouteDecisionTargetRecord {
                    target: AttemptTarget::Remote("peer-a".into()),
                    context_length: Some(32768),
                    required_tokens: Some(8192),
                    avg_tokens_per_second_milli: Some(41_000),
                    throughput_samples: 7,
                    reason_codes: vec![RouteDecisionReason::Selected],
                },
                RouteDecisionTargetRecord {
                    target: AttemptTarget::Endpoint(
                        "https://user:sk-secret-body@example.test/v1?api_key=sk-secret-query"
                            .into(),
                    ),
                    context_length: Some(4096),
                    required_tokens: Some(8192),
                    avg_tokens_per_second_milli: None,
                    throughput_samples: 0,
                    reason_codes: vec![
                        RouteDecisionReason::ContextTooSmall,
                        RouteDecisionReason::NotSelected,
                        RouteDecisionReason::HealthFiltered,
                    ],
                },
            ],
        });

        let snapshot = diagnostics.snapshot();

        assert_eq!(snapshot.len(), 1);
        assert_eq!(snapshot[0].reason_codes, vec!["selected"]);
        assert_eq!(snapshot[0].selected_target.as_deref(), Some("peer-a"));
        assert_eq!(snapshot[0].targets.len(), 2);
        assert_eq!(snapshot[0].targets[1].target, "https://example.test/v1");
        let json = serde_json::to_string(&snapshot).unwrap();
        assert!(!json.contains("sk-secret-body"));
        assert!(!json.contains("sk-secret-query"));
        assert_eq!(
            snapshot[0].targets[1].reason_codes,
            vec!["context_too_small", "not_selected"]
        );
    }
}
