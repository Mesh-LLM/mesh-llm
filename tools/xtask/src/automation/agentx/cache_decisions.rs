//! Bounded, content-free projection of cache telemetry for AgentX diagnostics.
use crate::process::{ObservedLine, Stream, retained::MemberId};
use serde::Serialize;
use serde_json::Value;

const MAX_EVENTS: usize = 512;

#[derive(Default, Serialize)]
pub(super) struct CacheDecisions {
    schema_version: u8,
    events: Vec<CacheDecision>,
    omitted_events: u64,
}

#[derive(Serialize)]
struct CacheDecision {
    kind: &'static str,
    decision: &'static str,
    at_unix_nanos: u64,
    position: Option<u64>,
    runtime_position: Option<u64>,
}

impl CacheDecisions {
    pub(super) fn new() -> Self {
        Self {
            schema_version: 1,
            ..Self::default()
        }
    }

    pub(super) fn observe(&mut self, member: MemberId, line: ObservedLine<'_>) {
        if member != MemberId::Seed || line.stream != Stream::Stderr {
            return;
        }
        let Some(event) = parse(line.bytes) else {
            return;
        };
        if self.events.len() < MAX_EVENTS {
            self.events.push(event);
        } else {
            self.omitted_events = self.omitted_events.saturating_add(1);
        }
    }
}

fn parse(line: &[u8]) -> Option<CacheDecision> {
    if !line.starts_with(b"{\"event\":") {
        return None;
    }
    let event: Value = serde_json::from_slice(line).ok()?;
    let kind = match event.get("event")?.as_str()? {
        "stage.openai_kv_record_decision" => "record",
        "stage.openai_kv_lookup_decision" => "lookup",
        _ => return None,
    };
    let attrs = event.get("attributes")?;
    let decision = allowed_decision(attrs.get("skippy.kv.decision")?.as_str()?);
    Some(CacheDecision {
        kind,
        decision,
        at_unix_nanos: event.get("start_time_unix_nanos")?.as_u64()?,
        position: attrs
            .get("skippy.exact_cache.recorded_tokens")
            .or_else(|| attrs.get("skippy.kv.checkpoint_token_count"))
            .and_then(Value::as_u64),
        runtime_position: attrs
            .get("skippy.kv.runtime_token_count")
            .and_then(Value::as_u64),
    })
}

fn allowed_decision(value: &str) -> &'static str {
    match value {
        "post_decode_checkpoint_recorded" => "post_decode_checkpoint_recorded",
        "post_decode_checkpoint_skipped" => "post_decode_checkpoint_skipped",
        "post_decode_checkpoint_error" => "post_decode_checkpoint_error",
        "post_decode_checkpoint_scheduler_error" => "post_decode_checkpoint_scheduler_error",
        "prefill_checkpoint_recorded" => "prefill_checkpoint_recorded",
        "prefill_checkpoint_skipped" => "prefill_checkpoint_skipped",
        "prefill_checkpoint_error" => "prefill_checkpoint_error",
        "prefill_checkpoint_scheduler_error" => "prefill_checkpoint_scheduler_error",
        "exact_hit" => "exact_hit",
        "resident_hit" => "resident_hit",
        "chain_restore_hit" => "chain_restore_hit",
        "miss" => "miss",
        "resident_error" => "resident_error",
        "exact_error" => "exact_error",
        _ => "other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projects_numeric_cache_evidence_without_retaining_raw_content() {
        let raw = br#"{"event":"stage.openai_kv_record_decision","attributes":{"skippy.kv.decision":"post_decode_checkpoint_recorded","skippy.exact_cache.recorded_tokens":110553,"authorization":"private-value","prompt":"token secret"},"start_time_unix_nanos":123}"#;
        let mut decisions = CacheDecisions::new();
        decisions.observe(
            MemberId::Seed,
            ObservedLine {
                stream: Stream::Stderr,
                bytes: raw,
                ending: crate::process::LineEnding::Lf,
            },
        );
        let projected = serde_json::to_string(&decisions).unwrap();
        assert!(projected.contains("110553"));
        assert!(projected.contains("post_decode_checkpoint_recorded"));
        assert!(!projected.contains("private-value"));
        assert!(!projected.contains("token secret"));
        assert!(!projected.contains("authorization"));
    }

    #[test]
    fn ignores_other_members_and_bounds_the_projection() {
        let raw = br#"{"event":"stage.openai_kv_lookup_decision","attributes":{"skippy.kv.decision":"miss"},"start_time_unix_nanos":123}"#;
        let mut decisions = CacheDecisions::new();
        decisions.observe(
            MemberId::WorkerOne,
            ObservedLine {
                stream: Stream::Stderr,
                bytes: raw,
                ending: crate::process::LineEnding::Lf,
            },
        );
        assert!(decisions.events.is_empty());
        for _ in 0..=MAX_EVENTS {
            decisions.observe(
                MemberId::Seed,
                ObservedLine {
                    stream: Stream::Stderr,
                    bytes: raw,
                    ending: crate::process::LineEnding::Lf,
                },
            );
        }
        assert_eq!(decisions.events.len(), MAX_EVENTS);
        assert_eq!(decisions.omitted_events, 1);
    }

    #[test]
    fn unknown_decisions_cannot_copy_untrusted_text() {
        let raw = br#"{"event":"stage.openai_kv_record_decision","attributes":{"skippy.kv.decision":"secret from untrusted source"},"start_time_unix_nanos":123}"#;
        let mut decisions = CacheDecisions::new();
        decisions.observe(
            MemberId::Seed,
            ObservedLine {
                stream: Stream::Stderr,
                bytes: raw,
                ending: crate::process::LineEnding::Lf,
            },
        );
        let projected = serde_json::to_string(&decisions).unwrap();
        assert!(projected.contains("\"decision\":\"other\""));
        assert!(!projected.contains("secret from untrusted source"));
    }
}
