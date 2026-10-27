//! Bounded, content-free projection of cache telemetry for AgentX diagnostics.
use crate::process::{ObservedLine, Stream, retained::MemberId};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

const MAX_EVENTS: usize = 512;

#[derive(Default, Serialize)]
pub(super) struct CacheDecisions {
    schema_version: u8,
    events: Vec<CacheDecision>,
    omitted_events: u64,
    exact_state_events: BTreeMap<&'static str, u64>,
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
            schema_version: 2,
            ..Self::default()
        }
    }

    pub(super) fn observe(&mut self, member: MemberId, line: ObservedLine<'_>) {
        if member != MemberId::Seed {
            return;
        }
        let exact_state = match line.stream {
            Stream::Stdout => exact_state_jsonl_event(line.bytes),
            Stream::Stderr => exact_state_event(line.bytes),
        };
        if let Some(event) = exact_state {
            *self.exact_state_events.entry(event).or_default() += 1;
            return;
        }
        if line.stream != Stream::Stderr {
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

fn exact_state_jsonl_event(line: &[u8]) -> Option<&'static str> {
    if !std::str::from_utf8(line)
        .ok()?
        .contains("Skippy exact-state ")
    {
        return None;
    }
    let event: Value = serde_json::from_slice(line).ok()?;
    if event.get("type")?.as_str()? != "info" {
        return None;
    }
    exact_state_event(event.pointer("/data/message")?.as_str()?.as_bytes())
}

fn exact_state_event(line: &[u8]) -> Option<&'static str> {
    match line.strip_prefix(b"Skippy exact-state ")? {
        b"demotion_credit_full" => Some("demotion_credit_full"),
        b"demotion_byte_full" => Some("demotion_byte_full"),
        b"demotion_queue_full" => Some("demotion_queue_full"),
        b"demotion_worker_stopped" => Some("demotion_worker_stopped"),
        b"demotion_duplicate_or_stopped" => Some("demotion_duplicate_or_stopped"),
        b"demotion_queued" => Some("demotion_queued"),
        b"demotion_no_native_prefix" => Some("demotion_no_native_prefix"),
        b"demotion_error" => Some("demotion_error"),
        b"evicted_l1_count" => Some("evicted_l1_count"),
        b"evicted_l1_soft" => Some("evicted_l1_soft"),
        b"evicted_l1_hard" => Some("evicted_l1_hard"),
        b"published_l1" => Some("published_l1"),
        _ => None,
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
    fn counts_exact_state_drops_without_retaining_context() {
        let mut decisions = CacheDecisions::new();
        for line in [
            b"Skippy exact-state demotion_credit_full".as_slice(),
            b"  position=155509 secret=private-prompt".as_slice(),
            b"Skippy exact-state demotion_credit_full".as_slice(),
            b"Skippy exact-state demotion_queued".as_slice(),
        ] {
            decisions.observe(
                MemberId::Seed,
                ObservedLine {
                    stream: Stream::Stderr,
                    bytes: line,
                    ending: crate::process::LineEnding::Lf,
                },
            );
        }
        let projected = serde_json::to_string(&decisions).unwrap();
        assert!(projected.contains("\"demotion_credit_full\":2"));
        assert!(projected.contains("\"demotion_queued\":1"));
        assert!(!projected.contains("private-prompt"));
    }

    #[test]
    fn counts_standalone_jsonl_exact_state_events_without_retaining_context() {
        let mut decisions = CacheDecisions::new();
        for bytes in [
            br#"{"data":{"context":"position=155509 secret=private-prompt","message":"Skippy exact-state demotion_byte_full"},"schema_version":1,"sequence":17,"type":"info"}"#.as_slice(),
            br#"{"data":{"message":"Skippy exact-state demotion_queued"},"type":"info"}"#.as_slice(),
            br#"{"data":{"message":"Skippy exact-state demotion_credit_full"},"type":"native_log"}"#.as_slice(),
        ] {
            decisions.observe(
                MemberId::Seed,
                ObservedLine {
                    stream: Stream::Stdout,
                    bytes,
                    ending: crate::process::LineEnding::Lf,
                },
            );
        }
        let projected = serde_json::to_string(&decisions).unwrap();
        assert!(projected.contains("\"demotion_byte_full\":1"));
        assert!(projected.contains("\"demotion_queued\":1"));
        assert!(!projected.contains("demotion_credit_full"));
        assert!(!projected.contains("private-prompt"));
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
