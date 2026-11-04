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
    exact_state_samples: Vec<ExactStateSample>,
    omitted_exact_state_samples: u64,
}

#[derive(Serialize)]
struct ExactStateSample {
    event: &'static str,
    sequence: Option<u64>,
    fields: BTreeMap<&'static str, u64>,
}

#[derive(Serialize)]
struct CacheDecision {
    kind: &'static str,
    decision: &'static str,
    source: Option<&'static str>,
    at_unix_nanos: u64,
    position: Option<u64>,
    runtime_position: Option<u64>,
}

impl CacheDecisions {
    pub(super) fn new() -> Self {
        Self {
            schema_version: 4,
            ..Self::default()
        }
    }

    pub(super) fn observe(&mut self, member: MemberId, line: ObservedLine<'_>) {
        if member != MemberId::Seed {
            return;
        }
        let exact_state = match line.stream {
            Stream::Stdout => exact_state_jsonl_event(line.bytes),
            Stream::Stderr => exact_state_event(line.bytes).map(|event| ExactStateSample {
                event,
                sequence: None,
                fields: BTreeMap::new(),
            }),
        };
        if let Some(sample) = exact_state {
            *self.exact_state_events.entry(sample.event).or_default() += 1;
            if self.exact_state_samples.len() < MAX_EVENTS {
                self.exact_state_samples.push(sample);
            } else {
                self.omitted_exact_state_samples =
                    self.omitted_exact_state_samples.saturating_add(1);
            }
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

fn exact_state_jsonl_event(line: &[u8]) -> Option<ExactStateSample> {
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
    let kind = exact_state_event(event.pointer("/data/message")?.as_str()?.as_bytes())?;
    let fields = event
        .pointer("/data/context")
        .and_then(Value::as_str)
        .map(exact_state_numeric_fields)
        .unwrap_or_default();
    Some(ExactStateSample {
        event: kind,
        sequence: event.get("sequence").and_then(Value::as_u64),
        fields,
    })
}

fn exact_state_numeric_fields(context: &str) -> BTreeMap<&'static str, u64> {
    context
        .split_whitespace()
        .filter_map(|field| {
            let (key, value) = field.split_once('=')?;
            let name = match key {
                "position" => "position",
                "logical_bytes" => "logical_bytes",
                "physical_bytes" => "physical_bytes",
                "limit_bytes" => "limit_bytes",
                "max_entries" => "max_entries",
                "kv_bytes" => "kv_bytes",
                "recurrent_bytes" => "recurrent_bytes",
                "pending_bytes" => "pending_bytes",
                _ => return None,
            };
            Some((name, value.parse().ok()?))
        })
        .collect()
}

fn exact_state_event(line: &[u8]) -> Option<&'static str> {
    match line.strip_prefix(b"Skippy exact-state ")? {
        b"demotion_credit_full" => Some("demotion_credit_full"),
        b"demotion_byte_full" => Some("demotion_byte_full"),
        b"demotion_queue_full" => Some("demotion_queue_full"),
        b"demotion_worker_stopped" => Some("demotion_worker_stopped"),
        b"demotion_duplicate_or_stopped" => Some("demotion_duplicate_or_stopped"),
        b"demotion_queued" => Some("demotion_queued"),
        b"demotion_queued_in_place" => Some("demotion_queued_in_place"),
        b"demotion_queue_full_in_place" => Some("demotion_queue_full_in_place"),
        b"resident_reset_unrecorded" => Some("resident_reset_unrecorded"),
        b"demotion_no_native_prefix" => Some("demotion_no_native_prefix"),
        b"demotion_error" => Some("demotion_error"),
        b"evicted_l1_count" => Some("evicted_l1_count"),
        b"evicted_l1_soft" => Some("evicted_l1_soft"),
        b"evicted_l1_hard" => Some("evicted_l1_hard"),
        b"published_l1" => Some("published_l1"),
        b"resident_branch_discarded" => Some("resident_branch_discarded"),
        b"resident_capacity_displaced" => Some("resident_capacity_displaced"),
        b"resident_displaced_lost" => Some("resident_displaced_lost"),
        b"resident_displaced_superseded" => Some("resident_displaced_superseded"),
        b"resident_discarded_cleanup" => Some("resident_discarded_cleanup"),
        b"resident_discarded_idle_pool_full" => Some("resident_discarded_idle_pool_full"),
        b"resident_discarded_memory_pressure" => Some("resident_discarded_memory_pressure"),
        b"resident_discarded_recorder_stopped" => Some("resident_discarded_recorder_stopped"),
        b"resident_discarded_retention_expired" => Some("resident_discarded_retention_expired"),
        b"resident_restore_suffix_discarded" => Some("resident_restore_suffix_discarded"),
        b"lane_recycle_retained" => Some("lane_recycle_retained"),
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
        source: attrs
            .get("skippy.exact_cache.source")
            .and_then(Value::as_str)
            .and_then(allowed_source),
        at_unix_nanos: event.get("start_time_unix_nanos")?.as_u64()?,
        position: attrs
            .get("skippy.exact_cache.recorded_tokens")
            .or_else(|| attrs.get("skippy.exact_cache.restored_tokens"))
            .or_else(|| attrs.get("skippy.kv.checkpoint_token_count"))
            .and_then(Value::as_u64),
        runtime_position: attrs
            .get("skippy.kv.runtime_token_count")
            .and_then(Value::as_u64),
    })
}

fn allowed_source(value: &str) -> Option<&'static str> {
    match value {
        "resident_recurrent" => Some("resident_recurrent"),
        "radix" => Some("radix"),
        "l2" => Some("l2"),
        "l3" => Some("l3"),
        "l3-cachegen" => Some("l3-cachegen"),
        _ => None,
    }
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
        "shared_prefill_checkpoint_recorded" => "shared_prefill_checkpoint_recorded",
        "shared_prefill_checkpoint_skipped" => "shared_prefill_checkpoint_skipped",
        "shared_prefill_checkpoint_error" => "shared_prefill_checkpoint_error",
        "shared_prefill_checkpoint_scheduler_error" => "shared_prefill_checkpoint_scheduler_error",
        "chat_prefix_checkpoint_recorded" => "chat_prefix_checkpoint_recorded",
        "chat_prefix_checkpoint_skipped" => "chat_prefix_checkpoint_skipped",
        "chat_prefix_checkpoint_error" => "chat_prefix_checkpoint_error",
        "chat_prefix_checkpoint_scheduler_error" => "chat_prefix_checkpoint_scheduler_error",
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
    fn distinguishes_prefill_checkpoints_from_other_record_decisions() {
        let mut decisions = CacheDecisions::new();
        for (decision, expected) in [
            (
                "shared_prefill_checkpoint_recorded",
                "shared_prefill_checkpoint_recorded",
            ),
            (
                "chat_prefix_checkpoint_skipped",
                "chat_prefix_checkpoint_skipped",
            ),
        ] {
            let raw = format!(
                r#"{{"event":"stage.openai_kv_record_decision","attributes":{{"skippy.kv.decision":"{decision}","prompt":"token secret"}},"start_time_unix_nanos":123}}"#
            );
            decisions.observe(
                MemberId::Seed,
                ObservedLine {
                    stream: Stream::Stderr,
                    bytes: raw.as_bytes(),
                    ending: crate::process::LineEnding::Lf,
                },
            );
            assert_eq!(decisions.events.last().unwrap().decision, expected);
        }
        let projected = serde_json::to_string(&decisions).unwrap();
        assert!(!projected.contains("token secret"));
    }

    #[test]
    fn projects_exact_hit_tier_and_restored_position_without_page_identity() {
        let raw = br#"{"event":"stage.openai_kv_lookup_decision","attributes":{"skippy.kv.decision":"exact_hit","skippy.exact_cache.source":"resident_recurrent","skippy.exact_cache.restored_tokens":26938,"skippy.exact_cache.hit_page_id":"secret-page"},"start_time_unix_nanos":123}"#;
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
        assert!(projected.contains("\"source\":\"resident_recurrent\""));
        assert!(projected.contains("\"position\":26938"));
        assert!(!projected.contains("secret-page"));
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
            br#"{"data":{"context":"position=155509 logical_bytes=123 physical_bytes=456 limit_bytes=789 secret=private-prompt","message":"Skippy exact-state demotion_byte_full"},"schema_version":1,"sequence":17,"type":"info"}"#.as_slice(),
            br#"{"data":{"message":"Skippy exact-state demotion_queued"},"type":"info"}"#.as_slice(),
            br#"{"data":{"context":"position=44805","message":"Skippy exact-state resident_branch_discarded"},"type":"info"}"#.as_slice(),
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
        assert!(projected.contains("\"resident_branch_discarded\":1"));
        assert_eq!(decisions.exact_state_samples[0].sequence, Some(17));
        assert_eq!(decisions.exact_state_samples[0].fields["position"], 155509);
        assert_eq!(
            decisions.exact_state_samples[0].fields["logical_bytes"],
            123
        );
        assert_eq!(
            decisions.exact_state_samples[0].fields["physical_bytes"],
            456
        );
        assert_eq!(decisions.exact_state_samples[0].fields["limit_bytes"], 789);
        assert!(!projected.contains("demotion_credit_full"));
        assert!(!projected.contains("private-prompt"));
    }

    #[test]
    fn bounds_exact_state_samples_while_counting_every_event() {
        let raw = br#"{"data":{"context":"position=42","message":"Skippy exact-state evicted_l1_soft"},"type":"info"}"#;
        let mut decisions = CacheDecisions::new();
        for _ in 0..=MAX_EVENTS {
            decisions.observe(
                MemberId::Seed,
                ObservedLine {
                    stream: Stream::Stdout,
                    bytes: raw,
                    ending: crate::process::LineEnding::Lf,
                },
            );
        }
        assert_eq!(decisions.exact_state_samples.len(), MAX_EVENTS);
        assert_eq!(decisions.omitted_exact_state_samples, 1);
        assert_eq!(decisions.exact_state_events["evicted_l1_soft"], 513);
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
