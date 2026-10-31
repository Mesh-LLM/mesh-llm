//! Content-free aggregates of serving scheduler debug timings.
use crate::process::{ObservedLine, Stream, retained::MemberId};
use serde::Serialize;
use serde_json::Value;

#[derive(Default, Serialize)]
struct Samples {
    count: u64,
    sum: f64,
    max: f64,
    at_least_10: u64,
    at_least_100: u64,
    at_least_1000: u64,
}

impl Samples {
    fn add(&mut self, value: f64) {
        if !value.is_finite() || value < 0.0 {
            return;
        }
        self.count += 1;
        self.sum += value;
        self.max = self.max.max(value);
        self.at_least_10 += u64::from(value >= 10.0);
        self.at_least_100 += u64::from(value >= 100.0);
        self.at_least_1000 += u64::from(value >= 1000.0);
    }
}

#[derive(Default, Serialize)]
pub(super) struct SchedulerProfile {
    schema_version: u8,
    iterations: u64,
    batch_size: Samples,
    token_count: Samples,
    batch_wait_max_ms: Samples,
    runtime_lock_wait_ms: Samples,
    runtime_lock_hold_ms: Samples,
    planned_iterations: u64,
    planned_decode_batches: PlannedDecodeBatches,
    planned_running_sequences: Samples,
    planned_waiting_sequences: Samples,
    planned_decode_tokens: Samples,
    planned_prefill_tokens: Samples,
    planned_step_ms: Samples,
    planned_execution_ms: Samples,
    planned_overhead_ms: Samples,
}

#[derive(Default, Serialize)]
struct PlannedDecodeBatches {
    none: u64,
    one: u64,
    two: u64,
    three: u64,
    four_or_more: u64,
}

impl PlannedDecodeBatches {
    fn add(&mut self, count: u64) {
        let bucket = match count {
            0 => &mut self.none,
            1 => &mut self.one,
            2 => &mut self.two,
            3 => &mut self.three,
            _ => &mut self.four_or_more,
        };
        *bucket += 1;
    }
}

enum SchedulerEvent {
    Direct,
    Planned,
}

impl SchedulerProfile {
    pub(super) fn new() -> Self {
        Self {
            schema_version: 2,
            ..Self::default()
        }
    }

    pub(super) fn observe(&mut self, member: MemberId, line: &ObservedLine<'_>) {
        if member != MemberId::Seed || line.stream != Stream::Stderr {
            return;
        }
        let Some((event, attributes)) = parse(line.bytes) else {
            return;
        };
        match event {
            SchedulerEvent::Direct => self.observe_direct(&attributes),
            SchedulerEvent::Planned => self.observe_planned(&attributes),
        }
    }

    fn observe_direct(&mut self, attributes: &serde_json::Map<String, Value>) {
        self.iterations += 1;
        for (name, samples) in [
            ("skippy.scheduler.batch_size", &mut self.batch_size),
            ("skippy.scheduler.token_count", &mut self.token_count),
            (
                "skippy.scheduler.batch_wait_max_ms",
                &mut self.batch_wait_max_ms,
            ),
            (
                "skippy.scheduler.runtime_lock_wait_ms",
                &mut self.runtime_lock_wait_ms,
            ),
            (
                "skippy.scheduler.runtime_lock_hold_ms",
                &mut self.runtime_lock_hold_ms,
            ),
        ] {
            if let Some(value) = attributes.get(name).and_then(Value::as_f64) {
                samples.add(value);
            }
        }
    }

    fn observe_planned(&mut self, attributes: &serde_json::Map<String, Value>) {
        self.planned_iterations += 1;
        if let Some(count) = attributes
            .get("skippy.scheduler.decode_tokens")
            .and_then(Value::as_u64)
        {
            self.planned_decode_batches.add(count);
        }
        for (name, samples) in [
            (
                "skippy.scheduler.running",
                &mut self.planned_running_sequences,
            ),
            (
                "skippy.scheduler.waiting",
                &mut self.planned_waiting_sequences,
            ),
            (
                "skippy.scheduler.decode_tokens",
                &mut self.planned_decode_tokens,
            ),
            (
                "skippy.scheduler.prefill_tokens",
                &mut self.planned_prefill_tokens,
            ),
            ("skippy.scheduler.step_ms", &mut self.planned_step_ms),
            (
                "skippy.scheduler.execution_ms",
                &mut self.planned_execution_ms,
            ),
            (
                "skippy.scheduler.overhead_ms",
                &mut self.planned_overhead_ms,
            ),
        ] {
            if let Some(value) = attributes.get(name).and_then(Value::as_f64) {
                samples.add(value);
            }
        }
    }
}

fn parse(line: &[u8]) -> Option<(SchedulerEvent, serde_json::Map<String, Value>)> {
    if !line.starts_with(b"{\"event\":") {
        return None;
    }
    let event: Value = serde_json::from_slice(line).ok()?;
    let kind = match event["event"].as_str()? {
        "stage.scheduler_feature_iteration" => SchedulerEvent::Direct,
        "stage.scheduler_iteration" => SchedulerEvent::Planned,
        _ => return None,
    };
    Some((kind, event["attributes"].as_object()?.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregates_only_allowlisted_numeric_scheduler_fields() {
        let mut profile = SchedulerProfile::new();
        let raw = br#"{"event":"stage.scheduler_feature_iteration","attributes":{"skippy.scheduler.batch_size":4,"skippy.scheduler.token_count":512,"skippy.scheduler.batch_wait_max_ms":12.5,"skippy.scheduler.runtime_lock_wait_ms":101.0,"skippy.scheduler.runtime_lock_hold_ms":1001.0,"prompt":"secret"}}"#;
        profile.observe(
            MemberId::Seed,
            &ObservedLine {
                stream: Stream::Stderr,
                bytes: raw,
                ending: crate::process::LineEnding::Lf,
            },
        );
        let value = serde_json::to_value(profile).unwrap();
        assert_eq!(value["iterations"], 1);
        assert_eq!(value["batch_size"]["sum"], 4.0);
        assert_eq!(value["runtime_lock_wait_ms"]["at_least_100"], 1);
        assert_eq!(value["runtime_lock_hold_ms"]["at_least_1000"], 1);
        assert!(!value.to_string().contains("secret"));
    }

    #[test]
    fn captures_planned_decode_occupancy_after_direct_prefill_handoff() {
        let mut profile = SchedulerProfile::new();
        for (decode, execution) in [(0, 120.0), (1, 35.0), (4, 51.0)] {
            let event = format!(
                r#"{{"event":"stage.scheduler_iteration","attributes":{{"skippy.scheduler.running":4,"skippy.scheduler.waiting":1,"skippy.scheduler.decode_tokens":{decode},"skippy.scheduler.prefill_tokens":0,"skippy.scheduler.step_ms":{},"skippy.scheduler.execution_ms":{execution},"skippy.scheduler.overhead_ms":2.0,"prompt":"private content"}}}}"#,
                execution + 2.0
            );
            profile.observe(
                MemberId::Seed,
                &ObservedLine {
                    stream: Stream::Stderr,
                    bytes: event.as_bytes(),
                    ending: crate::process::LineEnding::Lf,
                },
            );
        }
        let value = serde_json::to_value(profile).unwrap();
        assert_eq!(value["iterations"], 0);
        assert_eq!(value["planned_iterations"], 3);
        assert_eq!(value["planned_decode_batches"]["none"], 1);
        assert_eq!(value["planned_decode_batches"]["one"], 1);
        assert_eq!(value["planned_decode_batches"]["four_or_more"], 1);
        assert_eq!(value["planned_decode_tokens"]["sum"], 5.0);
        assert_eq!(value["planned_execution_ms"]["sum"], 206.0);
        assert!(!value.to_string().contains("private content"));
    }
}
