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
}

impl SchedulerProfile {
    pub(super) fn new() -> Self {
        Self {
            schema_version: 1,
            ..Self::default()
        }
    }

    pub(super) fn observe(&mut self, member: MemberId, line: &ObservedLine<'_>) {
        if member != MemberId::Seed || line.stream != Stream::Stderr {
            return;
        }
        let Some(attributes) = parse(line.bytes) else {
            return;
        };
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
}

fn parse(line: &[u8]) -> Option<serde_json::Map<String, Value>> {
    if !line.starts_with(b"{\"event\":") {
        return None;
    }
    let event: Value = serde_json::from_slice(line).ok()?;
    (event["event"] == "stage.scheduler_feature_iteration")
        .then(|| event["attributes"].as_object().cloned())?
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
}
