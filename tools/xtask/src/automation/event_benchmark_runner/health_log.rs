//! Final observed event-system health; absence remains unmeasured.
use serde_json::{Map, Value};

const COUNTERS: &[&str] = &[
    "version",
    "rebuild_generation",
    "reservation_exhausted",
    "cancelled_reservation_rejected",
    "terminal_delivery_failed",
    "dropped_progress",
    "coalesced_progress",
    "dropped_diagnostic",
    "replay_evicted",
    "subscriber_disconnected",
    "shutdown_degraded",
    "reducer_rejected",
    "state_transition_rejected",
    "state_degraded",
    "rebuild_required",
];

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Observation {
    pub health: Option<Map<String, Value>>,
    pub ingress_p99_us: Option<f64>,
}

pub(super) fn line(bytes: &[u8]) -> Option<Observation> {
    if bytes.len() > 16 * 1024 {
        return None;
    }
    let envelope: Value = serde_json::from_slice(bytes).ok()?;
    if envelope.get("context")?.as_str()? != "event_system_health" {
        return None;
    }
    let message = envelope.get("message")?.as_str()?;
    let mut health = Map::new();
    let mut p99 = None;
    for token in message.split_whitespace() {
        let Some((key, raw)) = token.split_once('=') else {
            continue;
        };
        if key == "ingress_p99_us" {
            p99 = serde_json::from_str::<Value>(raw)
                .ok()
                .and_then(|value| value.as_f64())
                .filter(|n| n.is_finite() && *n >= 0.0);
        } else if COUNTERS.contains(&key)
            && let Ok(value) = serde_json::from_str::<Value>(raw)
            && matches!(value, Value::Number(_) | Value::Bool(_) | Value::Null)
        {
            health.insert(key.into(), value);
        }
    }
    Some(Observation {
        health: (!health.is_empty()).then_some(health),
        ingress_p99_us: p99,
    })
}

#[cfg(test)]
pub(super) fn final_observation(log: &[u8]) -> Observation {
    log.split(|byte| *byte == b'\n')
        .filter_map(line)
        .next_back()
        .unwrap_or_default()
}

#[cfg(test)]
#[path = "health_log_tests.rs"]
mod tests;
