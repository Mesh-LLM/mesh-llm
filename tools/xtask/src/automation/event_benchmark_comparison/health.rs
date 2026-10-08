//! Health availability and exact event-loss expectations remain separate facts.
use serde::Deserialize;

#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub(super) struct Health {
    terminal_delivery_failed: u64,
    state_lane_evictions: u64,
    state_transition_rejected: u64,
    // This count is admitted as u64 but is not a state-loss predicate.
    #[serde(rename = "cancelled_reservation_rejected")]
    _cancelled_reservation_rejected: u64,
    state_degraded: bool,
    rebuild_required: bool,
    dropped_progress: u64,
    dropped_diagnostic: u64,
}

pub(super) struct Evidence {
    pub available: bool,
    pub violations: Vec<String>,
}

pub(super) fn evaluate(
    mode: &str,
    health: Option<&Health>,
    expected_progress: Option<u64>,
    expected_diagnostic: Option<u64>,
) -> Evidence {
    if mode == "off" {
        return Evidence {
            available: true,
            violations: Vec::new(),
        };
    }
    let Some(health) = health else {
        return Evidence {
            available: false,
            violations: Vec::new(),
        };
    };
    let mut violations = Vec::new();
    for (count, reason) in [
        (
            health.terminal_delivery_failed,
            "terminal_delivery_failed must be zero",
        ),
        (
            health.state_lane_evictions,
            "state-transition drops/evictions must be zero",
        ),
        (
            health.state_transition_rejected,
            "state_transition_rejected must be zero",
        ),
    ] {
        if count != 0 {
            violations.push(reason.into());
        }
    }
    for (flag, name) in [
        (health.state_degraded, "state_degraded"),
        (health.rebuild_required, "rebuild_required"),
    ] {
        if flag {
            violations.push(format!("{name} must be false"));
        }
    }
    for (name, actual, expected) in [
        (
            "dropped_progress",
            health.dropped_progress,
            expected_progress,
        ),
        (
            "dropped_diagnostic",
            health.dropped_diagnostic,
            expected_diagnostic,
        ),
    ] {
        if let Some(expected) = expected
            && actual != expected
        {
            violations.push(format!(
                "{name} {actual} does not exactly match the expected count {expected}"
            ));
        }
    }
    Evidence {
        available: true,
        violations,
    }
}

#[cfg(test)]
#[path = "health_tests.rs"]
mod tests;
