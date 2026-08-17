use std::{io, thread, time::Duration};

use anyhow::{Result, bail};
use skippy_protocol::binary::{StageWireMessage, WireActivationDType, write_stage_message};

#[derive(Clone, Copy, Debug)]
pub struct WireCondition {
    delay_ms: f64,
    mbps: Option<f64>,
}

impl WireCondition {
    pub fn new(delay_ms: f64, mbps: Option<f64>) -> Result<Self> {
        if !delay_ms.is_finite() || delay_ms < 0.0 {
            bail!("downstream wire delay must be finite and non-negative");
        }
        if mbps.is_some_and(|value| !value.is_finite() || value <= 0.0) {
            bail!("downstream wire mbps must be finite and greater than zero");
        }
        Ok(Self { delay_ms, mbps })
    }

    pub(crate) fn propagation_delay(&self) -> Duration {
        Duration::from_secs_f64(self.delay_ms / 1000.0)
    }

    fn sleep_for(&self, message: &StageWireMessage) {
        thread::sleep(self.propagation_delay());
        self.sleep_for_bandwidth(message);
    }

    fn sleep_for_bandwidth(&self, message: &StageWireMessage) {
        let bandwidth_seconds = self
            .mbps
            .map(|mbps| message.estimated_wire_bytes() as f64 / (mbps * 125_000.0))
            .unwrap_or(0.0);
        if bandwidth_seconds > 0.0 {
            thread::sleep(Duration::from_secs_f64(bandwidth_seconds));
        }
    }
}

pub(crate) fn write_stage_message_conditioned(
    writer: impl io::Write,
    message: &StageWireMessage,
    dtype: WireActivationDType,
    condition: WireCondition,
) -> io::Result<()> {
    condition.sleep_for(message);
    write_stage_message(writer, message, dtype)
}

pub(crate) fn write_stage_message_after_propagation(
    writer: impl io::Write,
    message: &StageWireMessage,
    dtype: WireActivationDType,
    condition: WireCondition,
) -> io::Result<()> {
    condition.sleep_for_bandwidth(message);
    write_stage_message(writer, message, dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_condition_rejects_non_finite_or_negative_delay() {
        for delay_ms in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(WireCondition::new(delay_ms, None).is_err());
        }
    }

    #[test]
    fn wire_condition_rejects_non_finite_or_non_positive_bandwidth() {
        for mbps in [-1.0, 0.0, f64::NAN, f64::INFINITY] {
            assert!(WireCondition::new(0.0, Some(mbps)).is_err());
        }
    }

    #[test]
    fn propagation_delay_is_exposed_without_bandwidth_serialization() {
        let condition = WireCondition::new(25.0, Some(100.0)).unwrap();

        assert_eq!(condition.propagation_delay(), Duration::from_millis(25));
    }
}
