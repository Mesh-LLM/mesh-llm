//! Frozen host classification and measurable callback p99 admission.
use serde::Deserialize;
use serde_json::{Value, json};

use crate::command::DynResult;

pub(super) const BUDGET_US: f64 = 100.0;

#[derive(Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Gate {
    Enforced,
    Informational,
}

#[derive(Clone, Deserialize)]
pub(super) struct Host {
    pub system: String,
    pub machine: String,
    pub certification_host: Option<String>,
    pub p99_gate: Gate,
}

impl Host {
    pub fn validate(&self) -> DynResult<()> {
        if self.system.trim().is_empty() || self.machine.trim().is_empty() {
            return Err("host system and architecture must be nonempty".into());
        }
        let expected = match (self.system.as_str(), self.machine.as_str()) {
            ("Darwin", "arm64") => Some("macos-arm64-metal"),
            ("Linux", "x86_64") => Some("linux-x86_64-cuda"),
            _ => None,
        };
        if self.certification_host.as_deref() != expected
            || (self.p99_gate == Gate::Enforced) != expected.is_some()
        {
            return Err(
                "host callback gate differs from the frozen certification classification".into(),
            );
        }
        Ok(())
    }
}

pub(super) fn evaluate(host: &Host, p99: Option<f64>, budget: f64) -> DynResult<(Value, bool)> {
    host.validate()?;
    if !budget.is_finite() || budget < 0.0 {
        return Err("callback budget must be finite and nonnegative".into());
    }
    let p99 = p99.filter(|v| v.is_finite() && *v >= 0.0);
    if host.p99_gate == Gate::Informational {
        return Ok((json!({"status":"informational","value":p99}), false));
    }
    let Some(p99) = p99 else {
        return Ok((
            json!({"status":"blocked","reason":"p99 unmeasurable on a certification host"}),
            true,
        ));
    };
    if p99 > budget {
        return Ok((
            json!({"status":"blocked","value":p99,"reason":format!("p99 {p99}us exceeds the {budget}us budget")}),
            true,
        ));
    }
    Ok((json!({"status":"ok","value":p99}), false))
}

#[cfg(test)]
#[path = "callback_latency_tests.rs"]
mod tests;
