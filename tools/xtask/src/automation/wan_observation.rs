//! Reduce supplied WAN observations; this owner never performs network measurement.
use crate::command::DynResult;
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    io::{Read, Write},
};

pub(crate) const USAGE: &str = "cargo xtool automation wan-observation {delay RTT_MS | bandwidth < IPERF_JSON | tensor-split N | latency-summary < MEASUREMENT_JSON}";
#[path = "wan_observation/research_projection.rs"]
mod research_projection;
const JSON_LIMIT: usize = 1024 * 1024;

#[derive(Default, Deserialize)]
#[serde(untagged)]
enum Bitrate {
    Number(f64),
    Null(()),
    #[default]
    #[serde(skip)]
    Missing,
}
#[derive(Default, Deserialize)]
struct Sender {
    #[serde(default)]
    bits_per_second: Bitrate,
    #[serde(flatten)]
    other: BTreeMap<String, serde_json::Value>,
}
impl Sender {
    fn empty(&self) -> bool {
        matches!(self.bits_per_second, Bitrate::Missing) && self.other.is_empty()
    }
}
#[derive(Default, Deserialize)]
struct End {
    sum_sent: Option<Sender>,
    sum: Option<Sender>,
}
#[derive(Deserialize)]
struct Iperf {
    end: Option<End>,
}
fn delay(value: &str) -> DynResult<String> {
    if value.len() > 128 {
        return Err("WAN RTT observation exceeds input bound".into());
    }
    let rtt: f64 = value
        .trim()
        .parse()
        .map_err(|_| "WAN RTT must be a finite nonnegative number")?;
    if !rtt.is_finite() || rtt < 0.0 {
        return Err("WAN RTT must be a finite nonnegative number".into());
    }
    Ok(format!("{:.3}\n", rtt / 2.0))
}
fn bandwidth(bytes: &[u8]) -> DynResult<String> {
    if bytes.len() > JSON_LIMIT {
        return Err("WAN iperf observation exceeds input bound".into());
    }
    let json: serde_json::Value =
        serde_json::from_slice(bytes).map_err(|_| "invalid typed WAN iperf observation")?;
    if !json.is_object() {
        return Err("WAN iperf observation must be an object".into());
    }
    if let Some(end) = json.get("end").filter(|value| !value.is_null()) {
        if !end.is_object() {
            return Err("WAN iperf end must be an object".into());
        }
        for field in ["sum_sent", "sum"] {
            if end
                .get(field)
                .is_some_and(|value| !value.is_null() && !value.is_object())
            {
                return Err("WAN iperf sender must be an object".into());
            }
        }
    }
    let data: Iperf =
        serde_json::from_value(json).map_err(|_| "invalid typed WAN iperf observation")?;
    let end = data.end.unwrap_or_default();
    let sender = end.sum_sent.filter(|s| !s.empty()).or(end.sum);
    let Some(bits) = sender.and_then(|s| match s.bits_per_second {
        Bitrate::Number(value) => Some(value),
        Bitrate::Null(()) | Bitrate::Missing => None,
    }) else {
        return Ok(String::new());
    };
    if !bits.is_finite() || bits < 0.0 {
        return Err("WAN bitrate must be finite and nonnegative".into());
    }
    if bits == 0.0 {
        return Ok(String::new());
    }
    let mbit = (bits / 1_000_000.0).round_ties_even().max(1.0);
    if !mbit.is_finite() || mbit >= u64::MAX as f64 {
        return Err("WAN bitrate exceeds integer output bound".into());
    }
    Ok(format!("{}\n", mbit as u64))
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let output = match args {
        [help] if matches!(help.as_str(), "--help" | "-h") => format!("{USAGE}\n"),
        [verb, value] if verb == "delay" => delay(value)?,
        [verb, value] if verb == "tensor-split" => research_projection::tensor_split(value)?,
        [verb] if matches!(verb.as_str(), "bandwidth" | "latency-summary") => {
            let mut bytes = Vec::new();
            std::io::stdin()
                .lock()
                .take((JSON_LIMIT + 1) as u64)
                .read_to_end(&mut bytes)?;
            if verb == "bandwidth" {
                bandwidth(&bytes)?
            } else {
                research_projection::latency_summary(&bytes)?
            }
        }
        _ => return Err(USAGE.into()),
    };
    std::io::stdout().lock().write_all(output.as_bytes())?;
    Ok(())
}
#[cfg(test)]
#[path = "wan_observation/tests.rs"]
mod tests;
