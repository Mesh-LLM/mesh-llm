//! The evidence half of `capture.py`: `json.loads` of the sccache stats
//! text, `sanitize_stats` (required counters and totalled count maps, every
//! other field dropped) and `assess_cache`.

use crate::ci_operations::python_json_decode::{DecodeError, Hooks, loads};
use crate::ci_plan::document::Json;

pub(crate) type Failure = String;

pub(crate) const REQUIRED_COUNTERS: [&str; 6] = [
    "compile_requests",
    "requests_executed",
    "compilations",
    "cache_writes",
    "cache_read_errors",
    "cache_write_errors",
];
pub(crate) const REQUIRED_COUNT_MAPS: [&str; 3] = ["cache_hits", "cache_misses", "cache_errors"];
pub(crate) const OUTPUT_COUNTERS: [&str; 7] = [
    "compile_requests",
    "requests_executed",
    "cache_hits",
    "cache_misses",
    "cache_writes",
    "cache_read_errors",
    "cache_write_errors",
];
const DECODE_STACK_BYTES: usize = 64 * 1024 * 1024;

/// Counters in legacy insertion order: the six scalars, then map totals.
pub(crate) struct Counters(Vec<(&'static str, i128)>);

impl Counters {
    pub(crate) fn get(&self, name: &str) -> i128 {
        self.0
            .iter()
            .find(|(key, _)| *key == name)
            .map_or(0, |(_, value)| *value)
    }
}

pub(crate) struct Assessment {
    pub(crate) expectation: &'static str,
    pub(crate) classification: &'static str,
    pub(crate) minimum_hit_rate: f64,
    pub(crate) hit_rate: Option<f64>,
    pub(crate) requests: i128,
    pub(crate) passed: bool,
}

/// `json.loads` keeps a repeated key at its first position with its last
/// value; the non-finite constants load as floats, which no field accepts.
fn pairs(items: Vec<(String, Json)>) -> Result<Json, String> {
    let mut merged: Vec<(String, Json)> = Vec::with_capacity(items.len());
    for (key, value) in items {
        match merged.iter_mut().find(|(seen, _)| *seen == key) {
            Some(slot) => slot.1 = value,
            None => merged.push((key, value)),
        }
    }
    Ok(Json::Object(merged))
}

fn constant(_: &str) -> Result<Json, String> {
    Ok(Json::Null)
}

/// Decodes on a large-stack thread: the scanner recurses per container.
pub(crate) fn decode(text: String) -> Result<Json, Failure> {
    let worker = std::thread::Builder::new()
        .stack_size(DECODE_STACK_BYTES)
        .spawn(move || decode_text(&text))
        .map_err(|error| error.to_string())?;
    worker
        .join()
        .unwrap_or_else(|_| Err("sccache returned invalid JSON".to_owned()))
}

fn decode_text(text: &str) -> Result<Json, Failure> {
    let invalid = |message: &str| format!("sccache returned invalid JSON: {message}");
    if text.starts_with('\u{feff}') {
        return Err(invalid(
            "Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1 (char 0)",
        ));
    }
    let hooks = Hooks { pairs, constant };
    loads(text.as_bytes(), &hooks).map_err(|error| match error {
        DecodeError::Value(message) => invalid(&message),
        DecodeError::Recursion => "maximum recursion depth exceeded".to_owned(),
    })
}

fn require_counter(stats: &Json, name: &str) -> Result<i128, Failure> {
    stats
        .get(name)
        .and_then(Json::as_int)
        .filter(|value| *value >= 0)
        .ok_or_else(|| format!("sccache JSON field stats.{name} must be a non-negative integer"))
}

fn count_tree(value: &Json, field: &str) -> Result<i128, Failure> {
    match value {
        Json::Bool(_) => Err(format!("sccache JSON field {field} contains a boolean")),
        Json::Number(_) if value.as_int().is_some() => {
            let count = value.as_int().unwrap_or_default();
            if count < 0 {
                return Err(format!(
                    "sccache JSON field {field} contains a negative counter"
                ));
            }
            Ok(count)
        }
        Json::Object(entries) => {
            let child = format!("{field} entry");
            entries.iter().try_fold(0_i128, |total, (_, item)| {
                Ok(total.saturating_add(count_tree(item, &child)?))
            })
        }
        _ => Err(format!(
            "sccache JSON field {field} must contain only counter maps and integers"
        )),
    }
}

fn count_map(stats: &Json, name: &str) -> Result<i128, Failure> {
    let value = stats
        .get(name)
        .filter(|value| value.as_object().is_some())
        .ok_or_else(|| format!("sccache JSON field stats.{name} must be an object"))?;
    let counts = value
        .get("counts")
        .filter(|counts| counts.as_object().is_some())
        .ok_or_else(|| format!("sccache JSON field stats.{name}.counts must be an object"))?;
    count_tree(counts, &format!("stats.{name}.counts"))
}

/// `sanitize_stats`: the required counters, in legacy check order.
pub(crate) fn sanitize(payload: &Json) -> Result<Counters, Failure> {
    if payload.as_object().is_none() {
        return Err("sccache JSON root must be an object".to_owned());
    }
    let stats = payload
        .get("stats")
        .filter(|stats| stats.as_object().is_some())
        .ok_or("sccache JSON field stats must be an object")?;
    let mut counters = Vec::new();
    for name in REQUIRED_COUNTERS {
        counters.push((name, require_counter(stats, name)?));
    }
    for name in REQUIRED_COUNT_MAPS {
        counters.push((name, count_map(stats, name)?));
    }
    Ok(Counters(counters))
}

/// `assess_cache`.
pub(crate) fn assess(
    expectation: &'static str,
    minimum_hit_rate: f64,
    counters: &Counters,
) -> Result<Assessment, Failure> {
    if !(0.0..=1.0).contains(&minimum_hit_rate) {
        return Err("minimum hit rate must be between 0 and 1".to_owned());
    }
    let hits = counters.get("cache_hits");
    let requests = hits + counters.get("cache_misses");
    let hit_rate = (requests != 0).then(|| true_divide(hits, requests));
    let (classification, passed) = match (expectation, hit_rate) {
        ("cold", _) => ("cold", true),
        ("opportunistic", _) => ("opportunistic", true),
        (_, None) if minimum_hit_rate == 0.0 => ("warm-pass", true),
        (_, Some(rate)) if rate >= minimum_hit_rate => ("warm-pass", true),
        _ => ("warm-failure", false),
    };
    Ok(Assessment {
        expectation,
        classification,
        minimum_hit_rate,
        hit_rate,
        requests,
        passed,
    })
}

/// Python `int / int` for counters that fit in an `f64` exactly.
fn true_divide(numerator: i128, denominator: i128) -> f64 {
    numerator as f64 / denominator as f64
}
