use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const MAX_COUNTER: u64 = (1 << 53) - 1;
const SCALARS: [&str; 14] = [
    "compile_requests",
    "requests_unsupported_compiler",
    "requests_not_compile",
    "requests_not_cacheable",
    "requests_executed",
    "cache_timeouts",
    "cache_read_errors",
    "non_cacheable_compilations",
    "forced_recaches",
    "cache_write_errors",
    "cache_writes",
    "compilations",
    "compile_fails",
    "dist_errors",
];

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct LanguageCounts {
    pub counts: BTreeMap<String, u64>,
    pub adv_counts: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct Snapshot {
    pub stats: Stats,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct Stats {
    pub cache_hits: LanguageCounts,
    pub cache_misses: LanguageCounts,
    pub cache_errors: LanguageCounts,
    pub not_cached: BTreeMap<String, u64>,
    #[serde(flatten)]
    pub scalars: BTreeMap<String, u64>,
}

impl Snapshot {
    pub fn parse(bytes: &[u8]) -> DynResult<Self> {
        let mut payload: serde_json::Value = serde_json::from_slice(bytes)?;
        let stats = payload
            .get_mut("stats")
            .and_then(serde_json::Value::as_object_mut)
            .ok_or("invalid stats object")?;
        stats.retain(|key, _| {
            SCALARS.contains(&key.as_str())
                || ["cache_hits", "cache_misses", "cache_errors", "not_cached"]
                    .contains(&key.as_str())
        });
        let snapshot: Self = serde_json::from_value(payload)?;
        snapshot.validate()?;
        Ok(snapshot)
    }

    pub fn validate(&self) -> DynResult<()> {
        for name in SCALARS {
            if self
                .stats
                .scalars
                .get(name)
                .is_none_or(|count| *count > MAX_COUNTER)
            {
                return Err(format!("missing or invalid counter: {name}").into());
            }
        }
        for counts in [
            &self.stats.cache_hits,
            &self.stats.cache_misses,
            &self.stats.cache_errors,
        ] {
            validate_map(&counts.counts)?;
            validate_map(&counts.adv_counts)?;
        }
        validate_map(&self.stats.not_cached)
    }

    pub fn measurement(&self) -> DynResult<Measurement> {
        self.validate()?;
        if self.stats.scalars["cache_read_errors"] != 0
            || self.stats.scalars["cache_write_errors"] != 0
            || total(&self.stats.cache_errors.counts)? != 0
        {
            return Err("cache errors: inconclusive".into());
        }
        let hits = &self.stats.cache_hits.counts;
        let misses = &self.stats.cache_misses.counts;
        let hit_total = total(hits)?;
        let requests = hit_total
            .checked_add(total(misses)?)
            .ok_or("counter overflow")?;
        if requests == 0 {
            return Err("no cacheable requests: inconclusive".into());
        }
        let native_hits = native(hits)?;
        let native_requests = native_hits
            .checked_add(native(misses)?)
            .ok_or("counter overflow")?;
        let rate = hit_total.to_string().parse::<f64>()? / requests.to_string().parse::<f64>()?;
        Ok(Measurement {
            hit_rate: rate,
            native_hits,
            native_requests,
        })
    }
}

pub(super) struct Measurement {
    pub hit_rate: f64,
    pub native_hits: u64,
    pub native_requests: u64,
}

fn validate_map(counts: &BTreeMap<String, u64>) -> DynResult<()> {
    if counts.len() > 128
        || counts
            .iter()
            .any(|(name, count)| name.len() > 128 || *count > MAX_COUNTER)
    {
        return Err("invalid counter map".into());
    }
    Ok(())
}

fn total(counts: &BTreeMap<String, u64>) -> DynResult<u64> {
    counts.values().try_fold(0u64, |sum, count| {
        sum.checked_add(*count)
            .ok_or_else(|| "counter overflow".into())
    })
}

fn native(counts: &BTreeMap<String, u64>) -> DynResult<u64> {
    ["C/C++", "C", "C++"]
        .into_iter()
        .try_fold(0u64, |sum, name| {
            sum.checked_add(counts.get(name).copied().unwrap_or(0))
                .ok_or_else(|| "counter overflow".into())
        })
}
