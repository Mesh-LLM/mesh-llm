//! Cache counters are observed usage, never inference from latency or slot occupancy.
use serde::Serialize;
use std::collections::BTreeSet;
#[derive(Clone, Serialize)]
pub(super) struct Observation {
    pub run: Option<u64>,
    pub elapsed_ms: f64,
    pub prompt_tokens: u64,
    pub cached_tokens: Option<u64>,
    pub cache_status: &'static str,
    pub hit_kind: &'static str,
    pub cacheable_prefix_tokens: u64,
    pub suffix_prefill_tokens: Option<u64>,
    pub cache_efficiency: Option<f64>,
    pub content_sha256: String,
}
impl Observation {
    pub fn new(
        run: Option<u64>,
        elapsed: f64,
        prompt: u64,
        cached: Option<u64>,
        skippy: bool,
        enabled: bool,
        content_sha256: String,
    ) -> Self {
        let cacheable = if skippy {
            prompt.saturating_sub(1)
        } else {
            prompt
        };
        let status = match cached {
            None => "unqualified",
            Some(_) if !enabled => "disabled",
            Some(0) => "miss",
            Some(_) => "hit",
        };
        Self {
            run,
            elapsed_ms: elapsed * 1000.0,
            prompt_tokens: prompt,
            cached_tokens: cached,
            cache_status: status,
            hit_kind: if status != "hit" {
                "none"
            } else if skippy {
                "usage_cached_tokens"
            } else {
                "llama_prompt_cache"
            },
            cacheable_prefix_tokens: cacheable,
            suffix_prefill_tokens: cached.map(|n| cacheable.saturating_sub(n)),
            cache_efficiency: cached.map(|n| {
                if cacheable == 0 {
                    0.0
                } else {
                    number(n) / number(cacheable)
                }
            }),
            content_sha256,
        }
    }
}
#[derive(Serialize)]
pub(super) struct Row {
    pub name: &'static str,
    pub backend: &'static str,
    pub cache_mode: &'static str,
    pub warmup: Option<Observation>,
    pub runs: Vec<Observation>,
    pub verdict: &'static str,
    pub cache_statuses: BTreeSet<&'static str>,
    pub median_elapsed_ms: Option<f64>,
    pub mean_cached_tokens: Option<f64>,
    pub min_cached_tokens: Option<u64>,
    pub max_cached_tokens: Option<u64>,
    pub max_prompt_tokens: u64,
    pub max_cacheable_prefix_tokens: u64,
    pub max_suffix_prefill_tokens: Option<u64>,
    pub max_prompt_cached_ratio: Option<f64>,
    pub min_cache_efficiency: Option<f64>,
    pub max_cache_efficiency: Option<f64>,
}
fn number(value: u64) -> f64 {
    super::super::openai_exchange::stream::number(value)
}
impl Row {
    pub fn summarize(
        name: &'static str,
        skippy: bool,
        enabled: bool,
        warmup: Option<Observation>,
        runs: Vec<Observation>,
    ) -> Self {
        let mut elapsed: Vec<_> = runs.iter().map(|row| row.elapsed_ms).collect();
        elapsed.sort_by(f64::total_cmp);
        let median = match elapsed.len() {
            0 => None,
            n if n % 2 == 1 => Some(elapsed[n / 2]),
            n => Some((elapsed[n / 2 - 1] + elapsed[n / 2]) / 2.0),
        };
        let qualified = !runs.is_empty() && runs.iter().all(|row| row.cached_tokens.is_some());
        let cached: Vec<_> = runs.iter().filter_map(|row| row.cached_tokens).collect();
        let efficiency: Vec<_> = runs.iter().filter_map(|row| row.cache_efficiency).collect();
        let maximum_prompt = runs.iter().map(|row| row.prompt_tokens).max().unwrap_or(0);
        let min_cached = qualified.then(|| cached.iter().copied().min().unwrap_or(0));
        let max_cached = qualified.then(|| cached.iter().copied().max().unwrap_or(0));
        let verdict = if !qualified {
            "UNQUALIFIED missing cache usage"
        } else if !enabled && cached.iter().all(|n| *n == 0) {
            "PASS disabled/no-cache"
        } else if !enabled {
            "FAIL disabled cached"
        } else if cached.iter().all(|n| *n > 0) {
            "PASS all-hit"
        } else {
            "FAIL missed-hit"
        };
        Self {
            name,
            backend: if skippy {
                "skippy-openai"
            } else {
                "llama-server"
            },
            cache_mode: match (skippy, enabled) {
                (false, false) => "cache_prompt=false",
                (false, true) => "cache_prompt=true",
                (true, false) => "prefix-cache-disabled",
                (true, true) => "prefix-cache-enabled",
            },
            cache_statuses: runs.iter().map(|row| row.cache_status).collect(),
            median_elapsed_ms: median,
            mean_cached_tokens: qualified.then(|| {
                cached.iter().map(|n| number(*n)).sum::<f64>() / number(cached.len() as u64)
            }),
            min_cached_tokens: min_cached,
            max_cached_tokens: max_cached,
            max_prompt_tokens: maximum_prompt,
            max_cacheable_prefix_tokens: runs
                .iter()
                .map(|row| row.cacheable_prefix_tokens)
                .max()
                .unwrap_or(0),
            max_suffix_prefill_tokens: qualified.then(|| {
                runs.iter()
                    .filter_map(|row| row.suffix_prefill_tokens)
                    .max()
                    .unwrap_or(0)
            }),
            max_prompt_cached_ratio: max_cached.map(|n| {
                if maximum_prompt == 0 {
                    0.0
                } else {
                    number(n) / number(maximum_prompt)
                }
            }),
            min_cache_efficiency: qualified.then(|| {
                efficiency
                    .iter()
                    .copied()
                    .min_by(f64::total_cmp)
                    .unwrap_or(0.0)
            }),
            max_cache_efficiency: qualified.then(|| {
                efficiency
                    .iter()
                    .copied()
                    .max_by(f64::total_cmp)
                    .unwrap_or(0.0)
            }),
            warmup,
            runs,
            verdict,
        }
    }
}
pub(super) fn markdown(rows: &[Row]) -> String {
    let mut result="| Mode | Verdict | Warmup | Statuses | Cache mode | Median ms | Prompt | Cacheable | Cached | Suffix/uncached | Prompt ratio | Cache efficiency |\n| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n".to_owned();
    for row in rows {
        let decimal =
            |value: Option<f64>| value.map_or_else(|| "n/a".into(), |n| format!("{n:.1}"));
        let percent = |value: Option<f64>| {
            value.map_or_else(|| "unqualified".into(), |n| format!("{:.1}%", n * 100.0))
        };
        let cached = match (row.min_cached_tokens, row.max_cached_tokens) {
            (Some(a), Some(b)) if a == b => a.to_string(),
            (Some(a), Some(b)) => format!("{a}-{b}"),
            _ => "unqualified".into(),
        };
        result.push_str(&format!(
            "| {} | {} | {} | {} | `{}` | {} | {} | {} | {} | {} | {} | {}-{} |\n",
            row.name,
            row.verdict,
            row.warmup.as_ref().map_or("none", |r| r.cache_status),
            row.cache_statuses
                .iter()
                .copied()
                .collect::<Vec<_>>()
                .join(", "),
            row.cache_mode,
            decimal(row.median_elapsed_ms),
            row.max_prompt_tokens,
            row.max_cacheable_prefix_tokens,
            cached,
            row.max_suffix_prefill_tokens
                .map_or_else(|| "unqualified".into(), |n| n.to_string()),
            percent(row.max_prompt_cached_ratio),
            percent(row.min_cache_efficiency),
            percent(row.max_cache_efficiency)
        ));
    }
    result
}
