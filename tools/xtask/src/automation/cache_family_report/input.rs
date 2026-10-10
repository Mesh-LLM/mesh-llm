use crate::command::DynResult;
use serde::Deserialize;
use std::{fs, io::Read, path::Path};

pub(super) const FAMILIES: &[&str] = &[
    "Qwen3Next",
    "Falcon-H1",
    "Llama",
    "Qwen3 dense",
    "DeepSeek2",
    "GLM-4.7 Flash",
    "GLM4",
    "Gemma4 A4B",
    "Gemma4 E4B",
    "Gemma3",
    "Gemma2",
    "OLMo",
    "MiniMax M2.7",
];
pub(super) const USE_CASES: &[&str] = &[
    "tool_calling",
    "text_to_sql",
    "coding_agent_loop",
    "issue_fixing",
    "code_refinement",
    "few_shot_reasoning",
    "open_chat",
    "summarization_rag",
];

#[derive(Default, Deserialize)]
#[serde(default)]
pub(super) struct Row {
    pub family: String,
    pub model_id: String,
    pub payload: Option<String>,
    pub stage_load_mode: String,
    pub use_case: Option<String>,
    pub use_case_label: Option<String>,
    pub prefix_tokens: Option<u64>,
    pub benchmark_prompt_token_count: Option<u64>,
    pub notes: String,
    pub skippy: Skippy,
    pub llama_server: Llama,
    pub case: Case,
}

#[derive(Default, Deserialize)]
#[serde(default)]
pub(super) struct Skippy {
    pub status: Option<String>,
    pub cache_storage_bytes: Option<u64>,
    pub cache_hit_import_ms: Option<Vec<Option<f64>>>,
    pub cache_hit_decode_ms: Option<Vec<Option<f64>>>,
    pub cache_hit_total_ms: Option<f64>,
    pub recompute_total_ms: Option<f64>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
pub(super) struct Llama {
    pub status: Option<String>,
    pub warm_median_ms: Option<f64>,
    pub warm_mean_ms: Option<f64>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
pub(super) struct Case {
    pub resident_kv_bytes_per_token: Option<u64>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
pub(super) struct Corpus {
    pub use_cases: Vec<UseCase>,
}

#[derive(Default, Deserialize)]
#[serde(default)]
pub(super) struct UseCase {
    pub key: String,
    pub label: Option<String>,
    pub source: Source,
}

#[derive(Default, Deserialize)]
#[serde(default)]
pub(super) struct Source {
    pub dataset: String,
    pub config: String,
    pub split: String,
    pub row_idx: Option<u64>,
}

pub(super) fn load<T: serde::de::DeserializeOwned>(path: &Path) -> DynResult<T> {
    if fs::metadata(path)?.len() > 16 * 1024 * 1024 {
        return Err("cache report input exceeds 16 MiB".into());
    }
    let mut bytes = Vec::new();
    fs::File::open(path)?
        .take(16 * 1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 16 * 1024 * 1024 {
        return Err("cache report input exceeds 16 MiB".into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}

pub(super) fn validate(row: &Row) -> DynResult<()> {
    let metrics = [
        &row.skippy.cache_hit_import_ms,
        &row.skippy.cache_hit_decode_ms,
    ];
    for values in metrics.into_iter().flatten() {
        if values.len() > 100_000 || values.iter().flatten().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("invalid or oversized cache timing series".into());
        }
    }
    for value in [
        row.skippy.cache_hit_total_ms,
        row.skippy.recompute_total_ms,
        row.llama_server.warm_median_ms,
        row.llama_server.warm_mean_ms,
    ]
    .into_iter()
    .flatten()
    {
        if !value.is_finite() || value < 0.0 {
            return Err("cache timing must be finite and nonnegative".into());
        }
    }
    if let (Some(imports), Some(decodes)) = (
        &row.skippy.cache_hit_import_ms,
        &row.skippy.cache_hit_decode_ms,
    ) && imports.len() != decodes.len()
    {
        return Err("paired cache timing series lengths differ".into());
    }
    if let (Some(imports), Some(decodes)) = (
        &row.skippy.cache_hit_import_ms,
        &row.skippy.cache_hit_decode_ms,
    ) && imports
        .iter()
        .zip(decodes)
        .any(|(a, b)| a.zip(*b).is_some_and(|(a, b)| !(a + b).is_finite()))
    {
        return Err("paired cache timing sum overflow".into());
    }
    Ok(())
}

pub(super) fn order<'a>(text: &'a str, roster: &[&str]) -> (usize, &'a str) {
    (
        roster
            .iter()
            .position(|value| *value == text)
            .unwrap_or(roster.len()),
        text,
    )
}
