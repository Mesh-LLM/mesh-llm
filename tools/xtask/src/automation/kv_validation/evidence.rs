use super::contract::Backend;
use crate::command::DynResult;
use std::collections::{BTreeMap, BTreeSet};

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RestartReceipt {
    status: String,
    source: String,
    seed_process_id: u32,
    restore_process_id: u32,
    state_integrity: String,
    fixed_input_predictions: Vec<i32>,
}

pub(super) fn process_restart(bytes: &[u8]) -> DynResult<()> {
    let receipt: RestartReceipt = serde_json::from_slice(bytes)?;
    if receipt.status != "passed"
        || receipt.source != "l3"
        || receipt.seed_process_id == 0
        || receipt.restore_process_id == 0
        || receipt.seed_process_id == receipt.restore_process_id
        || receipt.state_integrity != "sha256_exact"
        || receipt.fixed_input_predictions.len() != 4
    {
        return Err(
            "L3 restart lacks separate successful processes and exact state/content evidence"
                .into(),
        );
    }
    Ok(())
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CacheEvidence {
    status: String,
    content_parity: String,
    cold_content: String,
    repeat: CacheResponse,
    growing_chat: CacheResponse,
    growing_chat_repeat: CacheResponse,
}
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
struct CacheResponse {
    cached_tokens: u64,
    prompt_tokens: u64,
    content: String,
}
impl CacheEvidence {
    pub(super) fn verify(&self) -> DynResult<()> {
        if self.status != "pass"
            || self.content_parity != "exact"
            || self.cold_content.is_empty()
            || self.repeat.content != self.cold_content
            || self.growing_chat.content.is_empty()
            || self.growing_chat_repeat.content != self.growing_chat.content
            || [&self.repeat, &self.growing_chat, &self.growing_chat_repeat]
                .into_iter()
                .any(|r| r.cached_tokens == 0 || r.cached_tokens > r.prompt_tokens)
        {
            return Err("generic cache evidence lacks exact content parity and valid reuse".into());
        }
        Ok(())
    }
}

pub(super) fn doctor(value: &serde_json::Value, runtime: &std::path::Path) -> DynResult<()> {
    if value["native_runtime"]["path"].as_str() != runtime.to_str()
        || value
            .get("runtime_issue")
            .is_none_or(|issue| !issue.is_null())
    {
        return Err(
            "doctor requires exact runtime leaf and an explicitly clean runtime issue".into(),
        );
    }
    Ok(())
}

const TYPES: [&str; 8] = ["f32", "f16", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0"];
fn fields(line: &str) -> BTreeMap<&str, &str> {
    line.split_ascii_whitespace()
        .filter_map(|word| word.split_once('='))
        .collect()
}
pub(super) fn matrix(text: &str, backend: Backend) -> DynResult<()> {
    let mut pairs = BTreeSet::new();
    let mut models = 0;
    let mut summaries = 0;
    let mut observed = BTreeSet::new();
    let mut skips = BTreeSet::new();
    for line in text.lines() {
        let f = fields(line);
        if line.starts_with("MODEL ") {
            models += 1;
            if f.get("backend") != Some(&backend.device()) || f.get("loads") != Some(&"1") {
                return Err("KV matrix model load/backend evidence mismatch".into());
            }
        } else if line.starts_with("REAL_FA ")
            || line.starts_with("PASS ")
            || line.starts_with("SKIP ")
        {
            let k = *f.get("K").ok_or("matrix missing K")?;
            let v = *f.get("V").ok_or("matrix missing V")?;
            if !TYPES.contains(&k)
                || !TYPES.contains(&v)
                || f.get("backend") != Some(&backend.device())
            {
                return Err("KV matrix unexpected dtype/backend".into());
            }
            if line.starts_with("SKIP ") {
                if backend != Backend::Metal
                    || k == v
                    || f.get("reason") != Some(&"preexisting_Metal_requires_matching_KV_types")
                    || !skips.insert((k, v))
                {
                    return Err("KV matrix cannot skip required pair".into());
                }
            } else if line.starts_with("REAL_FA ") {
                if !observed.insert((k, v))
                    || f.get("q")
                        .and_then(|s| s.parse::<usize>().ok())
                        .is_none_or(|q| !(20..=64).contains(&q))
                {
                    return Err("KV matrix missing wide-prefill graph evidence".into());
                }
            } else {
                let completion = f.get("completion_hex").ok_or("matrix completion missing")?;
                let seconds = f
                    .get("seconds")
                    .ok_or("matrix timing missing")?
                    .parse::<f64>()?;
                if !seconds.is_finite()
                    || seconds < 0.0
                    || f.get("decoded") != Some(&"16")
                    || f.get("attention_layers")
                        .and_then(|v| v.parse::<usize>().ok())
                        .is_none_or(|n| n == 0)
                    || completion.is_empty()
                    || completion.len() % 2 != 0
                    || !completion.bytes().all(|b| b.is_ascii_hexdigit())
                    || !pairs.insert((k, v))
                {
                    return Err("invalid KV matrix completion/graph evidence".into());
                }
            }
        } else if line.starts_with("SUMMARY ") {
            summaries += 1;
            if f.get("backend") != Some(&backend.device())
                || f.get("model_loads") != Some(&"1")
                || f.get("failed") != Some(&"0")
                || f.get("passed").and_then(|v| v.parse::<usize>().ok()) != Some(backend.pairs())
            {
                return Err("KV matrix summary failed".into());
            }
        } else if line.starts_with("FAIL ") {
            return Err("KV matrix reported failure".into());
        }
    }
    let expected: BTreeSet<_> = TYPES
        .into_iter()
        .flat_map(|k| {
            TYPES
                .into_iter()
                .filter_map(move |v| (backend != Backend::Metal || k == v).then_some((k, v)))
        })
        .collect();
    if models != 1
        || summaries != 1
        || pairs != expected
        || observed != expected
        || skips.len() != if backend == Backend::Metal { 56 } else { 0 }
    {
        return Err("KV matrix incomplete/duplicate roster".into());
    }
    Ok(())
}

pub(super) fn rust_test(text: &str, test: &str) -> DynResult<()> {
    if !text.contains(&format!("test {test} ..."))
        || !text.contains("test result: ok. 1 passed; 0 failed; 0 ignored;")
    {
        return Err("required actual-model Rust cache test did not execute".into());
    }
    Ok(())
}
