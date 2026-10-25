use crate::{command::DynResult, process::Cancellation};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::{collections::BTreeSet, fs, path::PathBuf, time::Instant};
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    pub config: PathBuf,
    pub config_sha256: String,
    pub model_keys: Vec<String>,
    pub sources: Vec<Source>,
    pub output_directory: PathBuf,
    pub timeout_seconds: u64,
    pub maximum_source_bytes: u64,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Source {
    pub key: String,
    pub directory: PathBuf,
    pub source_tree_sha256: String,
    pub kind: Kind,
}
#[derive(Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Kind {
    ImmutableSnapshot,
    SuppliedDerivedExport,
}
pub(super) fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn pin(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
pub(super) fn key(value: &str) -> bool {
    !value.is_empty()
        && value != "."
        && value != ".."
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
}
impl Request {
    pub(super) fn validate(&self) -> DynResult<()> {
        if !self.config.is_absolute()
            || !pin(&self.config_sha256)
            || !self.output_directory.is_absolute()
            || !(1..=86400).contains(&self.timeout_seconds)
            || !(1..=1024 * 1024 * 1024 * 1024).contains(&self.maximum_source_bytes)
        {
            return Err("invalid bounded local materialization input".into());
        }
        match fs::symlink_metadata(&self.output_directory) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
            _ => return Err("output must be fresh".into()),
        }
        let parent = self.output_directory.parent().ok_or("output parent")?;
        if parent.canonicalize()? != parent {
            return Err("output parent must be existing canonical directory".into());
        }
        let mut names = BTreeSet::new();
        for source in &self.sources {
            if !key(&source.key)
                || !names.insert(&source.key)
                || !pin(&source.source_tree_sha256)
                || !source.directory.is_absolute()
                || source.directory.canonicalize()? != source.directory
                || !source.directory.is_dir()
            {
                return Err("invalid or duplicate supplied source".into());
            }
            if self.output_directory.starts_with(&source.directory)
                || source.directory.starts_with(&self.output_directory)
            {
                return Err("source and output ancestry must be disjoint".into());
            }
        }
        Ok(())
    }
}
pub(super) struct Budget<'a> {
    pub deadline: Instant,
    pub cancellation: &'a Cancellation,
}
impl Budget<'_> {
    pub(super) fn check(&self) -> DynResult<()> {
        if self.cancellation.is_cancelled() {
            return Err("materialization cancelled".into());
        }
        if Instant::now() >= self.deadline {
            return Err("materialization deadline".into());
        }
        Ok(())
    }
}
pub(super) fn selected<'a>(config: &'a Value, input: &Request) -> DynResult<Vec<&'a Value>> {
    let models = config["models"].as_array().ok_or("model roster")?;
    let mut keys = BTreeSet::new();
    for name in &input.model_keys {
        if !key(name)
            || !keys.insert(name.as_str())
            || !models.iter().any(|m| m["key"].as_str() == Some(name))
        {
            return Err("unknown or duplicate model selection".into());
        }
    }
    let rows: Vec<_> = models
        .iter()
        .filter(|m| input.model_keys.is_empty() || keys.contains(m["key"].as_str().unwrap_or("")))
        .collect();
    let chosen: BTreeSet<_> = rows.iter().filter_map(|m| m["key"].as_str()).collect();
    if chosen.len() != input.sources.len()
        || input
            .sources
            .iter()
            .any(|s| !chosen.contains(s.key.as_str()))
    {
        return Err("supplied source roster must exactly match selected families".into());
    }
    for model in &rows {
        let name = model["key"].as_str().ok_or("model key")?;
        if ![
            "llama32-dense",
            "deepseek-v2-moe",
            "falcon-h1-recurrent",
            "granite-h1-hybrid",
        ]
        .contains(&name)
        {
            return Err("unsupported tokenizer projection family".into());
        }
        let source = input
            .sources
            .iter()
            .find(|s| s.key == name)
            .ok_or("source")?;
        if name == "granite-h1-hybrid" && source.kind != Kind::ImmutableSnapshot {
            return Err("Granite requires complete supplied snapshot".into());
        }
        if ["deepseek-v2-moe", "falcon-h1-recurrent"].contains(&name)
            && source.kind != Kind::SuppliedDerivedExport
        {
            return Err("derived tokenizer export capability required".into());
        }
    }
    Ok(rows)
}

/// Typed immutable input projection; this is not a dispatched download receipt.
pub(super) fn acquisition_plan(config: &Value, selected: &[&Value]) -> Value {
    serde_json::json!({"status":"NOT_EXECUTED","models":selected.iter().map(|m|serde_json::json!({"key":m["key"],"repo":m["repo"],"revision":m["revision"],"filename":m["filename"],"sha256":m["sha256"],"tokenizer_snapshot":m["vllm_hf_config"],"tokenizer_sha256":m["tokenizer_sha256"]})).collect::<Vec<_>>(),"dataset":config["thoughtworks"]["dataset"],"manifest_selection":config["thoughtworks"]["selection"],"required_native_frontend":"competitive-inputs materialize/prefetch pending acquisition and semantic export implementation"})
}
