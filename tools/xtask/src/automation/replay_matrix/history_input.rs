use crate::command::DynResult;
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
pub(super) struct Model {
    pub family: String,
    pub class: String,
    pub quant: String,
    pub repo: String,
    pub revision: String,
    pub file: String,
    pub sha256: String,
}
#[derive(Deserialize)]
pub(super) struct Replay {
    pub mode: String,
    pub passes: u32,
    pub concurrency: Vec<usize>,
    pub sessions_per_concurrency: usize,
    pub minimum_context_tokens: u64,
    pub minimum_session_prompt_tokens: u64,
    pub max_output_tokens: u64,
    pub minimum_worker_waves: usize,
    pub warmup_turns: usize,
    pub dataset_revision: String,
    pub dataset_sha256: String,
}
pub(super) struct Input {
    pub models: Vec<(Model, Value)>,
    pub replay: Replay,
    pub replay_value: Value,
    pub hardware: Value,
    pub source_sha: String,
    pub backend_sha256: Option<String>,
    pub root: PathBuf,
    pub label: String,
}
pub(super) fn load(
    matrix: &Path,
    replay: &Path,
    hardware: &Path,
    root: PathBuf,
    label: String,
    source_sha: String,
    backend_sha256: Option<String>,
) -> DynResult<Input> {
    super::input::load(matrix).map_err(|error| error.to_string())?;
    let matrix: Value = read(matrix)?;
    let replay_value: Value = read(replay)?;
    if matrix["replay"] != replay_value {
        return Err("history replay parameters differ from matrix".into());
    }
    let replay: Replay = serde_json::from_value(replay_value.clone())?;
    if replay.mode != "all" {
        return Err("history requires complete recorded sessions".into());
    }
    pin(&replay.dataset_revision, 40)?;
    pin(&replay.dataset_sha256, 64)?;
    pin(&source_sha, 40)?;
    if let Some(digest) = &backend_sha256 {
        pin(digest, 64)?;
    }
    safe_component(&label)?;
    let hardware: Value = read(hardware)?;
    if hardware["machine_model"].as_str().is_none_or(str::is_empty) {
        return Err("hardware machine_model is missing".into());
    }
    for field in ["chip", "os_version"] {
        if hardware[field].as_str().is_none_or(str::is_empty) {
            return Err(format!("hardware fingerprint is missing {field}").into());
        }
    }
    for field in ["gpu_cores", "unified_memory_bytes"] {
        if hardware[field].as_u64().is_none() {
            return Err(format!("hardware fingerprint is missing {field}").into());
        }
    }
    let mut models = Vec::new();
    let mut families = BTreeSet::new();
    for value in matrix["models"]
        .as_array()
        .ok_or("matrix models are missing")?
    {
        let model: Model = serde_json::from_value(value.clone())?;
        safe_component(&model.family)?;
        pin(&model.revision, 40)?;
        pin(&model.sha256, 64)?;
        if !families.insert(model.family.clone())
            || model.repo.is_empty()
            || model.file.is_empty()
            || model.quant.is_empty()
            || !matches!(model.class.as_str(), "dense" | "moe" | "hybrid-recurrent")
        {
            return Err("invalid or duplicate history model".into());
        }
        models.push((model, value.clone()));
    }
    if models.is_empty() {
        return Err("history model roster is empty".into());
    }
    Ok(Input {
        models,
        replay,
        replay_value,
        hardware,
        source_sha,
        backend_sha256,
        root,
        label,
    })
}
pub(super) fn pin(value: &str, size: usize) -> DynResult<()> {
    if value.len() != size
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("history requires an immutable {size}-hex identity").into());
    }
    Ok(())
}
pub(super) fn safe_component(value: &str) -> DynResult<()> {
    if value.is_empty()
        || matches!(value, "." | "..")
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
    {
        return Err("history path component is unsafe".into());
    }
    Ok(())
}
pub(super) fn read<T: serde::de::DeserializeOwned>(path: &Path) -> DynResult<T> {
    if !std::fs::symlink_metadata(path)?.file_type().is_file() {
        return Err("history evidence must be a regular file".into());
    }
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}
