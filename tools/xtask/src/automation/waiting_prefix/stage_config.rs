//! Single-stage cache configuration for comparable old/new A/B cells.
use super::{options, publish};
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Payload {
    ResidentKv,
    KvRecurrent,
    FullState,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    model_id: String,
    model_path: PathBuf,
    source_model_sha256: String,
    layer_end: u32,
    ctx_size: u32,
    lane_count: u32,
    n_gpu_layers: i32,
    payload: Payload,
    cache_entries: u64,
}

impl Input {
    fn validate(&self) -> DynResult<()> {
        if self.model_id.trim().is_empty()
            || !self.model_path.is_absolute()
            || self.source_model_sha256.len() != 64
            || !self
                .source_model_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            || [self.layer_end, self.ctx_size, self.lane_count].contains(&0)
            || self.cache_entries == 0
            || self.n_gpu_layers < -1
        {
            return Err("invalid A/B stage identity, layer range or cache capacity".into());
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct Cache<'a> {
    mode: &'static str,
    payload: &'a Payload,
    max_entries: u64,
    max_bytes: u64,
    min_tokens: u64,
    shared_prefix_stride_tokens: u64,
    shared_prefix_record_limit: u64,
}

#[derive(Serialize)]
struct Config<'a> {
    run_id: &'static str,
    topology_id: &'static str,
    model_id: &'a str,
    model_path: &'a Path,
    source_model_sha256: &'a str,
    stage_id: &'static str,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    ctx_size: u32,
    lane_count: u32,
    n_gpu_layers: i32,
    load_mode: &'static str,
    execution_contract: &'static str,
    bind_addr: &'static str,
    upstream: Option<()>,
    downstream: Option<()>,
    kv_cache: Cache<'a>,
}

fn config(input: &Input) -> DynResult<Config<'_>> {
    input.validate()?;
    Ok(Config {
        run_id: "skippy-waiting-prefix-ab",
        topology_id: "skippy-waiting-prefix-ab-single-stage",
        model_id: &input.model_id,
        model_path: &input.model_path,
        source_model_sha256: &input.source_model_sha256,
        stage_id: "stage-0",
        stage_index: 0,
        layer_start: 0,
        layer_end: input.layer_end,
        ctx_size: input.ctx_size,
        lane_count: input.lane_count,
        n_gpu_layers: input.n_gpu_layers,
        load_mode: "runtime-slice",
        execution_contract: "",
        bind_addr: "127.0.0.1:0",
        upstream: None,
        downstream: None,
        kv_cache: Cache {
            mode: "lookup-record",
            payload: &input.payload,
            max_entries: input.cache_entries,
            max_bytes: 0,
            min_tokens: 64,
            shared_prefix_stride_tokens: 128,
            shared_prefix_record_limit: 1,
        },
    })
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let opts = options(args, &["--input", "--output"], &["--input", "--output"])?;
    let mut input: Input = serde_json::from_slice(&std::fs::read(opts["--input"])?)?;
    input.validate()?;
    input.model_path = input.model_path.canonicalize()?;
    if !input.model_path.is_file() {
        return Err("A/B source model must be a regular file".into());
    }
    let hash = crate::product::digest::file_sha256(&input.model_path).map_err(|e| e.error)?;
    if hash != input.source_model_sha256 {
        return Err("A/B source model SHA-256 mismatch".into());
    }
    let mut bytes = serde_json::to_vec_pretty(&config(&input)?)?;
    bytes.push(b'\n');
    publish(Path::new(opts["--output"]), &bytes)
}

#[cfg(test)]
#[path = "stage_config_tests.rs"]
mod tests;
