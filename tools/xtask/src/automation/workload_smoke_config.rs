//! Candidate-only unsplit stage configuration for workload HTTP certification.
use crate::{command::DynResult, repository::check_args::Grammar};
use serde::Serialize;
use std::{fs, io::Write};

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation workload-smoke-config --output PATH --model-id ID --model-path PATH --model-sha256 SHA --layer-end N --n-gpu-layers N [--projector-path PATH]",
    values: &[
        "--output",
        "--model-id",
        "--model-path",
        "--model-sha256",
        "--layer-end",
        "--n-gpu-layers",
        "--projector-path",
    ],
    flags: &["--help"],
};

#[derive(Serialize)]
struct Device {
    backend_device: &'static str,
}

#[derive(Serialize)]
struct Config<'a> {
    run_id: &'static str,
    topology_id: &'static str,
    model_id: &'a str,
    model_path: &'a str,
    source_model_sha256: &'a str,
    stage_id: &'static str,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    ctx_size: u32,
    lane_count: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_gpu_layers: i32,
    selected_device: Option<Device>,
    kv_offload: Option<bool>,
    op_offload: Option<bool>,
    resident_tensor_names: Vec<String>,
    execution_contract: &'static str,
    native_mtp_enabled: bool,
    load_mode: &'static str,
    bind_addr: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    projector_path: Option<&'a str>,
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(value) => value,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return crate::repository::check_report::CheckReport::success(format!(
            "{}\n",
            GRAMMAR.usage
        ))
        .emit();
    }
    if !parsed.positionals.is_empty() {
        return Err("unexpected positional arguments".into());
    }
    let required = |key| -> DynResult<&str> {
        parsed
            .last(key)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("missing {key}").into())
    };
    let sha = required("--model-sha256")?;
    if sha.len() != 64 || !sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("model SHA256 must contain 64 hexadecimal characters".into());
    }
    let layers = required("--n-gpu-layers")?.parse::<i32>()?;
    let cpu = layers == 0;
    let config = Config {
        run_id: "workload-http-smoke",
        topology_id: "workload-http-smoke-local",
        model_id: required("--model-id")?,
        model_path: required("--model-path")?,
        source_model_sha256: sha,
        stage_id: "stage-0",
        stage_index: 0,
        layer_start: 0,
        layer_end: required("--layer-end")?.parse()?,
        ctx_size: 2048,
        lane_count: 1,
        n_batch: 2048,
        n_ubatch: 2048,
        n_gpu_layers: layers,
        selected_device: cpu.then_some(Device {
            backend_device: "CPU",
        }),
        kv_offload: cpu.then_some(false),
        op_offload: cpu.then_some(false),
        resident_tensor_names: Vec::new(),
        execution_contract: "",
        native_mtp_enabled: false,
        load_mode: "runtime-slice",
        bind_addr: "127.0.0.1:0",
        projector_path: parsed
            .last("--projector-path")
            .filter(|value| !value.is_empty()),
    };
    let output = required("--output")?;
    let mut bytes = serde_json::to_vec_pretty(&config)?;
    bytes.push(b'\n');
    fs::File::create(output)?.write_all(&bytes)?;
    Ok(())
}
