//! Final comparison launch settings and pre-load resolved-capacity evidence.
use super::{
    Prepared, command,
    prepared_tools::{Backend, Tools},
    write,
};
use crate::{command::DynResult, process};
use serde_json::{Value, json};

pub(super) fn arguments(args: &mut Vec<String>, engine: &str, tools: &Tools) {
    if let Some(index) = args.iter().position(|arg| arg == "--n-gpu-layers") {
        args[index + 1] = if tools.backend == Backend::Cpu {
            "0"
        } else {
            "99"
        }
        .into();
    }
    args.extend([
        "--threads".into(),
        tools.threads.to_string(),
        "--threads-batch".into(),
        tools.threads.to_string(),
    ]);
    if engine == "skippy" {
        args.splice(0..0, ["--output".into(), "jsonl".into()]);
        args.extend([
            "--ctx-size".into(),
            "131072".into(),
            "--prefill-chunk-size".into(),
            "64".into(),
            "--prefill-chunk-policy".into(),
            "fixed".into(),
        ]);
    }
}

pub(super) fn doctor(
    tools: &Tools,
    root: &std::path::Path,
    output: &std::path::Path,
) -> DynResult<()> {
    let spec = command::spec(
        root,
        tools
            .skippy
            .path
            .to_str()
            .ok_or("CLI path Unicode required")?,
        &[
            "--output".into(),
            "json".into(),
            "--runtime-bundle".into(),
            tools.runtime.path.to_string_lossy().into(),
            "--runtime-selection".into(),
            tools.backend.name().into(),
            "doctor".into(),
        ],
    )?;
    command::run(&spec, 120, &output.join("doctor"))?;
    let value: Value = serde_json::from_slice(&std::fs::read(output.join("doctor.stdout.log"))?)?;
    if value["native_runtime"]["path"].as_str() != tools.runtime.path.to_str()
        || value
            .get("runtime_issue")
            .is_none_or(|issue| !issue.is_null())
    {
        return Err("doctor must select the exact prepared runtime leaf without issue".into());
    }
    Ok(())
}

pub(super) fn effective_config(input: &Prepared, server: &process::ProcessSpec) -> DynResult<()> {
    let mut probe = process::ProcessSpec {
        executable: server.executable.clone(),
        arguments: server.arguments.clone(),
        cwd: server.cwd.clone(),
        environment: server.environment.clone(),
    };
    if let Some(index) = probe
        .arguments
        .iter()
        .position(|arg| matches!(arg,process::Value::Public(v) if v=="--output"))
    {
        probe.arguments[index + 1] = process::Value::Public("json".into());
    }
    probe
        .arguments
        .push(process::Value::Public("--print-effective-config".into()));
    command::run(&probe, 900, &input.output.join("effective-config"))?;
    let value: Value = serde_json::from_slice(&std::fs::read(
        input.output.join("effective-config.stdout.log"),
    )?)?;
    validate_config(
        &value,
        input
            .tools
            .as_ref()
            .ok_or("prepared launch missing tools")?,
    )?;
    write(&input.output.join("effective-config.json"), &value)
}
pub(super) fn validate_config(value: &Value, tools: &Tools) -> DynResult<()> {
    let stage = &value["stage"];
    let lanes = stage["lane_count"]
        .as_u64()
        .filter(|n| *n > 0)
        .ok_or("resolved lanes unavailable")?;
    if stage["ctx_size"] != 131072
        || stage["n_batch"] != 2048
        || stage["n_ubatch"] != 512
        || stage["cache_type_k"] != "f16"
        || stage["cache_type_v"] != "f16"
        || stage["flash_attn_type"] != "enabled"
        || stage["kv_cache"]["l2_max_bytes"] != 8589934592u64
        || value["options"]["kv-cache-disk"] != "off"
        || value["execution"]["threads"] != tools.threads
        || value["execution"]["threads_batch"] != tools.threads
        || value["frontend"]["prefill_chunk_size"] != 64
        || value["frontend"]["prefill_chunk_policy"] != "fixed"
        || value["frontend"]["generation_concurrency"].as_u64() != Some(lanes)
        || value["constraints"]["unified_kv"] != "required"
        || stage["kv_unified"] == false
        || tools
            .diagnostic_generation_lanes
            .is_some_and(|n| u64::from(n) != lanes)
    {
        return Err("resolved Skippy settings disagree with the final comparison contract".into());
    }
    Ok(())
}
pub(super) fn configured(output: &std::path::Path) -> Option<Value> {
    let config: Value =
        serde_json::from_slice(&std::fs::read(output.join("effective-config.json")).ok()?).ok()?;
    let context = config["stage"]["ctx_size"].as_u64()?;
    let lanes = config["stage"]["lane_count"].as_u64()?;
    Some(
        json!({"status":"observed","source":"print-effective-config_before_model_load",
        "context_tokens_per_lane":context,"lanes":lanes,"configured_pool_tokens":context.checked_mul(lanes)?,
        "native_allocation":"separately observed; configured pool is not a memory residency measurement",
        "frontend":config["frontend"],"execution":config["execution"],"kv_cache":config["stage"]["kv_cache"]}),
    )
}
