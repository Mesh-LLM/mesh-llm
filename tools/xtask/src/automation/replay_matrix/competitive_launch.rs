//! Prepared competitive launch: exact planned capacity, bounded provenance, no shell input.
use crate::{
    command::DynResult,
    process::{ProcessSpec, Value as Argument},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::{Path, PathBuf},
};
#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Artifact {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Backend {
    pub executable: Artifact,
    pub version_sha256: String,
    pub cwd: PathBuf,
    pub runtime: Option<Artifact>,
    pub tokenizer: Option<Artifact>,
    pub hf_config: Option<Artifact>,
    pub comparison_model: Option<Artifact>,
    pub match_kv_capacity: bool,
}
pub(super) struct Prepared {
    pub spec: ProcessSpec,
    pub served: String,
    pub provenance: Value,
    pub command: Vec<String>,
}
pub(super) fn file(artifact: &Artifact) -> DynResult<String> {
    if !artifact.path.is_absolute() || !std::fs::symlink_metadata(&artifact.path)?.is_file() {
        return Err("competitive artifact must be an absolute regular file".into());
    }
    let actual =
        crate::product::digest::file_sha256(&artifact.path).map_err(|error| error.error)?;
    if actual != artifact.sha256 {
        return Err("competitive artifact SHA-256 mismatch".into());
    }
    Ok(actual)
}
pub(super) fn tree(artifact: &Artifact) -> DynResult<String> {
    crate::automation::waiting_prefix::native_identity::verify(&artifact.path, &artifact.sha256)
}
pub(super) fn environment(runtime: Option<&Artifact>, cache: bool) -> BTreeMap<OsString, Argument> {
    // Preserve login/user settings. Secret classification redacts values from diagnostics;
    // values too large/multiline cannot satisfy the supervisor's bounded secret contract.
    let mut env: BTreeMap<_, _> = std::env::vars_os()
        .map(|(key, value)| {
            (
                key,
                if value.is_empty() {
                    Argument::Public(value)
                } else {
                    Argument::Secret(value)
                },
            )
        })
        .collect();
    if let Some(runtime) = runtime {
        env.insert(
            "LLAMA_STAGE_BUILD_DIR".into(),
            Argument::Public(runtime.path.as_os_str().to_owned()),
        );
    }
    env.insert(
        "SKIPPY_TELEMETRY_STDERR".into(),
        Argument::Public(if cache { "1" } else { "0" }.into()),
    );
    env.insert(
        "SKIPPY_NATIVE_MTP_GREEDY_SAMPLING_FASTPATH".into(),
        Argument::Public("1".into()),
    );
    env
}
pub(super) fn prepare(
    document: &Value,
    cell: &Value,
    backend: &Backend,
    model_file: &Artifact,
    port: u16,
    directory: &Path,
) -> DynResult<Prepared> {
    let model = document["models"]
        .as_array()
        .ok_or("models")?
        .iter()
        .find(|model| model["key"] == cell["model"])
        .ok_or("model missing")?;
    let model_hash = file(model_file)?;
    if model["sha256"].as_str() != Some(&model_hash) {
        return Err("model bytes differ from competitive source pin".into());
    }
    let binary_hash = file(&backend.executable)?;
    if !backend.cwd.is_absolute() || !backend.cwd.is_dir() {
        return Err("backend cwd must exist and be absolute".into());
    }
    let arm = cell["arm"].as_str().ok_or("arm")?;
    let cache = cell["workload"] == "thoughtworks";
    let context = if cache {
        cell["context_size"].as_u64()
    } else {
        model["synthetic_context_size"].as_u64()
    }
    .ok_or("context")?;
    let lanes = if cache {
        cell["active_lanes"].as_u64()
    } else {
        document["synthetic"]["active_lanes"].as_u64()
    }
    .ok_or("lanes")?;
    let served = model["model_id"].as_str().ok_or("model ID")?;
    let served = if arm == "sglang" {
        served.replace(':', "-")
    } else {
        served.into()
    };
    let mut provenance = json!({"binary_sha256":binary_hash,"model_sha256":model_hash,"backend_version_sha256":backend.version_sha256});
    let selection = Selection {
        model,
        cell,
        backend,
        artifact: model_file,
        port,
        directory,
        capacity: Capacity { context, lanes },
        cache,
    };
    let command = match arm {
        "mesh" | "mesh-adaptive" => mesh(&selection, &mut provenance)?,
        "llama" | "vllm" | "sglang" => external(&selection, arm, &served, &mut provenance)?,
        _ => return Err("unsupported competitive arm".into()),
    };
    Ok(Prepared {
        spec: ProcessSpec {
            executable: backend.executable.path.clone(),
            arguments: command
                .iter()
                .skip(1)
                .map(|value| Argument::Public(value.into()))
                .collect(),
            cwd: backend.cwd.clone(),
            environment: environment(backend.runtime.as_ref(), cache),
        },
        served,
        provenance,
        command,
    })
}
struct Capacity {
    context: u64,
    lanes: u64,
}
struct Selection<'a> {
    model: &'a Value,
    cell: &'a Value,
    backend: &'a Backend,
    artifact: &'a Artifact,
    port: u16,
    directory: &'a Path,
    capacity: Capacity,
    cache: bool,
}
fn mesh(selected: &Selection<'_>, provenance: &mut Value) -> DynResult<Vec<String>> {
    let Selection {
        model,
        cell,
        backend,
        artifact: model_file,
        port,
        directory,
        capacity,
        cache,
    } = selected;
    let context = capacity.context;
    let lanes = capacity.lanes;
    let cache = *cache;
    let port = *port;
    let runtime = backend
        .runtime
        .as_ref()
        .ok_or("mesh requires pinned native runtime directory")?;
    provenance["runtime_directory_sha256"] = tree(runtime)?.into();
    let stage = stage(model, model_file, port, Capacity { context, lanes }, cache);
    let path = directory.join("stage.json");
    crate::command::write_json_file(&path, &stage)?;
    let mut command = vec![
        backend
            .executable
            .path
            .to_str()
            .ok_or("executable Unicode")?
            .into(),
        "serve-openai".into(),
        "--config".into(),
        path.to_str().ok_or("stage Unicode")?.into(),
        "--bind-addr".into(),
        format!("127.0.0.1:{port}"),
        "--model-id".into(),
        model["model_id"].as_str().ok_or("model ID")?.into(),
        "--generation-concurrency".into(),
        lanes.to_string(),
        "--generation-queue-capacity".into(),
        "256".into(),
        "--generation-admission-timeout-secs".into(),
        "600".into(),
        "--default-max-tokens".into(),
        cell["output_tokens"].as_u64().ok_or("output")?.to_string(),
        "--telemetry-level".into(),
        if cache { "summary" } else { "off" }.into(),
    ];
    if cell["arm"] == "mesh-adaptive" {
        command.extend(
            [
                "--adaptive-generation-concurrency",
                "--adaptive-generation-min-concurrency",
                "1",
            ]
            .map(str::to_owned),
        );
    }
    Ok(command)
}
fn stage(model: &Value, artifact: &Artifact, port: u16, capacity: Capacity, cache: bool) -> Value {
    let mut value = json!({"run_id":format!("competitive-{}",model["key"].as_str().unwrap_or("invalid")),"topology_id":"competitive-single-stage","model_id":model["model_id"],"model_path":artifact.path,"source_model_sha256":artifact.sha256,"stage_id":"stage-0","stage_index":0,"layer_start":0,"layer_end":model["layer_end"],"ctx_size":capacity.context,"lane_count":capacity.lanes,"n_batch":2048,"n_ubatch":512,"n_gpu_layers":-1,"cache_type_k":"f16","cache_type_v":"f16","native_mtp_enabled":false,"load_mode":"runtime-slice","execution_contract":"","bind_addr":format!("127.0.0.1:{port}"),"upstream":null,"downstream":null});
    if cache {
        value["kv_cache"] = json!({"mode":"lookup-record","payload":model["cache_payload"],"max_entries":512,"max_bytes":0,"min_tokens":64,"shared_prefix_stride_tokens":128,"shared_prefix_record_limit":2});
    }
    value
}
fn external(
    selected: &Selection<'_>,
    name: &str,
    served: &str,
    provenance: &mut Value,
) -> DynResult<Vec<String>> {
    let Selection {
        model,
        backend,
        port,
        capacity,
        cache,
        ..
    } = selected;
    let context = capacity.context;
    let lanes = capacity.lanes;
    let cache = *cache;
    let port = *port;
    use super::external_config::{Arm, Engine};
    let engine = match name {
        "llama" => Engine::Llama,
        "vllm" => Engine::Vllm,
        _ => Engine::Sglang,
    };
    let source = comparison_source(selected, name, provenance)?;
    let extra_args = capacity_args(model, name, context, backend.match_kv_capacity)?;
    let arm = Arm {
        label: name.into(),
        engine,
        executable: backend
            .executable
            .path
            .to_str()
            .ok_or("executable Unicode")?
            .into(),
        model: source.to_str().ok_or("model Unicode")?.into(),
        served_model: Some(served.into()),
        context_size: context,
        max_concurrency: usize::try_from(lanes)?,
        tokenizer: backend
            .tokenizer
            .as_ref()
            .map(|value| value.path.to_string_lossy().into_owned()),
        hf_config: backend
            .hf_config
            .as_ref()
            .map(hf_config_directory)
            .transpose()?
            .map(|value| value.to_string_lossy().into_owned()),
        prefix_cache: cache,
        batch_size: 2048,
        ubatch_size: 512,
        extra_args,
        cwd: backend.cwd.clone(),
    };
    arm.validate_prepared()?;
    super::external_command::server(&arm, &backend.executable.path, port)
}

fn comparison_source(
    selected: &Selection<'_>,
    name: &str,
    provenance: &mut Value,
) -> DynResult<PathBuf> {
    let model = selected.model;
    let backend = selected.backend;
    let artifact = selected.artifact;
    let mut source = artifact.path.clone();
    if name != "llama" {
        let tokenizer = backend
            .tokenizer
            .as_ref()
            .ok_or("optional engine requires pinned tokenizer")?;
        if model["tokenizer_sha256"].as_str() != Some(&tokenizer.sha256) {
            return Err("tokenizer source pin differs".into());
        }
        tree(tokenizer)?;
        if let Some(alternate) = &backend.comparison_model {
            let pin = &model["comparison_inputs"][name];
            if pin["sha256"].as_str() != Some(&alternate.sha256)
                || !pin["tensor_equivalence_sha256"].is_string()
            {
                return Err("alternate comparison input lacks pinned equivalence context".into());
            }
            provenance["comparison_input_sha256"] = tree(alternate)?.into();
            provenance["tensor_equivalence_sha256"] = pin["tensor_equivalence_sha256"].clone();
            source = alternate.path.clone();
        }
    }
    if name == "vllm" {
        let config = backend
            .hf_config
            .as_ref()
            .ok_or("vllm HF config pin missing")?;
        if model["vllm_hf_config"]["sha256"].as_str() != Some(&config.sha256) {
            return Err("HF config source pin differs".into());
        }
        hf_config_directory(config)?;
    }
    Ok(source)
}
fn capacity_args(model: &Value, name: &str, context: u64, matched: bool) -> DynResult<Vec<String>> {
    let mut extra_args = Vec::new();
    if matched && name == "vllm" {
        let capacity = &model["vllm_capacity"];
        let (tokens, blocks) = if capacity.is_null() {
            (16, 1)
        } else {
            (
                capacity["reference_tokens"]
                    .as_u64()
                    .ok_or("capacity tokens")?,
                capacity["reference_blocks"]
                    .as_u64()
                    .ok_or("capacity blocks")?,
            )
        };
        if tokens == 0 || blocks == 0 {
            return Err("capacity references must be positive".into());
        }
        let count = context
            .checked_mul(blocks)
            .and_then(|value| value.checked_add(tokens - 1))
            .ok_or("capacity overflow")?
            / tokens;
        extra_args.extend([
            "--block-size".into(),
            "16".into(),
            "--num-gpu-blocks-override".into(),
            count.to_string(),
        ]);
    } else if matched && name == "sglang" {
        extra_args.extend(["--max-total-tokens".into(), context.to_string()]);
    }
    Ok(extra_args)
}

// The materializer's config.json can be a symlink to a pinned tokenizer/config file.
// Resolve only this explicitly declared file, hash its canonical regular bytes, and
// retain the original config directory as the backend interface. Tree policy is unchanged.
pub(super) fn hf_config_directory(config: &Artifact) -> DynResult<PathBuf> {
    if !config.path.is_absolute()
        || config
            .path
            .file_name()
            .is_none_or(|name| name != "config.json")
    {
        return Err("HF config requires an absolute config.json file pin".into());
    }
    let canonical = config.path.canonicalize()?;
    file(&Artifact {
        path: canonical,
        sha256: config.sha256.clone(),
    })?;
    Ok(config
        .path
        .parent()
        .ok_or("HF config parent")?
        .to_path_buf())
}

#[cfg(test)]
#[path = "competitive_launch_tests.rs"]
mod tests;
