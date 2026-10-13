//! System One smoke stage inputs and the aggregate qualification receipt.
use crate::command::DynResult;
use serde::Serialize;
use serde_json::json;
use std::{fs, io::Write, path::Path};

const USAGE: &str = "cargo xtool automation system-one-smoke {stage OUTPUT MODEL PATH SHA LAYERS BIND LANES CONTEXT BATCH GPU_LAYERS | report OUTPUT STATUS CONTRACT READ ARTIFACT BACKEND ARTIFACT_PATH CACHE_CHECKED CERTIFIED REQUIRE_QUALIFIED SKIP_CONTRACT REASONS}";

#[derive(Serialize)]
struct Stage<'a> {
    run_id: &'static str,
    topology_id: &'static str,
    model_id: &'a str,
    model_path: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_model_sha256: Option<&'a str>,
    stage_id: &'static str,
    stage_index: u32,
    layer_start: u32,
    layer_end: u32,
    ctx_size: u32,
    lane_count: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_gpu_layers: i32,
    cache_type_k: &'static str,
    cache_type_v: &'static str,
    load_mode: &'static str,
    execution_contract: &'static str,
    bind_addr: &'a str,
    upstream: Option<&'a str>,
    downstream: Option<&'a str>,
}

fn positive(value: &str) -> DynResult<u32> {
    let number = value.parse::<u32>()?;
    if number == 0 {
        return Err("stage dimensions must be positive".into());
    }
    Ok(number)
}

fn stage(args: &[String]) -> DynResult<()> {
    let [
        output,
        model,
        path,
        sha,
        layers,
        bind,
        lanes,
        context,
        batch,
        gpu,
    ] = args
    else {
        return Err(USAGE.into());
    };
    if model.is_empty() || path.is_empty() {
        return Err("model identity and path are required".into());
    }
    if !sha.is_empty() && (sha.len() != 64 || !sha.bytes().all(|b| b.is_ascii_hexdigit())) {
        return Err("source model SHA256 must be 64 hexadecimal characters".into());
    }
    let address = bind.parse::<std::net::SocketAddr>()?;
    if !address.ip().is_loopback() || address.port() == 0 {
        return Err("stage bind must name a loopback address with a selected port".into());
    }
    let config = Stage {
        run_id: "skippy-system-one-smoke",
        topology_id: "skippy-system-one-smoke-single-stage",
        model_id: model,
        model_path: path,
        source_model_sha256: (!sha.is_empty()).then_some(sha.as_str()),
        stage_id: "stage-0",
        stage_index: 0,
        layer_start: 0,
        layer_end: positive(layers)?,
        ctx_size: positive(context)?,
        lane_count: positive(lanes)?,
        n_batch: positive(batch)?,
        n_ubatch: positive(batch)?,
        n_gpu_layers: gpu.parse()?,
        cache_type_k: "f16",
        cache_type_v: "f16",
        load_mode: "runtime-slice",
        execution_contract: "",
        bind_addr: bind,
        upstream: None,
        downstream: None,
    };
    publish(output, &config, false, false)
}

fn nullable(value: &str) -> Option<&str> {
    (!value.is_empty()).then_some(value)
}

fn enabled(value: &str) -> bool {
    matches!(value, "1" | "true")
}

fn report(args: &[String]) -> DynResult<()> {
    let [
        output,
        status,
        contract,
        read,
        artifact,
        backend,
        artifact_path,
        checked,
        certified,
        required,
        skipped,
        reasons,
    ] = args
    else {
        return Err(USAGE.into());
    };
    if !matches!(status.as_str(), "pass" | "fail" | "unqualified")
        || !matches!(contract.as_str(), "pass" | "fail" | "error" | "skipped")
        || !matches!(
            read.as_str(),
            "pass" | "fail" | "error" | "skipped" | "unqualified"
        )
    {
        return Err("invalid smoke qualification status".into());
    }
    let receipt = json!({
        "schema_version": 1,
        "status": status,
        "contract": {"status": contract},
        "full_model_read": {
            "status": read,
            "artifact": artifact,
            "backend": nullable(backend),
            "artifact_path": nullable(artifact_path),
            "artifact_cache_checked": enabled(checked),
            "certified_backends": certified.split(',').filter(|s| !s.is_empty()).collect::<Vec<_>>(),
            "require_qualified": enabled(required),
        },
        "contract_part_skipped": enabled(skipped),
        "reasons": reasons.split("; ").filter(|s| !s.is_empty()).collect::<Vec<_>>(),
    });
    publish(output, &receipt, true, true)
}

fn publish(path: &str, value: &impl Serialize, parents: bool, stdout: bool) -> DynResult<()> {
    if path.is_empty() {
        return Err("output path is required".into());
    }
    let path = Path::new(path);
    if let Ok(metadata) = fs::symlink_metadata(path)
        && !metadata.file_type().is_file()
    {
        return Err("output must be a regular file".into());
    }
    if parents && let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    let mut options = fs::OpenOptions::new();
    options.write(true).create(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err("output must be a regular file".into());
    }
    file.set_len(0)?;
    file.write_all(&bytes)?;
    if stdout {
        std::io::stdout().lock().write_all(&bytes)?;
    }
    Ok(())
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    match args {
        [verb, rest @ ..] if verb == "stage" => stage(rest),
        [verb, rest @ ..] if verb == "report" => report(rest),
        [help] if matches!(help.as_str(), "--help" | "-h") => {
            println!("{USAGE}");
            Ok(())
        }
        _ => Err(USAGE.into()),
    }
}
