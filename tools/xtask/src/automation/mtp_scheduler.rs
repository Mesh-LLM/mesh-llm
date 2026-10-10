//! Supplied old/new binary-stage scheduler comparison; no build or model acquisition.
mod execution;
mod final_publication;
mod metrics;
mod worker;
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    old_bin: PathBuf,
    new_bin: PathBuf,
    client_bin: PathBuf,
    package: PathBuf,
    native_build: PathBuf,
    output_dir: PathBuf,
    concurrency: Vec<usize>,
    requests: usize,
    model_id: String,
    layer_start: usize,
    layer_end: usize,
    activation_width: usize,
    startup_seconds: u64,
    client_seconds: u64,
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    use crate::repository::check_args::Grammar;
    const G: Grammar = Grammar {
        usage: "cargo xtool automation mtp-scheduler --old-bin PATH --new-bin PATH --client-bin PATH --package PATH --native-build PATH --output-dir PATH [--concurrency 1,2,4,8] [--requests 64]",
        values: &[
            "--old-bin",
            "--new-bin",
            "--client-bin",
            "--package",
            "--native-build",
            "--output-dir",
            "--concurrency",
            "--requests",
            "--model-id",
            "--layer-start",
            "--layer-end",
            "--activation-width",
            "--startup-seconds",
            "--client-seconds",
        ],
        flags: &["--help"],
    };
    let p = match G.parse(args) {
        Ok(p) => p,
        Err(r) => return r.emit(),
    };
    if p.flag("--help") {
        println!("{}", G.usage);
        return Ok(());
    }
    if !p.positionals.is_empty() {
        return G.error("unexpected positional arguments").emit();
    }
    let required =
        |name| -> DynResult<PathBuf> { Ok(p.last(name).ok_or("missing required path")?.into()) };
    let number =
        |name, default: &str| -> DynResult<usize> { Ok(p.last(name).unwrap_or(default).parse()?) };
    let input = Input {
        old_bin: required("--old-bin")?,
        new_bin: required("--new-bin")?,
        client_bin: required("--client-bin")?,
        package: required("--package")?,
        native_build: required("--native-build")?,
        output_dir: required("--output-dir")?,
        concurrency: p
            .last("--concurrency")
            .unwrap_or("1,2,4,8")
            .split(',')
            .map(str::parse)
            .collect::<Result<_, _>>()?,
        requests: number("--requests", "64")?,
        model_id: p
            .last("--model-id")
            .unwrap_or("meshllm/GLM-5.2-Q2_K-MTP-Q8-layers")
            .into(),
        layer_start: number("--layer-start", "74")?,
        layer_end: number("--layer-end", "78")?,
        activation_width: number("--activation-width", "6144")?,
        startup_seconds: number("--startup-seconds", "900")? as u64,
        client_seconds: number("--client-seconds", "1800")? as u64,
    };
    execution::execute(input)
}
pub(crate) fn run_worker(args: &[String]) -> DynResult<()> {
    worker::run(args)
}
pub(super) fn regular(path: &Path, maximum: u64) -> DynResult<PathBuf> {
    let path = path.canonicalize()?;
    let metadata = std::fs::metadata(&path)?;
    if !metadata.is_file() || metadata.len() > maximum {
        return Err("requires bounded regular file".into());
    }
    Ok(path)
}
pub(super) fn read(path: &Path) -> DynResult<Vec<u8>> {
    use std::io::Read;
    let path = regular(path, 16 * 1024 * 1024)?;
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > 16 * 1024 * 1024 {
        return Err("requires bounded regular input".into());
    }
    let mut bytes = Vec::new();
    file.take(16 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > 16 * 1024 * 1024 {
        return Err("input grew beyond bound".into());
    }
    Ok(bytes)
}
pub(super) fn limits(execution: Duration) -> crate::process::Limits {
    crate::process::Limits {
        execution,
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 65536,
        readiness: crate::process::Readiness::None,
        completion: crate::process::Completion::Exit,
    }
}
pub(super) fn environment(
    input: &Input,
) -> std::collections::BTreeMap<std::ffi::OsString, crate::process::Value> {
    use crate::process::Value;
    let mut environment = std::collections::BTreeMap::new();
    for key in ["PATH", "SystemRoot", "WINDIR", "TMP", "TEMP"] {
        if let Some(value) = std::env::var_os(key) {
            environment.insert(key.into(), Value::Public(value));
        }
    }
    for (key, value) in [
        (
            "HOME",
            input.output_dir.join("private-home").into_os_string(),
        ),
        (
            "LLAMA_STAGE_BUILD_DIR",
            input.native_build.clone().into_os_string(),
        ),
        ("SKIPPY_NATIVE_MTP_GREEDY_SAMPLING_FASTPATH", "1".into()),
        ("SKIPPY_TELEMETRY_STDERR", "1".into()),
        ("GIT_MASTER", "1".into()),
        ("GIT_OPTIONAL_LOCKS", "0".into()),
    ] {
        environment.insert(key.into(), Value::Public(value));
    }
    environment
}
pub(super) fn config(input: &Input, address: &str) -> serde_json::Value {
    let lanes = *input.concurrency.iter().max().expect("admitted sweep");
    serde_json::json!({"run_id":"mtp-scheduler-benchmark","topology_id":"mtp-scheduler-benchmark-final-stage","model_id":input.model_id,"package_ref":input.package,"model_path":input.package,
    "stage_id":"stage-final","stage_index":1,"layer_start":input.layer_start,"layer_end":input.layer_end,"ctx_size":256*lanes,"lane_count":lanes,"n_batch":8.max(lanes),"n_ubatch":8.max(lanes),"n_gpu_layers":0,"mmap":true,"mlock":false,"cache_type_k":"f16","cache_type_v":"f16","flash_attn_type":"disabled","selected_device":{"backend_device":"CPU"},"native_mtp_enabled":true,"load_mode":"layer-package","execution_contract":"","bind_addr":address,"upstream":{"stage_id":"stage-prev","stage_index":0,"endpoint":"tcp://127.0.0.1:19000"},"downstream":null})
}

pub(super) fn profile(input: &Input) -> DynResult<()> {
    if input.concurrency.is_empty()
        || input.concurrency.len() > 32
        || input.concurrency.iter().any(|n| !(1..=256).contains(n))
        || !(1..=10000).contains(&input.requests)
        || !(1..=65536).contains(&input.activation_width)
        || input.layer_start >= input.layer_end
        || input.model_id.is_empty()
        || !(1..=900).contains(&input.startup_seconds)
        || !(1..=1800).contains(&input.client_seconds)
    {
        return Err("invalid bounded MTP scheduler profile".into());
    }
    if input.model_id.len() > 4096
        || input.model_id.chars().any(char::is_control)
        || input.layer_end > 65536
    {
        return Err("invalid bounded MTP scheduler profile".into());
    }
    Ok(())
}
