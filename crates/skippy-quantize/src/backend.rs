use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

use crate::memory_budget::{MemoryPolicy, monitor_interval};

const SKIPPY_FEATURE_MODEL_INTROSPECTION: u64 = 1 << 3;
const SKIPPY_FEATURE_GGUF_SLICE_WRITE: u64 = 1 << 4;

#[derive(Debug, Parser)]
pub struct BackendArgs {
    #[arg(long = "skippy-runtime-library", value_name = "PATH")]
    skippy_runtime_libraries: Vec<PathBuf>,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Serialize)]
pub struct BackendCapabilities {
    pub external_process: ExternalProcessCapabilities,
    pub native_rust: NativeRustCapabilities,
    pub llama_api: LlamaApiCapabilities,
    pub skippy_abi: SkippyAbiCapabilities,
}

#[derive(Debug, Serialize)]
pub struct ExternalProcessCapabilities {
    pub convert_hf_to_gguf: bool,
    pub llama_quantize: bool,
    pub resumable_windows: bool,
    pub low_residency_staging: bool,
    pub watchdog: bool,
}

#[derive(Debug, Serialize)]
pub struct NativeRustCapabilities {
    pub convert_hf_to_gguf: bool,
    pub llama_quantize: bool,
    pub resumable_windows: bool,
    pub low_residency_streaming: bool,
    pub reason: String,
}

#[derive(Debug, Serialize)]
pub struct LlamaApiCapabilities {
    pub convert_hf_to_gguf: bool,
    pub llama_quantize: bool,
    pub runtime_loaded: bool,
    pub reason: String,
}

#[derive(Debug, Serialize)]
pub struct SkippyAbiCapabilities {
    pub convert_hf_to_gguf: bool,
    pub llama_quantize: bool,
    pub runtime_loaded: bool,
    pub feature_mask: Option<u64>,
    pub model_introspection: bool,
    pub gguf_slice_write: bool,
    pub load_error: Option<String>,
    pub reason: String,
}

pub fn capabilities(skippy_runtime_libraries: &[PathBuf]) -> BackendCapabilities {
    let skippy_abi = skippy_abi_capabilities(skippy_runtime_libraries);
    let llama_api = LlamaApiCapabilities {
        convert_hf_to_gguf: false,
        llama_quantize: skippy_ffi::native_runtime_loaded(),
        runtime_loaded: skippy_ffi::native_runtime_loaded(),
        reason: if skippy_ffi::native_runtime_loaded() {
            "loaded native runtime exposes llama_model_quantize".to_string()
        } else {
            "no native runtime library was loaded for llama API probing".to_string()
        },
    };
    BackendCapabilities {
        external_process: ExternalProcessCapabilities {
            convert_hf_to_gguf: true,
            llama_quantize: true,
            resumable_windows: true,
            low_residency_staging: true,
            watchdog: true,
        },
        native_rust: NativeRustCapabilities {
            convert_hf_to_gguf: true,
            llama_quantize: false,
            resumable_windows: true,
            low_residency_streaming: true,
            reason: "Rust SafeTensors-to-GGUF writer streams tensor payloads and materializes one split window per run".to_string(),
        },
        llama_api,
        skippy_abi,
    }
}

pub fn run_backends(args: BackendArgs) -> Result<()> {
    let capabilities = capabilities(&args.skippy_runtime_libraries);
    if args.json {
        println!("{}", serde_json::to_string_pretty(&capabilities)?);
    } else {
        println!(
            "external-process: convert_hf_to_gguf={} llama_quantize={} resumable_windows={} low_residency_staging={}",
            capabilities.external_process.convert_hf_to_gguf,
            capabilities.external_process.llama_quantize,
            capabilities.external_process.resumable_windows,
            capabilities.external_process.low_residency_staging
        );
        println!(
            "native-rust: convert_hf_to_gguf={} llama_quantize={} resumable_windows={} low_residency_streaming={} reason={}",
            capabilities.native_rust.convert_hf_to_gguf,
            capabilities.native_rust.llama_quantize,
            capabilities.native_rust.resumable_windows,
            capabilities.native_rust.low_residency_streaming,
            capabilities.native_rust.reason
        );
        println!(
            "llama-api: convert_hf_to_gguf={} llama_quantize={} runtime_loaded={} reason={}",
            capabilities.llama_api.convert_hf_to_gguf,
            capabilities.llama_api.llama_quantize,
            capabilities.llama_api.runtime_loaded,
            capabilities.llama_api.reason
        );
        println!(
            "skippy-abi: convert_hf_to_gguf={} llama_quantize={} runtime_loaded={} feature_mask={} model_introspection={} gguf_slice_write={} reason={}",
            capabilities.skippy_abi.convert_hf_to_gguf,
            capabilities.skippy_abi.llama_quantize,
            capabilities.skippy_abi.runtime_loaded,
            capabilities
                .skippy_abi
                .feature_mask
                .map_or_else(|| "unknown".to_string(), |value| format!("{value:#x}")),
            capabilities.skippy_abi.model_introspection,
            capabilities.skippy_abi.gguf_slice_write,
            capabilities.skippy_abi.reason
        );
        if let Some(load_error) = capabilities.skippy_abi.load_error.as_deref() {
            println!("skippy-abi-load-error: {load_error}");
        }
    }
    Ok(())
}

fn skippy_abi_capabilities(skippy_runtime_libraries: &[PathBuf]) -> SkippyAbiCapabilities {
    let load_error = load_skippy_runtime_for_probe(skippy_runtime_libraries);
    let runtime_loaded = skippy_ffi::native_runtime_loaded();
    let feature_mask = if runtime_loaded {
        std::panic::catch_unwind(skippy_ffi::skippy_abi_features).ok()
    } else {
        None
    };
    let model_introspection = feature_mask.is_some_and(|mask| {
        mask & SKIPPY_FEATURE_MODEL_INTROSPECTION == SKIPPY_FEATURE_MODEL_INTROSPECTION
    });
    let gguf_slice_write = feature_mask.is_some_and(|mask| {
        mask & SKIPPY_FEATURE_GGUF_SLICE_WRITE == SKIPPY_FEATURE_GGUF_SLICE_WRITE
    });
    SkippyAbiCapabilities {
        convert_hf_to_gguf: false,
        llama_quantize: runtime_loaded,
        runtime_loaded,
        feature_mask,
        model_introspection,
        gguf_slice_write,
        load_error,
        reason: skippy_abi_reason(runtime_loaded, feature_mask, gguf_slice_write),
    }
}

fn load_skippy_runtime_for_probe(skippy_runtime_libraries: &[PathBuf]) -> Option<String> {
    if skippy_runtime_libraries.is_empty() || skippy_ffi::native_runtime_loaded() {
        return None;
    }
    // The caller explicitly supplied these native runtime libraries for probing.
    // Loading arbitrary libraries would be unsafe, so the command never guesses.
    let result = unsafe { skippy_ffi::load_native_runtime_libraries(skippy_runtime_libraries) };
    result.err().map(|err| err.to_string())
}

fn skippy_abi_reason(
    runtime_loaded: bool,
    feature_mask: Option<u64>,
    gguf_slice_write: bool,
) -> String {
    if !runtime_loaded {
        return "no Skippy native runtime library was loaded for ABI probing".to_string();
    }
    if feature_mask.is_none() {
        return "loaded Skippy runtime does not expose skippy_abi_features".to_string();
    }
    if gguf_slice_write {
        return "loaded Skippy ABI exposes GGUF slice writing and the linked llama symbols can be used for GGUF quantization, but not HF checkpoint conversion".to_string();
    }
    "loaded Skippy ABI exposes staged inference/runtime entry points and linked llama symbols can be used for GGUF quantization, but not HF checkpoint conversion".to_string()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum BackendKind {
    ExternalProcess,
    NativeRust,
    LlamaApi,
    SkippyAbi,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ExternalProcess => "external-process",
            Self::NativeRust => "native-rust",
            Self::LlamaApi => "llama-api",
            Self::SkippyAbi => "skippy-abi",
        }
    }
}

pub fn ensure_convert_backend(kind: BackendKind) -> Result<()> {
    ensure!(
        matches!(kind, BackendKind::ExternalProcess | BackendKind::NativeRust),
        "backend {} cannot convert HF checkpoints yet: {}",
        kind.as_str(),
        capabilities(&[]).skippy_abi.reason
    );
    Ok(())
}

pub fn ensure_quant_backend(kind: BackendKind) -> Result<()> {
    ensure!(
        matches!(
            kind,
            BackendKind::ExternalProcess | BackendKind::LlamaApi | BackendKind::SkippyAbi
        ),
        "backend {} cannot quantize GGUFs yet: {}",
        kind.as_str(),
        capabilities(&[]).skippy_abi.reason
    );
    Ok(())
}

pub fn ensure_external_tool<'a>(
    kind: BackendKind,
    path: Option<&'a Path>,
    flag_name: &str,
    tool_name: &str,
) -> Result<&'a Path> {
    ensure!(
        kind == BackendKind::ExternalProcess,
        "{tool_name} path is only valid with --backend external-process"
    );
    let path =
        path.with_context(|| format!("{flag_name} is required for --backend external-process"))?;
    ensure!(
        path.is_file(),
        "{tool_name} does not exist: {}",
        path.display()
    );
    Ok(path)
}

pub fn resolve_external_tool(
    explicit: Option<&Path>,
    env_var: &str,
    relative_candidates: &[&str],
) -> Option<PathBuf> {
    if let Some(explicit) = explicit {
        return Some(explicit.to_path_buf());
    }
    if let Some(from_env) = std::env::var_os(env_var).map(PathBuf::from)
        && from_env.is_file()
    {
        return Some(from_env);
    }
    for root in candidate_roots() {
        for relative in relative_candidates {
            let candidate = root.join(relative);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn candidate_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(current) = std::env::current_dir() {
        roots.push(current);
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    roots.push(manifest_dir.clone());
    if let Some(repo_root) = manifest_dir.parent().and_then(Path::parent) {
        roots.push(repo_root.to_path_buf());
    }
    roots.dedup();
    roots
}

#[derive(Debug, Default)]
pub struct ExternalProcessOptions {
    pub watchdog_seconds: Option<u64>,
    pub max_memory_bytes: Option<u64>,
    pub memory_policy: MemoryPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendRunStatus {
    pub status_code: Option<i32>,
    pub success: bool,
}

impl BackendRunStatus {
    fn from_exit_status(status: ExitStatus) -> Self {
        Self {
            status_code: status.code(),
            success: status.success(),
        }
    }

    pub fn from_code(status_code: i32) -> Self {
        Self {
            status_code: Some(status_code),
            success: status_code == 0,
        }
    }
}

pub fn run_backend_command(
    kind: BackendKind,
    command: &[String],
    options: &ExternalProcessOptions,
) -> Result<BackendRunStatus> {
    ensure!(
        kind == BackendKind::ExternalProcess,
        "backend {} cannot execute prepared external commands",
        kind.as_str()
    );
    run_external_process(command, options).map(BackendRunStatus::from_exit_status)
}

fn run_external_process(
    command: &[String],
    options: &ExternalProcessOptions,
) -> Result<ExitStatus> {
    let (program, args) = command
        .split_first()
        .context("cannot run empty backend command")?;
    let mut process = ProcessCommand::new(program);
    process.args(args).stdin(Stdio::null());
    if let Some(max_memory_bytes) = options.max_memory_bytes {
        process.env(
            "LLAMA_QUANTIZE_MAX_MEMORY_BYTES",
            max_memory_bytes.to_string(),
        );
    }
    let mut child = process
        .spawn()
        .with_context(|| format!("run {}", shell_words(command)))?;
    let Some(interval) = monitor_interval(options.watchdog_seconds, options.max_memory_bytes)
    else {
        return child
            .wait()
            .with_context(|| format!("wait for {}", shell_words(command)));
    };
    let started = Instant::now();
    loop {
        if let Some(status) = child
            .try_wait()
            .with_context(|| format!("poll {}", shell_words(command)))?
        {
            return Ok(status);
        }
        thread::sleep(interval);
        if let Some(status) = child
            .try_wait()
            .with_context(|| format!("poll {}", shell_words(command)))?
        {
            return Ok(status);
        }
        print_watchdog_line(child.id(), started.elapsed());
        if let Some(max_memory_bytes) = options.max_memory_bytes {
            let memory_bytes = process_memory_bytes(child.id()).or_else(cgroup_memory_bytes);
            if let Some(memory_bytes) = memory_bytes
                && memory_bytes > max_memory_bytes
            {
                println!(
                    "backend_memory_budget_exceeded pid={} memory_bytes={} max_memory_bytes={} policy={:?}",
                    child.id(),
                    memory_bytes,
                    max_memory_bytes,
                    options.memory_policy
                );
                if options.memory_policy.is_hard() {
                    child
                        .kill()
                        .with_context(|| format!("kill {}", shell_words(command)))?;
                    return child
                        .wait()
                        .with_context(|| format!("wait for killed {}", shell_words(command)));
                }
            }
        }
    }
}

pub fn ensure_success(status: BackendRunStatus, command: &[String]) -> Result<()> {
    ensure!(
        status.success,
        "command failed with status_code {:?}: {}",
        status.status_code,
        shell_words(command)
    );
    Ok(())
}

fn shell_words(command: &[String]) -> String {
    command.join(" ")
}

fn print_watchdog_line(pid: u32, elapsed: Duration) {
    let memory_current = read_u64("/sys/fs/cgroup/memory.current");
    let memory_peak = read_u64("/sys/fs/cgroup/memory.peak");
    println!(
        "backend_watchdog pid={pid} elapsed_seconds={} memory_current_bytes={} memory_peak_bytes={}",
        elapsed.as_secs(),
        format_optional_u64(memory_current),
        format_optional_u64(memory_peak)
    );
}

fn process_memory_bytes(pid: u32) -> Option<u64> {
    let status_path = format!("/proc/{pid}/status");
    let status = fs::read_to_string(status_path).ok()?;
    for line in status.lines() {
        let Some(rest) = line.strip_prefix("VmRSS:") else {
            continue;
        };
        let kib = rest.split_whitespace().next()?.parse::<u64>().ok()?;
        return kib.checked_mul(1024);
    }
    None
}

fn cgroup_memory_bytes() -> Option<u64> {
    read_u64("/sys/fs/cgroup/memory.current")
}

fn read_u64(path: &str) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn format_optional_u64(value: Option<u64>) -> String {
    value.map_or_else(|| "unknown".to_string(), |value| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_current_backend_capabilities() {
        let capabilities = capabilities(&[]);
        assert!(capabilities.external_process.convert_hf_to_gguf);
        assert!(capabilities.external_process.llama_quantize);
        assert!(capabilities.external_process.resumable_windows);
        assert!(capabilities.external_process.low_residency_staging);
        assert!(capabilities.external_process.watchdog);
        assert!(!capabilities.llama_api.convert_hf_to_gguf);
        assert!(!capabilities.llama_api.llama_quantize);
        assert!(!capabilities.llama_api.runtime_loaded);
        assert!(!capabilities.skippy_abi.convert_hf_to_gguf);
        assert!(!capabilities.skippy_abi.llama_quantize);
        assert!(!capabilities.skippy_abi.runtime_loaded);
        assert_eq!(capabilities.skippy_abi.feature_mask, None);
        assert!(!capabilities.skippy_abi.model_introspection);
        assert!(!capabilities.skippy_abi.gguf_slice_write);
        assert_eq!(capabilities.skippy_abi.load_error, None);
        assert!(capabilities.skippy_abi.reason.contains("Skippy"));
    }

    #[test]
    fn formats_unknown_memory_values() {
        assert_eq!(format_optional_u64(None), "unknown");
        assert_eq!(format_optional_u64(Some(7)), "7");
    }

    #[test]
    fn rejects_skippy_abi_conversion_backend_until_supported() {
        assert!(ensure_convert_backend(BackendKind::SkippyAbi).is_err());
    }

    #[test]
    fn accepts_skippy_abi_quant_backend() {
        assert!(ensure_quant_backend(BackendKind::SkippyAbi).is_ok());
    }

    #[test]
    fn rejects_missing_external_tool_path() {
        assert!(
            ensure_external_tool(
                BackendKind::ExternalProcess,
                None,
                "--converter",
                "convert_hf_to_gguf.py",
            )
            .is_err()
        );
    }

    #[test]
    fn validates_backend_run_status() {
        let ok = BackendRunStatus {
            status_code: Some(0),
            success: true,
        };
        let failed = BackendRunStatus {
            status_code: Some(2),
            success: false,
        };
        assert!(ensure_success(ok, &["tool".to_string()]).is_ok());
        assert!(ensure_success(failed, &["tool".to_string()]).is_err());
    }

    #[test]
    fn resolves_explicit_external_tool_first() {
        let explicit = Path::new("/tmp/tool");
        assert_eq!(
            resolve_external_tool(Some(explicit), "NO_SUCH_ENV", &[]),
            Some(explicit.to_path_buf())
        );
    }
}
