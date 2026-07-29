use anyhow::{Context, Result, anyhow, bail};
pub use mesh_llm_gpu_bench::BenchmarkOutput;
use mesh_llm_native_runtime::{
    GPU_BENCHMARK_TOOL_PATH, InstalledNativeRuntime, NativeRuntimeBackendKind, RuntimeSelection,
    select_native_runtime_from_artifacts,
};
use mesh_llm_runtime_install::{
    CURRENT_MESH_VERSION, current_skippy_abi_version, default_native_runtime_cache,
    discover_local_native_runtimes, host_runtime_profile,
};
use serde::{Deserialize, Serialize};
#[cfg(any(test, target_os = "linux", target_os = "macos", windows))]
use std::env;
#[cfg(any(target_os = "linux", target_os = "macos"))]
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::hardware::HardwareSurvey;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuBandwidth {
    pub name: String,
    pub vram_bytes: u64,
    pub p50_gbps: f64,
    pub p90_gbps: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute_tflops_fp32: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compute_tflops_fp16: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkFingerprint {
    pub gpus: Vec<GpuBandwidth>, // per-GPU identity + bandwidth, in device order
    pub is_soc: bool,
    pub timestamp_secs: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkResult {
    pub mem_bandwidth_gbps: Vec<f64>,
    pub compute_tflops_fp32: Option<Vec<f64>>,
    pub compute_tflops_fp16: Option<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SavedBenchmark {
    pub path: PathBuf,
    pub result: BenchmarkResult,
}

pub const BENCHMARK_TIMEOUT: Duration = Duration::from_secs(25);

#[cfg(test)]
const BENCHMARK_TOOL_TEST_OVERRIDE_ENV: &str = "MESH_LLM_BENCHMARK_TOOL_TEST_OVERRIDE";

fn benchmark_backend_name(backend: mesh_llm_gpu_bench::BenchmarkBackend) -> &'static str {
    match backend {
        mesh_llm_gpu_bench::BenchmarkBackend::Metal => "metal",
        mesh_llm_gpu_bench::BenchmarkBackend::Cuda => "cuda",
        mesh_llm_gpu_bench::BenchmarkBackend::Hip => "hip",
        mesh_llm_gpu_bench::BenchmarkBackend::Intel => "intel",
    }
}

fn parse_benchmark_backend(name: &str) -> Option<mesh_llm_gpu_bench::BenchmarkBackend> {
    if name.eq_ignore_ascii_case("metal") {
        Some(mesh_llm_gpu_bench::BenchmarkBackend::Metal)
    } else if name.eq_ignore_ascii_case("cuda") {
        Some(mesh_llm_gpu_bench::BenchmarkBackend::Cuda)
    } else if name.eq_ignore_ascii_case("hip") {
        Some(mesh_llm_gpu_bench::BenchmarkBackend::Hip)
    } else if name.eq_ignore_ascii_case("intel") {
        Some(mesh_llm_gpu_bench::BenchmarkBackend::Intel)
    } else {
        None
    }
}

fn runtime_selection_for_benchmark(
    backend: mesh_llm_gpu_bench::BenchmarkBackend,
) -> Result<RuntimeSelection> {
    let kind = match backend {
        mesh_llm_gpu_bench::BenchmarkBackend::Metal => NativeRuntimeBackendKind::Metal,
        mesh_llm_gpu_bench::BenchmarkBackend::Cuda => NativeRuntimeBackendKind::Cuda,
        mesh_llm_gpu_bench::BenchmarkBackend::Hip => NativeRuntimeBackendKind::Rocm,
        mesh_llm_gpu_bench::BenchmarkBackend::Intel => {
            bail!(
                "Intel GPU benchmarking has no published native-runtime tool; use a supported \
                 CUDA, ROCm, or Metal runtime instead"
            )
        }
    };
    Ok(RuntimeSelection::Backend {
        kind,
        cuda_toolkit_major: None,
    })
}

fn runtimes_with_benchmark_tools(
    runtimes: &[InstalledNativeRuntime],
) -> Vec<&InstalledNativeRuntime> {
    runtimes
        .iter()
        .filter(|runtime| {
            runtime
                .manifest
                .runtime
                .tools
                .contains_key(GPU_BENCHMARK_TOOL_PATH)
        })
        .collect()
}

/// Resolve the selected installed native runtime's manifest-verified benchmark
/// tool. The neutral host never executes a benchmark binary from its own
/// directory or an unverified search path.
fn resolve_runtime_benchmark_tool(
    backend: mesh_llm_gpu_bench::BenchmarkBackend,
) -> Result<PathBuf> {
    #[cfg(test)]
    if let Some(path) = env::var_os(BENCHMARK_TOOL_TEST_OVERRIDE_ENV) {
        return Ok(PathBuf::from(path));
    }

    let cache = default_native_runtime_cache()?;
    let installed = discover_local_native_runtimes(&[], &cache)?;
    let installed_for_version = installed
        .iter()
        .filter(|runtime| runtime.mesh_version == CURRENT_MESH_VERSION)
        .cloned()
        .collect::<Vec<_>>();
    let installed_with_tools = runtimes_with_benchmark_tools(&installed_for_version);
    let artifacts = installed_with_tools
        .iter()
        .map(|runtime| runtime.manifest.runtime.clone())
        .collect::<Vec<_>>();
    let selection = runtime_selection_for_benchmark(backend)?;
    let selected = select_native_runtime_from_artifacts(
        &artifacts,
        &host_runtime_profile(),
        CURRENT_MESH_VERSION,
        Some(&current_skippy_abi_version()),
        &selection,
    )
    .with_context(|| {
        format!(
            "no compatible installed {} native runtime with a benchmark tool",
            benchmark_backend_name(backend)
        )
    })?;
    let runtime = installed_with_tools
        .into_iter()
        .find(|runtime| {
            runtime.native_runtime_id == selected.artifact.id
                && runtime.manifest.runtime.skippy_abi == selected.artifact.skippy_abi
        })
        .context("selected native runtime disappeared while resolving benchmark tool")?;
    runtime.gpu_benchmark_tool()
}

fn run_benchmark_subprocess(binary: &Path, timeout: Duration) -> Result<Vec<BenchmarkOutput>> {
    let mut command = Command::new(binary);
    command
        .arg("--json")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    configure_runtime_benchmark_library_lookup(&mut command, binary)?;

    let mut child = command.spawn().with_context(|| {
        format!(
            "failed to start runtime benchmark tool {}",
            binary.display()
        )
    })?;

    let started = Instant::now();
    while child.try_wait()?.is_none() {
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let output = child.wait_with_output()?;
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            if stderr.is_empty() {
                bail!("benchmark timed out after {:.1}s", timeout.as_secs_f64());
            }
            bail!(
                "benchmark timed out after {:.1}s: {stderr}",
                timeout.as_secs_f64()
            );
        }
        thread::sleep(Duration::from_millis(25));
    }

    let output = child.wait_with_output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        if stderr.is_empty() {
            bail!(
                "runtime benchmark tool exited with status {}",
                output.status
            );
        }
        bail!("runtime benchmark tool failed: {stderr}");
    }

    parse_benchmark_output(&output.stdout)
        .ok_or_else(|| anyhow!("benchmark child returned invalid output"))
}

fn configure_runtime_benchmark_library_lookup(command: &mut Command, binary: &Path) -> Result<()> {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    {
        let variable = if cfg!(target_os = "linux") {
            "LD_LIBRARY_PATH"
        } else {
            "DYLD_LIBRARY_PATH"
        };
        command.env(
            variable,
            runtime_benchmark_library_lookup(binary, env::var_os(variable))?,
        );
    }

    #[cfg(windows)]
    {
        let runtime_lib = binary
            .parent()
            .and_then(Path::parent)
            .context("native runtime benchmark tool has no runtime root")?
            .join("lib");
        let mut path_entries = vec![runtime_lib];
        if let Some(current_path) = env::var_os("PATH") {
            path_entries.extend(env::split_paths(&current_path));
        }
        command.env("PATH", env::join_paths(path_entries)?);
    }

    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
fn runtime_benchmark_library_lookup(binary: &Path, existing: Option<OsString>) -> Result<OsString> {
    let runtime_lib = binary
        .parent()
        .and_then(Path::parent)
        .context("native runtime benchmark tool has no runtime root")?
        .join("lib");
    let mut paths = vec![runtime_lib];
    if let Some(existing) = existing {
        paths.extend(env::split_paths(&existing));
    }
    env::join_paths(paths).context("join native runtime benchmark library lookup path")
}

pub fn run_backend_by_name(backend: &str) -> Result<Vec<BenchmarkOutput>> {
    let backend = parse_benchmark_backend(backend)
        .with_context(|| format!("unsupported benchmark backend {backend}"))?;
    let tool = resolve_runtime_benchmark_tool(backend)?;
    run_benchmark_subprocess(&tool, BENCHMARK_TIMEOUT)
}

/// Normalize `HardwareSurvey.gpu_name` into a per-GPU list of names.
/// - Splits on ',' and trims whitespace for robustness.
/// - Expands summarized forms like "8× NVIDIA A100" into 8 identical entries.
/// - If the expanded list length does not match `gpu_vram.len()` but `gpu_vram` is
///   non-empty, falls back to assuming all GPUs share the same summarized name and
///   returns `gpu_vram.len()` copies of it.
fn per_gpu_names(hw: &HardwareSurvey) -> Vec<String> {
    let raw = match hw.gpu_name.as_deref() {
        Some(s) => s.trim(),
        None => return Vec::new(),
    };

    if raw.is_empty() {
        return Vec::new();
    }

    let mut names: Vec<String> = Vec::new();

    for part in raw.split(',') {
        let part_trimmed = part.trim();
        if part_trimmed.is_empty() {
            continue;
        }

        // Handle summarized "N× name" form (e.g., "8× NVIDIA A100").
        let counted_name = part_trimmed.split_once('×').and_then(|(count_str, name)| {
            count_str
                .trim()
                .parse::<usize>()
                .ok()
                .map(|count| (count, name.trim()))
        });
        if let Some((count, name_trimmed)) = counted_name {
            for _ in 0..count {
                names.push(name_trimmed.to_string());
            }
            continue;
        }

        // Fallback: treat as a single GPU name.
        names.push(part_trimmed.to_string());
    }

    if names.len() == hw.gpu_vram.len() || hw.gpu_vram.is_empty() {
        return names;
    }

    // As a last resort, assume all GPUs share the same summarized name.
    let gpu_count = hw.gpu_vram.len();
    vec![raw.to_string(); gpu_count]
}

/// Returns true if the current hardware differs from the fingerprint's recorded hardware.
/// Compares GPU names, VRAM sizes (by index), and the is_soc flag.
pub fn hardware_changed(fingerprint: &BenchmarkFingerprint, hw: &HardwareSurvey) -> bool {
    if fingerprint.is_soc != hw.is_soc {
        return true;
    }

    let hw_names: Vec<String> = per_gpu_names(hw);

    if fingerprint.gpus.len() != hw_names.len() || fingerprint.gpus.len() != hw.gpu_vram.len() {
        return true;
    }

    for (i, cached) in fingerprint.gpus.iter().enumerate() {
        if cached.name != hw_names[i] || cached.vram_bytes != hw.gpu_vram[i] {
            return true;
        }
    }
    false
}

/// Returns the cache-backed benchmark fingerprint path, usually
/// `~/.cache/mesh-llm/benchmark-fingerprint.json`.
/// Falls back to `~/.cache` and then the platform temp directory if needed.
pub fn fingerprint_path() -> PathBuf {
    dirs::cache_dir()
        .or_else(|| dirs::home_dir().map(|home| home.join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
        .join("mesh-llm")
        .join("benchmark-fingerprint.json")
}

/// Load a `BenchmarkFingerprint` from disk.  Returns `None` on any error.
pub fn load_fingerprint(path: &Path) -> Option<BenchmarkFingerprint> {
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Atomically write a `BenchmarkFingerprint` to disk.
/// Uses a `.json.tmp` staging file + rename for crash safety.
/// Logs a warning on failure — never panics.
pub fn save_fingerprint(path: &Path, fp: &BenchmarkFingerprint) {
    if let Err(err) = try_save_fingerprint(path, fp) {
        tracing::warn!("benchmark: failed to persist fingerprint: {err}");
    }
}

pub fn try_save_fingerprint(path: &Path, fp: &BenchmarkFingerprint) -> Result<()> {
    let tmp = path.with_extension("json.tmp");

    std::fs::create_dir_all(path.parent().unwrap_or_else(|| Path::new(".")))
        .with_context(|| format!("failed to create cache dir for {}", path.display()))?;

    let json =
        serde_json::to_string_pretty(fp).context("failed to serialize benchmark fingerprint")?;

    std::fs::write(&tmp, &json)
        .with_context(|| format!("failed to write temporary fingerprint {}", tmp.display()))?;

    // On Windows, `rename` fails if the destination already exists.
    // Remove the destination first there; on Unix the rename stays atomic.
    #[cfg(windows)]
    if path.exists() {
        std::fs::remove_file(path)
            .with_context(|| format!("failed to remove existing fingerprint {}", path.display()))?;
    }

    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e).with_context(|| {
            format!(
                "failed to rename fingerprint into place at {}",
                path.display()
            )
        });
    }

    Ok(())
}

/// Determine whether this hardware maps to a benchmark backend.
pub fn detect_benchmark_binary(hw: &HardwareSurvey, _bin_dir: &Path) -> Option<PathBuf> {
    let runner = mesh_llm_gpu_bench::runner_for(
        std::env::consts::OS,
        hw.gpu_count,
        hw.gpu_name.as_deref(),
        hw.is_soc,
    )?;
    resolve_runtime_benchmark_tool(runner.backend).ok()
}

/// Parse raw stdout bytes from a benchmark run into a vec of per-device outputs.
///
/// Expects a JSON array of [`BenchmarkOutput`].  Returns `None` on any parse
/// failure or if the device list is empty.
pub fn parse_benchmark_output(stdout: &[u8]) -> Option<Vec<BenchmarkOutput>> {
    mesh_llm_gpu_bench::parse_benchmark_output(stdout)
}

/// Run a manifest-verified native-runtime benchmark tool and return per-device outputs.
pub fn run_benchmark(binary: &Path, timeout: Duration) -> Option<Vec<BenchmarkOutput>> {
    run_benchmark_subprocess(binary, timeout)
        .map_err(|err| tracing::warn!("benchmark failed: {err:#}"))
        .ok()
}

fn run_backend_for_hardware(
    hw: &HardwareSurvey,
    bin_dir: &Path,
    timeout: Duration,
) -> Result<Vec<BenchmarkOutput>> {
    let runner = detect_benchmark_binary(hw, bin_dir).with_context(|| {
        format!(
            "no supported benchmark backend found for detected GPU platform {:?}",
            hw.gpu_name
        )
    })?;

    run_benchmark_subprocess(&runner, timeout)
}

/// Load a cached fingerprint if hardware is unchanged, otherwise run the
/// compiled benchmark backend and persist the result.
///
/// Not `async` — intended for use inside `tokio::task::spawn_blocking`.
pub fn run_or_load(
    hw: &HardwareSurvey,
    bin_dir: &Path,
    timeout: Duration,
) -> Option<BenchmarkResult> {
    let path = fingerprint_path();

    // Cache-hit path
    match load_fingerprint(&path) {
        Some(ref cached) if !hardware_changed(cached, hw) => {
            let mem_bandwidth: Vec<f64> = cached.gpus.iter().map(|g| g.p90_gbps).collect();
            let compute_tflops_fp32 = cached
                .gpus
                .iter()
                .map(|g| g.compute_tflops_fp32)
                .collect::<Option<Vec<f64>>>();
            let compute_tflops_fp16 = cached
                .gpus
                .iter()
                .map(|g| g.compute_tflops_fp16)
                .collect::<Option<Vec<f64>>>();
            let result = BenchmarkResult {
                mem_bandwidth_gbps: mem_bandwidth,
                compute_tflops_fp32,
                compute_tflops_fp16,
            };
            tracing::info!(
                "Using cached bandwidth fingerprint: {} GPUs",
                result.mem_bandwidth_gbps.len()
            );
            return Some(result);
        }
        _ => {}
    }

    tracing::info!("Hardware changed or no cache — running memory bandwidth benchmark");

    let outputs = run_backend_for_hardware(hw, bin_dir, timeout)
        .map_err(|err| tracing::warn!("benchmark failed: {err:#}"))
        .ok()?;

    let (gpus, result) = build_benchmark_result(hw, &outputs);

    let fingerprint = BenchmarkFingerprint {
        gpus,
        is_soc: hw.is_soc,
        timestamp_secs: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    };

    save_fingerprint(&path, &fingerprint);
    Some(result)
}

pub fn run_and_save(
    hw: &HardwareSurvey,
    bin_dir: &Path,
    timeout: Duration,
) -> Result<SavedBenchmark> {
    run_and_save_to_path(hw, bin_dir, timeout, &fingerprint_path())
}

fn run_and_save_to_path(
    hw: &HardwareSurvey,
    bin_dir: &Path,
    timeout: Duration,
    path: &Path,
) -> Result<SavedBenchmark> {
    if hw.gpu_count == 0 {
        bail!("no GPUs detected on this node");
    }

    let outputs = run_backend_for_hardware(hw, bin_dir, timeout)?;

    let result = save_result_from_outputs(path, hw, &outputs)?;
    Ok(SavedBenchmark {
        path: path.to_path_buf(),
        result,
    })
}

fn save_result_from_outputs(
    path: &Path,
    hw: &HardwareSurvey,
    outputs: &[BenchmarkOutput],
) -> Result<BenchmarkResult> {
    let (gpus, result) = build_benchmark_result(hw, outputs);

    let fingerprint = BenchmarkFingerprint {
        gpus,
        is_soc: hw.is_soc,
        timestamp_secs: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    };

    try_save_fingerprint(path, &fingerprint)?;
    Ok(result)
}

fn build_benchmark_result(
    hw: &HardwareSurvey,
    outputs: &[BenchmarkOutput],
) -> (Vec<GpuBandwidth>, BenchmarkResult) {
    let hw_names = per_gpu_names(hw);

    let count = outputs
        .len()
        .min(hw.gpu_vram.len())
        .min(if hw_names.is_empty() {
            usize::MAX
        } else {
            hw_names.len()
        });

    let gpus: Vec<GpuBandwidth> = (0..count)
        .map(|i| GpuBandwidth {
            name: hw_names.get(i).cloned().unwrap_or_default(),
            vram_bytes: hw.gpu_vram.get(i).copied().unwrap_or(0),
            p50_gbps: outputs[i].p50_gbps,
            p90_gbps: outputs[i].p90_gbps,
            compute_tflops_fp32: outputs[i].compute_tflops_fp32,
            compute_tflops_fp16: outputs[i].compute_tflops_fp16,
        })
        .collect();

    let mem_bandwidth_gbps = gpus.iter().map(|g| g.p90_gbps).collect();
    let compute_tflops_fp32 = gpus
        .iter()
        .map(|g| g.compute_tflops_fp32)
        .collect::<Option<Vec<f64>>>();
    let compute_tflops_fp16 = gpus
        .iter()
        .map(|g| g.compute_tflops_fp16)
        .collect::<Option<Vec<f64>>>();

    (
        gpus,
        BenchmarkResult {
            mem_bandwidth_gbps,
            compute_tflops_fp32,
            compute_tflops_fp16,
        },
    )
}
#[cfg(test)]
mod tests;
