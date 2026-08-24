//! CUDA runtime capability detection and loader evidence.

use crate::environment::{env_string_set, env_u32, env_u32_set};
use crate::platform::command_output;
use mesh_llm_native_runtime::{HostCudaProfile, HostGpuProfile};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
pub(crate) fn detect_cuda_profile(gpus: &[HostGpuProfile]) -> Option<HostCudaProfile> {
    let mut toolkit_majors = env_u32_set("MESH_LLM_CUDA_TOOLKIT_MAJORS");
    if let Some(major) = env_u32("MESH_LLM_CUDA_TOOLKIT_MAJOR") {
        toolkit_majors.insert(major);
    }
    if toolkit_majors.is_empty() {
        toolkit_majors.extend(installed_cuda_toolkit_majors());
    }
    // `nvidia-smi` reports the newest CUDA the driver supports, which is often
    // ahead of the installed toolkit. Keep it separate so it is only used as an
    // upper bound during runtime selection.
    let driver_max_major = env_u32("MESH_LLM_CUDA_DRIVER_MAX_MAJOR")
        .or_else(|| cuda_majors_from_nvidia_smi().into_iter().next_back());
    let mut gpu_arches = env_string_set("MESH_LLM_CUDA_GPU_ARCHES");
    gpu_arches.extend(gpus.iter().filter_map(|gpu| gpu.cuda_sm.clone()));
    let has_cuda_label = gpus.iter().any(|gpu| {
        let label = gpu.display_name.to_ascii_lowercase();
        label.contains("nvidia") || label.contains("cuda")
    });
    if toolkit_majors.is_empty()
        && driver_max_major.is_none()
        && gpu_arches.is_empty()
        && !has_cuda_label
    {
        return None;
    }
    Some(HostCudaProfile {
        toolkit_majors,
        driver_max_major,
        driver_version: std::env::var("MESH_LLM_CUDA_DRIVER_VERSION").ok(),
        gpu_arches,
    })
}

/// CUDA toolkit majors that the dynamic loader can actually resolve on this
/// host.
///
/// Linux native runtimes link `libcudart`/`libcublas`/`libcublasLt` without
/// bundling them, so a runtime only loads when the loader itself can find a
/// matching-major copy of all three. Deliberately probe the loader's own view
/// — the `ldconfig` cache and `LD_LIBRARY_PATH` — rather than guessing from
/// installation directory names: a `/usr/local/cuda-13` directory can exist
/// with no usable libraries, and a toolkit that the loader cannot see cannot
/// be loaded no matter where it lives.
pub(crate) fn installed_cuda_toolkit_majors() -> BTreeSet<u32> {
    let mut evidence: BTreeMap<u32, CudaLibraryEvidence> = BTreeMap::new();
    if let Some(output) = command_output("ldconfig", &["-p"]) {
        for line in output.lines() {
            record_cuda_ldconfig_line(&mut evidence, line);
        }
    }
    for dir in loader_search_dirs() {
        let Ok(entries) = fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            record_cuda_soname_path(&mut evidence, &entry.file_name().to_string_lossy(), &path);
        }
    }
    evidence
        .into_iter()
        .filter(|(_, found)| found.is_complete())
        .map(|(major, _)| major)
        .collect()
}

/// Directories the loader searches ahead of its cache, plus common CUDA roots.
pub(crate) fn loader_search_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(value) = std::env::var_os("LD_LIBRARY_PATH") {
        dirs.extend(std::env::split_paths(&value));
    }
    for variable in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "CONDA_PREFIX"] {
        if let Some(root) = std::env::var_os(variable).filter(|value| !value.is_empty()) {
            append_cuda_root_dirs(&mut dirs, Path::new(&root));
        }
    }
    append_cuda_roots_under(&mut dirs, Path::new("/usr/local"));
    dirs.sort();
    dirs.dedup();
    dirs
}

pub(crate) fn append_cuda_root_dirs(dirs: &mut Vec<PathBuf>, root: &Path) {
    dirs.push(root.to_path_buf());
    for suffix in [
        "lib64",
        "lib",
        "targets/x86_64-linux/lib",
        "targets/aarch64-linux/lib",
    ] {
        dirs.push(root.join(suffix));
    }
}

pub(crate) fn append_cuda_roots_under(dirs: &mut Vec<PathBuf>, parent: &Path) {
    let Ok(entries) = fs::read_dir(parent) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        if path.is_dir() && (name == "cuda" || name.to_string_lossy().starts_with("cuda-")) {
            append_cuda_root_dirs(dirs, &path);
        }
    }
}

/// Per-major record of which valid, current-architecture CUDA libraries were found.
#[derive(Default)]
pub(crate) struct CudaLibraryEvidence {
    cudart: bool,
    cublas: bool,
    cublas_lt: bool,
}

impl CudaLibraryEvidence {
    /// A native runtime needs all three, so partial installs do not count.
    pub(crate) fn is_complete(&self) -> bool {
        self.cudart && self.cublas && self.cublas_lt
    }
}

pub(crate) fn record_cuda_ldconfig_line(
    evidence: &mut BTreeMap<u32, CudaLibraryEvidence>,
    line: &str,
) {
    let Some((library, target)) = line.rsplit_once("=>") else {
        return;
    };
    let Some(name) = library.split_whitespace().next() else {
        return;
    };
    record_cuda_soname_path(evidence, name, Path::new(target.trim()));
}

pub(crate) fn record_cuda_soname_path(
    evidence: &mut BTreeMap<u32, CudaLibraryEvidence>,
    token: &str,
    path: &Path,
) {
    let name = token.rsplit('/').next().unwrap_or(token);
    // `libcublasLt.so.` is checked first so it is not shadowed by `libcublas`.
    for prefix in ["libcudart.so.", "libcublasLt.so.", "libcublas.so."] {
        let Some(rest) = name.strip_prefix(prefix) else {
            continue;
        };
        let Some(major) = leading_major_version(rest) else {
            continue;
        };
        if !valid_cuda_library_target(path) {
            return;
        }
        let found = evidence.entry(major).or_default();
        match prefix {
            "libcudart.so." => found.cudart = true,
            "libcublasLt.so." => found.cublas_lt = true,
            _ => found.cublas = true,
        }
        return;
    }
}

pub(crate) fn valid_cuda_library_target(path: &Path) -> bool {
    let Ok(mut file) = fs::File::open(path) else {
        return false;
    };
    let mut bytes = [0_u8; 20];
    if file.read_exact(&mut bytes).is_err() || &bytes[..4] != b"\x7fELF" {
        return false;
    }
    let machine = match bytes[5] {
        1 => u16::from_le_bytes([bytes[18], bytes[19]]),
        2 => u16::from_be_bytes([bytes[18], bytes[19]]),
        _ => return false,
    };
    let identity = (bytes[4], machine);
    match current_elf_identity() {
        Some(expected) => identity == expected,
        None => true,
    }
}

pub(crate) fn current_elf_identity() -> Option<(u8, u16)> {
    #[cfg(target_arch = "x86_64")]
    {
        Some((2, 62))
    }
    #[cfg(target_arch = "aarch64")]
    {
        Some((2, 183))
    }
    #[cfg(target_arch = "x86")]
    {
        Some((1, 3))
    }
    #[cfg(target_arch = "arm")]
    {
        Some((1, 40))
    }
    #[cfg(target_arch = "riscv64")]
    {
        Some((2, 243))
    }
    #[cfg(target_arch = "powerpc64")]
    {
        Some((2, 21))
    }
    #[cfg(target_arch = "s390x")]
    {
        Some((2, 22))
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "x86",
        target_arch = "arm",
        target_arch = "riscv64",
        target_arch = "powerpc64",
        target_arch = "s390x"
    )))]
    {
        None
    }
}

pub(crate) fn cuda_majors_from_nvidia_smi() -> BTreeSet<u32> {
    let Some(output) = command_output("nvidia-smi", &[]) else {
        return BTreeSet::new();
    };
    cuda_majors_from_nvidia_smi_output(&output)
}

pub(crate) fn cuda_majors_from_nvidia_smi_output(output: &str) -> BTreeSet<u32> {
    let mut majors = BTreeSet::new();
    for token in output.split_whitespace() {
        if let Some(major) = cuda_major_from_token(token) {
            majors.insert(major);
        }
    }
    for line in output.lines() {
        for marker in ["CUDA Version:", "CUDA UMD Version:"] {
            if let Some((_, version)) = line.split_once(marker)
                && let Some(major) = leading_major_version(version)
            {
                majors.insert(major);
            }
        }
    }
    majors
}

pub(crate) fn cuda_major_from_token(token: &str) -> Option<u32> {
    token
        .strip_prefix("CUDA")?
        .trim_start_matches("Version:")
        .trim_matches(|ch: char| !ch.is_ascii_digit())
        .split('.')
        .next()
        .and_then(|value| value.parse::<u32>().ok())
}

pub(crate) fn leading_major_version(value: &str) -> Option<u32> {
    value
        .trim()
        .trim_start_matches(|ch: char| !ch.is_ascii_digit())
        .split('.')
        .next()
        .and_then(|value| value.parse::<u32>().ok())
}
