use mesh_llm_native_runtime::host::HostGpuProbe;
use mesh_llm_native_runtime::{
    HostCudaProfile, HostGpuProfile, HostRocmProfile, HostRuntimeProfile, HostVulkanProfile,
    NativeRuntimeBackendKind,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

mod rocm;

pub fn host_runtime_profile() -> HostRuntimeProfile {
    let mut gpus = detect_gpus();
    apply_gpu_arch_overrides(&mut gpus);
    let cuda = detect_cuda_profile(&gpus);
    let rocm = detect_rocm_profile(&gpus);
    let vulkan = detect_vulkan_profile();
    HostRuntimeProfile {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        target_triple: option_env!("TARGET").map(str::to_string),
        available_flavors: detected_native_runtime_flavors(
            &gpus,
            cuda.as_ref(),
            rocm.as_ref(),
            vulkan.as_ref(),
        ),
        gpus,
        cuda,
        rocm,
        vulkan,
    }
}

pub fn detected_native_runtime_flavors(
    gpus: &[HostGpuProfile],
    cuda: Option<&HostCudaProfile>,
    rocm: Option<&HostRocmProfile>,
    vulkan: Option<&HostVulkanProfile>,
) -> BTreeSet<NativeRuntimeBackendKind> {
    let mut flavors = BTreeSet::from([NativeRuntimeBackendKind::Cpu]);
    if cfg!(target_os = "macos") {
        flavors.insert(NativeRuntimeBackendKind::Metal);
    }
    if cuda.is_some() {
        flavors.insert(NativeRuntimeBackendKind::Cuda);
    }
    if rocm.is_some() {
        flavors.insert(NativeRuntimeBackendKind::Rocm);
    }
    if vulkan.is_some() {
        flavors.insert(NativeRuntimeBackendKind::Vulkan);
    }
    for gpu in gpus {
        insert_label_flavors(&mut flavors, &gpu.display_name);
        if let Some(device) = &gpu.backend_device {
            insert_label_flavors(&mut flavors, device);
        }
    }
    flavors
}

fn detect_gpus() -> Vec<HostGpuProfile> {
    merge_nvidia_and_fallback_gpus(detect_nvidia_gpu_profiles(), fallback_gpu_profiles())
}

fn merge_nvidia_and_fallback_gpus(
    mut nvidia_gpus: Vec<HostGpuProfile>,
    mut fallback_gpus: Vec<HostGpuProfile>,
) -> Vec<HostGpuProfile> {
    if nvidia_gpus.is_empty() {
        return fallback_gpus;
    }

    fallback_gpus.retain(|gpu| !looks_like_nvidia_gpu_label(&gpu.display_name));
    nvidia_gpus.extend(fallback_gpus);
    nvidia_gpus
}

fn fallback_gpu_profiles() -> Vec<HostGpuProfile> {
    gpu_labels()
        .into_iter()
        .map(fallback_gpu_profile_from_label)
        .collect()
}

fn fallback_gpu_profile_from_label(label: String) -> HostGpuProfile {
    HostGpuProfile {
        display_name: label,
        backend_device: None,
        stable_id: None,
        vram_bytes: None,
        unified_memory: cfg!(target_os = "macos"),
        probe: None,
        cuda_sm: None,
        rocm_gfx: None,
    }
}

fn looks_like_nvidia_gpu_label(label: &str) -> bool {
    let label = label.to_ascii_lowercase();
    label.contains("nvidia") || label.contains("cuda")
}

fn detect_nvidia_gpu_profiles() -> Vec<HostGpuProfile> {
    let Some(nvidia_smi) = command_output("nvidia-smi", &["-L"]) else {
        return Vec::new();
    };
    let compute_caps = command_output(
        "nvidia-smi",
        &[
            "--query-gpu=index,compute_cap",
            "--format=csv,noheader,nounits",
        ],
    )
    .map(|output| nvidia_compute_caps_by_index(&output))
    .unwrap_or_default();
    let lspci = command_output("lspci", &[]).unwrap_or_default();
    let proc_entries = linux_nvidia_proc_information_entries();
    let borrowed_entries: Vec<(&str, &str)> = proc_entries
        .iter()
        .map(|entry| (entry.path.as_str(), entry.info.as_str()))
        .collect();
    nvidia_gpu_profiles_from_probe_outputs(&nvidia_smi, &compute_caps, &lspci, &borrowed_entries)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NvidiaSmiGpu {
    index: usize,
    name: String,
    vendor_uuid: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NvidiaProcInformationEntry {
    path: String,
    info: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NvidiaProcProbe {
    pci_bdf: Option<String>,
    vendor_uuid: Option<String>,
    probe: HostGpuProbe,
}

fn nvidia_gpu_profiles_from_probe_outputs(
    nvidia_smi_output: &str,
    compute_caps: &BTreeMap<usize, String>,
    lspci_output: &str,
    proc_entries: &[(&str, &str)],
) -> Vec<HostGpuProfile> {
    let mut proc_probes = proc_entries
        .iter()
        .map(|(path, info)| nvidia_proc_probe(path, info))
        .collect::<Vec<_>>();

    parse_nvidia_smi_list(nvidia_smi_output)
        .into_iter()
        .map(|gpu| {
            let pci_bdf = nvidia_lspci_bdf_for_name(lspci_output, &gpu.name);
            let probe = take_matching_nvidia_probe(
                &mut proc_probes,
                gpu.vendor_uuid.as_deref(),
                pci_bdf.as_deref(),
            );
            HostGpuProfile {
                display_name: gpu.name,
                backend_device: Some(format!("CUDA{}", gpu.index)),
                stable_id: gpu
                    .vendor_uuid
                    .as_ref()
                    .map(|uuid| format!("uuid:{uuid}"))
                    .or_else(|| pci_bdf.as_ref().map(|bdf| format!("pci:{bdf}"))),
                vram_bytes: None,
                unified_memory: false,
                probe,
                cuda_sm: compute_caps.get(&gpu.index).cloned(),
                rocm_gfx: None,
            }
        })
        .collect()
}

fn nvidia_compute_caps_by_index(output: &str) -> BTreeMap<usize, String> {
    output
        .lines()
        .filter_map(|line| {
            let (index, compute_cap) = line.split_once(',')?;
            let index = index.trim().parse::<usize>().ok()?;
            let cuda_sm = cuda_sm_from_compute_cap(compute_cap.trim())?;
            Some((index, cuda_sm))
        })
        .collect()
}

fn cuda_sm_from_compute_cap(value: &str) -> Option<String> {
    let (major, minor) = value.split_once('.')?;
    let major = major.trim();
    let minor = minor.trim();
    if major.is_empty()
        || minor.is_empty()
        || !major.chars().all(|ch| ch.is_ascii_digit())
        || !minor.chars().all(|ch| ch.is_ascii_digit())
    {
        return None;
    }
    Some(format!("{major}{minor}"))
}

fn parse_nvidia_smi_list(output: &str) -> Vec<NvidiaSmiGpu> {
    output
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let body = line.strip_prefix("GPU ")?;
            let (index, rest) = body.split_once(':')?;
            let index = index.trim().parse::<usize>().ok()?;
            let rest = rest.trim();
            let (name, vendor_uuid) = match rest.rsplit_once(" (UUID: ") {
                Some((name, uuid)) => (name.trim(), uuid.strip_suffix(')').map(str::trim)),
                None => (rest, None),
            };
            (!name.is_empty()).then(|| NvidiaSmiGpu {
                index,
                name: name.to_string(),
                vendor_uuid: vendor_uuid.map(ToOwned::to_owned),
            })
        })
        .collect()
}

fn nvidia_lspci_bdf_for_name(output: &str, name: &str) -> Option<String> {
    let name = name.to_ascii_lowercase();
    output.lines().find_map(|line| {
        let line = line.trim();
        if !looks_like_display_controller(line) {
            return None;
        }
        let lower = line.to_ascii_lowercase();
        if !name
            .split_whitespace()
            .filter(|token| *token != "nvidia" && *token != "geforce")
            .all(|token| lower.contains(token))
        {
            return None;
        }
        line.split_whitespace().next().map(normalize_pci_bdf)
    })
}

fn normalize_pci_bdf(bdf: &str) -> String {
    if bdf.matches(':').count() == 1 {
        format!("0000:{bdf}")
    } else {
        bdf.to_ascii_lowercase()
    }
}

fn nvidia_proc_probe(path: &str, info: &str) -> NvidiaProcProbe {
    let fields = nvidia_proc_fields(info);
    NvidiaProcProbe {
        pci_bdf: fields
            .get("Bus Location")
            .map(String::as_str)
            .map(normalize_pci_bdf),
        vendor_uuid: fields.get("GPU UUID").cloned(),
        probe: HostGpuProbe {
            source: "linux_nvidia_proc".to_string(),
            path: Some(path.to_string()),
            fields,
            raw_lines: info.lines().map(str::to_string).collect(),
        },
    }
}

fn nvidia_proc_fields(info: &str) -> BTreeMap<String, String> {
    info.lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(':')?;
            let key = key.trim();
            if key.is_empty() {
                return None;
            }
            Some((key.to_string(), value.trim().to_string()))
        })
        .collect()
}

fn take_matching_nvidia_probe(
    probes: &mut Vec<NvidiaProcProbe>,
    vendor_uuid: Option<&str>,
    pci_bdf: Option<&str>,
) -> Option<HostGpuProbe> {
    let index = probes.iter().position(|probe| {
        vendor_uuid.is_some_and(|uuid| probe.vendor_uuid.as_deref() == Some(uuid))
            || pci_bdf.is_some_and(|bdf| probe.pci_bdf.as_deref() == Some(bdf))
    })?;
    Some(probes.remove(index).probe)
}

fn detect_cuda_profile(gpus: &[HostGpuProfile]) -> Option<HostCudaProfile> {
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
fn installed_cuda_toolkit_majors() -> BTreeSet<u32> {
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
fn loader_search_dirs() -> Vec<PathBuf> {
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

fn append_cuda_root_dirs(dirs: &mut Vec<PathBuf>, root: &Path) {
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

fn append_cuda_roots_under(dirs: &mut Vec<PathBuf>, parent: &Path) {
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
struct CudaLibraryEvidence {
    cudart: bool,
    cublas: bool,
    cublas_lt: bool,
}

impl CudaLibraryEvidence {
    /// A native runtime needs all three, so partial installs do not count.
    fn is_complete(&self) -> bool {
        self.cudart && self.cublas && self.cublas_lt
    }
}

fn record_cuda_ldconfig_line(evidence: &mut BTreeMap<u32, CudaLibraryEvidence>, line: &str) {
    let Some((library, target)) = line.rsplit_once("=>") else {
        return;
    };
    let Some(name) = library.split_whitespace().next() else {
        return;
    };
    record_cuda_soname_path(evidence, name, Path::new(target.trim()));
}

fn record_cuda_soname_path(
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

fn valid_cuda_library_target(path: &Path) -> bool {
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

fn current_elf_identity() -> Option<(u8, u16)> {
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

fn detect_rocm_profile(gpus: &[HostGpuProfile]) -> Option<HostRocmProfile> {
    let mut gpu_arches = env_string_set("MESH_LLM_ROCM_GPU_ARCHES");
    gpu_arches.extend(rocm::gpu_arches());
    detect_rocm_profile_with_arches(
        gpus,
        gpu_arches,
        std::env::var("MESH_LLM_ROCM_VERSION").ok(),
    )
}

fn detect_rocm_profile_with_arches(
    gpus: &[HostGpuProfile],
    mut gpu_arches: BTreeSet<String>,
    version: Option<String>,
) -> Option<HostRocmProfile> {
    gpu_arches.extend(gpus.iter().filter_map(|gpu| gpu.rocm_gfx.clone()));
    let has_rocm_label = gpus.iter().any(|gpu| {
        let label = gpu.display_name.to_ascii_lowercase();
        label.contains("amd") || label.contains("radeon") || label.contains("rocm")
    });
    if gpu_arches.is_empty() && version.is_none() && !has_rocm_label {
        return None;
    }
    Some(HostRocmProfile {
        version,
        gpu_arches,
    })
}

fn detect_vulkan_profile() -> Option<HostVulkanProfile> {
    let api_version = std::env::var("MESH_LLM_VULKAN_API_VERSION").ok();
    let enabled = std::env::var("MESH_LLM_VULKAN_AVAILABLE")
        .ok()
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
    if enabled || api_version.is_some() || command_output("vulkaninfo", &["--summary"]).is_some() {
        return Some(HostVulkanProfile { api_version });
    }
    None
}

fn apply_gpu_arch_overrides(gpus: &mut [HostGpuProfile]) {
    let cuda_arches = env_string_vec("MESH_LLM_CUDA_GPU_ARCHES");
    let rocm_arches = env_string_vec("MESH_LLM_ROCM_GPU_ARCHES");
    for (index, gpu) in gpus.iter_mut().enumerate() {
        if let Some(cuda_sm) = cuda_arches.get(index) {
            gpu.cuda_sm = Some(cuda_sm.clone());
        }
        if let Some(rocm_gfx) = rocm_arches.get(index) {
            gpu.rocm_gfx = Some(rocm_gfx.clone());
        }
    }
}

fn cuda_majors_from_nvidia_smi() -> BTreeSet<u32> {
    let Some(output) = command_output("nvidia-smi", &[]) else {
        return BTreeSet::new();
    };
    cuda_majors_from_nvidia_smi_output(&output)
}

fn cuda_majors_from_nvidia_smi_output(output: &str) -> BTreeSet<u32> {
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

fn cuda_major_from_token(token: &str) -> Option<u32> {
    token
        .strip_prefix("CUDA")?
        .trim_start_matches("Version:")
        .trim_matches(|ch: char| !ch.is_ascii_digit())
        .split('.')
        .next()
        .and_then(|value| value.parse::<u32>().ok())
}

fn leading_major_version(value: &str) -> Option<u32> {
    value
        .trim()
        .trim_start_matches(|ch: char| !ch.is_ascii_digit())
        .split('.')
        .next()
        .and_then(|value| value.parse::<u32>().ok())
}

fn gpu_labels() -> Vec<String> {
    let mut labels = Vec::new();
    append_command_lines(&mut labels, "vulkaninfo", &["--summary"]);
    append_platform_gpu_labels(&mut labels);
    labels.sort();
    labels.dedup();
    labels
}

#[cfg(target_os = "linux")]
fn append_platform_gpu_labels(labels: &mut Vec<String>) {
    append_command_lines(labels, "lspci", &[]);
}

#[cfg(target_os = "linux")]
fn linux_nvidia_proc_information_entries() -> Vec<NvidiaProcInformationEntry> {
    let Ok(entries) = std::fs::read_dir("/proc/driver/nvidia/gpus") else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path().join("information");
            let info = std::fs::read_to_string(&path).ok()?;
            Some(NvidiaProcInformationEntry {
                path: path.display().to_string(),
                info,
            })
        })
        .collect()
}

#[cfg(not(target_os = "linux"))]
fn linux_nvidia_proc_information_entries() -> Vec<NvidiaProcInformationEntry> {
    Vec::new()
}

#[cfg(target_os = "windows")]
fn append_platform_gpu_labels(labels: &mut Vec<String>) {
    append_command_lines(
        labels,
        "powershell",
        &[
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
        ],
    );
}

#[cfg(target_os = "macos")]
fn append_platform_gpu_labels(labels: &mut Vec<String>) {
    append_command_lines(labels, "system_profiler", &["SPDisplaysDataType"]);
}

#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
fn append_platform_gpu_labels(_labels: &mut Vec<String>) {}

fn append_command_lines(labels: &mut Vec<String>, program: &str, args: &[&str]) {
    let Some(output) = command_output(program, args) else {
        return;
    };
    labels.extend(gpu_labels_from_command_output(program, args, &output));
}

fn gpu_labels_from_command_output(program: &str, args: &[&str], output: &str) -> Vec<String> {
    match (program, args) {
        ("nvidia-smi", ["-L"]) => output
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with("GPU ") && line.contains(':'))
            .map(str::to_string)
            .collect(),
        ("vulkaninfo", ["--summary"]) => vulkaninfo_device_names(output),
        ("lspci", []) => output
            .lines()
            .map(str::trim)
            .filter(|line| looks_like_display_controller(line))
            .map(str::to_string)
            .collect(),
        _ => output
            .lines()
            .map(str::trim)
            .filter(|line| looks_like_gpu_label(line))
            .map(str::to_string)
            .collect(),
    }
}

fn vulkaninfo_device_names(output: &str) -> Vec<String> {
    output
        .lines()
        .map(str::trim)
        .filter_map(|line| line.strip_prefix("deviceName"))
        .filter_map(|line| line.split_once('=').map(|(_, value)| value.trim()))
        .filter(|value| !value.is_empty())
        .filter(|value| !looks_like_software_vulkan_adapter(value))
        .map(str::to_string)
        .collect()
}

fn looks_like_software_vulkan_adapter(value: &str) -> bool {
    let label = value.to_ascii_lowercase();
    [
        "llvmpipe",
        "swiftshader",
        "lavapipe",
        "softpipe",
        "software rasterizer",
    ]
    .iter()
    .any(|marker| label.contains(marker))
}

fn looks_like_display_controller(line: &str) -> bool {
    let label = line.to_ascii_lowercase();
    (label.contains("vga compatible controller")
        || label.contains("3d controller")
        || label.contains("display controller"))
        && looks_like_gpu_label(line)
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8(output.stdout).ok())
        .flatten()
}

fn looks_like_gpu_label(line: &str) -> bool {
    let label = line.to_ascii_lowercase();
    label.contains("gpu")
        || label.contains("nvidia")
        || label.contains("cuda")
        || label.contains("amd")
        || label.contains("radeon")
        || label.contains("rocm")
        || label.contains("vulkan")
        || label.contains("metal")
}

fn insert_label_flavors(flavors: &mut BTreeSet<NativeRuntimeBackendKind>, label: &str) {
    let label = label.to_ascii_lowercase();
    if label.contains("cuda") || label.contains("nvidia") {
        flavors.insert(NativeRuntimeBackendKind::Cuda);
    }
    if label.contains("rocm")
        || label.contains("hip")
        || label.contains("amd")
        || label.contains("radeon")
    {
        flavors.insert(NativeRuntimeBackendKind::Rocm);
    }
    if label.contains("vulkan") {
        flavors.insert(NativeRuntimeBackendKind::Vulkan);
    }
}

fn env_u32(name: &str) -> Option<u32> {
    std::env::var(name).ok()?.parse().ok()
}

fn env_u32_set(name: &str) -> BTreeSet<u32> {
    env_string_vec(name)
        .into_iter()
        .filter_map(|value| value.parse().ok())
        .collect()
}

fn env_string_set(name: &str) -> BTreeSet<String> {
    env_string_vec(name).into_iter().collect()
}

fn env_string_vec(name: &str) -> Vec<String> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EnvVarGuard {
        key: &'static str,
        previous: Option<String>,
    }

    impl EnvVarGuard {
        fn clear(key: &'static str) -> Self {
            let previous = std::env::var(key).ok();
            // SAFETY: this test module only mutates these override vars inside scoped guards.
            unsafe { std::env::remove_var(key) };
            Self { key, previous }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            match &self.previous {
                // SAFETY: restore the scoped test mutation before the guard leaves scope.
                Some(value) => unsafe { std::env::set_var(self.key, value) },
                // SAFETY: restore the scoped test mutation before the guard leaves scope.
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }

    fn write_minimal_elf(path: &Path, class: u8, machine: u16) {
        let mut bytes = vec![0; 20];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = class;
        bytes[5] = 1;
        bytes[18..20].copy_from_slice(&machine.to_le_bytes());
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn cuda_library_probe_includes_custom_roots_and_rejects_invalid_targets() {
        let Some((class, machine)) = current_elf_identity() else {
            return;
        };
        let root = std::env::temp_dir().join(format!("mesh-llm-cuda-probe-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        for name in ["libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"] {
            write_minimal_elf(&root.join(name), class, machine);
        }
        for name in ["libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13"] {
            write_minimal_elf(&root.join(name), class, machine.wrapping_add(1));
        }

        let mut evidence = BTreeMap::new();
        for name in ["libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"] {
            record_cuda_soname_path(&mut evidence, name, &root.join(name));
        }
        record_cuda_soname_path(&mut evidence, "libcudart.so.12", &root.join("missing.so"));
        for name in ["libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13"] {
            record_cuda_soname_path(&mut evidence, name, &root.join(name));
        }

        assert_eq!(
            evidence
                .into_iter()
                .filter(|(_, evidence)| evidence.is_complete())
                .map(|(major, _)| major)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([12])
        );
        let mut custom_root_dirs = Vec::new();
        append_cuda_root_dirs(&mut custom_root_dirs, Path::new("/opt/cuda-12"));
        assert!(custom_root_dirs.contains(&PathBuf::from("/opt/cuda-12/lib64")));
        assert!(custom_root_dirs.contains(&PathBuf::from("/opt/cuda-12/lib")));
        let _ = fs::remove_dir_all(root);
    }

    fn profile(label: &str) -> HostGpuProfile {
        HostGpuProfile {
            display_name: label.to_string(),
            backend_device: None,
            stable_id: None,
            vram_bytes: None,
            unified_memory: false,
            probe: None,
            cuda_sm: None,
            rocm_gfx: None,
        }
    }

    struct ExpectedNvidiaProcGpu<'a> {
        display_name: &'a str,
        backend_device: &'a str,
        cuda_sm: &'a str,
        stable_id: &'a str,
        probe_path: &'a str,
        irq: &'a str,
        dma_mask: &'a str,
    }

    fn assert_nvidia_proc_gpu(gpu: &HostGpuProfile, expected: ExpectedNvidiaProcGpu<'_>) {
        assert_eq!(gpu.display_name, expected.display_name);
        assert_eq!(gpu.backend_device.as_deref(), Some(expected.backend_device));
        assert_eq!(gpu.cuda_sm.as_deref(), Some(expected.cuda_sm));
        assert_eq!(gpu.stable_id.as_deref(), Some(expected.stable_id));
        let probe = gpu
            .probe
            .as_ref()
            .unwrap_or_else(|| panic!("{} probe details", expected.display_name));
        assert_eq!(probe.source, "linux_nvidia_proc");
        assert_eq!(probe.path.as_deref(), Some(expected.probe_path));
        assert_eq!(
            probe.fields.get("IRQ").map(String::as_str),
            Some(expected.irq)
        );
        assert_eq!(
            probe.fields.get("DMA Mask").map(String::as_str),
            Some(expected.dma_mask)
        );
    }

    #[test]
    fn nvidia_labels_enable_cuda() {
        let flavors = detected_native_runtime_flavors(
            &[profile("NVIDIA GeForce RTX 4090")],
            None,
            None,
            None,
        );

        assert!(flavors.contains(&NativeRuntimeBackendKind::Cpu));
        assert!(flavors.contains(&NativeRuntimeBackendKind::Cuda));
    }

    #[test]
    fn amd_labels_enable_rocm() {
        let flavors =
            detected_native_runtime_flavors(&[profile("AMD Radeon PRO W7900")], None, None, None);

        assert!(flavors.contains(&NativeRuntimeBackendKind::Rocm));
    }

    #[test]
    fn kfd_architecture_evidence_enables_rocm_without_inventory_synthesis() {
        let profile =
            detect_rocm_profile_with_arches(&[], BTreeSet::from(["gfx942".to_string()]), None)
                .expect("KFD architecture should enable a ROCm runtime profile");

        assert_eq!(profile.gpu_arches, BTreeSet::from(["gfx942".to_string()]));
    }

    #[test]
    fn mi300x_kfd_evidence_selects_rocm_over_cpu_runtime() {
        use mesh_llm_native_runtime::{
            NativeRuntimeArtifact, NativeRuntimeBackend, NativeRuntimePlatform, RuntimeSelection,
            select_native_runtime_from_artifacts,
        };

        let rocm =
            detect_rocm_profile_with_arches(&[], BTreeSet::from(["gfx942".to_string()]), None)
                .expect("MI300X KFD evidence should produce a ROCm profile");
        let runtime_profile = HostRuntimeProfile {
            os: "linux".to_string(),
            arch: "x86_64".to_string(),
            target_triple: None,
            available_flavors: detected_native_runtime_flavors(&[], None, Some(&rocm), None),
            gpus: Vec::new(),
            cuda: None,
            rocm: Some(rocm),
            vulkan: None,
        };
        let artifact = |id: &str, backend: NativeRuntimeBackend| NativeRuntimeArtifact {
            id: id.to_string(),
            mesh_version: Some("test".to_string()),
            skippy_abi: "test-abi".to_string(),
            platform: NativeRuntimePlatform {
                os: "linux".to_string(),
                arch: "x86_64".to_string(),
                target: None,
            },
            backend,
            rank: 0,
            libraries: vec!["lib/libmeshllm_ffi.so".to_string()],
            files: Default::default(),
            tools: Default::default(),
            url: None,
            sha256: None,
            signature: None,
        };
        let artifacts = vec![
            artifact("runtime-cpu", NativeRuntimeBackend::cpu()),
            artifact(
                "runtime-rocm",
                NativeRuntimeBackend::rocm(vec!["gfx942".to_string()]),
            ),
        ];

        let selected = select_native_runtime_from_artifacts(
            &artifacts,
            &runtime_profile,
            "test",
            Some("test-abi"),
            &RuntimeSelection::Recommended,
        )
        .expect("MI300X should select the compatible ROCm runtime");

        assert_eq!(
            selected.artifact.backend.kind,
            NativeRuntimeBackendKind::Rocm
        );
    }

    #[test]
    fn fallback_profiles_do_not_synthesize_backend_ordinals() {
        let gpu = fallback_gpu_profile_from_label("AMD Radeon PRO W7900".to_string());

        assert_eq!(gpu.display_name, "AMD Radeon PRO W7900");
        assert_eq!(gpu.backend_device, None);
        assert_eq!(gpu.stable_id, None);
        assert!(
            detected_native_runtime_flavors(&[gpu], None, None, None)
                .contains(&NativeRuntimeBackendKind::Rocm)
        );
    }

    #[test]
    fn parses_cuda_version_label_from_nvidia_smi_banner() {
        let output = "| NVIDIA-SMI 595.78 Driver Version: 595.78 CUDA Version: 13.2 |\n";

        assert_eq!(
            cuda_majors_from_nvidia_smi_output(output),
            BTreeSet::from([13])
        );
    }

    #[test]
    fn parses_cuda_umd_version_label_from_nvidia_smi_banner() {
        let output = "| NVIDIA-SMI 610.43.02 KMD Version: 610.43.02 CUDA UMD Version: 13.3 |\n";

        assert_eq!(
            cuda_majors_from_nvidia_smi_output(output),
            BTreeSet::from([13])
        );
    }

    #[test]
    fn parses_nvidia_compute_caps_as_cuda_arches() {
        let output = "\
0, 12.0
1, 8.6
";

        assert_eq!(
            nvidia_compute_caps_by_index(output),
            BTreeMap::from([(0, "120".to_string()), (1, "86".to_string())])
        );
    }

    #[test]
    fn empty_gpu_arch_overrides_preserve_detected_arches() {
        let _cuda_arches = EnvVarGuard::clear("MESH_LLM_CUDA_GPU_ARCHES");
        let _rocm_arches = EnvVarGuard::clear("MESH_LLM_ROCM_GPU_ARCHES");
        let mut gpus = vec![HostGpuProfile {
            cuda_sm: Some("120".to_string()),
            rocm_gfx: Some("gfx1200".to_string()),
            ..profile("NVIDIA GeForce RTX 5090")
        }];

        apply_gpu_arch_overrides(&mut gpus);

        assert_eq!(gpus[0].cuda_sm.as_deref(), Some("120"));
        assert_eq!(gpus[0].rocm_gfx.as_deref(), Some("gfx1200"));
    }

    #[test]
    fn vulkaninfo_labels_keep_only_device_names() {
        let output = "\
VULKANINFO
Vulkan Instance Version: 1.4.321
GPU0:
deviceName         = NVIDIA Tegra Orin (nvgpu)
deviceType         = PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
driverName         = NVIDIA
";

        assert_eq!(
            gpu_labels_from_command_output("vulkaninfo", &["--summary"], output),
            vec!["NVIDIA Tegra Orin (nvgpu)".to_string()]
        );
    }

    #[test]
    fn vulkaninfo_labels_ignore_software_adapters() {
        let output = "\
GPU0:
deviceName         = llvmpipe (LLVM 18.1.8, 256 bits)
GPU1:
deviceName         = SwiftShader Device (Subzero)
GPU2:
deviceName         = AMD Radeon PRO W7900
";

        assert_eq!(
            gpu_labels_from_command_output("vulkaninfo", &["--summary"], output),
            vec!["AMD Radeon PRO W7900".to_string()]
        );
    }

    #[test]
    fn lspci_labels_ignore_nvidia_pci_bridges() {
        let output = "\
0004:00:00.0 PCI bridge: NVIDIA Corporation Device 229c (rev a1)
0008:01:00.0 3D controller: NVIDIA Corporation GA102GL [RTX A6000] (rev a1)
";

        assert_eq!(
            gpu_labels_from_command_output("lspci", &[], output),
            vec![
                "0008:01:00.0 3D controller: NVIDIA Corporation GA102GL [RTX A6000] (rev a1)"
                    .to_string()
            ]
        );
    }

    #[test]
    fn nvidia_probe_results_merge_with_fallback_labels() {
        let nvidia_smi = "\
GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-80ded6bd-1a89-2628-3d94-902187dbab1d)
";
        let lspci = "\
01:00.0 VGA compatible controller: NVIDIA Corporation GB202 [GeForce RTX 5090] (rev a1)
";
        let compute_caps = BTreeMap::from([(0, "120".to_string())]);
        let nvidia_gpus =
            nvidia_gpu_profiles_from_probe_outputs(nvidia_smi, &compute_caps, lspci, &[]);
        let fallback_gpus = vec![
            profile("NVIDIA Corporation GB202 [GeForce RTX 5090]"),
            profile("AMD Radeon PRO W7900"),
        ];
        let merged = merge_nvidia_and_fallback_gpus(nvidia_gpus, fallback_gpus);

        let names = merged
            .iter()
            .map(|gpu| gpu.display_name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names, ["NVIDIA GeForce RTX 5090", "AMD Radeon PRO W7900"]);
        assert_eq!(merged[0].cuda_sm.as_deref(), Some("120"));
    }

    #[test]
    fn nvidia_proc_details_are_nested_under_matching_gpus() {
        let nvidia_smi = "\
GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-80ded6bd-1a89-2628-3d94-902187dbab1d)
GPU 1: NVIDIA GeForce RTX 3080 (UUID: GPU-6b7fe24c-5f15-4ac5-88d6-c8934135a4ea)
";
        let lspci = "\
01:00.0 VGA compatible controller: NVIDIA Corporation GB202 [GeForce RTX 5090] (rev a1)
06:00.0 VGA compatible controller: NVIDIA Corporation GA102 [GeForce RTX 3080] (rev a1)
";
        let proc_entries = vec![
            (
                "/proc/driver/nvidia/gpus/0000:01:00.0/information",
                "\
Model: \t\t NVIDIA GeForce RTX 5090
IRQ:   \t\t 16
GPU UUID: \t GPU-80ded6bd-1a89-2628-3d94-902187dbab1d
Video BIOS: \t 98.02.2e.40.7f
Bus Type: \t PCIe
DMA Size: \t 52 bits
DMA Mask: \t 0xfffffffffffff
Bus Location: \t 0000:01:00.0
Device Minor: \t 0
GPU Firmware: \t 610.43.02
GPU Excluded:\t No
",
            ),
            (
                "/proc/driver/nvidia/gpus/0000:06:00.0/information",
                "\
Model: \t\t NVIDIA GeForce RTX 3080
IRQ:   \t\t 184
GPU UUID: \t GPU-6b7fe24c-5f15-4ac5-88d6-c8934135a4ea
Video BIOS: \t 94.02.42.80.31
Bus Type: \t PCIe
DMA Size: \t 47 bits
DMA Mask: \t 0x7fffffffffff
Bus Location: \t 0000:06:00.0
Device Minor: \t 1
GPU Firmware: \t 610.43.02
GPU Excluded:\t No
",
            ),
        ];

        let compute_caps = BTreeMap::from([(0, "120".to_string()), (1, "86".to_string())]);
        let gpus =
            nvidia_gpu_profiles_from_probe_outputs(nvidia_smi, &compute_caps, lspci, &proc_entries);

        assert_eq!(gpus.len(), 2);
        assert_nvidia_proc_gpu(
            &gpus[0],
            ExpectedNvidiaProcGpu {
                display_name: "NVIDIA GeForce RTX 5090",
                backend_device: "CUDA0",
                cuda_sm: "120",
                stable_id: "uuid:GPU-80ded6bd-1a89-2628-3d94-902187dbab1d",
                probe_path: "/proc/driver/nvidia/gpus/0000:01:00.0/information",
                irq: "16",
                dma_mask: "0xfffffffffffff",
            },
        );
        assert_nvidia_proc_gpu(
            &gpus[1],
            ExpectedNvidiaProcGpu {
                display_name: "NVIDIA GeForce RTX 3080",
                backend_device: "CUDA1",
                cuda_sm: "86",
                stable_id: "uuid:GPU-6b7fe24c-5f15-4ac5-88d6-c8934135a4ea",
                probe_path: "/proc/driver/nvidia/gpus/0000:06:00.0/information",
                irq: "184",
                dma_mask: "0x7fffffffffff",
            },
        );

        let names: Vec<&str> = gpus.iter().map(|gpu| gpu.display_name.as_str()).collect();
        assert!(!names.iter().any(|name| name.contains("DMA Mask")));
        assert!(!names.iter().any(|name| name.contains("IRQ")));
        assert!(!names.iter().any(|name| name.contains("Bus Location")));
    }
}
