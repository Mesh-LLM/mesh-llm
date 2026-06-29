use mesh_llm_native_runtime::host::HostGpuProbe;
use mesh_llm_native_runtime::{
    HostCudaProfile, HostGpuProfile, HostRocmProfile, HostRuntimeProfile, HostVulkanProfile,
    NativeRuntimeBackendKind,
};
use std::collections::{BTreeMap, BTreeSet};
use std::process::Command;

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
    let lspci = command_output("lspci", &[]).unwrap_or_default();
    let proc_entries = linux_nvidia_proc_information_entries();
    let borrowed_entries: Vec<(&str, &str)> = proc_entries
        .iter()
        .map(|entry| (entry.path.as_str(), entry.info.as_str()))
        .collect();
    nvidia_gpu_profiles_from_probe_outputs(&nvidia_smi, &lspci, &borrowed_entries)
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
                cuda_sm: None,
                rocm_gfx: None,
            }
        })
        .collect()
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
        toolkit_majors.extend(cuda_majors_from_nvidia_smi());
    }
    let mut gpu_arches = env_string_set("MESH_LLM_CUDA_GPU_ARCHES");
    gpu_arches.extend(gpus.iter().filter_map(|gpu| gpu.cuda_sm.clone()));
    let has_cuda_label = gpus.iter().any(|gpu| {
        let label = gpu.display_name.to_ascii_lowercase();
        label.contains("nvidia") || label.contains("cuda")
    });
    if toolkit_majors.is_empty() && gpu_arches.is_empty() && !has_cuda_label {
        return None;
    }
    Some(HostCudaProfile {
        toolkit_majors,
        driver_version: std::env::var("MESH_LLM_CUDA_DRIVER_VERSION").ok(),
        gpu_arches,
    })
}

fn detect_rocm_profile(gpus: &[HostGpuProfile]) -> Option<HostRocmProfile> {
    let mut gpu_arches = env_string_set("MESH_LLM_ROCM_GPU_ARCHES");
    gpu_arches.extend(gpus.iter().filter_map(|gpu| gpu.rocm_gfx.clone()));
    let version = std::env::var("MESH_LLM_ROCM_VERSION").ok();
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
        gpu.cuda_sm = cuda_arches.get(index).cloned();
        gpu.rocm_gfx = rocm_arches.get(index).cloned();
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
        if let Some((_, version)) = line.split_once("CUDA Version:")
            && let Some(major) = leading_major_version(version)
        {
            majors.insert(major);
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
    append_command_lines(&mut labels, "rocminfo", &[]);
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
        let nvidia_gpus = nvidia_gpu_profiles_from_probe_outputs(nvidia_smi, lspci, &[]);
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

        let gpus = nvidia_gpu_profiles_from_probe_outputs(nvidia_smi, lspci, &proc_entries);

        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].display_name, "NVIDIA GeForce RTX 5090");
        assert_eq!(gpus[0].backend_device.as_deref(), Some("CUDA0"));
        assert_eq!(
            gpus[0].stable_id.as_deref(),
            Some("uuid:GPU-80ded6bd-1a89-2628-3d94-902187dbab1d")
        );
        let probe = gpus[0].probe.as_ref().expect("RTX 5090 probe details");
        assert_eq!(probe.source, "linux_nvidia_proc");
        assert_eq!(
            probe.path.as_deref(),
            Some("/proc/driver/nvidia/gpus/0000:01:00.0/information")
        );
        assert_eq!(probe.fields.get("IRQ").map(String::as_str), Some("16"));
        assert_eq!(
            probe.fields.get("DMA Mask").map(String::as_str),
            Some("0xfffffffffffff")
        );

        let names: Vec<&str> = gpus.iter().map(|gpu| gpu.display_name.as_str()).collect();
        assert!(!names.iter().any(|name| name.contains("DMA Mask")));
        assert!(!names.iter().any(|name| name.contains("IRQ")));
        assert!(!names.iter().any(|name| name.contains("Bus Location")));
    }
}
