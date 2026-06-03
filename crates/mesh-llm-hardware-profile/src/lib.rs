use mesh_llm_native_runtime::{
    HostCudaProfile, HostGpuProfile, HostRocmProfile, HostRuntimeProfile, HostVulkanProfile,
    NativeRuntimeBackendKind,
};
use std::collections::BTreeSet;
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
    let labels = gpu_labels();
    labels
        .into_iter()
        .enumerate()
        .map(|(index, label)| HostGpuProfile {
            display_name: label,
            backend_device: None,
            stable_id: Some(format!("detected-{index}")),
            vram_bytes: None,
            unified_memory: cfg!(target_os = "macos"),
            cuda_sm: None,
            rocm_gfx: None,
        })
        .collect()
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
    let mut majors = BTreeSet::new();
    let Some(output) = command_output("nvidia-smi", &[]) else {
        return majors;
    };
    for token in output.split_whitespace() {
        if let Some(major) = cuda_major_from_token(token) {
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

fn gpu_labels() -> Vec<String> {
    let mut labels = Vec::new();
    append_command_lines(&mut labels, "nvidia-smi", &["-L"]);
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
    append_linux_nvidia_proc_labels(labels);
}

#[cfg(target_os = "linux")]
fn append_linux_nvidia_proc_labels(labels: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir("/proc/driver/nvidia/gpus") else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path().join("information");
        let Ok(info) = std::fs::read_to_string(path) else {
            continue;
        };
        labels.extend(info.lines().map(str::to_string));
    }
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
    labels.extend(
        output
            .lines()
            .map(str::trim)
            .filter(|line| looks_like_gpu_label(line))
            .map(str::to_string),
    );
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
}
