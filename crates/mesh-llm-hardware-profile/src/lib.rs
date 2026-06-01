use mesh_llm_native_runtime::{HostGpuProfile, HostRuntimeProfile, NativeRuntimeFlavor};
use std::collections::BTreeSet;
use std::process::Command;

pub fn host_runtime_profile() -> HostRuntimeProfile {
    let gpus = detect_gpus();
    HostRuntimeProfile {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        target_triple: option_env!("TARGET").map(str::to_string),
        available_flavors: detected_native_runtime_flavors(&gpus),
        gpus,
    }
}

pub fn detected_native_runtime_flavors(gpus: &[HostGpuProfile]) -> BTreeSet<NativeRuntimeFlavor> {
    let mut flavors = BTreeSet::from([NativeRuntimeFlavor::Cpu]);
    if cfg!(target_os = "macos") {
        flavors.insert(NativeRuntimeFlavor::Metal);
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
        })
        .collect()
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

fn insert_label_flavors(flavors: &mut BTreeSet<NativeRuntimeFlavor>, label: &str) {
    let label = label.to_ascii_lowercase();
    if label.contains("cuda") || label.contains("nvidia") {
        flavors.insert(NativeRuntimeFlavor::Cuda);
    }
    if label.contains("blackwell")
        || label.contains("gb200")
        || label.contains("b200")
        || label.contains("rtx 50")
    {
        flavors.insert(NativeRuntimeFlavor::CudaBlackwell);
    }
    if label.contains("rocm")
        || label.contains("hip")
        || label.contains("amd")
        || label.contains("radeon")
    {
        flavors.insert(NativeRuntimeFlavor::Rocm);
    }
    if label.contains("vulkan") {
        flavors.insert(NativeRuntimeFlavor::Vulkan);
    }
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
        }
    }

    #[test]
    fn nvidia_labels_enable_cuda() {
        let flavors = detected_native_runtime_flavors(&[profile("NVIDIA GeForce RTX 4090")]);

        assert!(flavors.contains(&NativeRuntimeFlavor::Cpu));
        assert!(flavors.contains(&NativeRuntimeFlavor::Cuda));
    }

    #[test]
    fn blackwell_labels_enable_specific_flavor() {
        let flavors = detected_native_runtime_flavors(&[profile("NVIDIA B200")]);

        assert!(flavors.contains(&NativeRuntimeFlavor::Cuda));
        assert!(flavors.contains(&NativeRuntimeFlavor::CudaBlackwell));
    }

    #[test]
    fn amd_labels_enable_rocm() {
        let flavors = detected_native_runtime_flavors(&[profile("AMD Radeon PRO W7900")]);

        assert!(flavors.contains(&NativeRuntimeFlavor::Rocm));
    }
}
