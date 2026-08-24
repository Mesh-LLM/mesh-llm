//! Platform GPU label discovery and backend flavor hints.

use mesh_llm_native_runtime::NativeRuntimeBackendKind;
use std::collections::BTreeSet;
use std::process::Command;
pub(crate) fn gpu_labels() -> Vec<String> {
    let mut labels = Vec::new();
    append_command_lines(&mut labels, "vulkaninfo", &["--summary"]);
    append_platform_gpu_labels(&mut labels);
    labels.sort();
    labels.dedup();
    labels
}

#[cfg(target_os = "linux")]
pub(crate) fn append_platform_gpu_labels(labels: &mut Vec<String>) {
    append_command_lines(labels, "lspci", &[]);
}

#[cfg(target_os = "windows")]
pub(crate) fn append_platform_gpu_labels(labels: &mut Vec<String>) {
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
pub(crate) fn append_platform_gpu_labels(labels: &mut Vec<String>) {
    append_command_lines(labels, "system_profiler", &["SPDisplaysDataType"]);
}

#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
pub(crate) fn append_platform_gpu_labels(_labels: &mut Vec<String>) {}

pub(crate) fn append_command_lines(labels: &mut Vec<String>, program: &str, args: &[&str]) {
    let Some(output) = command_output(program, args) else {
        return;
    };
    labels.extend(gpu_labels_from_command_output(program, args, &output));
}

pub(crate) fn gpu_labels_from_command_output(
    program: &str,
    args: &[&str],
    output: &str,
) -> Vec<String> {
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

pub(crate) fn vulkaninfo_device_names(output: &str) -> Vec<String> {
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

pub(crate) fn looks_like_software_vulkan_adapter(value: &str) -> bool {
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

pub(crate) fn looks_like_display_controller(line: &str) -> bool {
    let label = line.to_ascii_lowercase();
    (label.contains("vga compatible controller")
        || label.contains("3d controller")
        || label.contains("display controller"))
        && looks_like_gpu_label(line)
}

pub(crate) fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8(output.stdout).ok())
        .flatten()
}

pub(crate) fn looks_like_gpu_label(line: &str) -> bool {
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

pub(crate) fn insert_label_flavors(flavors: &mut BTreeSet<NativeRuntimeBackendKind>, label: &str) {
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
