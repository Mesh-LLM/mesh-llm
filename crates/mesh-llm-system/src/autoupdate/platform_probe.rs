use std::path::Path;
#[cfg(windows)]
use std::path::PathBuf;

use crate::backend;
use crate::release_target::ReleaseTarget;

#[derive(Clone, Copy, Debug, Default)]
struct HostBackendProbe {
    cuda: bool,
    rocm: bool,
    vulkan: bool,
    metal: bool,
}

pub(super) fn installed_bundle_flavor(
    _dir: &Path,
    requested: Option<backend::BinaryFlavor>,
) -> Option<backend::BinaryFlavor> {
    if let Some(flavor) = requested {
        return Some(flavor);
    }

    preferred_bundle_flavor_for_current_host()
}

pub(super) fn preferred_bundle_flavor_for_current_host() -> Option<backend::BinaryFlavor> {
    preferred_bundle_flavor_for_platform(
        std::env::consts::OS,
        std::env::consts::ARCH,
        current_host_backend_probe(),
    )
}

fn preferred_bundle_flavor_for_platform(
    os: &str,
    arch: &str,
    probe: HostBackendProbe,
) -> Option<backend::BinaryFlavor> {
    // Keep this detection order and the probes below in sync with install.sh.
    const UPDATE_FLAVOR_PREFERENCE: [backend::BinaryFlavor; 5] = [
        backend::BinaryFlavor::Cuda,
        backend::BinaryFlavor::Rocm,
        backend::BinaryFlavor::Vulkan,
        backend::BinaryFlavor::Metal,
        backend::BinaryFlavor::Cpu,
    ];

    UPDATE_FLAVOR_PREFERENCE
        .into_iter()
        .find(|flavor| flavor_supported_for_update(*flavor, os, arch, probe))
}

fn flavor_supported_for_update(
    flavor: backend::BinaryFlavor,
    os: &str,
    arch: &str,
    probe: HostBackendProbe,
) -> bool {
    let Ok(target) = ReleaseTarget::from_raw(os, arch, flavor) else {
        return false;
    };
    if !target.support_status().is_supported() {
        return false;
    }

    match flavor {
        backend::BinaryFlavor::Cuda => probe.cuda,
        backend::BinaryFlavor::Rocm => probe.rocm,
        backend::BinaryFlavor::Vulkan => probe.vulkan,
        backend::BinaryFlavor::Metal => probe.metal,
        backend::BinaryFlavor::Cpu => true,
    }
}

fn current_host_backend_probe() -> HostBackendProbe {
    HostBackendProbe {
        cuda: probe_nvidia_backend(),
        rocm: probe_rocm_backend(),
        vulkan: probe_vulkan_backend(),
        metal: cfg!(target_os = "macos"),
    }
}

fn probe_nvidia_backend() -> bool {
    command_exists("nvidia-smi")
        || command_exists("nvcc")
        || Path::new("/dev/nvidiactl").exists()
        || Path::new("/proc/driver/nvidia/gpus").is_dir()
        || Path::new("/dev/nvhost-gpu").exists()
        || Path::new("/dev/nvhost-ctrl-gpu").exists()
        || nvidia_device_tree_models()
            .iter()
            .any(|model| is_tegra_nvidia_model(model))
}

#[cfg(test)]
fn is_blackwell_compute_capability(capability: &str) -> bool {
    let normalized = capability
        .chars()
        .filter(|ch| ch.is_ascii_digit())
        .collect::<String>();
    normalized
        .parse::<u16>()
        .is_ok_and(|sm| (100..200).contains(&sm))
}

fn nvidia_device_tree_models() -> Vec<String> {
    [
        "/proc/device-tree/model",
        "/proc/device-tree/compatible",
        "/sys/firmware/devicetree/base/model",
        "/sys/firmware/devicetree/base/compatible",
    ]
    .iter()
    .filter_map(|path| std::fs::read(path).ok())
    .map(|bytes| {
        String::from_utf8_lossy(&bytes)
            .replace('\0', "\n")
            .trim()
            .to_string()
    })
    .filter(|model| !model.is_empty())
    .collect()
}

fn is_tegra_nvidia_model(model: &str) -> bool {
    const TEGRA_MODEL_MARKERS: [&str; 5] = ["JETSON", "TEGRA", "ORIN", "NVGPU", "THOR"];

    let upper = model.to_ascii_uppercase();
    TEGRA_MODEL_MARKERS
        .iter()
        .any(|marker| upper.contains(marker))
}

#[cfg(test)]
fn is_blackwell_nvidia_model(model: &str) -> bool {
    const BLACKWELL_MODEL_MARKERS: [&str; 14] = [
        "BLACKWELL",
        "GB300",
        "B300",
        "GB200",
        "B200",
        "B100",
        "GB10",
        "RTX 5090",
        "RTX 5080",
        "RTX 5070",
        "RTX 5060",
        "RTX 5050",
        "RTX PRO 6000",
        "THOR",
    ];

    let upper = model.to_ascii_uppercase();
    BLACKWELL_MODEL_MARKERS
        .iter()
        .any(|marker| upper.contains(marker))
}

fn probe_rocm_backend() -> bool {
    command_exists("rocm-smi")
        || command_exists("rocminfo")
        || command_exists("hipcc")
        || env_path_exists("HIP_PATH")
        || env_path_exists("ROCM_PATH")
        || windows_program_files_path_exists(&["AMD", "ROCm"])
        || windows_program_files_path_exists(&["AMD", "HIP"])
        || Path::new("/opt/rocm/bin/hipcc").is_file()
}

fn probe_vulkan_backend() -> bool {
    if command_success("vulkaninfo", &["--summary"]) {
        return true;
    }

    command_exists("glslc")
        && (command_success("pkg-config", &["--exists", "vulkan"])
            || Path::new("/usr/include/vulkan/vulkan.h").is_file()
            || Path::new("/usr/local/include/vulkan/vulkan.h").is_file()
            || std::env::var_os("VULKAN_SDK").is_some_and(|value| !value.is_empty()))
        || env_path_exists("VULKAN_SDK")
        || windows_vulkan_sdk_root_exists()
}

fn env_path_exists(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|value| !value.is_empty() && Path::new(&value).exists())
}

#[cfg(windows)]
fn windows_program_files_path_exists(parts: &[&str]) -> bool {
    std::env::var_os("ProgramFiles").is_some_and(|root| {
        let mut path = PathBuf::from(root);
        for part in parts {
            path.push(part);
        }
        path.exists()
    })
}

#[cfg(not(windows))]
fn windows_program_files_path_exists(_parts: &[&str]) -> bool {
    false
}

#[cfg(windows)]
fn windows_vulkan_sdk_root_exists() -> bool {
    std::env::var_os("ProgramFiles").is_some_and(|root| {
        let sdk_base = PathBuf::from(root).join("VulkanSDK");
        sdk_base.read_dir().is_ok_and(|mut entries| {
            entries.any(|entry| entry.is_ok_and(|entry| entry.path().is_dir()))
        })
    })
}

#[cfg(not(windows))]
fn windows_vulkan_sdk_root_exists() -> bool {
    false
}

fn command_exists(name: &str) -> bool {
    let path = Path::new(name);
    if path.components().count() > 1 {
        return command_file_exists(path);
    }

    std::env::var_os("PATH").is_some_and(|paths| {
        std::env::split_paths(&paths).any(|dir| command_exists_in_dir(&dir, name))
    })
}

#[cfg(windows)]
fn command_exists_in_dir(dir: &Path, name: &str) -> bool {
    let pathext = std::env::var_os("PATHEXT")
        .map(|value| {
            value
                .to_string_lossy()
                .split(';')
                .filter(|ext| !ext.is_empty())
                .map(|ext| ext.trim_start_matches('.').to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| vec!["exe".to_string(), "bat".to_string(), "cmd".to_string()]);
    if dir.join(name).is_file() {
        return true;
    }
    pathext
        .iter()
        .any(|ext| dir.join(format!("{name}.{ext}")).is_file())
}

#[cfg(not(windows))]
fn command_exists_in_dir(dir: &Path, name: &str) -> bool {
    command_file_exists(&dir.join(name))
}

#[cfg(windows)]
fn command_file_exists(path: &Path) -> bool {
    path.is_file()
}

#[cfg(unix)]
fn command_file_exists(path: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;

    path.metadata()
        .map(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(all(not(unix), not(windows)))]
fn command_file_exists(path: &Path) -> bool {
    path.is_file()
}

fn command_success(name: &str, args: &[&str]) -> bool {
    std::process::Command::new(name)
        .args(args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;
    #[cfg(unix)]
    use std::time::{SystemTime, UNIX_EPOCH};

    #[cfg(unix)]
    fn temp_dir(name: &str) -> std::path::PathBuf {
        let unique = format!(
            "mesh-llm-{name}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn test_update_flavor_preference_uses_backend_order() {
        let probe = HostBackendProbe {
            cuda: true,
            rocm: true,
            vulkan: true,
            metal: false,
        };
        assert_eq!(
            preferred_bundle_flavor_for_platform("linux", "x86_64", probe),
            Some(backend::BinaryFlavor::Cuda)
        );

        let probe = HostBackendProbe {
            cuda: false,
            rocm: true,
            vulkan: true,
            metal: false,
        };
        assert_eq!(
            preferred_bundle_flavor_for_platform("linux", "x86_64", probe),
            Some(backend::BinaryFlavor::Rocm)
        );

        let probe = HostBackendProbe {
            cuda: false,
            rocm: false,
            vulkan: true,
            metal: false,
        };
        assert_eq!(
            preferred_bundle_flavor_for_platform("linux", "x86_64", probe),
            Some(backend::BinaryFlavor::Vulkan)
        );
    }

    #[cfg(unix)]
    #[test]
    fn test_unix_command_exists_rejects_non_executable_regular_file() {
        let dir = temp_dir("command-probe-non-executable");
        let command = dir.join("probe-tool");
        std::fs::write(&command, b"#!/bin/sh\n").unwrap();
        let mut permissions = std::fs::metadata(&command).unwrap().permissions();
        permissions.set_mode(0o644);
        std::fs::set_permissions(&command, permissions).unwrap();

        assert!(!command_exists_in_dir(&dir, "probe-tool"));
        assert!(!command_exists(command.to_string_lossy().as_ref()));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[cfg(unix)]
    #[test]
    fn test_unix_command_exists_accepts_executable_regular_file_only() {
        let dir = temp_dir("command-probe-executable");
        let command = dir.join("probe-tool");
        let directory = dir.join("directory-tool");
        std::fs::write(&command, b"#!/bin/sh\n").unwrap();
        std::fs::create_dir(&directory).unwrap();
        let mut permissions = std::fs::metadata(&command).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&command, permissions).unwrap();
        let mut permissions = std::fs::metadata(&directory).unwrap().permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&directory, permissions).unwrap();

        assert!(command_exists_in_dir(&dir, "probe-tool"));
        assert!(command_exists(command.to_string_lossy().as_ref()));
        assert!(!command_exists_in_dir(&dir, "directory-tool"));
        assert!(!command_exists(directory.to_string_lossy().as_ref()));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_update_flavor_preference_filters_by_published_platform() {
        let probe = HostBackendProbe {
            cuda: true,
            rocm: true,
            vulkan: true,
            metal: true,
        };
        assert_eq!(
            preferred_bundle_flavor_for_platform("macos", "aarch64", probe),
            Some(backend::BinaryFlavor::Metal)
        );
        assert_eq!(
            preferred_bundle_flavor_for_platform("linux", "aarch64", probe),
            Some(backend::BinaryFlavor::Cuda)
        );

        let probe = HostBackendProbe {
            cuda: false,
            rocm: true,
            vulkan: true,
            metal: true,
        };
        assert_eq!(
            preferred_bundle_flavor_for_platform("linux", "aarch64", probe),
            Some(backend::BinaryFlavor::Cpu)
        );
        assert_eq!(
            preferred_bundle_flavor_for_platform("linux", "armv7l", probe),
            None
        );
    }

    #[test]
    fn test_blackwell_cuda_detection_from_compute_capability() {
        for capability in ["10.0", "10.3", "12.0", "12.1", "100", "103", "120", "121"] {
            assert!(
                is_blackwell_compute_capability(capability),
                "{capability} requires CUDA 13.x"
            );
        }
        for capability in ["7.5", "8.0", "8.9", "9.0", "75", "80", "89", "90"] {
            assert!(
                !is_blackwell_compute_capability(capability),
                "{capability} should select primary cuda"
            );
        }
    }

    #[test]
    fn test_blackwell_cuda_detection_from_model_name() {
        for model in [
            "NVIDIA B200",
            "NVIDIA GB200",
            "NVIDIA GeForce RTX 5090",
            "NVIDIA RTX PRO 6000 Blackwell",
            "NVIDIA GB10",
        ] {
            assert!(
                is_blackwell_nvidia_model(model),
                "{model} requires CUDA 13.x"
            );
        }
        for model in ["NVIDIA H100", "NVIDIA A100", "NVIDIA RTX 4090"] {
            assert!(
                !is_blackwell_nvidia_model(model),
                "{model} should select primary cuda"
            );
        }
    }

    #[test]
    fn test_tegra_nvidia_detection_from_model_name() {
        for model in [
            "NVIDIA Jetson AGX Orin",
            "Orin (nvgpu)",
            "nvidia,tegra234",
            "Jetson Thor",
        ] {
            assert!(is_tegra_nvidia_model(model), "{model} should select cuda");
        }
        for model in ["Raspberry Pi 5", "Apple M4", "AMD Radeon"] {
            assert!(
                !is_tegra_nvidia_model(model),
                "{model} should not select cuda"
            );
        }
    }

    #[test]
    fn test_update_flavor_preference_falls_back_to_cpu() {
        assert_eq!(
            preferred_bundle_flavor_for_platform("windows", "x86_64", HostBackendProbe::default()),
            Some(backend::BinaryFlavor::Cpu)
        );
    }

    #[test]
    fn test_is_tegra_nvidia_model_positive_orin_agx() {
        assert!(is_tegra_nvidia_model(
            "NVIDIA Jetson AGX Orin Developer Kit"
        ));
    }

    #[test]
    fn test_is_tegra_nvidia_model_positive_orin_nano() {
        assert!(is_tegra_nvidia_model(
            "NVIDIA Jetson Orin Nano Developer Kit"
        ));
    }

    #[test]
    fn test_is_tegra_nvidia_model_positive_xavier() {
        assert!(is_tegra_nvidia_model("NVIDIA Jetson Xavier NX"));
    }

    #[test]
    fn test_is_tegra_nvidia_model_positive_lowercase() {
        assert!(is_tegra_nvidia_model("nvidia tegra234"));
    }

    #[test]
    fn test_is_tegra_nvidia_model_negative_raspberry_pi() {
        assert!(!is_tegra_nvidia_model("Raspberry Pi 4 Model B Rev 1.5"));
    }

    #[test]
    fn test_is_tegra_nvidia_model_negative_amd() {
        assert!(!is_tegra_nvidia_model("AMD EPYC Server"));
    }

    #[test]
    fn test_is_tegra_nvidia_model_empty() {
        assert!(!is_tegra_nvidia_model(""));
    }

    #[test]
    fn test_tegra_selects_cuda_on_linux_aarch64() {
        // Simulate a Tegra/Jetson probe: cuda=true (set by tegra), everything else false.
        let probe = HostBackendProbe {
            cuda: true,
            rocm: false,
            vulkan: false,
            metal: false,
        };
        assert_eq!(
            preferred_bundle_flavor_for_platform("linux", "aarch64", probe),
            Some(backend::BinaryFlavor::Cuda),
            "Tegra on Linux aarch64 must select CUDA bundle"
        );
    }

    #[test]
    fn test_tegra_falls_back_to_cpu_when_cuda_unsupported() {
        // If the release has no CUDA asset for aarch64, CPU is the fallback.
        let probe = HostBackendProbe {
            cuda: true,
            rocm: false,
            vulkan: false,
            metal: false,
        };
        // On armv7l (no published assets), even with cuda=true, there's nothing to match.
        assert_eq!(
            preferred_bundle_flavor_for_platform("linux", "armv7l", probe),
            None,
            "armv7l has no published assets regardless of backend"
        );
    }
}
