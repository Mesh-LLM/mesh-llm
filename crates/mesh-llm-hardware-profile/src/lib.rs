use mesh_llm_native_runtime::{
    HostCudaProfile, HostGpuProfile, HostRocmProfile, HostRuntimeProfile, HostVulkanProfile,
    NativeRuntimeBackendKind,
};
use std::collections::BTreeSet;

mod cuda;
mod environment;
mod gpu_discovery;
mod platform;
mod rocm;

use cuda::*;
use environment::*;
use gpu_discovery::*;
use platform::*;
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
    apply_gpu_arch_overrides_from(gpus, &cuda_arches, &rocm_arches);
}

fn apply_gpu_arch_overrides_from(
    gpus: &mut [HostGpuProfile],
    cuda_arches: &[String],
    rocm_arches: &[String],
) {
    for (index, gpu) in gpus.iter_mut().enumerate() {
        if let Some(cuda_sm) = cuda_arches.get(index) {
            gpu.cuda_sm = Some(cuda_sm.clone());
        }
        if let Some(rocm_gfx) = rocm_arches.get(index) {
            gpu.rocm_gfx = Some(rocm_gfx.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::{Path, PathBuf};

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
        let mut gpus = vec![HostGpuProfile {
            cuda_sm: Some("120".to_string()),
            rocm_gfx: Some("gfx1200".to_string()),
            ..profile("NVIDIA GeForce RTX 5090")
        }];

        apply_gpu_arch_overrides_from(&mut gpus, &[], &[]);

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
