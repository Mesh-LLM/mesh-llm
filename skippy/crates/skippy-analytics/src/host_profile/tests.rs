use super::*;
use std::collections::BTreeSet;

fn gpu(name: &str, vram_gb: u64, unified: bool) -> HostGpuProfile {
    HostGpuProfile {
        display_name: name.to_owned(),
        backend_device: None,
        stable_id: Some("GPU-1234-secret-uuid".to_owned()),
        vram_bytes: Some(vram_gb << 30),
        unified_memory: unified,
        probe: None,
        cuda_sm: None,
        rocm_gfx: None,
    }
}

fn profile(gpus: Vec<HostGpuProfile>, flavors: &[NativeRuntimeBackendKind]) -> HostRuntimeProfile {
    HostRuntimeProfile {
        os: "macos".to_owned(),
        arch: "aarch64".to_owned(),
        target_triple: None,
        glibc_version: None,
        available_flavors: flavors.iter().cloned().collect::<BTreeSet<_>>(),
        gpus,
        cuda: None,
        rocm: None,
        vulkan: None,
    }
}

#[test]
fn a_gpu_is_described_by_family_and_whole_gib_never_by_its_id() {
    let profile = profile(
        vec![gpu("Apple M4 Max", 36, true)],
        &[
            NativeRuntimeBackendKind::Cpu,
            NativeRuntimeBackendKind::Metal,
        ],
    );
    let properties = properties(&profile, Some(64 << 30));
    let rendered = format!("{properties:?}");

    assert!(rendered.contains("apple-m4-max"), "{rendered}");
    assert_eq!(
        properties.get("gpu_family"),
        Some(&Value::Text(Label::sanitize("apple-m4").unwrap()))
    );
    assert_eq!(properties.get("vram_gb"), Some(&Value::Int(36)));
    assert_eq!(properties.get("system_ram_gb"), Some(&Value::Int(64)));
    assert_eq!(properties.get("backend"), Some(&Value::Static("metal")));
    assert!(!rendered.contains("secret"), "stable id leaked: {rendered}");
}

#[test]
fn a_machine_without_a_gpu_says_none_and_reports_no_vram() {
    let properties = properties(&profile(Vec::new(), &[NativeRuntimeBackendKind::Cpu]), None);
    assert_eq!(properties.get("gpu_model"), Some(&Value::Static("none")));
    assert_eq!(properties.get("backend"), Some(&Value::Static("cpu")));
    assert!(properties.get("vram_gb").is_none());
}

#[test]
fn device_names_reduce_to_families() {
    for (slug, family) in [
        ("apple-m4-pro", "apple-m4"),
        ("nvidia-geforce-rtx-4090-laptop-gpu", "rtx-4090"),
        ("nvidia-gb10", "gb10"),
        ("2-nvidia-geforce-rtx-3090", "rtx-3090"),
    ] {
        assert_eq!(gpu_family(slug).unwrap().as_str(), family, "{slug}");
    }
}
