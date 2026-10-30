use super::*;
use crate::system::{backend, hardware};

#[test]
fn pinned_gpu_startup_preflight_uses_config_gpu_id() {
    let options = runtime_options_for_test(&["mesh-llm"]);
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            host_ram_offload: None,
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        models: vec![plugin::ModelConfigEntry {
            model: "Qwen3-8B-Q4_K_M".into(),
            mmproj: None,
            ctx_size: Some(8192),
            gpu_id: Some("pci:0000:65:00.0".into()),
            parallel: None,
            cache_type_k: None,
            cache_type_v: None,
            batch: None,
            ubatch: None,
            flash_attention: None,
            ..Default::default()
        }],
        ..plugin::MeshConfig::default()
    };
    let specs = build_startup_model_specs(&options, &config).unwrap();
    let mut plans = vec![StartupModelPlan {
        model_source: String::new(),
        declared_ref: "Qwen3-8B-Q4_K_M".into(),
        resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
        preindexed_split_package: None,
        mmproj_path: None,
        ctx_size: Some(8192),
        gpu_id: specs[0].gpu_id.clone(),
        config_model_id: specs[0].config_model_id.clone(),
        pinned_gpu: None,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let gpus = vec![
        synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0")),
        synthetic_gpu(1, Some("pci:0000:b3:00.0"), Some("CUDA1")),
    ];

    preflight_pinned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None).unwrap();

    assert_eq!(plans[0].gpu_id.as_deref(), Some("pci:0000:65:00.0"));
    assert_eq!(
        plans[0].pinned_gpu,
        Some(StartupPinnedGpuTarget {
            index: 0,
            stable_id: "pci:0000:65:00.0".into(),
            backend_device: "CUDA0".into(),
            vram_bytes: 24_000_000_000,
            reserved_bytes: None,
        })
    );
}

#[test]
fn pinned_gpu_startup_preflight_synthesizes_backend_from_binary_flavor() {
    let mut gpus = vec![
        synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0")),
        synthetic_gpu(1, Some("pci:0000:b3:00.0"), Some("ROCm1")),
    ];

    apply_backend_devices_for_flavor(&mut gpus, Some(backend::BinaryFlavor::Vulkan));

    assert_eq!(gpus[0].backend_device.as_deref(), Some("Vulkan0"));
    assert_eq!(gpus[1].backend_device.as_deref(), Some("Vulkan1"));
}

#[test]
fn pinned_gpu_startup_preflight_rejects_synthesized_backend_missing_from_probe() {
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            host_ram_offload: None,
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        ..plugin::MeshConfig::default()
    };
    let specs = vec![StartupModelSpec {
        model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
        declared_ref: None,
        config_model_id: None,
        mmproj_ref: None,
        ctx_size: Some(4096),
        gpu_id: Some("pci:0000:b3:00.0".into()),
        cli_device_override: false,
        resolve_pinned_gpu: true,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let mut plans = vec![StartupModelPlan {
        model_source: String::new(),
        declared_ref: "Qwen3-8B-Q4_K_M".into(),
        config_model_id: None,
        resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
        preindexed_split_package: None,
        mmproj_path: None,
        ctx_size: Some(4096),
        gpu_id: Some("pci:0000:b3:00.0".into()),
        pinned_gpu: None,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let gpus = vec![synthetic_gpu(1, Some("pci:0000:b3:00.0"), Some("Vulkan1"))];
    let backend_probe = backend::BinaryBackendDeviceProbe {
        path: PathBuf::from("/tmp/backend-vulkan"),
        flavor: Some(backend::BinaryFlavor::Vulkan),
        available_devices: vec!["Vulkan0".into(), "CPU".into()],
    };

    let err = preflight_pinned_startup_models_with_gpus(
        &config,
        &specs,
        &mut plans,
        &gpus,
        Some(&backend_probe),
    )
    .unwrap_err();
    let message = format!("{err:#}");

    assert!(message.contains("failed pinned GPU preflight"));
    assert!(message.contains("requested device Vulkan1 is not supported"));
    assert!(message.contains("Available devices: Vulkan0, CPU"));
}

#[test]
fn pinned_gpu_startup_preflight_canonicalizes_rocm_hip_alias_from_probe() {
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            host_ram_offload: None,
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        ..plugin::MeshConfig::default()
    };
    let specs = vec![StartupModelSpec {
        model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
        declared_ref: None,
        config_model_id: None,
        mmproj_ref: None,
        ctx_size: Some(4096),
        gpu_id: Some("pci:0000:b3:00.0".into()),
        cli_device_override: false,
        resolve_pinned_gpu: true,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let mut plans = vec![StartupModelPlan {
        model_source: String::new(),
        declared_ref: "Qwen3-8B-Q4_K_M".into(),
        config_model_id: None,
        resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
        preindexed_split_package: None,
        mmproj_path: None,
        ctx_size: Some(4096),
        gpu_id: Some("pci:0000:b3:00.0".into()),
        pinned_gpu: None,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let gpus = vec![synthetic_gpu(1, Some("pci:0000:b3:00.0"), Some("ROCm1"))];
    let backend_probe = backend::BinaryBackendDeviceProbe {
        path: PathBuf::from("/tmp/backend-rocm"),
        flavor: Some(backend::BinaryFlavor::Rocm),
        available_devices: vec!["HIP1".into(), "CPU".into()],
    };

    preflight_pinned_startup_models_with_gpus(
        &config,
        &specs,
        &mut plans,
        &gpus,
        Some(&backend_probe),
    )
    .unwrap();

    assert_eq!(plans[0].pinned_gpu.as_ref().unwrap().backend_device, "HIP1");
}

#[test]
fn pinned_gpu_startup_preflight_keeps_detected_backend_without_resolved_flavor() {
    let mut gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

    apply_backend_devices_for_flavor(&mut gpus, None);

    assert_eq!(gpus[0].backend_device.as_deref(), Some("CUDA0"));
}

#[test]
fn pinned_gpu_startup_preflight_requests_per_gpu_vram_metrics() {
    let metrics = pinned_startup_preflight_metrics();

    assert_eq!(metrics.len(), 4);
    assert!(metrics.contains(&hardware::Metric::GpuName));
    assert!(metrics.contains(&hardware::Metric::GpuFacts));
    assert!(metrics.contains(&hardware::Metric::VramBytes));
    assert!(metrics.contains(&hardware::Metric::IsSoc));
}
#[test]
fn pinned_gpu_startup_preflight_unmatched_cli_models_bypass_config_gpu_id() {
    let options = runtime_options_for_test(&["mesh-llm", "--model", "Qwen3-8B-Q4_K_M"]);
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            host_ram_offload: None,
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        models: vec![plugin::ModelConfigEntry {
            model: "Ignored-Model".into(),
            mmproj: None,
            ctx_size: Some(8192),
            gpu_id: Some("pci:0000:65:00.0".into()),
            parallel: None,
            cache_type_k: None,
            cache_type_v: None,
            batch: None,
            ubatch: None,
            flash_attention: None,
            ..Default::default()
        }],
        ..plugin::MeshConfig::default()
    };
    let specs = build_startup_model_specs(&options, &config).unwrap();
    let mut plans = vec![StartupModelPlan {
        model_source: String::new(),
        declared_ref: "Qwen3-8B-Q4_K_M".into(),
        resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
        preindexed_split_package: None,
        mmproj_path: None,
        ctx_size: None,
        gpu_id: specs[0].gpu_id.clone(),
        config_model_id: specs[0].config_model_id.clone(),
        pinned_gpu: None,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

    preflight_pinned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None).unwrap();

    assert_eq!(specs[0].gpu_id, None);
    assert!(!specs[0].resolve_pinned_gpu);
    assert_eq!(plans[0].gpu_id, None);
    assert_eq!(plans[0].pinned_gpu, None);
}

#[test]
fn pinned_gpu_startup_preflight_missing_gpu_id_fails_closed() {
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            host_ram_offload: None,
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        ..plugin::MeshConfig::default()
    };
    let specs = vec![StartupModelSpec {
        model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
        declared_ref: None,
        config_model_id: None,
        mmproj_ref: None,
        ctx_size: None,
        gpu_id: None,
        cli_device_override: false,
        resolve_pinned_gpu: true,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let mut plans = vec![StartupModelPlan {
        model_source: String::new(),
        declared_ref: "Qwen3-8B-Q4_K_M".into(),
        config_model_id: None,
        resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
        preindexed_split_package: None,
        mmproj_path: None,
        ctx_size: None,
        gpu_id: None,
        pinned_gpu: None,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

    let err = preflight_pinned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None)
        .unwrap_err();
    let message = format!("{err:#}");

    assert!(message.contains("failed pinned GPU preflight"));
    assert!(message.contains("missing configured gpu_id"));
}

#[test]
fn pinned_gpu_startup_preflight_stores_resolved_pinned_target_in_plan() {
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            host_ram_offload: None,
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        ..plugin::MeshConfig::default()
    };
    let specs = vec![StartupModelSpec {
        model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
        declared_ref: None,
        config_model_id: None,
        mmproj_ref: None,
        ctx_size: Some(4096),
        gpu_id: Some("uuid:GPU-123".into()),
        cli_device_override: false,
        resolve_pinned_gpu: true,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let mut plans = vec![StartupModelPlan {
        model_source: String::new(),
        declared_ref: "Qwen3-8B-Q4_K_M".into(),
        config_model_id: None,
        resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
        preindexed_split_package: None,
        mmproj_path: None,
        ctx_size: Some(4096),
        gpu_id: Some("uuid:GPU-123".into()),
        pinned_gpu: None,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let mut gpus = vec![synthetic_gpu(3, Some("uuid:GPU-123"), Some("CUDA3"))];
    gpus[0].reserved_bytes = Some(500_000_000);

    preflight_pinned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None).unwrap();

    let pinned_gpu = plans[0].pinned_gpu.as_ref().unwrap();
    assert_eq!(pinned_gpu.index, 3);
    assert_eq!(pinned_gpu.stable_id, "uuid:GPU-123");
    assert_eq!(pinned_gpu.backend_device, "CUDA3");
    assert_eq!(pinned_gpu.vram_bytes, 24_000_000_000);
    assert_eq!(pinned_gpu.reserved_bytes, Some(500_000_000));
}

#[test]
fn pinned_gpu_startup_preflight_rejects_resolved_gpu_without_backend_device() {
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            host_ram_offload: None,
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        ..plugin::MeshConfig::default()
    };
    let specs = vec![StartupModelSpec {
        model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
        declared_ref: None,
        config_model_id: None,
        mmproj_ref: None,
        ctx_size: Some(4096),
        gpu_id: Some("uuid:GPU-123".into()),
        cli_device_override: false,
        resolve_pinned_gpu: true,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let mut plans = vec![StartupModelPlan {
        model_source: String::new(),
        declared_ref: "Qwen3-8B-Q4_K_M".into(),
        config_model_id: None,
        resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
        preindexed_split_package: None,
        mmproj_path: None,
        ctx_size: Some(4096),
        gpu_id: Some("uuid:GPU-123".into()),
        pinned_gpu: None,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let gpus = vec![synthetic_gpu(3, Some("uuid:GPU-123"), None)];

    let err = preflight_pinned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None)
        .unwrap_err();
    let message = format!("{err:#}");

    assert!(message.contains("failed pinned GPU preflight"));
    assert!(message.contains("without a backend_device"));
}

#[test]
fn pinned_gpu_startup_preflight_unresolvable_gpu_id_fails_closed() {
    let config = plugin::MeshConfig {
        gpu: plugin::GpuConfig {
            host_ram_offload: None,
            assignment: plugin::GpuAssignment::Pinned,
            parallel: None,
        },
        ..plugin::MeshConfig::default()
    };
    let specs = vec![StartupModelSpec {
        model_ref: PathBuf::from("Qwen3-8B-Q4_K_M"),
        declared_ref: None,
        config_model_id: None,
        mmproj_ref: None,
        ctx_size: None,
        gpu_id: Some("pci:0000:b3:00.0".into()),
        cli_device_override: false,
        resolve_pinned_gpu: true,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let mut plans = vec![StartupModelPlan {
        model_source: String::new(),
        declared_ref: "Qwen3-8B-Q4_K_M".into(),
        config_model_id: None,
        resolved_path: PathBuf::from("/tmp/Qwen3-8B-Q4_K_M.gguf"),
        preindexed_split_package: None,
        mmproj_path: None,
        ctx_size: None,
        gpu_id: Some("pci:0000:b3:00.0".into()),
        pinned_gpu: None,
        parallel: None,
        cache_type_k: None,
        cache_type_v: None,
        n_batch: None,
        n_ubatch: None,
        flash_attention: FlashAttentionType::Auto,
        local_source_required: false,
        profile: String::new(),
    }];
    let gpus = vec![synthetic_gpu(0, Some("pci:0000:65:00.0"), Some("CUDA0"))];

    let err = preflight_pinned_startup_models_with_gpus(&config, &specs, &mut plans, &gpus, None)
        .unwrap_err();
    let message = format!("{err:#}");

    assert!(message.contains("failed pinned GPU preflight"));
    assert!(message.contains("did not match any available pinnable GPU"));
}
