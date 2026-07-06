use super::super::{
    mlock::{TuneMlockLimit, TuneMlockProbe},
    types::{TuneDeviceSelectionSource, TuneDeviceTarget, TuneGpuTarget, TuneMemorySource},
};
use super::helpers::{
    config_with_defaults_and_model, config_with_model, evaluate_with_probe, sample_gpu,
    sample_target, survey_with_gpus,
};
use crate::gpus::tune::{TuneDiagnosticCode, TuneField, TuneFieldStatus};
use mesh_llm_config::{HardwareConfig, MeshConfig, ModelConfigEntry};

#[test]
fn gpu_tune_prefers_model_hardware_device_over_defaults_and_legacy_gpu_id() {
    let config = config_with_defaults_and_model(
        HardwareConfig {
            device: Some("CUDA0".to_string()),
            ..HardwareConfig::default()
        },
        ModelConfigEntry {
            gpu_id: Some("pci:0000:00:00.0".to_string()),
            hardware: Some(HardwareConfig {
                device: Some("CUDA1".to_string()),
                ..HardwareConfig::default()
            }),
            ..ModelConfigEntry::default()
        },
    );
    let target = sample_target(true);
    let survey = survey_with_gpus(vec![
        sample_gpu(0, "pci:0000:00:00.0", Some("CUDA0")),
        sample_gpu(1, "pci:0000:01:00.0", Some("CUDA1")),
    ]);

    let evaluation = evaluate_with_probe(
        &config,
        &target,
        &survey,
        TuneMlockProbe::Supported {
            limit: TuneMlockLimit::Unlimited,
        },
    )
    .unwrap();

    assert_eq!(
        evaluation.evaluated_device.source,
        TuneDeviceSelectionSource::ModelHardwareDevice
    );
    assert_eq!(
        evaluation.evaluated_device.target,
        TuneDeviceTarget::Gpu(TuneGpuTarget {
            index: 1,
            display_name: "GPU 1".to_string(),
            stable_id: Some("pci:0000:01:00.0".to_string()),
            backend_device: Some("CUDA1".to_string()),
        })
    );
}

#[test]
fn gpu_tune_uses_defaults_hardware_device_before_legacy_gpu_id() {
    let config = config_with_defaults_and_model(
        HardwareConfig {
            device: Some("CUDA1".to_string()),
            ..HardwareConfig::default()
        },
        ModelConfigEntry {
            gpu_id: Some("pci:0000:00:00.0".to_string()),
            ..ModelConfigEntry::default()
        },
    );
    let target = sample_target(true);
    let survey = survey_with_gpus(vec![
        sample_gpu(0, "pci:0000:00:00.0", Some("CUDA0")),
        sample_gpu(1, "pci:0000:01:00.0", Some("CUDA1")),
    ]);

    let evaluation = evaluate_with_probe(
        &config,
        &target,
        &survey,
        TuneMlockProbe::Supported {
            limit: TuneMlockLimit::Unlimited,
        },
    )
    .unwrap();

    assert_eq!(
        evaluation.evaluated_device.source,
        TuneDeviceSelectionSource::DefaultsHardwareDevice
    );
    assert_eq!(
        evaluation.evaluated_device.target,
        TuneDeviceTarget::Gpu(TuneGpuTarget {
            index: 1,
            display_name: "GPU 1".to_string(),
            stable_id: Some("pci:0000:01:00.0".to_string()),
            backend_device: Some("CUDA1".to_string()),
        })
    );
}

#[test]
fn gpu_tune_uses_legacy_gpu_id_when_no_effective_device() {
    let config = config_with_model(ModelConfigEntry {
        gpu_id: Some("pci:0000:01:00.0".to_string()),
        ..ModelConfigEntry::default()
    });
    let target = sample_target(true);
    let survey = survey_with_gpus(vec![
        sample_gpu(0, "pci:0000:00:00.0", Some("CUDA0")),
        sample_gpu(1, "pci:0000:01:00.0", Some("CUDA1")),
    ]);

    let evaluation = evaluate_with_probe(
        &config,
        &target,
        &survey,
        TuneMlockProbe::Supported {
            limit: TuneMlockLimit::Unlimited,
        },
    )
    .unwrap();

    assert_eq!(
        evaluation.evaluated_device.source,
        TuneDeviceSelectionSource::LegacyGpuId
    );
    assert_eq!(
        evaluation.evaluated_device.target,
        TuneDeviceTarget::Gpu(TuneGpuTarget {
            index: 1,
            display_name: "GPU 1".to_string(),
            stable_id: Some("pci:0000:01:00.0".to_string()),
            backend_device: Some("CUDA1".to_string()),
        })
    );
}

#[test]
fn gpu_tune_ignores_main_gpu_for_selection_and_records_it() {
    let config = config_with_model(ModelConfigEntry {
        hardware: Some(HardwareConfig {
            main_gpu: Some(1),
            ..HardwareConfig::default()
        }),
        ..ModelConfigEntry::default()
    });
    let target = sample_target(true);
    let survey = survey_with_gpus(vec![
        sample_gpu(0, "pci:0000:00:00.0", Some("CUDA0")),
        sample_gpu(1, "pci:0000:01:00.0", Some("CUDA1")),
    ]);

    let evaluation = evaluate_with_probe(
        &config,
        &target,
        &survey,
        TuneMlockProbe::Supported {
            limit: TuneMlockLimit::Unlimited,
        },
    )
    .unwrap();

    assert_eq!(
        evaluation.evaluated_device.source,
        TuneDeviceSelectionSource::SurveyDefault
    );
    assert_eq!(evaluation.evaluated_device.report_only_main_gpu, Some(1));
    match evaluation.device_field_status() {
        TuneFieldStatus::ReportOnly { reason, .. } => assert!(reason.contains("main_gpu=1")),
        other => panic!("expected report-only device status, got {other:?}"),
    }
}

#[test]
fn gpu_tune_reports_missing_configured_gpu() {
    let config = config_with_model(ModelConfigEntry {
        hardware: Some(HardwareConfig {
            device: Some("CUDA9".to_string()),
            ..HardwareConfig::default()
        }),
        ..ModelConfigEntry::default()
    });
    let target = sample_target(true);
    let survey = survey_with_gpus(vec![sample_gpu(0, "pci:0000:00:00.0", Some("CUDA0"))]);

    let error = evaluate_with_probe(
        &config,
        &target,
        &survey,
        TuneMlockProbe::Supported {
            limit: TuneMlockLimit::Unlimited,
        },
    )
    .unwrap_err();

    assert_eq!(error.code, TuneDiagnosticCode::MissingConfiguredDevice);
    assert_eq!(error.field, Some(TuneField::Device));
    assert!(error.message.contains("CUDA9"));
    assert!(error.message.contains("Available backend devices: CUDA0"));
}

#[test]
fn gpu_tune_falls_back_to_backend_device_for_non_pinnable_hardware_default() {
    let config = config_with_defaults_and_model(
        HardwareConfig {
            device: Some("nvidia-cuda-0".to_string()),
            ..HardwareConfig::default()
        },
        ModelConfigEntry::default(),
    );
    let target = sample_target(true);
    let survey = survey_with_gpus(vec![sample_gpu(0, "uuid:GPU-abc123-def456", Some("nvidia-cuda-0"))]);

    let evaluation = evaluate_with_probe(
        &config,
        &target,
        &survey,
        TuneMlockProbe::Supported {
            limit: TuneMlockLimit::Unlimited,
        },
    )
    .unwrap();

    assert_eq!(
        evaluation.evaluated_device.source,
        TuneDeviceSelectionSource::DefaultsHardwareDevice
    );
    assert_eq!(
        evaluation.evaluated_device.target,
        TuneDeviceTarget::Gpu(TuneGpuTarget {
            index: 0,
            display_name: "GPU 0".to_string(),
            stable_id: Some("uuid:GPU-abc123-def456".to_string()),
            backend_device: Some("nvidia-cuda-0".to_string()),
        })
    );
}

#[test]
fn gpu_tune_backend_fallback_failure_merges_diagnostics_for_non_pinnable_hardware_default() {
    let config = config_with_defaults_and_model(
        HardwareConfig {
            device: Some("nvidia-cuda-0".to_string()),
            ..HardwareConfig::default()
        },
        ModelConfigEntry::default(),
    );
    let target = sample_target(true);
    let survey = survey_with_gpus(vec![sample_gpu(
        0,
        "uuid:GPU-abc123-def456",
        Some("CUDA0"),
    )]);

    let error = evaluate_with_probe(
        &config,
        &target,
        &survey,
        TuneMlockProbe::Supported {
            limit: TuneMlockLimit::Unlimited,
        },
    )
    .unwrap_err();

    assert_eq!(error.code, TuneDiagnosticCode::MissingConfiguredDevice);
    assert_eq!(error.field, Some(TuneField::Device));
    assert!(
        error.message.contains("nvidia-cuda-0"),
        "expected the requested device to appear in the merged diagnostic, got: {}",
        error.message
    );
    assert!(
        error.message.contains("backend fallback also failed:"),
        "expected the backend-fallback suffix in the merged diagnostic, got: {}",
        error.message
    );
    assert!(
        error.message.contains("requested backend device was not present in the survey"),
        "expected the backend resolver's detail string in the merged diagnostic, got: {}",
        error.message
    );
}

#[test]
fn gpu_tune_falls_back_to_cpu_system_ram_when_no_selectable_gpu() {
    let config = MeshConfig::default();
    let target = sample_target(false);
    let survey = mesh_llm_system::hardware::HardwareSurvey {
        vram_bytes: 14 * 1024 * 1024 * 1024,
        gpus: vec![sample_gpu(0, "pci:0000:00:00.0", None)],
        ..mesh_llm_system::hardware::HardwareSurvey::default()
    };

    let evaluation = evaluate_with_probe(
        &config,
        &target,
        &survey,
        TuneMlockProbe::Supported {
            limit: TuneMlockLimit::Unlimited,
        },
    )
    .unwrap();

    assert_eq!(
        evaluation.evaluated_device.source,
        TuneDeviceSelectionSource::CpuSystemRamFallback
    );
    assert_eq!(evaluation.evaluated_device.target, TuneDeviceTarget::Cpu);
    assert_eq!(
        evaluation.memory.source,
        TuneMemorySource::SystemRamFallback
    );
    assert_eq!(evaluation.memory.allocatable_bytes, 14 * 1024 * 1024 * 1024);
}
