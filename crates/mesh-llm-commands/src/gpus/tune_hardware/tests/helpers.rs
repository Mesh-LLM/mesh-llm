use super::super::{
    evaluate::evaluate_tune_hardware_with_mlock_probe_for_tests,
    mlock::TuneMlockProbe,
    types::{TuneHardwareEvaluation, TuneHardwareEvaluationInput},
};
use crate::gpus::tune_resolver::{ConfigModelMatch, LocalTargetSource, TuneTargetSelection};
use mesh_llm_config::{HardwareConfig, MeshConfig, ModelConfigDefaults, ModelConfigEntry};
use mesh_llm_system::hardware::{GpuFacts, HardwareSurvey};
use std::path::PathBuf;

pub(super) fn sample_gpu(index: usize, stable_id: &str, backend_device: Option<&str>) -> GpuFacts {
    GpuFacts {
        index,
        display_name: format!("GPU {index}"),
        backend_device: backend_device.map(str::to_string),
        vram_bytes: 24 * 1024 * 1024 * 1024,
        reserved_bytes: Some(1024 * 1024 * 1024),
        mem_bandwidth_gbps: None,
        compute_tflops_fp32: None,
        compute_tflops_fp16: None,
        unified_memory: false,
        stable_id: Some(stable_id.to_string()),
        pci_bdf: None,
        vendor_uuid: None,
        metal_registry_id: None,
        dxgi_luid: None,
        pnp_instance_id: None,
    }
}

pub(super) fn sample_target(configured: bool) -> crate::gpus::tune_resolver::ResolvedTuneTarget {
    crate::gpus::tune_resolver::ResolvedTuneTarget {
        requested_input: "hf://mesh/example.gguf".to_string(),
        canonical_model_ref: "hf://mesh/example.gguf".to_string(),
        resolved_path: PathBuf::from("/tmp/example.gguf"),
        local_source: LocalTargetSource::FilesystemPath {
            synthetic_model_ref: "local-gguf/example".to_string(),
        },
        config_matches: if configured {
            vec![ConfigModelMatch {
                row_index: 0,
                configured_model: "hf://mesh/example.gguf".to_string(),
            }]
        } else {
            Vec::new()
        },
        selection: if configured {
            TuneTargetSelection::Configured
        } else {
            TuneTargetSelection::Explicit { configured: false }
        },
    }
}

pub(super) fn config_with_model(model: ModelConfigEntry) -> MeshConfig {
    MeshConfig {
        models: vec![model],
        ..MeshConfig::default()
    }
}

pub(super) fn config_with_defaults_and_model(
    defaults_hardware: HardwareConfig,
    model: ModelConfigEntry,
) -> MeshConfig {
    MeshConfig {
        defaults: Some(ModelConfigDefaults {
            hardware: Some(defaults_hardware),
            ..ModelConfigDefaults::default()
        }),
        models: vec![model],
        ..MeshConfig::default()
    }
}

pub(super) fn survey_with_gpus(gpus: Vec<GpuFacts>) -> HardwareSurvey {
    HardwareSurvey {
        vram_bytes: 12 * 1024 * 1024 * 1024,
        gpus,
        ..HardwareSurvey::default()
    }
}

pub(super) fn evaluate_with_probe(
    config: &MeshConfig,
    target: &crate::gpus::tune_resolver::ResolvedTuneTarget,
    survey: &HardwareSurvey,
    probe: TuneMlockProbe,
) -> Result<TuneHardwareEvaluation, crate::gpus::tune::TuneDiagnostic> {
    evaluate_tune_hardware_with_mlock_probe_for_tests(
        TuneHardwareEvaluationInput {
            config,
            target,
            survey,
        },
        probe,
    )
}
