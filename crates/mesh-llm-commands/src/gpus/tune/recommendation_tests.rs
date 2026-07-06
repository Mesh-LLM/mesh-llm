use crate::gpus::tune_hardware::types::{
    EvaluatedTuneDevice, TuneDeviceSelectionSource, TuneDeviceTarget, TuneGpuTarget,
    TuneHardwareEvaluation, TuneMemoryBudget, TuneMemorySource, TuneMlockEvaluation,
};
use crate::gpus::tune_resolver::{
    ConfigModelMatch, LocalTargetSource, ResolvedTuneTarget, TuneTargetSelection,
};
use mesh_llm_system::hardware::{GpuFacts, HardwareSurvey};

use super::*;

pub(crate) fn sample_metadata(
    model_bytes: u64,
    layer_count: u32,
    context_length: u32,
    expert_count: u32,
) -> TuneGgufMetadata {
    TuneGgufMetadata {
        compact_meta: model_artifact::gguf::GgufCompactMeta {
            architecture: "llama".to_string(),
            context_length,
            head_count: 32,
            kv_head_count: 8,
            layer_count,
            key_length: 128,
            value_length: 128,
            ..Default::default()
        },
        tensor_profile: TuneTensorProfile::Exact(model_artifact::gguf::GgufTensorByteProfile {
            expert_count,
            expert_used_count: expert_count.min(2),
            full_model_bytes: model_bytes,
            base_resident_bytes: model_bytes,
            expert_tensor_bytes: 0,
            file_overhead_bytes: 0,
        }),
        model_bytes,
    }
}

pub(crate) fn recommendation_target(configured: bool) -> ResolvedTuneTarget {
    ResolvedTuneTarget {
        requested_input: "hf://mesh/example.gguf".to_string(),
        canonical_model_ref: "hf://mesh/example.gguf".to_string(),
        resolved_path: std::path::PathBuf::from("/tmp/example.gguf"),
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

pub(crate) fn survey_with_gpu(gpu_allocatable_bytes: u64, system_ram_bytes: u64) -> HardwareSurvey {
    let total_bytes = gpu_allocatable_bytes.saturating_add(1024 * 1024 * 1024);
    HardwareSurvey {
        vram_bytes: system_ram_bytes,
        gpus: vec![GpuFacts {
            index: 0,
            display_name: "GPU 0".to_string(),
            backend_device: Some("CUDA0".to_string()),
            vram_bytes: total_bytes,
            reserved_bytes: Some(total_bytes.saturating_sub(gpu_allocatable_bytes)),
            mem_bandwidth_gbps: None,
            compute_tflops_fp32: None,
            compute_tflops_fp16: None,
            unified_memory: false,
            stable_id: Some("pci:0000:00:00.0".to_string()),
            pci_bdf: None,
            vendor_uuid: None,
            metal_registry_id: None,
            dxgi_luid: None,
            pnp_instance_id: None,
        }],
        ..HardwareSurvey::default()
    }
}

pub(crate) fn gpu_hardware(allocatable_bytes: u64) -> TuneHardwareEvaluation {
    TuneHardwareEvaluation {
        evaluated_device: EvaluatedTuneDevice {
            target: TuneDeviceTarget::Gpu(TuneGpuTarget {
                index: 0,
                display_name: "GPU 0".to_string(),
                stable_id: Some("pci:0000:00:00.0".to_string()),
                backend_device: Some("CUDA0".to_string()),
            }),
            source: TuneDeviceSelectionSource::SurveyDefault,
            report_only_main_gpu: None,
        },
        memory: TuneMemoryBudget {
            source: TuneMemorySource::EvaluatedGpuVram,
            total_bytes: allocatable_bytes,
            reserved_bytes: Some(0),
            allocatable_bytes,
        },
        mlock: TuneMlockEvaluation {
            available: false,
            reason: "mlock unavailable in test".to_string(),
        },
    }
}

pub(crate) fn status_for(plan: &TunePlan, field: TuneField) -> &TuneFieldStatus {
    plan.field_statuses
        .iter()
        .find(|status| match status {
            TuneFieldStatus::Applied { recommendation, .. }
            | TuneFieldStatus::ReportOnly { recommendation, .. } => recommendation.field == field,
            TuneFieldStatus::Preserved {
                field: candidate, ..
            }
            | TuneFieldStatus::Unsupported {
                field: candidate, ..
            }
            | TuneFieldStatus::Error {
                field: candidate, ..
            } => *candidate == field,
        })
        .unwrap_or_else(|| panic!("missing field status for {field:?}"))
}

pub(crate) fn assert_applied_kv(plan: &TunePlan, field: TuneField, value: TuneKvCacheType) {
    match status_for(plan, field) {
        TuneFieldStatus::Applied { recommendation, .. } => {
            assert_eq!(
                recommendation.value,
                TuneRecommendedValue::KvCacheType(value)
            );
        }
        other => panic!("expected applied kv status, got {other:?}"),
    }
}

pub(crate) fn assert_applied_flash_attention(plan: &TunePlan, value: TuneFlashAttentionValue) {
    match status_for(plan, TuneField::FlashAttention) {
        TuneFieldStatus::Applied { recommendation, .. } => {
            assert_eq!(
                recommendation.value,
                TuneRecommendedValue::FlashAttention(value)
            );
        }
        other => panic!("expected applied flash_attention, got {other:?}"),
    }
}

pub(crate) fn assert_applied_context(plan: &TunePlan, value: u32) {
    match status_for(plan, TuneField::CtxSize) {
        TuneFieldStatus::Applied { recommendation, .. } => {
            assert_eq!(
                recommendation.value,
                TuneRecommendedValue::ContextSize(value)
            );
        }
        other => panic!("expected applied ctx_size, got {other:?}"),
    }
}

pub(crate) fn assert_applied_batch(plan: &TunePlan, value: u32) {
    match status_for(plan, TuneField::Batch) {
        TuneFieldStatus::Applied { recommendation, .. } => {
            assert_eq!(recommendation.value, TuneRecommendedValue::Batch(value));
        }
        other => panic!("expected applied batch, got {other:?}"),
    }
}

pub(crate) fn assert_applied_ubatch(plan: &TunePlan, value: u32) {
    match status_for(plan, TuneField::Ubatch) {
        TuneFieldStatus::Applied { recommendation, .. } => {
            assert_eq!(recommendation.value, TuneRecommendedValue::Ubatch(value));
        }
        other => panic!("expected applied ubatch, got {other:?}"),
    }
}

pub(crate) fn assert_applied_gpu_layers(plan: &TunePlan, value: TuneGpuLayersValue) {
    match status_for(plan, TuneField::GpuLayers) {
        TuneFieldStatus::Applied { recommendation, .. } => {
            assert_eq!(recommendation.value, TuneRecommendedValue::GpuLayers(value));
        }
        other => panic!("expected applied gpu_layers, got {other:?}"),
    }
}

pub(crate) fn assert_applied_fit_target(plan: &TunePlan, value: u64) {
    match status_for(plan, TuneField::FitTargetMib) {
        TuneFieldStatus::Applied { recommendation, .. } => {
            assert_eq!(
                recommendation.value,
                TuneRecommendedValue::FitTargetMib(value)
            );
        }
        other => panic!("expected applied fit_target_mib, got {other:?}"),
    }
}

pub(crate) fn assert_preserved(plan: &TunePlan, field: TuneField, expected_path: &str) {
    match status_for(plan, field) {
        TuneFieldStatus::Preserved { reason, .. } => assert!(reason.contains(expected_path)),
        other => panic!("expected preserved status, got {other:?}"),
    }
}

pub(crate) const fn gib() -> u64 {
    1024 * 1024 * 1024
}
