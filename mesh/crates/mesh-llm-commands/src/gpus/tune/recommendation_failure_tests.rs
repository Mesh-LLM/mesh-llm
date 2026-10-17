use mesh_llm_config::MeshConfig;
use mesh_llm_system::hardware::HardwareSurvey;

use super::*;

#[test]
fn gpu_tune_fails_when_context_cannot_fit_even_with_quantized_kv() {
    let plan = build_tune_plan(TuneRecommendationInput {
        apply_mode: TuneApplyMode::ApplyMissing,
        config: &MeshConfig::default(),
        target: &recommendation_target(false),
        metadata: &sample_metadata(10 * gib(), 32, 131_072, 0),
        hardware: &gpu_hardware(11 * gib()),
        survey: &survey_with_gpu(11 * gib(), 11 * gib()),
    });

    assert!(plan.config_edits().is_empty());
    assert!(
        plan.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == TuneDiagnosticCode::InsufficientMemory)
    );
    assert!(matches!(
        status_for(&plan, TuneField::CtxSize),
        TuneFieldStatus::Error { .. }
    ));
    assert!(matches!(
        status_for(&plan, TuneField::GpuLayers),
        TuneFieldStatus::Error { .. }
    ));
}

#[test]
fn gpu_tune_recommends_partial_gpu_layers_when_full_offload_is_unsafe() {
    let plan = build_tune_plan(TuneRecommendationInput {
        apply_mode: TuneApplyMode::ApplyMissing,
        config: &MeshConfig::default(),
        target: &recommendation_target(false),
        metadata: &sample_metadata(30 * gib(), 60, 65_536, 0),
        hardware: &gpu_hardware(18 * gib()),
        survey: &survey_with_gpu(18 * gib(), 64 * gib()),
    });

    match status_for(&plan, TuneField::GpuLayers) {
        TuneFieldStatus::Applied { recommendation, .. } => match &recommendation.value {
            TuneRecommendedValue::GpuLayers(TuneGpuLayersValue::Count(count)) => {
                assert!(*count > 0);
            }
            other => panic!("expected partial gpu layer count, got {other:?}"),
        },
        other => panic!("expected applied gpu_layers, got {other:?}"),
    }
}

/// A 40 GiB mixture-of-experts model whose routed experts are 32 GiB of it.
fn moe_metadata() -> TuneGgufMetadata {
    let mut metadata = sample_metadata(40 * gib(), 32, 65_536, 128);
    if let TuneTensorProfile::Exact(profile) = &mut metadata.tensor_profile {
        profile.expert_tensor_bytes = 32 * gib();
        profile.base_resident_bytes = 8 * gib();
        profile.expert_used_count = 8;
    }
    metadata
}

#[test]
fn gpu_tune_keeps_experts_on_the_accelerator_when_the_model_fits() {
    let plan = build_tune_plan(TuneRecommendationInput {
        apply_mode: TuneApplyMode::ApplyMissing,
        config: &MeshConfig::default(),
        target: &recommendation_target(false),
        metadata: &sample_metadata(8 * gib(), 32, 65_536, 16),
        hardware: &gpu_hardware(24 * gib()),
        survey: &survey_with_gpu(24 * gib(), 64 * gib()),
    });

    match status_for(&plan, TuneField::CpuMoe) {
        TuneFieldStatus::ReportOnly { recommendation, .. } => assert_eq!(
            recommendation.value,
            TuneRecommendedValue::BoolOrAuto(TuneBoolOrAutoValue::Auto)
        ),
        other => panic!("expected cpu_moe report-only status, got {other:?}"),
    }
    match status_for(&plan, TuneField::NCpuMoe) {
        TuneFieldStatus::ReportOnly { recommendation, .. } => {
            assert_eq!(
                recommendation.value,
                TuneRecommendedValue::HostExpertBlocks(0)
            );
        }
        other => panic!("expected n_cpu_moe report-only status, got {other:?}"),
    }
    assert!(
        !plan
            .config_edits()
            .iter()
            .any(|edit| matches!(edit, TuneConfigEdit::SetHardwareNCpuMoe(_)))
    );
}

#[test]
fn gpu_tune_writes_n_cpu_moe_when_host_experts_let_every_layer_fit() {
    let plan = build_tune_plan(TuneRecommendationInput {
        apply_mode: TuneApplyMode::ApplyMissing,
        config: &MeshConfig::default(),
        target: &recommendation_target(false),
        metadata: &moe_metadata(),
        hardware: &gpu_hardware(24 * gib()),
        survey: &HardwareSurvey {
            system_ram_bytes: Some(128 * gib()),
            ..survey_with_gpu(24 * gib(), 128 * gib())
        },
    });

    match status_for(&plan, TuneField::GpuLayers) {
        TuneFieldStatus::Applied { recommendation, .. } => assert_eq!(
            recommendation.value,
            TuneRecommendedValue::GpuLayers(TuneGpuLayersValue::All)
        ),
        other => panic!("expected every layer on the accelerator, got {other:?}"),
    }
    let blocks = match status_for(&plan, TuneField::NCpuMoe) {
        TuneFieldStatus::Applied {
            recommendation:
                TuneRecommendation {
                    value: TuneRecommendedValue::HostExpertBlocks(blocks),
                    ..
                },
            edit: TuneConfigEdit::SetHardwareNCpuMoe(edit_blocks),
        } => {
            assert_eq!(blocks, edit_blocks);
            *blocks
        }
        other => panic!("expected an applied n_cpu_moe, got {other:?}"),
    };
    // 40 GiB of weights against a ~22 GiB budget needs well over half of the
    // 1 GiB-per-block expert banks moved, but not all of them.
    assert!(
        blocks > 16 && blocks < 32,
        "unexpected block count {blocks}"
    );
}

#[test]
fn gpu_tune_preserves_an_operator_n_cpu_moe() {
    let config = MeshConfig {
        defaults: Some(mesh_llm_config::ModelConfigDefaults {
            hardware: Some(mesh_llm_config::HardwareConfig {
                n_cpu_moe: Some(4),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let plan = build_tune_plan(TuneRecommendationInput {
        apply_mode: TuneApplyMode::ApplyMissing,
        config: &config,
        target: &recommendation_target(false),
        metadata: &moe_metadata(),
        hardware: &gpu_hardware(24 * gib()),
        survey: &HardwareSurvey {
            system_ram_bytes: Some(128 * gib()),
            ..survey_with_gpu(24 * gib(), 128 * gib())
        },
    });

    assert!(matches!(
        status_for(&plan, TuneField::NCpuMoe),
        TuneFieldStatus::Preserved { .. }
    ));
}

#[test]
fn gpu_tune_reports_dense_models_as_having_no_routed_experts() {
    let plan = build_tune_plan(TuneRecommendationInput {
        apply_mode: TuneApplyMode::ApplyMissing,
        config: &MeshConfig::default(),
        target: &recommendation_target(false),
        metadata: &sample_metadata(8 * gib(), 32, 65_536, 0),
        hardware: &gpu_hardware(24 * gib()),
        survey: &survey_with_gpu(24 * gib(), 64 * gib()),
    });

    for field in [TuneField::CpuMoe, TuneField::NCpuMoe] {
        match status_for(&plan, field) {
            TuneFieldStatus::Unsupported { reason, .. } => {
                assert!(reason.contains("no routed experts"));
            }
            other => panic!("expected {field:?} unsupported for a dense model, got {other:?}"),
        }
    }
}
