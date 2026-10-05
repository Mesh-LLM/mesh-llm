use mesh_llm_config::MeshConfig;

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

#[test]
fn gpu_tune_reports_cpu_moe_as_report_only_for_expert_models() {
    let plan = build_tune_plan(TuneRecommendationInput {
        apply_mode: TuneApplyMode::ApplyMissing,
        config: &MeshConfig::default(),
        target: &recommendation_target(false),
        metadata: &sample_metadata(8 * gib(), 32, 65_536, 16),
        hardware: &gpu_hardware(24 * gib()),
        survey: &survey_with_gpu(24 * gib(), 64 * gib()),
    });

    match status_for(&plan, TuneField::CpuMoe) {
        TuneFieldStatus::ReportOnly { reason, .. } => assert!(reason.contains("report-only")),
        other => panic!("expected cpu_moe report-only status, got {other:?}"),
    }
}
