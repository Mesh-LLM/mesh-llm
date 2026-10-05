use crate::gpus::tune_apply::PreparedTunePlan;
use mesh_llm_config::MeshConfig;

use super::*;

pub(crate) fn sample_output_plan() -> TunePlan {
    TunePlan {
        target: sample_target(),
        apply_mode: TuneApplyMode::Review,
        field_statuses: vec![
            TuneFieldStatus::Applied {
                recommendation: TuneRecommendation {
                    field: TuneField::CacheTypeK,
                    value: TuneRecommendedValue::KvCacheType(TuneKvCacheType::Q8_0),
                    rationale: "stable kv fit".to_string(),
                },
                edit: TuneConfigEdit::SetModelFitCacheTypeK(TuneKvCacheType::Q8_0),
            },
            TuneFieldStatus::Applied {
                recommendation: TuneRecommendation {
                    field: TuneField::Mlock,
                    value: TuneRecommendedValue::Bool(false),
                    rationale: "current lock limit is 64.0 KiB; enable IPC_LOCK or raise RLIMIT_MEMLOCK to lock the evaluated working set"
                        .to_string(),
                },
                edit: TuneConfigEdit::SetHardwareMlock(false),
            },
            TuneFieldStatus::Unsupported {
                field: TuneField::TensorSplit,
                reason: "tensor_split remains unsupported by the pinned runtime in v1".to_string(),
            },
            TuneFieldStatus::Error {
                field: TuneField::CtxSize,
                diagnostic: TuneDiagnostic {
                    severity: TuneDiagnosticSeverity::Error,
                    code: TuneDiagnosticCode::InsufficientMemory,
                    field: Some(TuneField::CtxSize),
                    message: "no safe startup plan fits".to_string(),
                },
            },
        ],
        diagnostics: vec![TuneDiagnostic {
            severity: TuneDiagnosticSeverity::Warning,
            code: TuneDiagnosticCode::MlockUnavailable,
            field: Some(TuneField::Mlock),
            message:
                "current lock limit is 64.0 KiB; enable IPC_LOCK or raise RLIMIT_MEMLOCK to lock the evaluated working set"
                    .to_string(),
        }],
    }
}

#[test]
fn gpu_tune_human_output_names_targets_and_reasons() {
    let report = TuneRunReport {
        command: "gpu_tune",
        apply_mode: TuneApplyMode::Review,
        summary: TuneResultSummary {
            total_targets: 2,
            ready_targets: 1,
            failed_targets: 1,
            written_targets: 0,
            skipped_targets: 0,
            fields: TunePlanSummary {
                applied: 1,
                preserved: 0,
                report_only: 1,
                unsupported: 1,
                error: 0,
            },
        },
        global_blockers: Vec::new(),
        benchmarks: Vec::new(),
        targets: vec![
            TuneTargetReport {
                target: sample_target(),
                status: TuneTargetStatus::Ready,
                canonical_model_ref: Some("hf://mesh/example.gguf".to_string()),
                selection: "configured".to_string(),
                reason: Some("prepared 1 writable tune edits for review".to_string()),
                field_summary: Some(TunePlanSummary {
                    applied: 1,
                    preserved: 0,
                    report_only: 1,
                    unsupported: 1,
                    error: 0,
                }),
                diagnostics: vec![TuneDiagnostic {
                    severity: TuneDiagnosticSeverity::Warning,
                    code: TuneDiagnosticCode::MlockUnavailable,
                    field: Some(TuneField::Mlock),
                    message: "mlock unavailable".to_string(),
                }],
                settings: vec![TuneRenderedSetting {
                    field: TuneField::CacheTypeK,
                    support: TuneFieldSupport::Writable,
                    status: TuneRenderedSettingStatus::Applied,
                    config_path: "models.<model-ref>.model_fit.cache_type_k".to_string(),
                    value: Some(TuneRecommendedValue::KvCacheType(TuneKvCacheType::Q8_0)),
                    rationale: Some("stable kv fit".to_string()),
                    reason: None,
                    diagnostic: None,
                    edit: Some(TuneConfigEdit::SetModelFitCacheTypeK(TuneKvCacheType::Q8_0)),
                    applied_write: true,
                }],
                config_edits: vec![TuneRenderedSetting {
                    field: TuneField::CacheTypeK,
                    support: TuneFieldSupport::Writable,
                    status: TuneRenderedSettingStatus::Applied,
                    config_path: "models.<model-ref>.model_fit.cache_type_k".to_string(),
                    value: Some(TuneRecommendedValue::KvCacheType(TuneKvCacheType::Q8_0)),
                    rationale: Some("stable kv fit".to_string()),
                    reason: None,
                    diagnostic: None,
                    edit: Some(TuneConfigEdit::SetModelFitCacheTypeK(TuneKvCacheType::Q8_0)),
                    applied_write: true,
                }],
                launch: None,
            },
            TuneTargetReport {
                target: TuneTarget {
                    requested: "missing.gguf".to_string(),
                    resolved: None,
                    config_model_ref: None,
                    derived_profile: None,
                },
                status: TuneTargetStatus::Failed,
                canonical_model_ref: None,
                selection: "unresolved".to_string(),
                reason: Some(
                    "requested target `missing.gguf`: target is not an existing local path or installed cache ref"
                        .to_string(),
                ),
                field_summary: None,
                diagnostics: Vec::new(),
                settings: Vec::new(),
                config_edits: Vec::new(),
                launch: None,
            },
        ],
    };

    let rendered = render_tune_human_output(&report);

    assert!(rendered.contains("Target: hf://mesh/example.gguf"));
    assert!(rendered.contains("Reason: prepared 1 writable tune edits for review"));
    assert!(rendered.contains("Target: missing.gguf"));
    assert!(rendered.contains("installed cache ref"));
}

#[test]
fn gpu_tune_output_never_marks_unsupported_fields_as_applied() {
    let report = build_tune_run_report(
        "gpu_tune",
        &MeshConfig::default(),
        TuneApplyMode::Review,
        &[PreparedTunePlan::new(
            recommendation_target(false),
            sample_output_plan(),
        )],
        &[],
        &[],
        &[],
    );

    let target = &report.targets[0];
    assert!(
        target
            .config_edits
            .iter()
            .all(|setting| setting.field != TuneField::TensorSplit)
    );
    assert!(
        target
            .settings
            .iter()
            .any(|setting| setting.field == TuneField::TensorSplit
                && setting.status == TuneRenderedSettingStatus::Unsupported
                && !setting.applied_write)
    );
}

#[test]
fn gpu_tune_human_output_explains_mlock_unavailable() {
    let report = build_tune_run_report(
        "gpu_tune",
        &MeshConfig::default(),
        TuneApplyMode::Review,
        &[PreparedTunePlan::new(
            recommendation_target(false),
            sample_output_plan(),
        )],
        &[],
        &[],
        &[],
    );

    let rendered = render_tune_human_output(&report);

    assert!(rendered.contains("RLIMIT_MEMLOCK"));
    assert!(rendered.contains("mlock"));
}

#[test]
fn gpu_tune_json_reports_per_model_errors_without_silent_failures_output_builder() {
    let report = build_tune_run_report(
        "gpu_tune",
        &MeshConfig::default(),
        TuneApplyMode::Review,
        &[PreparedTunePlan::new(
            recommendation_target(false),
            sample_output_plan(),
        )],
        &[TuneTargetFailure {
            requested_input: "missing.gguf".to_string(),
            reason: "requested target `missing.gguf`: target is not an existing local path or installed cache ref"
                .to_string(),
        }],
        &[],
        &[],
    );

    let value = serde_json::to_value(&report).expect("report should serialize");

    assert_eq!(value["summary"]["total_targets"], serde_json::json!(2));
    assert_eq!(value["summary"]["failed_targets"], serde_json::json!(2));
    assert_eq!(value["targets"][0]["status"], serde_json::json!("failed"));
    assert_eq!(value["targets"][1]["status"], serde_json::json!("failed"));
    assert!(
        value["targets"][1]["reason"]
            .as_str()
            .expect("reason should be a string")
            .contains("installed cache ref")
    );
}
