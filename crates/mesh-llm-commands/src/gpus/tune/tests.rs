use super::*;
use serde_json::json;

pub(crate) fn sample_target() -> TuneTarget {
    TuneTarget {
        requested: "hf://mesh/example.gguf".to_string(),
        resolved: Some("/models/example.gguf".to_string()),
        config_model_ref: Some("hf://mesh/example.gguf".to_string()),
        derived_profile: Some("abc12345".to_string()),
    }
}

#[test]
fn tune_plan_field_statuses_are_serializable_and_stable() {
    let plan = TunePlan {
        target: sample_target(),
        apply_mode: TuneApplyMode::ApplyMissing,
        field_statuses: vec![
            TuneFieldStatus::Applied {
                recommendation: TuneRecommendation {
                    field: TuneField::CacheTypeK,
                    value: TuneRecommendedValue::KvCacheType(TuneKvCacheType::Q8_0),
                    rationale: "stable kv fit".to_string(),
                },
                edit: TuneConfigEdit::SetModelFitCacheTypeK(TuneKvCacheType::Q8_0),
            },
            TuneFieldStatus::Preserved {
                field: TuneField::CtxSize,
                reason: "existing defaults.ctx_size remains authoritative".to_string(),
            },
            TuneFieldStatus::Applied {
                recommendation: TuneRecommendation {
                    field: TuneField::Mlock,
                    value: TuneRecommendedValue::Bool(true),
                    rationale: "would reduce paging on supported hosts".to_string(),
                },
                edit: TuneConfigEdit::SetHardwareMlock(true),
            },
            TuneFieldStatus::Unsupported {
                field: TuneField::CpuMoe,
                reason: "pinned skippy resolver rejects cpu_moe".to_string(),
            },
            TuneFieldStatus::Error {
                field: TuneField::Device,
                diagnostic: TuneDiagnostic {
                    severity: TuneDiagnosticSeverity::Error,
                    code: TuneDiagnosticCode::MissingConfiguredDevice,
                    field: Some(TuneField::Device),
                    message: "configured device gpu-7 was not present in the survey".to_string(),
                },
            },
        ],
        diagnostics: vec![TuneDiagnostic {
            severity: TuneDiagnosticSeverity::Warning,
            code: TuneDiagnosticCode::ReportOnlyField,
            field: Some(TuneField::Mlock),
            message: "mlock was reviewed but not emitted as a config write".to_string(),
        }],
    };

    assert_eq!(
        serde_json::to_value(&plan).expect("plan should serialize"),
        json!({
            "target": {
                "requested": "hf://mesh/example.gguf",
                "resolved": "/models/example.gguf",
                "config_model_ref": "hf://mesh/example.gguf",
                "derived_profile": "abc12345"
            },
            "apply_mode": "apply_missing",
            "field_statuses": [
                {
                    "kind": "applied",
                    "recommendation": {
                        "field": "cache_type_k",
                        "value": { "kind": "kv_cache_type", "value": "q8_0" },
                        "rationale": "stable kv fit"
                    },
                    "edit": { "kind": "set_model_fit_cache_type_k", "value": "q8_0" }
                },
                {
                    "kind": "preserved",
                    "field": "ctx_size",
                    "reason": "existing defaults.ctx_size remains authoritative"
                },
                {
                    "kind": "applied",
                    "recommendation": {
                        "field": "mlock",
                        "value": { "kind": "bool", "value": true },
                        "rationale": "would reduce paging on supported hosts"
                    },
                    "edit": { "kind": "set_hardware_mlock", "value": true }
                },
                {
                    "kind": "unsupported",
                    "field": "cpu_moe",
                    "reason": "pinned skippy resolver rejects cpu_moe"
                },
                {
                    "kind": "error",
                    "field": "device",
                    "diagnostic": {
                        "severity": "error",
                        "code": "missing_configured_device",
                        "field": "device",
                        "message": "configured device gpu-7 was not present in the survey"
                    }
                }
            ],
            "diagnostics": [
                {
                    "severity": "warning",
                    "code": "report_only_field",
                    "field": "mlock",
                    "message": "mlock was reviewed but not emitted as a config write"
                }
            ]
        })
    );

    assert_eq!(
        plan.summary(),
        TunePlanSummary {
            applied: 2,
            preserved: 1,
            report_only: 0,
            unsupported: 1,
            error: 1,
        }
    );
}

#[test]
fn tune_plan_unsupported_fields_do_not_emit_config_edits_but_mmap_is_writable() {
    let plan = TunePlan {
        target: sample_target(),
        apply_mode: TuneApplyMode::Review,
        field_statuses: vec![
            TuneFieldStatus::Applied {
                recommendation: TuneRecommendation {
                    field: TuneField::FitTargetMib,
                    value: TuneRecommendedValue::FitTargetMib(28_672),
                    rationale: "allocatable vram after safety margin".to_string(),
                },
                edit: TuneConfigEdit::SetHardwareFitTargetMib(28_672),
            },
            TuneFieldStatus::Applied {
                recommendation: TuneRecommendation {
                    field: TuneField::Mmap,
                    value: TuneRecommendedValue::BoolOrAuto(TuneBoolOrAutoValue::Auto),
                    rationale: "visible in schema but not proven end-to-end".to_string(),
                },
                edit: TuneConfigEdit::SetHardwareMmap(TuneBoolOrAutoValue::Auto),
            },
            TuneFieldStatus::Unsupported {
                field: TuneField::TensorSplit,
                reason: "pinned skippy resolver rejects tensor_split".to_string(),
            },
            TuneFieldStatus::Unsupported {
                field: TuneField::Placement,
                reason: "pinned skippy resolver rejects placement".to_string(),
            },
            TuneFieldStatus::Error {
                field: TuneField::CpuMoe,
                diagnostic: TuneDiagnostic {
                    severity: TuneDiagnosticSeverity::Error,
                    code: TuneDiagnosticCode::UnsupportedField,
                    field: Some(TuneField::CpuMoe),
                    message: "cpu_moe is not writable in v1".to_string(),
                },
            },
        ],
        diagnostics: Vec::new(),
    };

    assert_eq!(
        plan.config_edits(),
        vec![
            TuneConfigEdit::SetHardwareFitTargetMib(28_672),
            TuneConfigEdit::SetHardwareMmap(TuneBoolOrAutoValue::Auto),
        ]
    );
    assert_eq!(
        TuneField::TensorSplit.spec().support,
        TuneFieldSupport::Unsupported
    );
    assert_eq!(
        TuneField::Placement.spec().support,
        TuneFieldSupport::Unsupported
    );
    assert_eq!(TuneField::Mmap.spec().support, TuneFieldSupport::Writable);
    assert_eq!(
        TuneField::Defaults.spec().support,
        TuneFieldSupport::PreserveOnly
    );
}
