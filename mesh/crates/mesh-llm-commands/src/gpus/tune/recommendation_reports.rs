use crate::gpus::tune_hardware::types::TuneMemorySource;

use super::*;

pub(crate) fn push_cpu_moe_statuses(plan: &mut TunePlan, metadata: &TuneGgufMetadata) {
    let expert_count = match &metadata.tensor_profile {
        TuneTensorProfile::Exact(profile) => profile.expert_count,
        TuneTensorProfile::DegradedFallback { .. } => 0,
    };
    if expert_count > 0 {
        plan.field_statuses.push(TuneFieldStatus::ReportOnly {
            recommendation: TuneRecommendation {
                field: TuneField::CpuMoe,
                value: TuneRecommendedValue::BoolOrAuto(TuneBoolOrAutoValue::Auto),
                rationale: format!(
                    "GGUF advertises {expert_count} experts, but cpu_moe is not writable in v1"
                ),
            },
            reason: "cpu_moe remains report-only until the pinned runtime supports it end-to-end"
                .to_string(),
        });
    } else {
        plan.field_statuses.push(unsupported_status(
            TuneField::CpuMoe,
            "cpu_moe remains unsupported by the pinned runtime in v1",
        ));
    }
    plan.field_statuses.push(unsupported_status(
        TuneField::NCpuMoe,
        "n_cpu_moe remains unsupported by the pinned runtime in v1",
    ));
}

pub(crate) fn unsupported_status(field: TuneField, reason: &str) -> TuneFieldStatus {
    TuneFieldStatus::Unsupported {
        field,
        reason: reason.to_string(),
    }
}

pub(crate) fn invalid_existing_value_diagnostic(field: TuneField, value: &str) -> TuneDiagnostic {
    TuneDiagnostic {
        severity: TuneDiagnosticSeverity::Error,
        code: TuneDiagnosticCode::InvalidExistingValue,
        field: Some(field),
        message: format!("existing value `{value}` is not a supported v1 tune setting"),
    }
}

pub(crate) fn insufficient_memory_diagnostic(
    source: &TuneMemorySource,
    budget_bytes: u64,
    model_bytes: u64,
    kv_bytes_per_token: u64,
) -> TuneDiagnostic {
    TuneDiagnostic {
        severity: TuneDiagnosticSeverity::Error,
        code: TuneDiagnosticCode::InsufficientMemory,
        field: Some(TuneField::CtxSize),
        message: format!(
            "no safe startup plan fits within {}: budget={} bytes, model={} bytes, minimum quantized KV={} bytes at 512 context",
            match source {
                TuneMemorySource::EvaluatedGpuVram => "GPU VRAM",
                TuneMemorySource::SystemRamFallback => "system RAM",
            },
            budget_bytes,
            model_bytes,
            kv_bytes_per_token.saturating_mul(512),
        ),
    }
}
