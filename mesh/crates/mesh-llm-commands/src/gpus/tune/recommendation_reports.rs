use crate::gpus::tune_hardware::types::TuneMemorySource;
use mesh_llm_config::{ModelConfigDefaults, ModelConfigEntry};

use super::*;

/// Inputs the routed-expert recommendations need beyond the model metadata.
pub(crate) struct CpuMoeTuneContext<'a> {
    pub(crate) apply_mode: TuneApplyMode,
    pub(crate) model_entry: Option<&'a ModelConfigEntry>,
    pub(crate) defaults: Option<&'a ModelConfigDefaults>,
    /// Blocks whose routed experts must move to host memory for the model to
    /// fit; `None` when it fits without moving any.
    pub(crate) host_expert_blocks: Option<u32>,
}

pub(crate) fn push_cpu_moe_statuses(
    plan: &mut TunePlan,
    metadata: &TuneGgufMetadata,
    context: CpuMoeTuneContext<'_>,
) {
    let expert_count = match &metadata.tensor_profile {
        TuneTensorProfile::Exact(profile) => profile.expert_count,
        TuneTensorProfile::DegradedFallback { .. } => 0,
    };
    if expert_count == 0 {
        plan.field_statuses.push(unsupported_status(
            TuneField::CpuMoe,
            "the model has no routed experts",
        ));
        plan.field_statuses.push(unsupported_status(
            TuneField::NCpuMoe,
            "the model has no routed experts",
        ));
        return;
    }
    push_cpu_moe_status(plan, &context, expert_count);
    push_n_cpu_moe_status(plan, &context);
}

fn push_cpu_moe_status(plan: &mut TunePlan, context: &CpuMoeTuneContext<'_>, expert_count: u32) {
    if let Some(source) = existing_cpu_moe_source(context.model_entry, context.defaults) {
        plan.field_statuses.push(TuneFieldStatus::Preserved {
            field: TuneField::CpuMoe,
            reason: preserve_reason(source, TuneField::CpuMoe),
        });
        return;
    }
    plan.field_statuses.push(TuneFieldStatus::ReportOnly {
        recommendation: TuneRecommendation {
            field: TuneField::CpuMoe,
            value: TuneRecommendedValue::BoolOrAuto(TuneBoolOrAutoValue::Auto),
            rationale: format!(
                "GGUF advertises {expert_count} experts; auto lets split planning move routed experts into host memory when a node set cannot otherwise hold the model"
            ),
        },
        reason: "cpu_moe stays auto; n_cpu_moe carries the single-node placement".to_string(),
    });
}

fn push_n_cpu_moe_status(plan: &mut TunePlan, context: &CpuMoeTuneContext<'_>) {
    if let Some(source) = existing_n_cpu_moe_source(context.model_entry, context.defaults)
        && context.apply_mode != TuneApplyMode::ReplaceExisting
    {
        plan.field_statuses.push(TuneFieldStatus::Preserved {
            field: TuneField::NCpuMoe,
            reason: preserve_reason(source, TuneField::NCpuMoe),
        });
        return;
    }
    let Some(blocks) = context.host_expert_blocks else {
        plan.field_statuses.push(TuneFieldStatus::ReportOnly {
            recommendation: TuneRecommendation {
                field: TuneField::NCpuMoe,
                value: TuneRecommendedValue::HostExpertBlocks(0),
                rationale: "every routed expert fits on the accelerator".to_string(),
            },
            reason: "no blocks need host-resident experts on this hardware".to_string(),
        });
        return;
    };
    plan.field_statuses.push(TuneFieldStatus::Applied {
        recommendation: TuneRecommendation {
            field: TuneField::NCpuMoe,
            value: TuneRecommendedValue::HostExpertBlocks(blocks),
            rationale: format!(
                "moving the routed experts of the first {blocks} block(s) into host memory keeps every layer on the accelerator"
            ),
        },
        edit: TuneConfigEdit::SetHardwareNCpuMoe(blocks),
    });
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
