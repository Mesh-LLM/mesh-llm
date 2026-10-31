//! Memory planning for the default DFlash draft of a direct local load: the
//! draft the resolver will attach, and a plan that holds room for its weights
//! and rollback snapshots when they fit beside the minimum context.

use super::context_planning::{
    PlannedSpeculativeDraft, RuntimeResourcePlan, RuntimeResourcePlanInput, plan_runtime_resources,
    plan_runtime_resources_with_draft,
};
use super::local::{LocalOpenAiModelStartSpec, resolve_local_openai_skippy_config};
use crate::inference::skippy;
use crate::models;

/// Plans with room for the default DFlash draft, and reports whether that
/// room was held. A draft that leaves no room for the minimum context is left
/// out of the plan, and the model serves without it.
pub(super) fn plan_with_default_dflash_draft(
    input: RuntimeResourcePlanInput<'_>,
    draft: Option<PlannedSpeculativeDraft>,
) -> (RuntimeResourcePlan, bool) {
    if draft.is_some() {
        let plan = plan_runtime_resources_with_draft(input, draft);
        if plan.fits() {
            return (plan, true);
        }
    }
    (plan_runtime_resources(input), false)
}

/// The draft the resolver will choose by default for this load, so the memory
/// plan can hold room for its weights and rollback snapshots. `None` when no
/// default applies or the draft is not cached yet. Resolution errors surface
/// later, when the load resolves for real.
pub(super) fn planned_default_dflash_draft(
    spec: &LocalOpenAiModelStartSpec<'_>,
    model_name: &str,
    compact_meta: Option<&models::gguf::GgufCompactMeta>,
) -> Option<PlannedSpeculativeDraft> {
    let resolved = resolve_local_openai_skippy_config(
        spec,
        model_name,
        spec.model_bytes,
        0,
        1,
        None,
        compact_meta,
    )
    .ok()?;
    let pairing = resolved.speculative.default_dflash?;
    let dflash = resolved.speculative.decode.dflash.as_ref()?;
    // A draft that is not cached yet downloads in the background; this load
    // plans no room for it and serves without it.
    if !skippy::default_dflash_draft_cached(pairing) {
        return None;
    }
    Some(PlannedSpeculativeDraft {
        weight_bytes: pairing.draft.bytes,
        recurrent_rollback_tokens: dflash.recurrent_rollback_tokens(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::context_planning::RuntimeResourcePlanningProfile;

    fn qwen35_9b_meta() -> crate::models::gguf::GgufCompactMeta {
        crate::models::gguf::GgufCompactMeta {
            architecture: "qwen35".to_string(),
            context_length: 262_144,
            embedding_size: 4096,
            head_count: 16,
            kv_head_count: 4,
            layer_count: 32,
            key_length: 256,
            value_length: 256,
            ssm_conv_kernel: 4,
            ssm_inner_size: 4096,
            ssm_state_size: 128,
            ssm_group_count: 16,
            recurrent_layers: (0..32).map(|layer| layer % 4 != 3).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn a_default_dflash_draft_that_does_not_fit_is_left_out_of_the_plan() {
        let meta = qwen35_9b_meta();
        let input = |vram_bytes| RuntimeResourcePlanInput {
            ctx_size_override: None,
            parallel_override: None,
            model_bytes: 5_680_522_464,
            projector_bytes: 0,
            vram_bytes,
            metadata: Some(&meta),
            kv_cache_quant: crate::models::gguf::GgufKvCacheQuant::F16,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: None,
        };
        let draft = Some(PlannedSpeculativeDraft {
            weight_bytes: 1_383_768_224,
            recurrent_rollback_tokens: 16,
        });

        let (roomy, planned) = plan_with_default_dflash_draft(input(64 << 30), draft);
        assert!(planned);
        assert!(roomy.breakdown.unwrap().speculative_bytes > 1_383_768_224);

        let (tight, planned) = plan_with_default_dflash_draft(input(10_500_000_000), draft);
        assert!(!planned);
        assert!(tight.fits());
        assert_eq!(tight.breakdown.unwrap().speculative_bytes, 0);

        let (_, planned) = plan_with_default_dflash_draft(input(64 << 30), None);
        assert!(!planned);
    }
}
