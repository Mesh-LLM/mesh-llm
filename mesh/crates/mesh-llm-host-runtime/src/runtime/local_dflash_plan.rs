//! Memory planning for the DFlash draft of a direct local load: the draft the
//! resolver will attach, and a plan that holds room for its weights, context
//! and rollback snapshots. A default draft is kept only when that room fits
//! beside the minimum context; an explicit draft is always planned for.

use std::path::Path;

use super::context_planning::{
    PlannedSpeculativeDraft, RuntimeResourcePlan, RuntimeResourcePlanInput, plan_runtime_resources,
    plan_runtime_resources_with_draft,
};
use super::local::{LocalOpenAiModelStartSpec, resolve_local_openai_skippy_config};
use crate::inference::skippy;
use crate::models;

/// The DFlash draft a load will attach, as the memory plan charges it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PlannedDFlashDraft {
    pub(super) draft: PlannedSpeculativeDraft,
    /// Chosen by default rather than configured, so it may be left out.
    pub(super) default: bool,
}

/// Microbatch rows a draft context's compute buffers are reserved for: the
/// target's default `n_ubatch`, which the draft context inherits.
const DRAFT_COMPUTE_ROWS: u64 = 512;

/// Plans with room for the DFlash draft, and reports whether that room was
/// held. A default draft that leaves no room for the minimum context is left
/// out of the plan, and the model serves without it. An explicit draft is
/// always planned for; it shortens the context, and on auto lanes it also
/// gives up lanes until it fits.
pub(super) fn plan_with_default_dflash_draft(
    input: RuntimeResourcePlanInput<'_>,
    draft: Option<PlannedDFlashDraft>,
) -> (RuntimeResourcePlan, bool) {
    if let Some(PlannedDFlashDraft { draft, default }) = draft {
        let plan = plan_runtime_resources_with_draft(input, Some(draft));
        if plan.fits() {
            return (plan, true);
        }
        if !default {
            let fewer_lanes = fewer_lanes_for_draft(input, draft, plan.slots);
            return (fewer_lanes.unwrap_or(plan), true);
        }
    }
    (plan_runtime_resources(input), false)
}

/// The draft's rollback snapshots are held per lane on a hybrid target, so a
/// shorter context cannot make room for them: only fewer lanes can. On auto
/// lanes, the widest plan under `slots` that fits; a configured lane count is
/// kept as asked.
fn fewer_lanes_for_draft(
    input: RuntimeResourcePlanInput<'_>,
    draft: PlannedSpeculativeDraft,
    slots: usize,
) -> Option<RuntimeResourcePlan> {
    if input.parallel_override.is_some() {
        return None;
    }
    (1..slots).rev().find_map(|lanes| {
        let mut plan = plan_runtime_resources_with_draft(
            RuntimeResourcePlanInput {
                parallel_override: Some(lanes),
                ..input
            },
            Some(draft),
        );
        // The planner chose this lane count, not the operator.
        if let Some(breakdown) = plan.breakdown.as_mut() {
            breakdown.slots_auto = true;
        }
        plan.fits().then_some(plan)
    })
}

/// The DFlash draft the resolver will attach for this load, explicit or
/// default, so the memory plan can hold room for its weights, context and
/// rollback snapshots. `None` without a draft, or when a default draft is not
/// cached yet. Resolution errors surface later, when the load resolves for
/// real.
pub(super) fn planned_default_dflash_draft(
    spec: &LocalOpenAiModelStartSpec<'_>,
    model_name: &str,
    compact_meta: Option<&models::gguf::GgufCompactMeta>,
) -> Option<PlannedDFlashDraft> {
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
    let dflash = resolved.speculative.decode.dflash.as_ref()?;
    let recurrent_rollback_tokens = dflash.recurrent_rollback_tokens();
    let (path, weight_bytes, default) = match resolved.speculative.default_dflash.as_ref() {
        Some(chosen) => {
            // A draft that is not cached yet downloads in the background; this
            // load plans no room for it and serves without it.
            if !skippy::default_dflash_draft_cached(chosen) {
                return None;
            }
            (
                mesh_llm_skippy_adapter::config::dflash_draft_cache_path(&chosen.pairing.draft),
                chosen.pairing.draft.bytes,
                true,
            )
        }
        None => {
            let path = dflash.draft_model_path.clone();
            let bytes = std::fs::metadata(&path).ok()?.len();
            (path, bytes, false)
        }
    };
    // The draft context takes its own cache types (`draft_cache_type_k/v`,
    // F16 by default), not the target's.
    let draft_kv_cache_quant = models::gguf::GgufKvCacheQuant::from_llama_args(
        &resolved.speculative.decode.draft_cache_type_k,
        &resolved.speculative.decode.draft_cache_type_v,
    )
    .unwrap_or(models::gguf::GgufKvCacheQuant::F16);
    let (kv_bytes_per_token, compute_bytes) = draft_context_bytes(&path, draft_kv_cache_quant);
    Some(PlannedDFlashDraft {
        draft: PlannedSpeculativeDraft {
            weight_bytes,
            recurrent_rollback_tokens,
            kv_bytes_per_token,
            compute_bytes,
        },
        default,
    })
}

/// KV bytes per token and compute bytes of the draft context at `path`. The
/// draft context has the target's context size and lanes, the draft cache
/// types, and
/// reserves logits for a full microbatch. Unreadable metadata charges nothing;
/// the attach then fails and the load falls back without a default draft.
fn draft_context_bytes(path: &Path, kv_cache_quant: models::gguf::GgufKvCacheQuant) -> (u64, u64) {
    let Some(meta) = models::gguf::scan_gguf_compact_meta(path) else {
        return (0, 0);
    };
    let kv_bytes_per_token = kv_cache_quant.kv_cache_bytes_per_token(&meta).unwrap_or(0);
    let vocab = models::gguf::scan_gguf_vocab_token_count(path).unwrap_or(0);
    let row_bytes = vocab
        .saturating_add(u64::from(meta.embedding_size))
        .saturating_mul(4);
    (
        kv_bytes_per_token,
        row_bytes.saturating_mul(DRAFT_COMPUTE_ROWS),
    )
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
        let draft = PlannedSpeculativeDraft {
            weight_bytes: 1_383_768_224,
            recurrent_rollback_tokens: 16,
            kv_bytes_per_token: 0,
            compute_bytes: 0,
        };
        let default_draft = Some(PlannedDFlashDraft {
            draft,
            default: true,
        });
        let explicit_draft = Some(PlannedDFlashDraft {
            draft,
            default: false,
        });

        let (roomy, planned) = plan_with_default_dflash_draft(input(64 << 30), default_draft);
        assert!(planned);
        assert!(roomy.breakdown.unwrap().speculative_bytes > 1_383_768_224);

        let (tight, planned) = plan_with_default_dflash_draft(input(10_500_000_000), default_draft);
        assert!(!planned);
        assert!(tight.fits());
        assert_eq!(tight.breakdown.unwrap().speculative_bytes, 0);

        // An explicit draft keeps its room and its rollback planes on the same
        // device, shortening the context instead of being dropped.
        let (explicit, planned) =
            plan_with_default_dflash_draft(input(10_500_000_000), explicit_draft);
        assert!(planned);
        assert!(explicit.breakdown.unwrap().speculative_bytes > 1_383_768_224);

        let (_, planned) = plan_with_default_dflash_draft(input(64 << 30), None);
        assert!(!planned);
    }

    /// Measured on an RTX 4070 Ti through Vulkan: with four auto lanes the
    /// target's rollback snapshots did not fit beside the 9B weights and the
    /// load failed allocating the recurrent-state cache; one lane served.
    #[test]
    fn an_explicit_dflash_draft_gives_up_auto_lanes_rather_than_overrun() {
        let meta = qwen35_9b_meta();
        let input = |parallel_override| RuntimeResourcePlanInput {
            ctx_size_override: Some(4096),
            parallel_override,
            model_bytes: 5_680_522_464,
            projector_bytes: 0,
            vram_bytes: 11_226 << 20,
            metadata: Some(&meta),
            kv_cache_quant: crate::models::gguf::GgufKvCacheQuant::F16,
            local_layer_fraction: None,
            planning_profile: RuntimeResourcePlanningProfile::DedicatedLocal,
            measured_buffers: None,
        };
        let explicit_draft = Some(PlannedDFlashDraft {
            draft: PlannedSpeculativeDraft {
                weight_bytes: 1_383_768_224,
                recurrent_rollback_tokens: 16,
                kv_bytes_per_token: 0,
                compute_bytes: 0,
            },
            default: false,
        });

        let four_lanes =
            plan_runtime_resources_with_draft(input(None), explicit_draft.map(|d| d.draft));
        assert_eq!(four_lanes.slots, 4);
        assert!(!four_lanes.fits(), "four lanes overrun the device");

        let (plan, planned) = plan_with_default_dflash_draft(input(None), explicit_draft);
        assert!(planned);
        assert!(plan.fits());
        assert!((1..4).contains(&plan.slots), "{} lanes", plan.slots);
        assert_eq!(plan.context_length, 4096);
        assert!(
            plan.breakdown.unwrap().slots_auto,
            "the planner chose the lanes"
        );

        // A lane count the operator configured is kept, even when tight.
        let (configured, planned) = plan_with_default_dflash_draft(input(Some(4)), explicit_draft);
        assert!(planned);
        assert_eq!(configured.slots, 4);
    }
}
