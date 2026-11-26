use skippy_runtime::ModelStateKind;

use crate::frontend::SpeculativeDecodeConfig;

const RECURRENT_NATIVE_MTP_MAX_DRAFT_TOKENS: usize = 3;

/// Hybrid and recurrent models can roll back a limited number of verify rows
/// in place; a wider window copies the whole recurrent state to host on every
/// window (#2318). The rollback budget the planner sizes is
/// `max(draft, verify) + 1` — the anchor plus the larger of the draft and
/// verify windows — so the draft cap alone does not determine whether a
/// window stays in place. Two draft tokens improved throughput on a
/// split-serving measurement; the actual recovery path (in-place vs
/// checkpoint/replay) was not observed. Deeper windows may fall back to the
/// checkpoint/replay path.
pub(in crate::frontend) fn native_mtp_draft_cap(
    requested: usize,
    state_kind: Option<ModelStateKind>,
) -> usize {
    match state_kind {
        Some(ModelStateKind::Recurrent | ModelStateKind::Hybrid) => {
            requested.min(RECURRENT_NATIVE_MTP_MAX_DRAFT_TOKENS)
        }
        _ => requested,
    }
}

pub(in crate::frontend) fn apply_native_mtp_draft_cap(
    speculative: &mut SpeculativeDecodeConfig,
    state_kind: Option<ModelStateKind>,
) {
    let requested = speculative.native_mtp.max_draft_tokens;
    let capped = native_mtp_draft_cap(requested, state_kind);
    if capped == requested {
        return;
    }
    speculative.native_mtp.max_draft_tokens = capped;
    speculative.native_mtp.min_draft_tokens = speculative.native_mtp.min_draft_tokens.min(capped);
    if speculative.native_mtp.enabled {
        let _ = skippy_events::diagnostics::emit(
            skippy_events::diagnostics::ServingDiagnostic::Warning {
                message: format!(
                    "native MTP draft tokens capped from {requested} to {capped}: this model has \
                     recurrent state, and more than two draft tokens may copy the whole recurrent \
                     state to host on every window; the planner sizes rollback from \
                     max(draft, verify) + 1, so the in-place budget depends on the verify \
                     window, not just the draft count (#2318)"
                ),
                context: Some(format!("model_state_kind={state_kind:?}")),
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_and_recurrent_models_cap_native_mtp_to_three_drafts() {
        for kind in [ModelStateKind::Hybrid, ModelStateKind::Recurrent] {
            assert_eq!(native_mtp_draft_cap(4, Some(kind)), 3, "{kind:?}");
            assert_eq!(native_mtp_draft_cap(3, Some(kind)), 3, "{kind:?}");
            assert_eq!(native_mtp_draft_cap(1, Some(kind)), 1, "{kind:?}");
        }
    }

    #[test]
    fn dense_models_keep_the_requested_native_mtp_depth() {
        assert_eq!(native_mtp_draft_cap(3, Some(ModelStateKind::Dense)), 3);
        assert_eq!(native_mtp_draft_cap(1, Some(ModelStateKind::Dense)), 1);
        assert_eq!(native_mtp_draft_cap(3, None), 3);
    }

    #[test]
    fn capping_a_hybrid_plan_keeps_min_within_max() {
        let mut speculative = SpeculativeDecodeConfig::default();
        speculative.native_mtp.enabled = true;
        speculative.native_mtp.max_draft_tokens = 4;
        speculative.native_mtp.min_draft_tokens = 3;

        apply_native_mtp_draft_cap(&mut speculative, Some(ModelStateKind::Hybrid));

        assert_eq!(speculative.native_mtp.max_draft_tokens, 3);
        assert_eq!(speculative.native_mtp.min_draft_tokens, 3);
        speculative.validate().expect("capped plan stays valid");
    }
}
