use skippy_runtime::ModelStateKind;

use crate::frontend::SpeculativeDecodeConfig;

const RECURRENT_NATIVE_MTP_MAX_DRAFT_TOKENS: usize = 1;

/// Hybrid and recurrent models roll back at most two verify rows in place, so
/// a wider native MTP window copies the whole recurrent state to host on every
/// window (#2318). Cap them to one draft until rollback depth follows the
/// speculative window.
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
                     recurrent state, and more than one draft token is about 7x slower on hybrid \
                     and recurrent models until the rollback depth is sized from the speculative \
                     window (#2318)"
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
    fn hybrid_and_recurrent_models_cap_native_mtp_to_one_draft() {
        for kind in [ModelStateKind::Hybrid, ModelStateKind::Recurrent] {
            assert_eq!(native_mtp_draft_cap(3, Some(kind)), 1, "{kind:?}");
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
        speculative.native_mtp.max_draft_tokens = 3;
        speculative.native_mtp.min_draft_tokens = 2;

        apply_native_mtp_draft_cap(&mut speculative, Some(ModelStateKind::Hybrid));

        assert_eq!(speculative.native_mtp.max_draft_tokens, 1);
        assert_eq!(speculative.native_mtp.min_draft_tokens, 1);
        speculative.validate().expect("capped plan stays valid");
    }
}
