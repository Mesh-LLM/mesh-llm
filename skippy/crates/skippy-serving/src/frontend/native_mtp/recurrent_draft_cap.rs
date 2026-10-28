use skippy_runtime::ModelStateKind;

use crate::frontend::SpeculativeDecodeConfig;

/// Effective native MTP draft depth for the loaded model.
pub(in crate::frontend) fn native_mtp_draft_cap(
    requested: usize,
    _state_kind: Option<ModelStateKind>,
) -> usize {
    requested
}

/// Applies [`native_mtp_draft_cap`] to a speculative plan.
pub(in crate::frontend) fn apply_native_mtp_draft_cap(
    _speculative: &mut SpeculativeDecodeConfig,
    _state_kind: Option<ModelStateKind>,
) {
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
