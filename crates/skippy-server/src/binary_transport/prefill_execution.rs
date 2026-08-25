use skippy_protocol::binary::WireMessageKind;

pub(in crate::binary_transport) fn executable_prefill_start(
    kind: WireMessageKind,
    restored_tokens: usize,
    token_count: usize,
    layer_start: u32,
    has_downstream: bool,
) -> usize {
    let partial_restore = restored_tokens > 0 && restored_tokens < token_count;
    if kind.is_prefill() && partial_restore && (layer_start == 0 || !has_downstream) {
        restored_tokens
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_non_first_stage_executes_only_suffix_after_partial_restore() {
        assert_eq!(
            executable_prefill_start(WireMessageKind::PrefillEmbd, 3, 5, 8, false),
            3
        );
    }

    #[test]
    fn intermediate_non_first_stage_preserves_full_activation_range() {
        assert_eq!(
            executable_prefill_start(WireMessageKind::PrefillEmbd, 3, 5, 8, true),
            0
        );
    }
}
