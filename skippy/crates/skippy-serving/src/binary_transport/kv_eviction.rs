use std::{collections::BTreeMap, sync::Arc};

use anyhow::{Context, Result};
use serde_json::Value;
use skippy_protocol::binary::WireMessageKind;

use crate::{
    kv_integration::{KvStageIntegration, proactive_eviction_attrs},
    runtime_state::RuntimeState,
};

#[derive(Debug, Clone)]
pub(in crate::binary_transport) struct BinaryProactiveEviction {
    status: &'static str,
    error_kind: Option<&'static str>,
    target_tokens: u64,
    evicted_entries: usize,
    evicted_tokens: u64,
}

impl BinaryProactiveEviction {
    fn disabled() -> Self {
        Self {
            status: "disabled",
            error_kind: None,
            target_tokens: 0,
            evicted_entries: 0,
            evicted_tokens: 0,
        }
    }

    pub(in crate::binary_transport) fn attrs(&self) -> BTreeMap<String, Value> {
        proactive_eviction_attrs(
            self.status,
            self.error_kind,
            self.target_tokens,
            self.evicted_entries,
            self.evicted_tokens,
        )
    }

    pub(in crate::binary_transport) fn insert_attrs(&self, attrs: &mut BTreeMap<String, Value>) {
        attrs.extend(self.attrs());
    }

    pub(super) fn should_emit_summary(&self) -> bool {
        self.error_kind.is_some() || self.evicted_entries > 0 || self.evicted_tokens > 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::binary_transport) struct BinaryProactiveEvictionPlan {
    pub(in crate::binary_transport) required: bool,
    pub(in crate::binary_transport) ensure_session_before_eviction: bool,
    pub(in crate::binary_transport) target_tokens: Option<u64>,
}

pub(super) fn binary_proactive_eviction_plan(
    kind: WireMessageKind,
    pos_start: i32,
    restored_prefill: bool,
    executable_token_count: usize,
    remaining_prefill_tokens: usize,
) -> BinaryProactiveEvictionPlan {
    let required =
        binary_proactive_eviction_required(kind, restored_prefill, executable_token_count);
    BinaryProactiveEvictionPlan {
        required,
        ensure_session_before_eviction: required
            && (kind.is_prefill() || (kind == WireMessageKind::DecodeEmbd && pos_start == 0)),
        target_tokens: kind.is_prefill().then(|| {
            u64::try_from(remaining_prefill_tokens.max(executable_token_count)).unwrap_or(u64::MAX)
        }),
    }
}

pub(super) fn binary_proactive_eviction_required(
    kind: WireMessageKind,
    restored_prefill: bool,
    executable_token_count: usize,
) -> bool {
    !restored_prefill
        && executable_token_count > 0
        && matches!(
            kind,
            WireMessageKind::PrefillEmbd
                | WireMessageKind::PrefillFinalEmbd
                | WireMessageKind::DecodeEmbd
                | WireMessageKind::DecodeReplayEmbd
                | WireMessageKind::DecodeReplayFinalEmbd
                | WireMessageKind::DecodeReadout
                | WireMessageKind::DecodeLightCtx
                | WireMessageKind::VerifyWindow
        )
}

pub(in crate::binary_transport) fn evict_binary_resident_prefix_for_decode(
    runtime: &mut RuntimeState,
    kv: Option<&Arc<KvStageIntegration>>,
    session_id: &str,
    plan: BinaryProactiveEvictionPlan,
) -> Result<BinaryProactiveEviction> {
    let Some(kv) = kv else {
        return Ok(BinaryProactiveEviction::disabled());
    };
    if plan.ensure_session_before_eviction {
        // Prefill chunks and the first decode of a one-token prompt can reach
        // eviction before execution has activated a runtime session. Eviction
        // needs that session for native resident-prefix sequence drops and
        // decode-batch discovery.
        runtime.ensure_session_active(session_id).with_context(|| {
            format!("activate binary session {session_id} before resident-prefix eviction")
        })?;
    }
    let eviction = (if let Some(target_tokens) = plan.target_tokens {
        kv.evict_resident_prefix_for_tokens(runtime, session_id, target_tokens)
    } else {
        kv.evict_resident_prefix_for_decode_batch(runtime, session_id)
    })
    .with_context(|| {
        format!("evict resident-prefix KV before binary execution for session {session_id}")
    })?;
    Ok(BinaryProactiveEviction {
        status: if eviction.evicted_entries > 0 {
            "evicted"
        } else {
            "noop"
        },
        error_kind: None,
        target_tokens: eviction.target_tokens,
        evicted_entries: eviction.evicted_entries,
        evicted_tokens: eviction.evicted_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    type BinaryEvictionFn = fn(
        &mut RuntimeState,
        Option<&std::sync::Arc<KvStageIntegration>>,
        &str,
        BinaryProactiveEvictionPlan,
    ) -> anyhow::Result<BinaryProactiveEviction>;
    #[test]
    fn binary_decode_work_requires_proactive_resident_eviction() {
        assert!(
            binary_proactive_eviction_plan(WireMessageKind::PrefillFinalEmbd, 0, false, 128, 4096)
                .required
        );
        assert!(
            binary_proactive_eviction_plan(WireMessageKind::DecodeEmbd, 1, false, 1, 0).required
        );
        assert!(
            binary_proactive_eviction_plan(WireMessageKind::DecodeReplayEmbd, 1, false, 64, 0)
                .required
        );
        let prefill =
            binary_proactive_eviction_plan(WireMessageKind::PrefillEmbd, 0, false, 128, 2048);
        assert!(prefill.required);
        assert!(prefill.ensure_session_before_eviction);
        assert_eq!(prefill.target_tokens, Some(2048));
        assert!(
            !binary_proactive_eviction_plan(WireMessageKind::DecodeEmbd, 1, true, 1, 0).required
        );
        assert!(
            !binary_proactive_eviction_plan(WireMessageKind::DecodeEmbd, 1, false, 0, 0).required
        );
        assert!(
            !binary_proactive_eviction_plan(
                WireMessageKind::TryRestorePrefillDecode,
                0,
                false,
                1,
                0
            )
            .required
        );
    }

    #[test]
    fn one_chunk_prefill_final_admits_session_before_proactive_eviction() {
        let plan =
            binary_proactive_eviction_plan(WireMessageKind::PrefillFinalEmbd, 0, false, 1, 1);

        assert!(plan.required);
        assert!(plan.ensure_session_before_eviction);
    }

    #[test]
    fn single_token_prompt_decode_admits_session_before_proactive_eviction() {
        // A fresh one-token prompt sends no prefill frame. Its first downstream
        // work is DecodeEmbd at position zero, so eviction must admit the lane.
        let plan = binary_proactive_eviction_plan(WireMessageKind::DecodeEmbd, 0, false, 1, 1);

        assert!(plan.required);
        assert!(plan.ensure_session_before_eviction);
        assert_eq!(plan.target_tokens, None);
    }

    #[test]
    fn decode_admission_is_limited_to_executable_zero_position_decode() {
        for (pos_start, restored_prefill, executable_tokens) in [
            (1, false, 1),
            (128, false, 1),
            (-1, false, 1),
            (0, true, 1),
            (0, false, 0),
        ] {
            let plan = binary_proactive_eviction_plan(
                WireMessageKind::DecodeEmbd,
                pos_start,
                restored_prefill,
                executable_tokens,
                0,
            );
            assert!(!plan.ensure_session_before_eviction);
        }
        for kind in [
            WireMessageKind::DecodeReplayEmbd,
            WireMessageKind::DecodeReplayFinalEmbd,
            WireMessageKind::DecodeReadout,
            WireMessageKind::DecodeLightCtx,
            WireMessageKind::VerifyWindow,
        ] {
            let plan = binary_proactive_eviction_plan(kind, 0, false, 1, 0);
            assert!(plan.required);
            assert!(!plan.ensure_session_before_eviction);
        }
        let continued_prefill =
            binary_proactive_eviction_plan(WireMessageKind::PrefillEmbd, 128, false, 1, 16);
        assert!(continued_prefill.ensure_session_before_eviction);
        assert_eq!(continued_prefill.target_tokens, Some(16));
    }

    #[test]
    fn required_binary_proactive_eviction_is_fallible_before_decode() {
        fn accepts_fallible_eviction(_evict: BinaryEvictionFn) {}

        accepts_fallible_eviction(evict_binary_resident_prefix_for_decode);
    }

    #[test]
    fn disabled_and_noop_evictions_are_debug_only() {
        assert!(!BinaryProactiveEviction::disabled().should_emit_summary());
        assert!(
            !BinaryProactiveEviction {
                status: "noop",
                error_kind: None,
                target_tokens: 1024,
                evicted_entries: 0,
                evicted_tokens: 0,
            }
            .should_emit_summary()
        );
    }

    #[test]
    fn actionable_evictions_stay_summary_visible() {
        assert!(
            BinaryProactiveEviction {
                status: "evicted",
                error_kind: None,
                target_tokens: 1024,
                evicted_entries: 1,
                evicted_tokens: 512,
            }
            .should_emit_summary()
        );
        assert!(
            BinaryProactiveEviction {
                status: "error",
                error_kind: Some("runtime"),
                target_tokens: 1024,
                evicted_entries: 0,
                evicted_tokens: 0,
            }
            .should_emit_summary()
        );
    }
}
