use std::{collections::BTreeMap, sync::Arc};

use anyhow::{Context, Result};
use serde_json::Value;
use skippy_protocol::binary::WireMessageKind;

use crate::{
    kv_integration::{KvStageIntegration, proactive_eviction_attrs},
    runtime_state::RuntimeState,
};

#[derive(Debug, Clone)]
pub(super) struct BinaryProactiveEviction {
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

    pub(super) fn attrs(&self) -> BTreeMap<String, Value> {
        proactive_eviction_attrs(
            self.status,
            self.error_kind,
            self.target_tokens,
            self.evicted_entries,
            self.evicted_tokens,
        )
    }

    pub(super) fn insert_attrs(&self, attrs: &mut BTreeMap<String, Value>) {
        attrs.extend(self.attrs());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BinaryProactiveEvictionPlan {
    pub(super) required: bool,
    pub(super) ensure_session_before_eviction: bool,
}

pub(super) fn binary_proactive_eviction_plan(
    kind: WireMessageKind,
    restored_prefill: bool,
    executable_token_count: usize,
) -> BinaryProactiveEvictionPlan {
    let required =
        binary_proactive_eviction_required(kind, restored_prefill, executable_token_count);
    BinaryProactiveEvictionPlan {
        required,
        ensure_session_before_eviction: required && kind == WireMessageKind::PrefillFinalEmbd,
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
            WireMessageKind::PrefillFinalEmbd
                | WireMessageKind::DecodeEmbd
                | WireMessageKind::DecodeReplayEmbd
                | WireMessageKind::DecodeReplayFinalEmbd
                | WireMessageKind::DecodeReadout
                | WireMessageKind::DecodeLightCtx
                | WireMessageKind::VerifySpan
        )
}

pub(super) fn evict_binary_resident_prefix_for_decode(
    runtime: &mut RuntimeState,
    kv: Option<&Arc<KvStageIntegration>>,
    session_id: &str,
    plan: BinaryProactiveEvictionPlan,
) -> Result<BinaryProactiveEviction> {
    let Some(kv) = kv else {
        return Ok(BinaryProactiveEviction::disabled());
    };
    if plan.ensure_session_before_eviction {
        // One-chunk final-prefill can reach eviction before the prefill call
        // has activated a runtime session. Eviction needs that session for
        // both n_batch discovery and native resident-prefix sequence drops.
        runtime.ensure_session_active(session_id).with_context(|| {
            format!("activate binary session {session_id} before resident-prefix eviction")
        })?;
    }
    let eviction = kv
        .evict_resident_prefix_for_decode_batch(runtime, session_id)
        .with_context(|| {
            format!("evict resident-prefix KV before binary decode for session {session_id}")
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
