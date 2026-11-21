//! Disk metadata preparation before binary work enters the runtime scheduler.
use super::*;
use crate::kv_integration::{PreparedExactStateRestore, ResidentActivationRestore};

pub(in crate::binary_transport) struct PreparedPrefixControl {
    pub(super) kv: Arc<KvStageIntegration>,
    pub(super) exact: Option<PreparedExactStateRestore>,
}

pub(in crate::binary_transport) struct PreparedBinaryPrefill {
    pub(super) kv: Arc<KvStageIntegration>,
    pub(super) identities: Vec<PrefillKvIdentity>,
    pub(super) activation: Option<ResidentActivationRestore>,
    pub(super) exact: PreparedExactStateRestore,
}

pub(in crate::binary_transport) fn prepare_prefix_cache_control(
    config: &StageConfig,
    kv: Option<&Arc<KvStageIntegration>>,
    session_id: &str,
    message: &StageWireMessage,
    tokens: &[i32],
) -> Option<PreparedPrefixControl> {
    let kv = kv?;
    let restoring = matches!(
        message.kind,
        WireMessageKind::RestorePrefill
            | WireMessageKind::TryRestorePrefill
            | WireMessageKind::TryRestorePrefillDecode
    );
    let exact = (restoring
        && kv.should_lookup()
        && !tokens.is_empty()
        && (message.pos_start <= 0 || message.kind == WireMessageKind::TryRestorePrefillDecode))
        .then(|| {
            let base = binary_message_base(config, session_id, message);
            let identity = kv.prefill_identity(config, &base, 0, tokens);
            kv.prepare_exact_state_restore(&[identity], false)
        });
    Some(PreparedPrefixControl {
        kv: Arc::clone(kv),
        exact,
    })
}

pub(in crate::binary_transport) fn prepare_binary_prefill(
    config: &StageConfig,
    kv: Option<&Arc<KvStageIntegration>>,
    session_id: &str,
    message: &StageWireMessage,
    tokens: &[i32],
    activation_width: i32,
) -> Option<PreparedBinaryPrefill> {
    let kv = kv?;
    let start = message.pos_start.max(0) as u64;
    let base = binary_message_base(config, session_id, message);
    let identities = kv.lookup_identities(config, &base, start, tokens);
    let activation = (config.downstream.is_some() && start == 0)
        .then(|| kv.restore_resident_activation(config, &base, start, tokens, activation_width))
        .flatten();
    let candidates: &[PrefillKvIdentity] = if config.downstream.is_some() {
        activation
            .as_ref()
            .map(|hit| std::slice::from_ref(&hit.identity))
            .unwrap_or_default()
    } else if start == 0 {
        &identities
    } else {
        &[]
    };
    let exact = kv.prepare_exact_state_restore(candidates, false);
    Some(PreparedBinaryPrefill {
        kv: Arc::clone(kv),
        identities,
        activation,
        exact,
    })
}
