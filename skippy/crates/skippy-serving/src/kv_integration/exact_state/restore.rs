//! Transaction boundaries and sampler-history completion for every exact tier.
use super::*;

impl KvStageIntegration {
    pub fn restore_exact_state(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identities: &[PrefillKvIdentity],
    ) -> Result<Option<ExactStateRestore>> {
        self.restore_exact_state_with_cold_cost(runtime, session_id, identities, None)
    }

    /// Constrain a cold L3 fallback to the checkpoint shared by every stage.
    pub(crate) fn restore_exact_state_with_shared_durable_fallback(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identities: &[PrefillKvIdentity],
    ) -> Result<Option<ExactStateRestore>> {
        runtime.restore_transaction(session_id, |runtime| {
            self.restore_exact_state_inner(runtime, session_id, identities, None, true)
        })
    }

    pub fn restore_exact_state_with_cold_cost(
        &self,
        runtime: &mut RuntimeState,
        session_id: &str,
        identities: &[PrefillKvIdentity],
        cold_prefill_cost: Option<f64>,
    ) -> Result<Option<ExactStateRestore>> {
        runtime.restore_transaction(session_id, |runtime| {
            self.restore_exact_state_inner(
                runtime,
                session_id,
                identities,
                cold_prefill_cost,
                false,
            )
        })
    }
}

pub(super) fn finish_restore(
    runtime: &mut RuntimeState,
    session_id: &str,
    identity: &PrefillKvIdentity,
    restored: ExactStateRestore,
) -> Result<Option<ExactStateRestore>> {
    // A longest-prefix lookup may restore fewer tokens than the candidate.
    // Keep this inside the restore transaction: a history failure must retire
    // the imported state rather than allow a partially restored cache hit.
    let tokens = identity
        .token_ids
        .get(..restored.token_count)
        .context("restored state extends beyond its prefix token identity")?;
    runtime.restore_token_history(session_id, tokens)?;
    Ok(Some(restored))
}
