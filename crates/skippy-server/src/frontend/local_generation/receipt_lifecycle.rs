//! Generation-receipt begin/commit bookkeeping for one local token-generation
//! call.
//!
//! Finalization (`finalize_generation_receipt`) lives in the parent
//! `local_generation` module; this module owns the start hook:
//! [`begin_generation_receipt`] marks a generation as started once, before
//! decode begins. Canonical tokens are recorded by
//! `generation_commit_batcher::GenerationCommitBatcher`, which the decode
//! loop owns directly so it can batch them to the progress cadence.

use std::sync::Arc;

use crate::frontend::generation::OpenAiGenerationIds;
use crate::frontend::generation_receipt::GenerationStart;

/// Marks a generation as started with the receipt sink, when a receipt
/// config is configured. Returns the prompt token ids captured for the
/// receipt (`None` when no receipt config is present); the caller carries
/// this through to `finalize_generation_receipt`.
pub(super) fn begin_generation_receipt(
    config: Option<&crate::frontend::GenerationReceiptConfig>,
    ids: &OpenAiGenerationIds,
    prompt_token_ids: &[i32],
) -> Option<Arc<[i32]>> {
    let receipt_prompt_token_ids = config.map(|_| Arc::<[i32]>::from(prompt_token_ids));
    if let Some(config) = config {
        config.begin(GenerationStart {
            request_id: ids.request_id,
            session_id: ids.session_id,
            agent_session_id: ids.agent_session_id.clone(),
            prompt_token_ids: Arc::clone(
                receipt_prompt_token_ids
                    .as_ref()
                    .expect("receipt prompt exists when receipt config exists"),
            ),
            frontend_request_id: ids.frontend_request_id,
        });
    }
    receipt_prompt_token_ids
}
