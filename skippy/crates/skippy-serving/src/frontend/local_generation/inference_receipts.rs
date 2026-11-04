//! Inference receipts for a turn this stage served alone
//! ([`crate::receipts`]). Not the generation receipts of
//! `generation_receipt`, which record what the frontend emitted.
//!
//! The receipt is committed after the turn and before its session is
//! cleaned up, while the recorder still maps the session to its sequence.
//! Only a turn that asks for a bound receipt is committed: a single stage
//! serves openings only on its OpenAI address, which opens bound receipts
//! alone, so an unbound one would be of no use. Other turns take no runtime
//! lock here.

use crate::frontend::generation::{LocalGeneration, StageOpenAiBackend};

impl StageOpenAiBackend {
    /// Commits and returns the turn's receipt, when the stage issues receipts
    /// and the request asked for a bound one. `final_token` is the last
    /// sampled token, which no ubatch computed, of the `sampled` tokens.
    pub(super) fn finish_local_receipt_turn(
        &self,
        request: &LocalGeneration<'_>,
        session: &str,
        final_token: Option<i32>,
        sampled: u32,
    ) -> Option<serde_json::Value> {
        let bound = request.ids.receipt?;
        crate::receipts::finish_stage_zero_turn(
            &self.runtime,
            request.ids.frontend_request_id,
            Some(bound),
            request.ids.request_id,
            session,
            crate::receipts::TurnTokens {
                generation_start: u32::try_from(request.prompt_token_ids.len()).unwrap_or(u32::MAX),
                final_token,
                sampled,
                sampler: crate::receipts::sampler_statement(
                    request.sampling,
                    request.chat_sampling_metadata,
                ),
            },
        )
    }
}
