//! Inference receipts for the turns stage 0 serves ([`crate::receipts`]):
//! what the generation paths call to commit a turn's receipt, and what a
//! stage that issues receipts does differently while it generates. Not the
//! generation receipts of `generation_receipt`, which record what the
//! frontend emitted.
//!
//! The receipt is committed after the turn and before its session is
//! cleaned up, while the session still holds its native sequence. A turn
//! this stage serves alone whose receipt is committed here therefore does
//! not run on the scheduler's built-in driver, which resets the session and
//! hands its lane to the next request as the sequence ends; it runs on the
//! feature loop, whose session lives until the request cleans it up. Were a
//! commit to come late all the same, it names its turn, and the recorder
//! gives it that turn's leaves or none
//! ([`skippy_receipts::recorder::TurnId`]).
//!
//! Of the turns this stage serves alone, only one that asks for a bound
//! receipt is committed: a single stage serves openings only on its OpenAI
//! address, which opens bound receipts alone, so an unbound one would be of
//! no use. Other such turns take no runtime lock here. A turn of a split is
//! committed either way.

use std::cell::Cell;

use crate::frontend::generation::{
    EmbeddedStageZeroGeneration, InferenceBackendMode, InferenceGenerationIds, LocalGeneration,
    StageOpenAiBackend,
};
use crate::receipts::TurnTokens;
use crate::runtime_state::panic_recovery::lock_runtime;

/// The tokens a turn sampled, as its receipt needs them: a receipt names
/// the last sampled token, which no ubatch computes, and ends at the last
/// token fed.
#[derive(Default)]
pub(super) struct SampledTokens {
    /// The last sampled token.
    last: Cell<Option<i32>>,
    /// How many tokens the turn sampled, the last one included.
    count: Cell<u32>,
}

impl SampledTokens {
    /// The turn sampled `token`.
    pub(super) fn note(&self, token: i32) {
        self.last.set(Some(token));
        self.count.set(self.count.get().saturating_add(1));
    }

    /// `on_token`, noting every token it is given first.
    pub(super) fn observing<'a, R>(
        &'a self,
        mut on_token: impl FnMut(i32) -> R + 'a,
    ) -> impl FnMut(i32) -> R + 'a {
        move |token| {
            self.note(token);
            on_token(token)
        }
    }

    /// What stage 0 knows about the tokens of a turn whose prompt had
    /// `prompt_tokens` tokens.
    fn of_turn(
        &self,
        prompt_tokens: usize,
        sampling: &skippy_runtime::SamplingConfig,
        chat_sampling_metadata: Option<&str>,
    ) -> TurnTokens {
        TurnTokens {
            generation_start: u32::try_from(prompt_tokens).unwrap_or(u32::MAX),
            final_token: self.last.get(),
            sampled: self.count.get(),
            sampler: crate::receipts::sampler_statement(sampling, chat_sampling_metadata),
        }
    }
}

impl StageOpenAiBackend {
    /// Whether [`Self::finish_local_receipt_turn`] commits a receipt for
    /// the request: it asked for a bound one, and the stage issues
    /// receipts.
    pub(super) fn commits_local_receipt(&self, request: &LocalGeneration<'_>) -> bool {
        request.ids.receipt.is_some() && lock_runtime(&self.runtime).receipts().is_some()
    }

    /// Commits and returns the receipt of a turn this stage served alone,
    /// when the stage issues receipts and the request asked for a bound one.
    /// `sampled` holds the tokens the turn sampled. Call it before the
    /// turn's session is cleaned up.
    pub(super) fn finish_local_receipt_turn(
        &self,
        request: &LocalGeneration<'_>,
        session: &str,
        sampled: &SampledTokens,
    ) -> Option<serde_json::Value> {
        let bound = request.ids.receipt?;
        crate::receipts::finish_stage_zero_turn(
            &self.runtime,
            request.ids.frontend_request_id,
            Some(bound),
            request.ids.request_id,
            session,
            sampled.of_turn(
                request.prompt_token_ids.len(),
                request.sampling,
                request.chat_sampling_metadata,
            ),
        )
    }

    /// Commits the receipt of a turn of a split on every stage, when the
    /// stage issues receipts, and returns it when the request asked for a
    /// bound one. `sampled` holds the tokens the turn sampled. Call it
    /// before the turn's session is cleaned up.
    pub(super) fn finish_split_receipt_turn(
        &self,
        request: &EmbeddedStageZeroGeneration<'_>,
        session: &str,
        sampled: &SampledTokens,
    ) -> Option<serde_json::Value> {
        crate::receipts::finish_stage_zero_turn(
            &self.runtime,
            request.ids.frontend_request_id,
            request.ids.receipt,
            request.ids.request_id,
            session,
            sampled.of_turn(
                request.prompt_token_ids.len(),
                request.sampling,
                request.chat_sampling_metadata,
            ),
        )
    }

    /// Gives a sampled turn that will get a receipt a draw its client can
    /// replay ([`crate::receipts::make_draw_replayable`]). A turn this stage
    /// serves alone gets a receipt when the request asks for a bound one,
    /// and a turn of a split either way. A greedy turn takes no runtime lock
    /// here.
    pub(super) fn prepare_receipt_draw(
        &self,
        sampling: &mut skippy_runtime::SamplingConfig,
        ids: &InferenceGenerationIds,
        chat_sampling_metadata: Option<&str>,
    ) {
        let asked = match self.mode {
            InferenceBackendMode::LocalRuntime => ids.receipt.is_some(),
            InferenceBackendMode::EmbeddedStageZero { .. } => true,
        };
        if !asked || !crate::receipts::draws_from_the_chain(sampling, chat_sampling_metadata) {
            return;
        }
        if lock_runtime(&self.runtime).receipts().is_none() {
            return;
        }
        let client_seed = ids.receipt.and_then(|receipt| receipt.seed);
        crate::receipts::make_draw_replayable(sampling, chat_sampling_metadata, client_seed);
    }

    /// A stage that issues receipts does not reuse tokens an earlier turn
    /// generated. Those tokens would sit in the turn's cached prefix, and an
    /// audit checks no sampled token there: a prefix holds only tokens a
    /// request fed, and a turn whose prefix is longer than its prompt gets
    /// no receipt (`StageReceipts::commit_turn`).
    pub(super) fn receipts_forbid_generated_token_reuse(&self) -> bool {
        lock_runtime(&self.runtime).receipts().is_some()
    }
}
