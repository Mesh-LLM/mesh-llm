//! Gateway configuration and policy controls.

use crate::backend::{ModelBackend, ModelEntry};
use std::time::Duration;

// ─── Configuration ───────────────────────────────────────────────────

/// Gateway configuration.
pub struct GatewayConfig {
    /// Available backends.  Models reference these by index.
    pub backends: Vec<std::sync::Arc<dyn ModelBackend>>,
    /// Available models for fan-out.
    pub models: Vec<ModelEntry>,
    /// Per-worker timeout.
    pub worker_timeout: Duration,
    /// Per-candidate wait before hedging a second reducer candidate. When the
    /// primary candidate is slow (e.g. cold KV) we don't want to wait the full
    /// reducer_timeout before kicking off candidate 2 — start the next one
    /// after hedge_delay and race them. Cost: up to 2× tokens for the rare
    /// slow case; zero cost on the happy path (candidate 1 returns first).
    pub hedge_delay: Duration,
    /// Reducer timeout.
    pub reducer_timeout: Duration,
    /// Chat-only grace: after this long since dispatch, if a single answer
    /// (conf >= 0.5) is in, accept it instead of waiting for consensus.
    /// Disabled for tool turns. Zero disables entirely.
    pub first_answer_grace: Duration,
    /// Tier-gate patience: when the worker pool mixes a big-tier Strong
    /// worker with small-tier workers, small-tier-only answers and
    /// consensus are held for up to this long after dispatch to give the
    /// strong worker a chance to weigh in. A hard bound — once it lapses,
    /// all decision rules revert to ungated behavior, so a stuck strong
    /// worker can never hold the turn hostage. Zero disables the gate.
    /// Has no effect when all workers are the same tier. Tool proposals
    /// are never held.
    pub strong_patience: Duration,
    /// Override for whether reasoning workers should think. Propagated to
    /// every worker and the reducer as `chat_template_kwargs.enable_thinking`
    /// (and `reasoning_effort: "none"` when disabled).
    ///
    /// `None` (the default) leaves each model's default behavior alone —
    /// existing callers see no behavior change. The MoA HTTP gateway
    /// populates this from the caller's `reasoning_effort` / `enable_thinking`
    /// / `reasoning.enabled` knobs so MoA users get a single switch.
    pub enable_thinking: Option<bool>,
    /// Actor priority order for tool turns and synthesis, as indices into
    /// [`Self::models`], best actor first.
    ///
    /// In the asymmetric (Hermes-style) tool path the *actor* is the model
    /// that actually emits the tool call; references only advise. The actor
    /// should be the best available tool-caller, which is a host-side judgement
    /// combining gossiped `tool_use` capability, model size, and recent peer
    /// health — signals the engine crate cannot see. The host computes the
    /// ordering and passes it here.
    ///
    /// Empty (the default) means "no host guidance": the engine falls back to
    /// its name-derived size tier (big-tier first), preserving prior behaviour
    /// for tests and any caller that doesn't populate it.
    pub actor_candidates: Vec<usize>,
    /// Whether tool turns gather advisory references before the actor acts.
    pub reference_policy: ReferencePolicy,
    /// Whether text turns run a cross-peer refinement round before synthesis.
    pub refinement_policy: RefinementPolicy,
}

/// When a text turn should run a cross-peer refinement round (Together's
/// `layers`): every worker sees all round-1 drafts and rewrites its answer
/// before the reducer synthesizes.
///
/// Measured over 40 preregistered reasoning prompts x 3 draws
/// (`evals/moa-openrouter/RESULTS.md`): a pool of four 8B-class models beat its
/// own best member **only** with this round — single-round synthesis was
/// indistinguishable from the aggregator acting solo (26/75/19, p=0.37) while
/// refine-then-synthesize won (42/66/12, p=5.2e-05). With a 32B aggregator the
/// round adds much less over single-round synthesis (p=0.015), so it is not
/// worth a second fan-out there.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum RefinementPolicy {
    /// Refine when the pool is all small-tier — the case where the round is
    /// what makes the collective beat its best member.
    #[default]
    Auto,
    /// Always run the refinement round.
    Always,
    /// Never refine: synthesize the round-1 drafts directly.
    Never,
}

/// When the asymmetric tool path should gather advisory references.
///
/// Measured on 40 preregistered tool tasks x 10 draws (see
/// `evals/moa-openrouter/RESULTS.md`): with correct advisor packing, references
/// are worth +0.017 net uplift to a weak actor but -0.037 to a strong one. They
/// help where the actor has headroom and cost where it is already reliable, so
/// the useful default is to gate on actor strength rather than always or never.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ReferencePolicy {
    /// Gather references only when the acting model looks weak enough to
    /// benefit. Cheapest correct default.
    #[default]
    Auto,
    /// Always gather references, regardless of actor strength.
    Always,
    /// Never gather references: the actor acts alone (Hermes' `enabled: false`).
    Never,
}
