//! Asymmetric tool turn: the best tool-caller (the "actor") acts with the real
//! tools; every other model advises tool-free. Tool authority tracks capability,
//! not a majority vote — which removes the majority-of-weakness failure class
//! (weak models outvoting the one strong tool-caller) by construction.
//!
//! Stays a stateless `/v1/chat/completions` turn: references are rebuilt from
//! the transcript each request, and the client still owns tool execution.

use crate::backend::{SamplingParams, call_backend};
use crate::context;
use crate::fanout::{DispatchedWorker, gather_references};
use crate::normalize::{self, WorkerOutput};
use crate::reducer::{self, hedged_reducer_call, reducer_candidates};
use crate::session::Session;
use crate::worker::{self, WorkerRole};
use crate::{
    ForcedToolChoice, GatewayConfig, MOA_ERR_ALL_REDUCERS_FAILED, ReferencePolicy, TurnKind,
    TurnResult, WorkerSummary, chat_response, enforce_tool_call_contract, error_response,
    fallback_worker_response, tool_call_response, tool_names_for_turn, tool_proposal_response,
};
use serde_json::Value;
use std::time::Instant;

/// How much conversation prose advisors see. Enough for continuity on a
/// multi-turn session, short enough to stay cheap and cacheable.
const REFERENCE_HISTORY_MESSAGES: usize = 6;

/// Handle a fresh, tool-bearing query with the asymmetric actor design.
pub(crate) async fn handle_tool_query(
    config: &GatewayConfig,
    session: &Session,
    allowed_tools: &[String],
    forced_tool: Option<&ForcedToolChoice>,
    start: Instant,
) -> TurnResult {
    // Best tool-caller first (host `actor_candidates`, else name-derived tier).
    let candidates = reducer_candidates(config);
    let actor_top = candidates.first().map(|(name, _)| name.clone());

    // Advisors are gathered only when they're likely to pay for themselves.
    // Actor excluded from advisors so it doesn't pay a redundant advisory pass.
    let (references, mut summaries) = if should_gather_references(config, actor_top.as_deref()) {
        dispatch_and_gather_references(config, session, actor_top.as_deref()).await
    } else {
        tracing::debug!(
            "moa: skipping advisory references (policy={:?}, actor={:?})",
            config.reference_policy,
            actor_top,
        );
        (Vec::new(), Vec::new())
    };

    let selected = forced_tool.map_or_else(
        || tool_names_for_turn(session, allowed_tools),
        |tool| vec![tool.name.clone()],
    );
    let (messages, tools) = context::pack_for_actor(session, &references, true, &selected);

    let hedge = hedged_reducer_call(
        &config.backends,
        candidates.clone(),
        messages,
        tools,
        config.reducer_timeout,
        config.hedge_delay,
        config.enable_thinking,
    )
    .await;

    let fallback_name = actor_top.unwrap_or_default();
    let (response_body, actor_name, actor_ok, attempts) = finalize_actor_output(
        session,
        allowed_tools,
        forced_tool,
        &references,
        fallback_name,
        hedge,
    );

    // Reducer role marks the acting pass, distinct from advisory summaries.
    summaries.push(WorkerSummary {
        model: actor_name,
        role: WorkerRole::Reducer,
        succeeded: actor_ok,
        elapsed_ms: start.elapsed().as_millis() as u64,
        output_kind: None,
        confidence: None,
    });

    TurnResult {
        response_body,
        worker_summaries: summaries,
        reducer_used: true,
        reducer_attempts: attempts,
        turn_kind: TurnKind::Fanout,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

/// Should this tool turn gather advisory references?
///
/// Under [`ReferencePolicy::Auto`] the answer tracks *actor headroom*. Measured
/// over 40 preregistered tool tasks x 10 draws with correct advisor packing
/// (`evals/moa-openrouter/RESULTS.md`):
///
/// * weak actor (qwen3-8b): +0.017 net uplift — references help
/// * strong actor (qwen3-32b): -0.037 net uplift — references cost
///
/// Per-stratum the split is sharper still: references gained where the actor
/// had headroom (search +10, execute +4) and lost where it was already perfect
/// (inspect -7). So we advise a small-tier actor and let a big-tier one act
/// alone.
///
/// Size tier is a coarse proxy for tool-calling strength; it is the same signal
/// the host already ranks actors by, and it needs no extra round trip. If a
/// pool has only one model there is nobody to advise, so we skip regardless.
fn should_gather_references(config: &GatewayConfig, actor: Option<&str>) -> bool {
    if config.models.len() < 2 {
        return false;
    }
    match config.reference_policy {
        ReferencePolicy::Never => false,
        ReferencePolicy::Always => true,
        // Unknown actor: fall back to advising, which is the prior behaviour.
        ReferencePolicy::Auto => actor.is_none_or(worker::model_name_is_small_tier),
    }
}

/// Fan out every non-actor model as a tool-free advisor and collect their
/// advice within a bounded window.
async fn dispatch_and_gather_references(
    config: &GatewayConfig,
    session: &Session,
    exclude: Option<&str>,
) -> (Vec<WorkerOutput>, Vec<WorkerSummary>) {
    let assignments = worker::assign_roles(&config.models);
    let mut join_set = tokio::task::JoinSet::new();
    let mut dispatched: Vec<DispatchedWorker> = Vec::new();
    let enable_thinking = config.enable_thinking;

    for a in &assignments {
        if Some(a.model_name.as_str()) == exclude {
            continue; // the actor advises itself when it acts
        }
        // Advisor packing: conversation prose only, no agent system prompt, no
        // tool transcript, no request for a tool call. Measured: the old
        // worker packing cost -0.102 net uplift vs -0.037 for this one (same
        // actor, same 40 tasks) — see evals/moa-openrouter/RESULTS.md.
        let packed = context::pack_for_reference(session, REFERENCE_HISTORY_MESSAGES);
        let model_name = a.model_name.clone();
        let role = a.role;
        let backend = config.backends[a.backend_index].clone();
        let timeout = config.worker_timeout;

        dispatched.push(DispatchedWorker {
            model: model_name.clone(),
            role,
        });

        join_set.spawn(async move {
            let t0 = Instant::now();
            let result = call_backend(
                &*backend,
                &model_name,
                &packed.messages,
                packed.tools.as_ref(),
                packed.max_tokens,
                timeout,
                SamplingParams::worker().with_thinking(enable_thinking),
            )
            .await;
            (model_name, role, result, t0.elapsed().as_millis() as u64)
        });
    }

    if dispatched.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Bounded wait: proceed at a majority of advisors so slow/absent peers on a
    // mixed mesh can't hold up the actor.
    let min_references = dispatched.len().div_ceil(2).max(1);
    gather_references(
        &mut join_set,
        &dispatched,
        config.worker_timeout,
        min_references,
    )
    .await
}

/// Turn the actor's hedged result into a response body + accounting.
fn finalize_actor_output(
    session: &Session,
    allowed_tools: &[String],
    forced_tool: Option<&ForcedToolChoice>,
    references: &[WorkerOutput],
    fallback_actor_name: String,
    hedge: Result<reducer::HedgedReducerOk, reducer::HedgedReducerErr>,
) -> (Value, String, bool, u32) {
    match hedge {
        Ok(reducer::HedgedReducerOk {
            winner,
            text,
            attempts,
        }) => {
            let mut acted =
                normalize::normalize_worker_output(&text, &winner, WorkerRole::Reducer, 0);
            enforce_tool_call_contract(&mut acted, allowed_tools, session.tools(), &winner);
            (
                actor_body(&acted, forced_tool, references),
                winner,
                true,
                attempts,
            )
        }
        Err(reducer::HedgedReducerErr { err, attempts }) => {
            tracing::warn!("moa: all {attempts} actor candidate(s) failed: {err}");
            let body = if let Some(t) = forced_tool {
                // A forced tool call is honoured even if the actor died.
                tool_call_response(&t.name, &t.fallback_arguments)
            } else if !references.is_empty() {
                // Degrade to the best advisory answer rather than fail outright.
                fallback_worker_response(references)
            } else {
                error_response(
                    &format!("Actor failed (tried {attempts}): {err}"),
                    MOA_ERR_ALL_REDUCERS_FAILED,
                )
            };
            (body, fallback_actor_name, false, attempts)
        }
    }
}

/// Map the actor's classified output to an OpenAI response body.
fn actor_body(
    acted: &WorkerOutput,
    forced_tool: Option<&ForcedToolChoice>,
    references: &[WorkerOutput],
) -> Value {
    match acted.kind {
        // The whole point: the actor emits the executable tool call.
        normalize::OutputKind::ToolProposal => tool_proposal_response(acted, true),
        normalize::OutputKind::Uncertainty => match forced_tool {
            Some(t) => tool_call_response(&t.name, &t.fallback_arguments),
            None => fallback_worker_response(references),
        },
        // Actor chose to answer directly (tool available but not needed).
        _ => match forced_tool {
            Some(t) => tool_call_response(&t.name, &t.fallback_arguments),
            None => chat_response(&acted.payload),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::ModelEntry;
    use std::time::Duration;

    fn config_with(models: &[&str], policy: ReferencePolicy) -> GatewayConfig {
        GatewayConfig {
            backends: Vec::new(),
            models: models
                .iter()
                .map(|n| ModelEntry {
                    name: (*n).to_string(),
                    backend_index: 0,
                })
                .collect(),
            worker_timeout: Duration::from_secs(60),
            hedge_delay: Duration::from_secs(5),
            reducer_timeout: Duration::from_secs(60),
            first_answer_grace: Duration::ZERO,
            strong_patience: Duration::ZERO,
            enable_thinking: Some(false),
            actor_candidates: Vec::new(),
            reference_policy: policy,
            refinement_policy: Default::default(),
        }
    }

    const POOL: &[&str] = &["Qwen3-32B", "Qwen3-8B", "Ministral-8B"];

    /// A strong actor is measurably worse with advice (-0.037 net uplift), so
    /// Auto must let it act alone.
    #[test]
    fn auto_skips_references_for_a_big_tier_actor() {
        let config = config_with(POOL, ReferencePolicy::Auto);
        assert!(!should_gather_references(&config, Some("Qwen3-32B")));
    }

    /// A weak actor has headroom advice can fill (+0.017), so Auto advises it.
    #[test]
    fn auto_gathers_references_for_a_small_tier_actor() {
        let config = config_with(POOL, ReferencePolicy::Auto);
        assert!(should_gather_references(&config, Some("Qwen3-8B")));
    }

    /// Unknown actor keeps the prior behaviour rather than silently degrading.
    #[test]
    fn auto_advises_when_the_actor_is_unknown() {
        let config = config_with(POOL, ReferencePolicy::Auto);
        assert!(should_gather_references(&config, None));
    }

    #[test]
    fn explicit_policies_override_actor_strength() {
        let never = config_with(POOL, ReferencePolicy::Never);
        assert!(!should_gather_references(&never, Some("Qwen3-8B")));

        let always = config_with(POOL, ReferencePolicy::Always);
        assert!(should_gather_references(&always, Some("Qwen3-32B")));
    }

    /// Nobody left to advise once the actor is excluded.
    #[test]
    fn a_single_model_pool_never_gathers_references() {
        for policy in [
            ReferencePolicy::Auto,
            ReferencePolicy::Always,
            ReferencePolicy::Never,
        ] {
            let config = config_with(&["Qwen3-8B"], policy);
            assert!(
                !should_gather_references(&config, Some("Qwen3-8B")),
                "{policy:?} must not advise a one-model pool"
            );
        }
    }

    #[test]
    fn auto_is_the_default_policy() {
        assert_eq!(ReferencePolicy::default(), ReferencePolicy::Auto);
    }
}
