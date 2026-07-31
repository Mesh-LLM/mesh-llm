//! Asymmetric (Hermes-style) tool turn.
//!
//! For a fresh request that carries tools, MoA no longer fans out tools to
//! every worker and votes on the result. Instead:
//!
//! 1. **References** — every model *except* the actor runs **tool-free** and
//!    only advises in prose.
//! 2. **Actor** — the single best tool-caller acts on that advice with the
//!    real tools and emits the tool call (or a direct answer).
//!
//! Why: tool authority should track *capability*, not *popularity*. The old
//! majority vote let a popular-but-wrong tool choice from several weak models
//! outvote the one strong tool-caller that picked correctly (observed:
//! `qwen3-32b` alone chose `run_command` for a failing-test triage while the
//! smaller models all chose `list_dir`, and the vote shipped `list_dir`). An
//! actor model removes that failure class rather than patching the arithmetic.
//!
//! This is the shape Together's MoA and Nous's Hermes both use, and it stays a
//! pure stateless `/v1/chat/completions` turn: references are regenerated from
//! the caller's transcript each request, and the external client still owns the
//! tool-execution loop.
//!
//! Mesh flavour: references are gathered with a bounded wait (proceed at a
//! majority of advisors, never block on the slow tail), and the actor is called
//! through the hedged ladder so a slow or broken best-candidate falls through
//! to the next tool-capable peer instead of stalling the turn.

use crate::backend::{SamplingParams, call_backend};
use crate::context;
use crate::fanout::{DispatchedWorker, gather_references};
use crate::normalize::{self, WorkerOutput};
use crate::reducer::{self, hedged_reducer_call, reducer_candidates};
use crate::session::Session;
use crate::worker::{self, WorkerRole};
use crate::{
    ForcedToolChoice, GatewayConfig, MOA_ERR_ALL_REDUCERS_FAILED, TurnKind, TurnResult,
    WorkerSummary, chat_response, enforce_tool_call_contract, error_response,
    fallback_worker_response, selected_tool_names_for_turn, tool_call_response,
    tool_proposal_response,
};
use serde_json::Value;
use std::time::Instant;

/// Handle a fresh, tool-bearing query with the asymmetric actor design.
pub(crate) async fn handle_tool_query(
    config: &GatewayConfig,
    session: &Session,
    allowed_tools: &[String],
    forced_tool: Option<&ForcedToolChoice>,
    start: Instant,
) -> TurnResult {
    // Actor priority order: best tool-caller first. `reducer_candidates`
    // honours the host-supplied `actor_candidates` (gossiped `tool_use` +
    // size + health) and falls back to name-derived size tier.
    let candidates = reducer_candidates(config);
    let actor_top = candidates.first().map(|(name, _)| name.clone());

    // References advise tool-free; the actor is excluded so it doesn't burn a
    // redundant advisory pass on the critical path.
    let (references, mut summaries) =
        dispatch_and_gather_references(config, session, actor_top.as_deref()).await;

    // Actor acts on the advice, this time WITH the real tools.
    let selected = selected_tool_names_for_turn(session, allowed_tools);
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

    // Record the actor as a distinct Reducer-role summary so observability can
    // tell advisory passes from the acting pass.
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
            continue; // the actor advises itself implicitly when it acts
        }
        // Tool-free: has_tools=false, no selected tool names. References
        // reason in prose; they never emit an executable call.
        let packed = context::pack_for_worker_selected(session, a.role, false, &[]);
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

    // Bounded wait: proceed once a majority of advisors are in, capped at the
    // worker timeout. On a mixed/public mesh this is the "don't wait for
    // perfect when good-enough advice is already in hand" guardrail — slow or
    // absent peers can't hold up the actor.
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
