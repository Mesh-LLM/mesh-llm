//! Arbiter decision resolution and reducer escalation.

use crate::MOA_ERR_NO_USABLE_ANSWER;
use crate::arbiter;
use crate::config::GatewayConfig;
use crate::context;
use crate::normalize;
use crate::reducer::{self, hedged_reducer_call, reducer_candidates};
use crate::response::{
    chat_response, error_response, fallback_worker_response, tool_call_response,
    tool_proposal_response,
};
use crate::tool_guard::enforce_tool_call_contract;
use crate::turn::DecisionResolution;
use crate::worker::WorkerRole;
use serde_json::Value;

// ─── Decision resolution ─────────────────────────────────────────────

/// Returns (response body, reducer_used, reducer_attempts).
pub(crate) async fn resolve_decision(
    config: &GatewayConfig,
    request: DecisionResolution<'_>,
) -> (Value, bool, u32) {
    let DecisionResolution {
        session,
        decision,
        outputs,
        has_tools,
        selected_tool_names,
        forced_tool,
        allowed_tools,
    } = request;

    match decision {
        arbiter::Decision::Answer(text) => {
            if let Some(tool) = forced_tool.filter(|_| has_tools) {
                (
                    tool_call_response(&tool.name, &tool.fallback_arguments),
                    false,
                    0,
                )
            } else {
                (chat_response(&text), false, 0)
            }
        }
        arbiter::Decision::ToolCall { name, arguments } => {
            if has_tools {
                (tool_call_response(&name, &arguments), false, 0)
            } else {
                (
                    error_response(
                        "MoA selected a tool call, but tools are disabled for this turn",
                        MOA_ERR_NO_USABLE_ANSWER,
                    ),
                    false,
                    0,
                )
            }
        }
        arbiter::Decision::NeedsReducer { reason } => {
            tracing::info!("moa: reducer — {reason}");
            let candidates = reducer_candidates(config);
            let (messages, tools) = context::pack_for_reducer_selected(
                session,
                outputs,
                &reason,
                has_tools,
                selected_tool_names,
            );

            // Hedged ladder over the ordered candidates (see hedged_reducer_call).
            let hedge_result = hedged_reducer_call(
                &config.backends,
                candidates,
                messages,
                tools,
                config.reducer_timeout,
                config.hedge_delay,
                config.enable_thinking,
            )
            .await;

            let (attempts, chosen): (u32, Option<normalize::WorkerOutput>) = match hedge_result {
                Ok(reducer::HedgedReducerOk {
                    winner,
                    text,
                    attempts: spawned,
                }) => {
                    let mut reduced =
                        normalize::normalize_worker_output(&text, &winner, WorkerRole::Reducer, 0);
                    enforce_tool_call_contract(
                        &mut reduced,
                        allowed_tools,
                        session.tools(),
                        &winner,
                    );
                    (spawned, Some(reduced))
                }
                Err(reducer::HedgedReducerErr {
                    err: _,
                    attempts: spawned,
                }) => (spawned, None),
            };

            match chosen {
                Some(reduced) => match reduced.kind {
                    normalize::OutputKind::ToolProposal => {
                        // See the matching block in `handle_tool_result`:
                        // emit `tool_calls` whenever `tool_name` is present,
                        // defaulting `arguments` to `{}` via
                        // `tool_call_response`. Agent harnesses key on
                        // `tool_calls` rather than scanning prose, so the
                        // previous "both name AND args required" gate would
                        // silently fall back to a chat_response and break
                        // the calling agent's tool loop.
                        (tool_proposal_response(&reduced, has_tools), true, attempts)
                    }
                    normalize::OutputKind::Uncertainty => {
                        if let Some(tool) = forced_tool.filter(|_| has_tools) {
                            (
                                tool_call_response(&tool.name, &tool.fallback_arguments),
                                true,
                                attempts,
                            )
                        } else {
                            (fallback_worker_response(outputs), true, attempts)
                        }
                    }
                    _ => {
                        if let Some(tool) = forced_tool.filter(|_| has_tools) {
                            (
                                tool_call_response(&tool.name, &tool.fallback_arguments),
                                true,
                                attempts,
                            )
                        } else {
                            (chat_response(&reduced.payload), true, attempts)
                        }
                    }
                },
                None => {
                    tracing::warn!("moa: all reducer candidates failed, using best worker");
                    // reducer_used=false here because the reducer did NOT
                    // produce the output we're returning — we fell back to
                    // a worker. attempts still reflects what was spawned so
                    // observability can see "we tried N times and all failed".
                    if let Some(tool) = forced_tool.filter(|_| has_tools) {
                        (
                            tool_call_response(&tool.name, &tool.fallback_arguments),
                            false,
                            attempts,
                        )
                    } else {
                        (fallback_worker_response(outputs), false, attempts)
                    }
                }
            }
        }
    }
}
