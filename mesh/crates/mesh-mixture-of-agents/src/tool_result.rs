//! Reducer handling for completed tool-result turns.

use crate::config::GatewayConfig;
use crate::context;
use crate::normalize;
use crate::reducer::{self, hedged_reducer_call, reducer_candidates};
use crate::response::{chat_response, error_response, tool_proposal_response};
use crate::session::Session;
use crate::tool_guard::enforce_tool_call_contract;
use crate::tool_names_for_turn;
use crate::turn::{TurnKind, TurnResult, WorkerSummary};
use crate::worker::WorkerRole;
use crate::{MOA_ERR_ALL_REDUCERS_FAILED, MOA_ERR_NO_USABLE_ANSWER};
use serde_json::Value;
use std::time::Instant;

const SAME_TOOL_FORCE_ANSWER_THRESHOLD: usize = 3;

// ─── Tool result handling ────────────────────────────────────────────

pub(crate) async fn handle_tool_result(
    config: &GatewayConfig,
    session: &Session,
    has_tools: bool,
    allowed_tools: &[String],
    start: Instant,
) -> TurnResult {
    let candidates = reducer_candidates(config);
    let candidate_count = candidates.len();
    let repeated_tool = repeated_identical_tool_results(session);
    let force_answer = repeated_tool.is_some();
    let selected_tool_names = if force_answer {
        Vec::new()
    } else {
        tool_names_for_turn(session, allowed_tools)
    };
    let tools_enabled_for_reducer = has_tools && !force_answer;
    let (mut messages, tools) = context::pack_for_tool_result_turn_selected(
        session,
        tools_enabled_for_reducer,
        &selected_tool_names,
    );
    if let Some((tool, count)) = repeated_tool {
        tracing::info!("moa: forcing answer after {count} consecutive completed {tool} tool calls");
        append_tool_loop_answer_instruction(&mut messages, &tool, count);
    }

    // Hedged ladder: start candidate 0, hedge to candidate 1 after hedge_delay
    // (or immediately on candidate 0 error), race for the first OK. Rescues
    // tool-result turns when the first strong peer is broken (e.g. stale
    // binary that 502s on tool grammars) without paying N×timeout serially.
    tracing::info!("moa: tool result → hedged reducer over {candidate_count} candidate(s)");
    let hedge_result = hedged_reducer_call(
        &config.backends,
        candidates.clone(),
        messages,
        tools,
        config.reducer_timeout,
        config.hedge_delay,
        config.enable_thinking,
    )
    .await;

    let mut last_err: Option<String> = None;
    let (attempts, chosen): (u32, Option<(String, normalize::WorkerOutput)>) = match hedge_result {
        Ok(reducer::HedgedReducerOk {
            winner,
            text,
            attempts: spawned,
        }) => {
            let mut reduced =
                normalize::normalize_worker_output(&text, &winner, WorkerRole::Reducer, 0);
            enforce_tool_call_contract(&mut reduced, allowed_tools, session.tools(), &winner);
            (spawned, Some((winner, reduced)))
        }
        Err(reducer::HedgedReducerErr {
            err,
            attempts: spawned,
        }) => {
            last_err = Some(err);
            (spawned, None)
        }
    };

    let (reducer_name, succeeded, response_body) = match chosen {
        Some((name, reduced)) => {
            // Be consistent with the fanout/arbiter path: emit a real
            // `tool_calls` response whenever the reducer named a tool,
            // even if `arguments` is missing. The fanout path emits `{}`
            // for empty arguments; this path used to fall back to a
            // chat_response carrying the reducer's prose, which broke
            // agent harnesses (Goose, OpenCode) that only act on
            // `tool_calls`. `tool_call_response` already collapses
            // missing / non-object arguments to `"{}"`.
            let body = match reduced.kind {
                normalize::OutputKind::ToolProposal => {
                    tool_proposal_response(&reduced, tools_enabled_for_reducer)
                }
                normalize::OutputKind::Uncertainty => error_response(
                    "MoA reducer returned no usable answer",
                    MOA_ERR_NO_USABLE_ANSWER,
                ),
                _ => chat_response(&repair_tool_result_answer(session, &reduced.payload)),
            };
            (name, true, body)
        }
        None => {
            let err = last_err.unwrap_or_else(|| "no reducer candidates".into());
            tracing::warn!("moa: all {attempts} reducer candidates failed");
            (
                candidates.first().map(|c| c.0.clone()).unwrap_or_default(),
                false,
                error_response(
                    &format!("Reducer failed (tried {attempts}): {err}"),
                    MOA_ERR_ALL_REDUCERS_FAILED,
                ),
            )
        }
    };

    TurnResult {
        response_body,
        worker_summaries: vec![WorkerSummary {
            model: reducer_name,
            role: WorkerRole::Reducer,
            succeeded,
            elapsed_ms: start.elapsed().as_millis() as u64,
            output_kind: None,
            confidence: None,
        }],
        reducer_used: succeeded,
        reducer_attempts: attempts,
        turn_kind: TurnKind::ToolResult,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

pub(crate) fn repair_tool_result_answer(session: &Session, answer: &str) -> String {
    let missing = missing_tool_evidence_values(session, answer);
    if missing.is_empty() {
        return answer.to_string();
    }

    let mut repaired = answer.trim().to_string();
    if !repaired.is_empty() {
        repaired.push_str("\n\n");
    }
    repaired.push_str("Tool facts: ");
    repaired.push_str(&missing.join(", "));
    repaired
}

fn missing_tool_evidence_values(session: &Session, answer: &str) -> Vec<String> {
    let mut missing = Vec::new();
    for (_, result) in session.recent_tool_results() {
        for value in short_tool_result_values(&result) {
            if !answer.contains(&value) && !missing.iter().any(|seen| seen == &value) {
                missing.push(value);
            }
        }
    }
    missing
}

fn short_tool_result_values(result: &str) -> Vec<String> {
    let Ok(parsed) = serde_json::from_str::<Value>(result) else {
        return Vec::new();
    };

    let mut values = Vec::new();
    collect_short_tool_result_values(&parsed, &mut values);
    values
}

fn collect_short_tool_result_values(value: &Value, values: &mut Vec<String>) {
    match value {
        Value::Object(map) => {
            for (key, nested) in map {
                if tool_result_value_key_is_evidence(key)
                    && let Some(scalar) = nested.as_str().filter(|s| short_exact_value(s))
                {
                    values.push(scalar.to_string());
                } else if nested.is_object() || nested.is_array() {
                    collect_short_tool_result_values(nested, values);
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                collect_short_tool_result_values(item, values);
            }
        }
        _ => {}
    }
}

fn tool_result_value_key_is_evidence(key: &str) -> bool {
    matches!(
        key.to_ascii_lowercase().as_str(),
        "value" | "fact" | "result" | "answer"
    )
}

fn short_exact_value(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty() && trimmed.len() <= 160 && !trimmed.contains('\n')
}

pub(crate) fn repeated_identical_tool_results(session: &Session) -> Option<(String, usize)> {
    let calls = session.pending_tool_calls();
    let last = calls.last()?;
    last.result.as_ref()?;

    let tool_name = last.function_name.as_str();
    let arguments = &last.arguments;
    let count = calls
        .iter()
        .rev()
        .take_while(|call| {
            call.function_name == tool_name && call.arguments == *arguments && call.result.is_some()
        })
        .count();

    (count >= SAME_TOOL_FORCE_ANSWER_THRESHOLD).then(|| (tool_name.to_string(), count))
}

fn append_tool_loop_answer_instruction(messages: &mut [Value], tool: &str, count: usize) {
    let instruction = format!(
        "\n\nTool loop guard: the last {count} completed tool calls all used `{tool}`. \
         Answer now from the gathered tool results. Do not call another tool. \
         If the evidence is incomplete, say what can be determined and what is missing."
    );

    let system_content = messages
        .iter_mut()
        .find(|msg| msg.get("role").and_then(Value::as_str) == Some("system"))
        .and_then(|system| {
            let content = system.get("content").and_then(Value::as_str)?.to_string();
            Some((system, content))
        });
    if let Some((system, content)) = system_content {
        let mut updated = content.to_string();
        updated.push_str(&instruction);
        system["content"] = Value::String(updated);
    }
}
