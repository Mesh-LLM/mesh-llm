//! Query dispatch and worker fan-out orchestration.

use crate::arbiter;
use crate::backend::{SamplingParams, call_backend};
use crate::config::GatewayConfig;
use crate::context;
use crate::fanout::{self, GraceMode, gather_workers_incremental};
use crate::refinement;
use crate::resolve::resolve_decision;
use crate::response::error_response;
use crate::session::Session;
use crate::tool_result::handle_tool_result;
use crate::turn::{DecisionResolution, ForcedToolChoice, TurnKind, TurnResult};
use crate::worker;
use crate::{MOA_ERR_ALL_WORKERS_FAILED, session, tool_turn};
use mesh_llm_guardrails::sanitize_tool_arguments_for_tool;
use serde_json::{Value, json};
use std::time::Instant;

// ─── Gateway entry point ─────────────────────────────────────────────

/// Process one MoA turn.
///
/// Stateless per request.  Multi-turn state is managed by the agent client
/// which sends the full conversation on each request.
pub async fn handle_turn(config: &GatewayConfig, body: &Value) -> TurnResult {
    let start = Instant::now();

    let mut session = Session::new();
    let incoming_messages = body
        .get("messages")
        .and_then(|m| m.as_array())
        .cloned()
        .unwrap_or_default();
    let tools = body.get("tools").cloned();
    let has_tools = tools
        .as_ref()
        .and_then(|t| t.as_array())
        .map(|a| !a.is_empty())
        .unwrap_or(false);

    session.ingest(&incoming_messages, &tools);

    let turn_type = session.classify_turn();
    let forced_tool = forced_tool_choice(body, &session, &tools);
    tracing::info!(
        "moa: turn={:?}, {} models, tools={}",
        turn_type,
        config.models.len(),
        has_tools,
    );

    let allowed_tools = session.tool_names();

    match turn_type {
        session::TurnType::ToolResult => {
            handle_tool_result(config, &session, has_tools, &allowed_tools, start).await
        }
        session::TurnType::Fresh => {
            handle_query(
                config,
                &session,
                has_tools,
                &allowed_tools,
                forced_tool.as_ref(),
                start,
            )
            .await
        }
    }
}

// ─── Query handling ──────────────────────────────────────────────────

async fn handle_query(
    config: &GatewayConfig,
    session: &Session,
    has_tools: bool,
    allowed_tools: &[String],
    forced_tool: Option<&ForcedToolChoice>,
    start: Instant,
) -> TurnResult {
    // Tool-bearing turns take the asymmetric (Hermes-style) path: references
    // advise tool-free, the best tool-caller acts. Tool authority tracks
    // capability, not a majority vote — see `tool_turn`. Text-only turns keep
    // the symmetric fan-out + synthesis-on-divergence path below.
    if has_tools || forced_tool.is_some() {
        return tool_turn::handle_tool_query(config, session, allowed_tools, forced_tool, start)
            .await;
    }

    let assignments = worker::assign_roles(&config.models);
    let grace_mode = grace_mode_for_turn(session, has_tools);

    // If the caller gave us tools, the workers get tools. Full stop.
    //
    // This used to be `matches!(grace_mode, GraceMode::Tool)`, which routed
    // the decision through `looks_like_tool_intent` — a keyword match against
    // the user's text ("read ", "search ", "file", "directory", ...). Two
    // separate concerns were riding on one flag: whether tools are *available*
    // and whether the chat-only answer grace applies.
    //
    // Recorded agentic traces show how badly that misfires
    // (`evals/moa-openrouter/agentic.jsonl`). "The test suite is failing. Find
    // out which test fails and why" matches no phrase, so every worker was
    // dispatched without tool schemas — while the same prompt, given tools,
    // produced 53 tool calls across 9 models. Same for "Is this project's test
    // suite passing?" (32) and "Find every place MeshError::Timeout is
    // constructed in this repo." (20). Five of ten recorded tool scenarios had
    // tools silently withheld.
    //
    // Worse, this flag is also passed to the arbiter as its `has_tools`, so a
    // pool that unanimously proposed a tool fell through to the answer path and
    // leaked the proposal's payload text — an agent harness received the prose
    // "calling search" instead of a `search` tool call.
    //
    // Tool availability is now the caller's declaration. `grace_mode` keeps
    // using the heuristic, which is where a guess is actually appropriate: it
    // only tunes how long we wait before shipping a partial answer.
    let query_uses_tools = forced_tool.is_some() || has_tools;
    let selected_tool_names = if let Some(tool) = forced_tool {
        vec![tool.name.clone()]
    } else if query_uses_tools {
        tool_names_for_turn(session, allowed_tools)
    } else {
        Vec::new()
    };

    tracing::info!(
        "moa: dispatching to {} workers: [{}]",
        assignments.len(),
        assignments
            .iter()
            .map(|a| format!("{}({})", a.model_name, a.role.label()))
            .collect::<Vec<_>>()
            .join(", ")
    );

    let mut join_set = tokio::task::JoinSet::new();
    let mut dispatched: Vec<fanout::DispatchedWorker> = Vec::with_capacity(assignments.len());

    let enable_thinking = config.enable_thinking;
    // Role tiers (Fast 256 tokens, Specialist 512, Strong 1024) encode a
    // capability spread so the cheap worker can answer the grace path quickly.
    // A homogeneous pool has no such spread, and when refinement is expected
    // every draft is an *input* to the round — a 256-token draft is a ~1000-char
    // stub that drags the refined answer down. Measured end-to-end, production's
    // tiered budgets produced ~3.1k-char answers against a ~4.1k-char solo
    // baseline; the study that showed the gain gave every peer the full budget.
    // Full-budget drafts for every answer-turn worker. Role tiers (Fast 256 /
    // Specialist 512 / Strong 1024) existed only so the cheap worker could
    // answer the grace fast-path quickly with a short reply — but grace no
    // longer finalizes answer turns (it collects, then synthesizes), so a
    // truncated draft is now pure downside: it is an *input* to synthesis, and
    // a 256-token stub drags the aggregated answer down. Measured: an all-small
    // pool lost 5W/31L through the shipped path with role-tiered drafts
    // (3399-char MoA vs 4064 solo) while the full-budget harness won 12W/2L on
    // the same pool. A big pool tolerated tiering only because its aggregator
    // is strong enough to rebuild a full answer from stubs. See
    // `evals/moa-openrouter/RESULTS.md`.
    let uniform_packing = grace_mode == GraceMode::Answer;
    for assignment in &assignments {
        let pack_role = if uniform_packing {
            worker::WorkerRole::Generalist
        } else {
            assignment.role
        };
        let packed = context::pack_for_worker_selected(
            session,
            pack_role,
            query_uses_tools,
            &selected_tool_names,
        );
        let model_name = assignment.model_name.clone();
        let role = assignment.role;
        let backend = config.backends[assignment.backend_index].clone();
        let timeout = config.worker_timeout;

        dispatched.push(fanout::DispatchedWorker {
            model: model_name.clone(),
            role,
            small_tier: assignment.small_tier,
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
            let elapsed = t0.elapsed().as_millis() as u64;
            (model_name, role, result, elapsed)
        });
    }

    let (mut outputs, mut summaries, early_decision) = gather_workers_incremental(
        &mut join_set,
        &dispatched,
        allowed_tools,
        session.tools(),
        fanout::GatherPolicy {
            first_answer_grace: config.first_answer_grace,
            grace_mode,
            strong_patience: config.strong_patience,
            // Refinement quality scales with how many perspectives it gets.
            // MIN_DRAFTS (2) is the minimum for the round to *run*, not a good
            // target to collect: on a 4-model pool it let grace stop gathering
            // at 2 of 4, while the study that measured the gain refined over
            // every draft. Wait for all but one, so a single straggler still
            // cannot hold the turn and `worker_timeout` still bounds the wait.
            // Synthesis quality scales with WIDTH — that is the whole small-pool
            // finding (6x 8B beats its best member, 4x is marginal, 2x is null).
            // Arming grace on the first answer let it stop collecting at ~2 of 6,
            // synthesizing a committee too narrow to win. Wait for all but one
            // draft on any multi-worker answer turn, so a single straggler still
            // cannot hold the turn and `worker_timeout` remains the hard bound.
            // Tool turns keep the fast path (first valid proposal wins).
            min_grace_answers: if refinement::refinement_expected(config) {
                config
                    .models
                    .len()
                    .saturating_sub(1)
                    .max(refinement::MIN_DRAFTS)
            } else {
                // 1, deliberately: `min_grace_answers` gates whether grace can
                // ARM at all, so any higher value is a liveness hazard. Setting
                // it to N-1 on an answer turn meant a 6-worker pool where two
                // public-mesh peers never returned could never arm grace (only
                // 4 answers ever arrive), so the turn rode `worker_timeout`
                // instead — measured 61s on a public mesh against 3-11s for the
                // released build, for a SHORTER answer.
                //
                // Width comes from the grace WINDOW (10s), not from a count
                // gate: healthy peers all land inside it, and a dead peer costs
                // 10s rather than 60s. This is also the configuration the
                // capable-pool win was measured under (71W/8T/1L).
                1
            },
            // Grace finalizes (ships one worker's answer and stops) ONLY on
            // tool turns, where a fast validated tool call keeps agent loops
            // snappy. On answer turns grace is a *collection deadline*: stop
            // waiting for the slow tail, but still synthesize what arrived
            // instead of shipping one worker's answer. Finalizing answer turns
            // shipped a single (often role-truncated) answer and skipped
            // synthesis — measured 80/80 early-exit at capable scale, turning a
            // 61W/3L committee win into a 26W/40L loss. See
            // `evals/moa-openrouter/RESULTS.md`.
            grace_finalizes: grace_mode == GraceMode::Tool,
        },
    )
    .await;

    // Cross-peer refinement (Together's `layers`): each worker rewrites its
    // answer after seeing the others'. Runs only for pool shapes where it
    // measurably pays (`evals/moa-openrouter/RESULTS.md`), and is best-effort —
    // on shortfall we keep the round-1 outputs.
    //
    // An `early_decision` here means the workers actually *agreed* — that is a
    // real signal and the cheap path, so it still short-circuits. (The other
    // early-exit source, the answer grace, is a timeout rather than a quality
    // signal; when refinement is expected it is configured above to bound the
    // gather without finalizing, so it no longer pre-empts this round.)
    if early_decision.is_none()
        && refinement::should_refine(config, outputs.len())
        && let Some((refined, refine_summaries)) =
            refinement::refine_round(config, session, &outputs).await
    {
        tracing::info!(
            "moa: refinement round produced {} draft(s) from {}",
            refined.len(),
            outputs.len(),
        );
        outputs = refined;
        summaries.extend(refine_summaries);
    }

    if outputs.is_empty() {
        return TurnResult {
            response_body: error_response("All MoA workers failed", MOA_ERR_ALL_WORKERS_FAILED),
            worker_summaries: summaries,
            reducer_used: false,
            reducer_attempts: 0,
            turn_kind: TurnKind::Failed,
            elapsed_ms: start.elapsed().as_millis() as u64,
        };
    }

    // Capture whether we took the early-exit path BEFORE we resolve the
    // decision: the arbiter never runs when early_decision is Some.
    let took_early_exit = early_decision.is_some();
    let decision = early_decision.unwrap_or_else(|| arbiter::arbitrate(&outputs));

    // Always synthesize a multi-worker answer turn — never ship one worker's
    // text verbatim.
    //
    // `arbitrate` returns `Answer(payload)` when the drafts agree, which ships
    // whichever single worker represented the cluster. That is the whole
    // harness-vs-shipped gap: the eval rig always synthesizes (draft ->
    // aggregate) and a 6x8B pool wins 12W/2L, while the shipped path shipped one
    // 8B verbatim on agreement and lost (6W/19L on the same pool). A weak 8B
    // answer alone loses to a full-budget solo; an aggregation of six beats it.
    // The synthesizer sees every draft and produces the fuller, better answer —
    // agreeing drafts are the best possible input to it, not a reason to skip.
    //
    // Applies to answer turns with >=2 successful workers. Tool turns keep
    // their own routing (single best actor). A single surviving worker has
    // nothing to synthesize, so it still ships directly.
    let is_answer_turn = grace_mode == GraceMode::Answer;
    let decision =
        if is_answer_turn && outputs.len() >= 2 && matches!(decision, arbiter::Decision::Answer(_))
        {
            arbiter::Decision::NeedsReducer {
                reason: format!("{} drafts to synthesize", outputs.len()),
            }
        } else {
            decision
        };
    let (response_body, reducer_used, reducer_attempts) = resolve_decision(
        config,
        DecisionResolution {
            session,
            decision,
            outputs: &outputs,
            has_tools: query_uses_tools,
            selected_tool_names: &selected_tool_names,
            forced_tool,
            allowed_tools,
        },
    )
    .await;

    // turn_kind is "early-exit" only when we genuinely short-circuited via
    // consensus AND didn't need to escalate to the reducer. A reducer-
    // escalated turn is "fanout" even if early_decision was set, because
    // we still did the expensive serial call.
    let turn_kind = if took_early_exit && !reducer_used {
        TurnKind::EarlyExit
    } else {
        TurnKind::Fanout
    };

    TurnResult {
        response_body,
        worker_summaries: summaries,
        reducer_used,
        reducer_attempts,
        turn_kind,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

pub(crate) fn grace_mode_for_turn(session: &Session, has_tools: bool) -> GraceMode {
    if !has_tools {
        return GraceMode::Answer;
    }
    if looks_like_tool_intent(&session.last_user_text()) {
        GraceMode::Tool
    } else {
        GraceMode::Answer
    }
}

fn looks_like_tool_intent(text: &str) -> bool {
    let text = text.to_ascii_lowercase();
    if contains_any(
        &text,
        &[
            "no tool",
            "without tool",
            "do not use tool",
            "don't use tool",
            "no web",
            "without web",
            "do not browse",
            "don't browse",
            "no lookup",
            "without lookup",
        ],
    ) {
        return false;
    }

    let tool_intent_phrases = [
        "use a tool",
        "using a tool",
        "read ",
        "inspect ",
        "open ",
        "fetch ",
        "search ",
        "look up",
        "browse",
        "web",
        "url",
        "http://",
        "https://",
        "file",
        "directory",
        "folder",
        "list ",
        "run ",
        "execute",
        "terminal",
        "shell",
        "github",
        "issue",
        "pull request",
        "pr ",
        "weather",
    ];
    tool_intent_phrases
        .iter()
        .any(|phrase| text.contains(phrase))
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

pub(crate) fn tool_names_for_turn(session: &Session, allowed_tools: &[String]) -> Vec<String> {
    if allowed_tools.is_empty() {
        session.tool_names()
    } else {
        allowed_tools.to_vec()
    }
}

pub(crate) fn forced_tool_choice(
    body: &Value,
    session: &Session,
    tools: &Option<Value>,
) -> Option<ForcedToolChoice> {
    let name = body
        .get("tool_choice")?
        .get("function")?
        .get("name")?
        .as_str()?;
    if name.is_empty() || !session.tool_names().iter().any(|tool| tool == name) {
        return None;
    }

    let inferred =
        infer_tool_arguments_from_prompt(name, tools.as_ref(), &session.last_user_text());
    let fallback_arguments =
        sanitize_tool_arguments_for_tool(name, &inferred, tools.as_ref()).unwrap_or(inferred);

    Some(ForcedToolChoice {
        name: name.to_string(),
        fallback_arguments,
    })
}

pub(crate) fn infer_tool_arguments_from_prompt(
    name: &str,
    tools: Option<&Value>,
    prompt: &str,
) -> Value {
    let Some(parameters) = tool_parameters(name, tools) else {
        return json!({});
    };
    let Some(required) = parameters.get("required").and_then(Value::as_array) else {
        return json!({});
    };
    let Some(properties) = parameters.get("properties").and_then(Value::as_object) else {
        return json!({});
    };

    let mut args = serde_json::Map::new();
    for field in required.iter().filter_map(Value::as_str) {
        let Some(schema) = properties.get(field) else {
            continue;
        };
        if let Some(value) = infer_string_argument(field, schema, prompt) {
            args.insert(field.to_string(), Value::String(value));
        }
    }
    Value::Object(args)
}

fn infer_string_argument(field: &str, schema: &Value, prompt: &str) -> Option<String> {
    if !schema_allows_string(schema) {
        return None;
    }

    infer_enum_argument(schema, prompt).or_else(|| infer_assignment_argument(field, prompt))
}

fn schema_allows_string(schema: &Value) -> bool {
    match schema.get("type") {
        Some(Value::String(value)) => value == "string",
        Some(Value::Array(values)) => values.iter().any(|value| value.as_str() == Some("string")),
        None => true,
        _ => false,
    }
}

fn infer_enum_argument(schema: &Value, prompt: &str) -> Option<String> {
    let prompt_lc = prompt.to_ascii_lowercase();
    schema
        .get("enum")?
        .as_array()?
        .iter()
        .filter_map(Value::as_str)
        .find(|candidate| prompt_lc.contains(&candidate.to_ascii_lowercase()))
        .map(str::to_string)
}

fn infer_assignment_argument(field: &str, prompt: &str) -> Option<String> {
    let prompt_lc = prompt.to_ascii_lowercase();
    let field_lc = field.to_ascii_lowercase();
    let marker = format!("{field_lc}=");
    let start = prompt_lc.find(&marker)? + marker.len();
    let tail = prompt.get(start..)?;
    let value = tail
        .split(|c: char| c.is_whitespace() || c == ',' || c == ';')
        .next()
        .unwrap_or("")
        .trim_matches(|c| c == '"' || c == '\'' || c == '`')
        .trim_end_matches('.');
    (!value.is_empty()).then(|| value.to_string())
}

fn tool_parameters<'a>(tool_name: &str, tools: Option<&'a Value>) -> Option<&'a Value> {
    tools?
        .as_array()?
        .iter()
        .find(|tool| {
            tool.pointer("/function/name")
                .and_then(Value::as_str)
                .is_some_and(|name| name == tool_name)
        })?
        .pointer("/function/parameters")
}
