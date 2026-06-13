//! Mixture-of-Agents (MoA) gateway.
//!
//! Fan out to N heterogeneous LLM backends in parallel, arbitrate their
//! outputs with deterministic logic, and return one coherent OpenAI-
//! compatible response.  The client thinks it talks to one model.
//!
//! Transport is abstracted behind the [`ModelBackend`] trait (see
//! [`backend`]). The default [`HttpBackend`] talks to any
//! OpenAI-compatible HTTP endpoint and is suitable for standalone/test
//! use. The mesh host-runtime provides mesh-native backends that
//! dispatch local models via direct HTTP and remote models via QUIC
//! tunnel.
//!
//! ```text
//! Agent / Goose / pi
//!     │
//!     │  POST /v1/chat/completions { "model": "mesh" }
//!     ▼
//!  MoA Gateway  (handle_turn)
//!   ├─ session / context packing (role-shaped)        — context::*
//!   ├─ parallel fan-out via ModelBackend              — fanout::gather_workers_incremental
//!   ├─ incremental gathering with early-exit          — arbiter::try_early_decision
//!   ├─ deterministic arbiter (code, not models)       — arbiter::arbitrate
//!   └─ reducer escalation only on genuine conflict    — reducer::hedged_reducer_call
//! ```
//!
//! Modules:
//! - [`backend`] — `ModelBackend` trait, `HttpBackend`, `SamplingParams`,
//!   `ModelEntry`
//! - [`reducer`] — reducer candidate ordering, hedged ladder
//! - [`fanout`] — incremental worker gathering with early-exit
//! - [`arbiter`] — deterministic arbitration + early-exit consensus
//! - [`normalize`] — 3-tier dirty-output parsing
//! - [`session`] — canonical transcript + turn classification
//! - [`context`] — role-shaped context packing
//! - [`worker`] — role assignment, think-tag stripping

pub mod arbiter;
pub mod backend;
pub mod context;
mod fanout;
pub mod normalize;
mod reducer;
pub mod session;
mod tool_guard;
pub mod worker;

pub use backend::{HttpBackend, ModelBackend, ModelEntry, SamplingParams, apply_enable_thinking};
pub(crate) use tool_guard::enforce_tool_call_contract;

use backend::call_backend;
use fanout::{GraceMode, gather_workers_incremental};
use mesh_llm_guardrails::{sanitize_tool_arguments_for_tool, tool_arguments_wire_string};
use normalize::WorkerOutput;
use reducer::{hedged_reducer_call, reducer_candidates};
use serde_json::{Value, json};
use session::Session;
use std::time::{Duration, Instant};
use worker::WorkerRole;
pub use worker::{strip_thinking, truncate_chars};

const SAME_TOOL_FORCE_ANSWER_THRESHOLD: usize = 3;

/// The virtual model name that triggers MoA routing.
pub const VIRTUAL_MODEL_NAME: &str = "mesh";

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
    /// Override for whether reasoning workers should think. Propagated to
    /// every worker and the reducer as `chat_template_kwargs.enable_thinking`
    /// (and `reasoning_effort: "none"` when disabled).
    ///
    /// `None` (the default) leaves each model's default behavior alone —
    /// existing callers see no behavior change. The MoA HTTP gateway
    /// populates this from the caller's `reasoning_effort` / `enable_thinking`
    /// / `reasoning.enabled` knobs so MoA users get a single switch.
    pub enable_thinking: Option<bool>,
}

// ─── Turn result ─────────────────────────────────────────────────────

/// Which gateway path produced this turn's response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnKind {
    /// Fan-out path: arbiter decided from full worker outputs.
    Fanout,
    /// Fan-out path with early-exit consensus before all workers returned.
    EarlyExit,
    /// Tool-result turn: skipped fan-out, went straight to reducer.
    ToolResult,
    /// All workers failed and no reducer recovery happened.
    Failed,
}

impl TurnKind {
    /// Lowercase header-friendly label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Fanout => "fanout",
            Self::EarlyExit => "early-exit",
            Self::ToolResult => "tool-result",
            Self::Failed => "failed",
        }
    }
}

/// What the gateway returns for a single turn.
#[derive(Debug)]
pub struct TurnResult {
    /// OpenAI chat.completion response body.
    pub response_body: Value,
    /// Per-worker details for observability.
    pub worker_summaries: Vec<WorkerSummary>,
    /// Whether the reducer was invoked.
    pub reducer_used: bool,
    /// How many reducer candidates were spawned (0 if reducer didn't run,
    /// 1 on the happy reducer path, ≥2 if the hedge fired or a fast-fail
    /// cascaded to the next candidate).
    pub reducer_attempts: u32,
    /// Which gateway path produced this response.
    pub turn_kind: TurnKind,
    /// Wall-clock time for this turn.
    pub elapsed_ms: u64,
}

#[derive(Debug)]
pub struct WorkerSummary {
    pub model: String,
    pub role: WorkerRole,
    pub succeeded: bool,
    pub elapsed_ms: u64,
    pub output_kind: Option<normalize::OutputKind>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone)]
struct ForcedToolChoice {
    name: String,
    fallback_arguments: Value,
}

struct DecisionResolution<'a> {
    session: &'a Session,
    decision: arbiter::Decision,
    outputs: &'a [WorkerOutput],
    has_tools: bool,
    selected_tool_names: &'a [String],
    forced_tool: Option<&'a ForcedToolChoice>,
    allowed_tools: &'a [String],
}

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
    let assignments = worker::assign_roles(&config.models);
    let grace_mode = grace_mode_for_turn(session, has_tools);
    let query_uses_tools = forced_tool.is_some() || matches!(grace_mode, GraceMode::Tool);
    let selected_tool_names = if let Some(tool) = forced_tool {
        vec![tool.name.clone()]
    } else if query_uses_tools {
        selected_tool_names_for_turn(session, allowed_tools)
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
    for assignment in &assignments {
        let packed = context::pack_for_worker_selected(
            session,
            assignment.role,
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

    let (outputs, summaries, early_decision) = gather_workers_incremental(
        &mut join_set,
        &dispatched,
        query_uses_tools,
        allowed_tools,
        session.tools(),
        config.first_answer_grace,
        grace_mode,
    )
    .await;

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
    let decision = early_decision.unwrap_or_else(|| arbiter::arbitrate(&outputs, query_uses_tools));
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

fn grace_mode_for_turn(session: &Session, has_tools: bool) -> GraceMode {
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

fn selected_tool_names_for_turn(session: &Session, allowed_tools: &[String]) -> Vec<String> {
    let available = if allowed_tools.is_empty() {
        session.tool_names()
    } else {
        allowed_tools.to_vec()
    };
    if available.is_empty() {
        return Vec::new();
    }

    let text = session.last_user_text().to_ascii_lowercase();
    let explicit = explicitly_requested_tool_names(&available, &text);
    if !explicit.is_empty() {
        return with_recent_tool_chain_names(session, &available, explicit);
    }

    let mut selected = Vec::new();
    for tool in &available {
        let tool_lc = tool.to_ascii_lowercase();
        if tool_is_relevant_to_text(&tool_lc, &text) {
            selected.push(tool.clone());
        }
    }

    with_recent_tool_chain_names(session, &available, selected)
}

fn with_recent_tool_chain_names(
    session: &Session,
    available: &[String],
    mut selected: Vec<String>,
) -> Vec<String> {
    for tool in recent_tool_chain_names(session) {
        if available.iter().any(|available| available == &tool) && !selected.contains(&tool) {
            selected.push(tool);
        }
    }

    if selected.is_empty() && available.len() == 1 {
        available.to_vec()
    } else {
        selected
    }
}

fn explicitly_requested_tool_names(available: &[String], text: &str) -> Vec<String> {
    available
        .iter()
        .filter(|tool| tool_name_is_explicitly_requested(&tool.to_ascii_lowercase(), text))
        .cloned()
        .collect()
}

fn tool_name_is_explicitly_requested(tool: &str, text: &str) -> bool {
    if tool.len() < 3 {
        return false;
    }

    let spaced = tool.replace('_', " ");
    let patterns = [
        format!("use {tool}"),
        format!("use the {tool}"),
        format!("{tool} tool"),
        format!("tool {tool}"),
        format!("call {tool}"),
        format!("call the {tool}"),
        format!("use {spaced}"),
        format!("use the {spaced}"),
        format!("{spaced} tool"),
        format!("tool {spaced}"),
        format!("call {spaced}"),
        format!("call the {spaced}"),
    ];

    patterns.iter().any(|pattern| text.contains(pattern))
}

fn recent_tool_chain_names(session: &Session) -> Vec<String> {
    let all = session.all_messages();
    let Some(latest_tool_idx) = all.iter().rposition(|msg| message_role(msg) == "tool") else {
        return Vec::new();
    };
    let start_idx = all[..=latest_tool_idx]
        .iter()
        .rposition(|msg| message_role(msg) == "user")
        .unwrap_or(0);

    let mut names = Vec::new();
    for msg in &all[start_idx..=latest_tool_idx] {
        let Some(tool_calls) = msg.get("tool_calls").and_then(Value::as_array) else {
            continue;
        };
        for tool_call in tool_calls {
            let Some(name) = tool_call
                .pointer("/function/name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
            else {
                continue;
            };
            if !names.iter().any(|existing| existing == name) {
                names.push(name.to_string());
            }
        }
    }

    names
}

fn message_role(msg: &Value) -> &str {
    msg.get("role").and_then(Value::as_str).unwrap_or("")
}

fn tool_is_relevant_to_text(tool_name: &str, text: &str) -> bool {
    if text.contains(tool_name) {
        return true;
    }

    match tool_name {
        "read" | "read_file" | "file_fetch" => contains_any(
            text,
            &["read ", "open ", "inspect ", "fetch file", "show file"],
        ),
        "edit" | "edit_file" | "file_write" | "write" => {
            contains_any(text, &["edit ", "change ", "modify ", "write ", "create "])
        }
        "exec" | "run_command" | "process" => contains_any(
            text,
            &[
                "run ", "execute ", "shell", "terminal", "command", "process",
            ],
        ),
        "web_search" => contains_any(
            text,
            &[
                "search ",
                "look up",
                "lookup",
                "web",
                "google",
                "github",
                "issue",
                "pull request",
                "pr ",
                "weather",
                "forecast",
                "current",
                "latest",
                "today",
                "tomorrow",
            ],
        ),
        "web_fetch" => contains_any(
            text,
            &[
                "fetch ",
                "browse",
                "url",
                "http://",
                "https://",
                "github.com/",
            ],
        ),
        "dir_list" | "dir_fetch" | "list_files" => {
            contains_any(text, &["list ", "directory", "folder", "dir "])
        }
        "image" | "image_generate" => contains_any(text, &["image", "picture", "generate"]),
        "pdf" => text.contains("pdf"),
        "memory_search" | "memory_get" => text.contains("memory"),
        _ => false,
    }
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

fn forced_tool_choice(
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

fn infer_tool_arguments_from_prompt(name: &str, tools: Option<&Value>, prompt: &str) -> Value {
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
    let tail = prompt_lc.get(start..)?;
    let value = tail
        .split(|c: char| c.is_whitespace() || c == ',' || c == '.' || c == ';')
        .next()
        .unwrap_or("")
        .trim_matches(|c| c == '"' || c == '\'' || c == '`');
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

// ─── Tool result handling ────────────────────────────────────────────

async fn handle_tool_result(
    config: &GatewayConfig,
    session: &Session,
    has_tools: bool,
    allowed_tools: &[String],
    start: Instant,
) -> TurnResult {
    let candidates = reducer_candidates(config);
    let candidate_count = candidates.len();
    let repeated_tool = repeated_same_tool_results(session);
    let force_answer = repeated_tool.is_some();
    let selected_tool_names = if force_answer {
        Vec::new()
    } else {
        selected_tool_names_for_turn(session, allowed_tools)
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
        reducer_used: true,
        reducer_attempts: attempts,
        turn_kind: TurnKind::ToolResult,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

fn repair_tool_result_answer(session: &Session, answer: &str) -> String {
    if !tool_evidence_should_be_preserved(&session.last_user_text()) {
        return answer.to_string();
    }

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

fn tool_evidence_should_be_preserved(user_text: &str) -> bool {
    let text = user_text.to_ascii_lowercase();
    contains_any(
        &text,
        &[
            "tool fact",
            "tool facts",
            "tool result",
            "tool output",
            "final recall",
            "include both",
            "include all",
            "answer with the tool",
            "return the tool",
        ],
    )
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
                if !tool_result_value_key_is_evidence(key) {
                    continue;
                }
                if let Some(scalar) = nested.as_str().filter(|s| short_exact_value(s)) {
                    values.push(scalar.to_string());
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

fn repeated_same_tool_results(session: &Session) -> Option<(String, usize)> {
    let calls = session.pending_tool_calls();
    let last = calls.last()?;
    last.result.as_ref()?;

    let tool_name = last.function_name.as_str();
    let count = calls
        .iter()
        .rev()
        .take_while(|call| call.function_name == tool_name && call.result.is_some())
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

// ─── Decision resolution ─────────────────────────────────────────────

/// Returns (response body, reducer_used, reducer_attempts).
async fn resolve_decision(
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

// ─── Response builders ───────────────────────────────────────────────

fn best_answer(outputs: &[WorkerOutput]) -> String {
    outputs
        .iter()
        .filter(|o| {
            matches!(o.kind, normalize::OutputKind::Answer)
                && !normalize::is_silent_reply_sentinel(&o.payload)
        })
        // `total_cmp` is total over all f32 (including NaN/Inf); `partial_cmp`
        // can return `None` on NaN, which would panic on `unwrap`.
        // `normalize_worker_output` now sanitizes non-finite confidences
        // before they reach here, but using `total_cmp` keeps this site
        // panic-free even if a future caller skips the normalizer.
        .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
        .map(|o| o.payload.clone())
        .unwrap_or_default()
}

fn fallback_worker_response(outputs: &[WorkerOutput]) -> Value {
    let answer = best_answer(outputs);
    if answer.is_empty() {
        error_response(
            "MoA could not produce a usable answer",
            MOA_ERR_NO_USABLE_ANSWER,
        )
    } else {
        chat_response(&answer)
    }
}

fn tool_proposal_response(output: &WorkerOutput, has_tools: bool) -> Value {
    if let (true, Some(name)) = (has_tools, output.tool_name.as_ref()) {
        let args = output.tool_arguments.as_ref().unwrap_or(&Value::Null);
        return tool_call_response(name, args);
    }

    if output.payload.trim().is_empty() || normalize::is_silent_reply_sentinel(&output.payload) {
        return error_response(
            "MoA reducer returned no usable answer",
            MOA_ERR_NO_USABLE_ANSWER,
        );
    }

    chat_response(&output.payload)
}

/// Build a response body that signals MoA-level failure to the client.
///
/// Distinguishable from a successful `chat.completion` in three ways:
///
///   * Top-level `error` object (OpenAI error-shape) so SDKs that read
///     `response.error` see the failure without parsing `choices`.
///   * `choices[0].finish_reason == "error"` (instead of `"stop"`) so
///     SDKs that branch on `finish_reason` see the failure too.
///   * The error text is still placed in `choices[0].message.content`
///     so unstructured clients still surface a useful string to the
///     human, just not as a successful assistant reply.
///
/// `code` is the machine-parseable failure mode that clients can branch
/// on. Callers pass one of the [`MOA_ERR_*`] constants so distinct
/// failure modes (all-workers-failed vs all-reducers-failed vs future
/// kinds) surface accurately to the caller rather than being collapsed
/// to a single string.
///
/// The ingress layer is responsible for choosing the HTTP status; this
/// body is the in-band signal.
fn error_response(message: &str, code: &str) -> Value {
    json!({
        "id": format!("chatcmpl-moa-{}", short_id()),
        "object": "chat.completion",
        "model": VIRTUAL_MODEL_NAME,
        "error": {
            "message": message,
            "type": "moa_failure",
            "code": code,
        },
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": message },
            "finish_reason": "error"
        }],
        "usage": usage_for_content(message)
    })
}

/// Estimate `completion_tokens` from output chars (OpenAI's ~chars/4 rule).
/// Returns at least 1 for non-empty so UI tok/s never divides by zero.
fn estimate_completion_tokens(content: &str) -> u64 {
    if content.is_empty() {
        return 0;
    }
    let chars = content.chars().count() as u64;
    chars.div_ceil(4).max(1)
}

fn usage_for_content(content: &str) -> Value {
    let completion = estimate_completion_tokens(content);
    json!({
        "prompt_tokens": 0,
        "completion_tokens": completion,
        "total_tokens": completion,
    })
}

/// All fanned-out workers failed before the arbiter could pick a winner.
pub const MOA_ERR_ALL_WORKERS_FAILED: &str = "all_workers_failed";
/// Every reducer candidate failed (in both the tool-result and the
/// arbiter-escalated paths).
pub const MOA_ERR_ALL_REDUCERS_FAILED: &str = "all_reducers_failed";
/// MoA only received silence directives or uncertainty after reduction.
pub const MOA_ERR_NO_USABLE_ANSWER: &str = "no_usable_answer";

fn chat_response(content: &str) -> Value {
    json!({
        "id": format!("chatcmpl-moa-{}", short_id()),
        "object": "chat.completion",
        "model": VIRTUAL_MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": content },
            "finish_reason": "stop"
        }],
        "usage": usage_for_content(content)
    })
}

fn tool_call_response(name: &str, arguments: &Value) -> Value {
    // OpenAI tool-call `arguments` is a JSON-object *string*. Three input
    // shapes have to collapse to a valid object string here:
    //
    //   * String form: trust the caller's JSON (worker already passed
    //     through `extract_tool_arguments` so the inner shape is sane).
    //   * Null / non-object: emit `"{}"` rather than `"null"` or
    //     `"\"foo\""`. The previous shape would serialize `Value::Null`
    //     to the literal four-char string `"null"`, which downstream
    //     OpenAI tool-call consumers reject.
    //   * Object: serialize as JSON.
    let args_str = tool_arguments_wire_string(arguments);

    // For tool-call responses, the user-visible output is the
    // arguments JSON, not free-form text. Use it as the basis of the
    // token estimate so callers still see a non-zero count.
    json!({
        "id": format!("chatcmpl-moa-{}", short_id()),
        "object": "chat.completion",
        "model": VIRTUAL_MODEL_NAME,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": format!("call_{}", short_id()),
                    "type": "function",
                    "function": { "name": name, "arguments": args_str }
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": usage_for_content(&args_str)
    })
}

fn short_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:x}", t)
}

#[cfg(test)]
mod response_builder_tests {
    use super::*;
    use crate::normalize::{OutputKind, WorkerOutput};
    use crate::worker::WorkerRole;

    fn answer(model: &str, confidence: f32, payload: &str) -> WorkerOutput {
        WorkerOutput {
            kind: OutputKind::Answer,
            confidence,
            tool_name: None,
            tool_arguments: None,
            payload: payload.to_string(),
            model: model.to_string(),
            role: WorkerRole::Fast,
            elapsed_ms: 1,
        }
    }

    #[test]
    fn best_answer_does_not_panic_on_nan_confidence() {
        // Regression for PR #566 review: `partial_cmp(...).unwrap()` could
        // panic if any confidence reached this site as NaN. After switching
        // to `total_cmp`, this is safe even if normalize is bypassed.
        let outputs = vec![
            answer("a", f32::NAN, "nan-answer"),
            answer("b", 0.7, "good-answer"),
            answer("c", f32::NAN, "another-nan"),
        ];
        let picked = best_answer(&outputs);
        // `total_cmp` treats NaN as greater than any finite; the assertion
        // here is *not* about which specific answer wins, only that we do
        // not panic and we return *some* answer.
        assert!(!picked.is_empty());
    }

    #[test]
    fn best_answer_ignores_silent_reply_sentinel() {
        let outputs = vec![
            answer("a", 0.99, "NO_REPLY"),
            answer("b", 0.6, "Here is a real response."),
        ];
        assert_eq!(best_answer(&outputs), "Here is a real response.");
    }

    #[test]
    fn fallback_worker_response_errors_when_only_silent_sentinel_remains() {
        let outputs = vec![answer("a", 0.99, "NO_REPLY")];
        let resp = fallback_worker_response(&outputs);
        assert_eq!(
            resp.pointer("/error/code").and_then(Value::as_str),
            Some(MOA_ERR_NO_USABLE_ANSWER)
        );
        assert_eq!(
            resp.pointer("/choices/0/finish_reason")
                .and_then(Value::as_str),
            Some("error")
        );
    }

    fn tool_proposal(payload: &str) -> WorkerOutput {
        WorkerOutput {
            kind: normalize::OutputKind::ToolProposal,
            confidence: 0.8,
            tool_name: Some("read_file".to_string()),
            tool_arguments: Some(json!({"path": "README.md"})),
            payload: payload.to_string(),
            model: "reducer".to_string(),
            role: WorkerRole::Reducer,
            elapsed_ms: 1,
        }
    }

    #[test]
    fn tool_proposal_response_emits_tool_call_when_tools_enabled() {
        let resp = tool_proposal_response(&tool_proposal("Need to read."), true);
        assert_eq!(
            resp.pointer("/choices/0/message/tool_calls/0/function/name")
                .and_then(Value::as_str),
            Some("read_file")
        );
    }

    #[test]
    fn tool_proposal_response_does_not_emit_tool_call_when_tools_disabled() {
        let resp = tool_proposal_response(&tool_proposal("I need to read README.md."), false);
        assert!(
            resp.pointer("/choices/0/message/tool_calls").is_none(),
            "disabled tools must not leak tool_calls: {resp}"
        );
        assert_eq!(
            resp.pointer("/choices/0/message/content")
                .and_then(Value::as_str),
            Some("I need to read README.md.")
        );
    }

    #[test]
    fn tool_call_response_emits_object_args_for_null() {
        // Regression: `Value::Null` previously serialized to the literal
        // string "null", which downstream OpenAI tool-call consumers reject.
        let resp = tool_call_response("list", &Value::Null);
        let args_str = resp
            .pointer("/choices/0/message/tool_calls/0/function/arguments")
            .and_then(|v| v.as_str())
            .expect("arguments is string");
        assert_eq!(args_str, "{}");
    }

    #[test]
    fn tool_call_response_emits_object_args_for_primitive() {
        let resp = tool_call_response("list", &Value::from(42));
        let args_str = resp
            .pointer("/choices/0/message/tool_calls/0/function/arguments")
            .and_then(|v| v.as_str())
            .expect("arguments is string");
        assert_eq!(args_str, "{}");
    }

    #[test]
    fn tool_call_response_passes_through_string_form_when_valid() {
        let resp = tool_call_response(
            "read_file",
            &Value::String("{\"path\":\"README.md\"}".to_string()),
        );
        let args_str = resp
            .pointer("/choices/0/message/tool_calls/0/function/arguments")
            .and_then(|v| v.as_str())
            .expect("arguments is string");
        let parsed: Value = serde_json::from_str(args_str).unwrap();
        assert_eq!(parsed["path"], "README.md");
    }

    #[test]
    fn tool_call_response_rejects_invalid_string_form() {
        // If the caller hands us a bare non-JSON string, fall back to `{}`.
        let resp = tool_call_response("x", &Value::String("not json at all".to_string()));
        let args_str = resp
            .pointer("/choices/0/message/tool_calls/0/function/arguments")
            .and_then(|v| v.as_str())
            .expect("arguments is string");
        assert_eq!(args_str, "{}");
    }

    // Regression for #637.

    #[test]
    fn estimate_completion_tokens_returns_zero_for_empty_content() {
        assert_eq!(estimate_completion_tokens(""), 0);
    }

    #[test]
    fn estimate_completion_tokens_returns_at_least_one_for_non_empty() {
        assert_eq!(estimate_completion_tokens("a"), 1);
    }

    #[test]
    fn estimate_completion_tokens_is_roughly_chars_over_four() {
        assert_eq!(estimate_completion_tokens("sixteen chars!!!"), 4);
        assert_eq!(estimate_completion_tokens(&"x".repeat(40)), 10);
    }

    #[test]
    fn chat_response_reports_non_zero_completion_tokens() {
        let resp = chat_response("Hi there! How can I help you today?");
        let tokens = resp
            .pointer("/usage/completion_tokens")
            .and_then(serde_json::Value::as_u64)
            .expect("completion_tokens is u64");
        assert!(tokens > 0);
        assert_eq!(
            resp.pointer("/usage/total_tokens").and_then(|v| v.as_u64()),
            Some(tokens),
        );
    }

    #[test]
    fn tool_call_response_reports_non_zero_completion_tokens() {
        let resp = tool_call_response("read_file", &serde_json::json!({"path": "/etc/hostname"}));
        let tokens = resp
            .pointer("/usage/completion_tokens")
            .and_then(serde_json::Value::as_u64)
            .expect("completion_tokens is u64");
        assert!(tokens > 0);
    }

    #[test]
    fn forced_tool_choice_infers_enum_argument_from_prompt() {
        let body = serde_json::json!({
            "tool_choice": {
                "type": "function",
                "function": {"name": "lookup_probe_fact"}
            },
            "messages": [{
                "role": "user",
                "content": "Use lookup_probe_fact with primary and report the result."
            }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup_probe_fact",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "key": {
                                "type": "string",
                                "enum": ["primary", "secondary"]
                            }
                        },
                        "required": ["key"]
                    }
                }
            }]
        });
        let tools = body.get("tools").cloned();
        let mut session = Session::new();
        let messages = body
            .get("messages")
            .and_then(Value::as_array)
            .cloned()
            .unwrap();
        session.ingest(&messages, &tools);

        let forced = forced_tool_choice(&body, &session, &tools).expect("forced tool");

        assert_eq!(forced.name, "lookup_probe_fact");
        assert_eq!(forced.fallback_arguments, json!({"key": "primary"}));
    }

    #[test]
    fn forced_tool_choice_infers_assignment_argument_from_prompt() {
        let tools = Some(serde_json::json!([{
            "type": "function",
            "function": {
                "name": "lookup_probe_fact",
                "parameters": {
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"]
                }
            }
        }]));

        let args = infer_tool_arguments_from_prompt(
            "lookup_probe_fact",
            tools.as_ref(),
            "Use key=Primary",
        );

        assert_eq!(args, json!({"key": "primary"}));
    }

    #[tokio::test]
    async fn forced_tool_choice_overrides_answer_decision() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Use lookup_probe_fact with primary"
            })],
            &Some(serde_json::json!([{
                "type": "function",
                "function": {"name": "lookup_probe_fact"}
            }])),
        );
        let config = GatewayConfig {
            backends: Vec::new(),
            models: Vec::new(),
            worker_timeout: Duration::from_secs(1),
            reducer_timeout: Duration::from_secs(1),
            hedge_delay: Duration::from_millis(10),
            first_answer_grace: Duration::from_millis(10),
            enable_thinking: Some(false),
        };
        let forced_tool = ForcedToolChoice {
            name: "lookup_probe_fact".to_string(),
            fallback_arguments: json!({"key": "primary"}),
        };
        let selected_tool_names = ["lookup_probe_fact".to_string()];
        let allowed_tools = ["lookup_probe_fact".to_string()];
        let (resp, reducer_used, attempts) = resolve_decision(
            &config,
            DecisionResolution {
                session: &session,
                decision: arbiter::Decision::Answer("I would call the tool.".to_string()),
                outputs: &[],
                has_tools: true,
                selected_tool_names: &selected_tool_names,
                forced_tool: Some(&forced_tool),
                allowed_tools: &allowed_tools,
            },
        )
        .await;

        assert!(!reducer_used);
        assert_eq!(attempts, 0);
        assert_eq!(
            resp.pointer("/choices/0/finish_reason")
                .and_then(Value::as_str),
            Some("tool_calls")
        );
        assert_eq!(
            resp.pointer("/choices/0/message/tool_calls/0/function/name")
                .and_then(Value::as_str),
            Some("lookup_probe_fact")
        );
        assert_eq!(
            resp.pointer("/choices/0/message/tool_calls/0/function/arguments")
                .and_then(Value::as_str),
            Some("{\"key\":\"primary\"}")
        );
    }

    #[test]
    fn error_response_reports_message_based_completion_tokens() {
        let resp = error_response("All MoA workers failed", MOA_ERR_ALL_WORKERS_FAILED);
        let tokens = resp
            .pointer("/usage/completion_tokens")
            .and_then(serde_json::Value::as_u64)
            .expect("completion_tokens is u64");
        assert!(tokens > 0);
    }

    #[test]
    fn tool_enabled_chat_uses_answer_grace() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({"role": "user", "content": "How are you?"})],
            &Some(serde_json::json!([{"type": "function", "function": {"name": "read"}}])),
        );
        assert_eq!(grace_mode_for_turn(&session, true), GraceMode::Answer);
    }

    #[test]
    fn tool_intent_uses_tool_grace() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Use a tool to read /tmp/openclaw-tool-baseline.txt",
            })],
            &Some(serde_json::json!([{"type": "function", "function": {"name": "read"}}])),
        );
        assert_eq!(grace_mode_for_turn(&session, true), GraceMode::Tool);
    }

    #[test]
    fn negated_web_prompt_uses_answer_grace() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Plain check with no web lookup: reply OK",
            })],
            &Some(serde_json::json!([{"type": "function", "function": {"name": "web_search"}}])),
        );
        assert_eq!(grace_mode_for_turn(&session, true), GraceMode::Answer);
    }

    #[test]
    fn no_tools_uses_answer_grace() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({"role": "user", "content": "Reply OK"})],
            &None,
        );
        assert_eq!(grace_mode_for_turn(&session, false), GraceMode::Answer);
    }

    #[test]
    fn read_prompt_selects_only_read_tool() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({"role": "user", "content": "Read /tmp/file.txt"})],
            &Some(serde_json::json!([
                {"type": "function", "function": {"name": "read"}},
                {"type": "function", "function": {"name": "web_search"}},
                {"type": "function", "function": {"name": "exec"}}
            ])),
        );
        assert_eq!(
            selected_tool_names_for_turn(&session, &[]),
            vec!["read".to_string()]
        );
    }

    #[test]
    fn weather_prompt_selects_web_search_tool() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Check the current Melbourne weather forecast for today",
            })],
            &Some(serde_json::json!([
                {"type": "function", "function": {"name": "read"}},
                {"type": "function", "function": {"name": "web_search"}},
                {"type": "function", "function": {"name": "exec"}}
            ])),
        );
        assert_eq!(
            selected_tool_names_for_turn(&session, &[]),
            vec!["web_search".to_string()]
        );
    }

    #[test]
    fn tool_result_turn_keeps_active_tool_selected() {
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": "check local auth"}),
                serde_json::json!({
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_exec",
                        "type": "function",
                        "function": {"name": "exec", "arguments": "{\"command\":\"echo ok\"}"}
                    }]
                }),
                serde_json::json!({
                    "role": "tool",
                    "tool_call_id": "call_exec",
                    "content": "logged in"
                }),
            ],
            &Some(serde_json::json!([
                {"type": "function", "function": {"name": "read"}},
                {"type": "function", "function": {"name": "web_search"}},
                {"type": "function", "function": {"name": "exec"}}
            ])),
        );

        assert_eq!(
            selected_tool_names_for_turn(&session, &[]),
            vec!["exec".to_string()]
        );
    }

    #[test]
    fn explicit_exec_request_suppresses_url_broadened_tools() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Use exec exactly once. Command: curl https://api.github.com/repos/Mesh-LLM/mesh-llm/issues"
            })],
            &Some(serde_json::json!([
                {"type": "function", "function": {"name": "exec"}},
                {"type": "function", "function": {"name": "web_search"}},
                {"type": "function", "function": {"name": "web_fetch"}}
            ])),
        );

        assert_eq!(
            selected_tool_names_for_turn(&session, &[]),
            vec!["exec".to_string()]
        );
    }

    #[test]
    fn explicit_multiple_tool_request_keeps_named_tools() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Use the exec tool once to run pwd, then use read to inspect USER.md."
            })],
            &Some(serde_json::json!([
                {"type": "function", "function": {"name": "exec"}},
                {"type": "function", "function": {"name": "read"}},
                {"type": "function", "function": {"name": "web_search"}}
            ])),
        );

        assert_eq!(
            selected_tool_names_for_turn(&session, &[]),
            vec!["exec".to_string(), "read".to_string()]
        );
    }

    #[test]
    fn two_same_tool_results_do_not_force_answer() {
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": "search"}),
                tool_call_msg("call_1", "web_search"),
                tool_result_msg("call_1", "result 1"),
                tool_call_msg("call_2", "web_search"),
                tool_result_msg("call_2", "result 2"),
            ],
            &Some(serde_json::json!([
                {"type": "function", "function": {"name": "web_search"}}
            ])),
        );

        assert_eq!(repeated_same_tool_results(&session), None);
    }

    #[test]
    fn three_same_tool_results_force_answer() {
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": "search"}),
                tool_call_msg("call_1", "web_search"),
                tool_result_msg("call_1", "result 1"),
                tool_call_msg("call_2", "web_search"),
                tool_result_msg("call_2", "result 2"),
                tool_call_msg("call_3", "web_search"),
                tool_result_msg("call_3", "result 3"),
            ],
            &Some(serde_json::json!([
                {"type": "function", "function": {"name": "web_search"}}
            ])),
        );

        assert_eq!(
            repeated_same_tool_results(&session),
            Some(("web_search".to_string(), 3))
        );
    }

    #[test]
    fn repair_tool_result_answer_preserves_short_json_values_on_recall() {
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": "search"}),
                tool_call_msg("call_1", "lookup"),
                tool_result_msg("call_1", r#"{"key":"primary","value":"PRIMARY-FACT-123"}"#),
                tool_call_msg("call_2", "lookup"),
                tool_result_msg(
                    "call_2",
                    r#"{"key":"secondary","value":"SECONDARY-FACT-456"}"#,
                ),
                serde_json::json!({
                    "role": "user",
                    "content": "Final recall: include both tool facts."
                }),
            ],
            &None,
        );

        let repaired =
            repair_tool_result_answer(&session, "The secondary fact is SECONDARY-FACT-456.");

        assert!(repaired.contains("PRIMARY-FACT-123"));
        assert!(repaired.contains("SECONDARY-FACT-456"));
        assert!(!repaired.contains("primary"));
    }

    #[test]
    fn repair_tool_result_answer_ignores_large_or_non_evidence_tool_values() {
        let huge = "x".repeat(200);
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": "search"}),
                tool_call_msg("call_1", "lookup"),
                tool_result_msg(
                    "call_1",
                    &serde_json::json!({
                        "value": huge,
                        "debug": "SHORT-BUT-NOT-EVIDENCE",
                        "result": "multi\nline",
                    })
                    .to_string(),
                ),
                serde_json::json!({
                    "role": "user",
                    "content": "Final recall: include tool facts."
                }),
            ],
            &None,
        );

        let repaired = repair_tool_result_answer(&session, "Done.");

        assert_eq!(repaired, "Done.");
    }

    fn tool_call_msg(id: &str, name: &str) -> Value {
        serde_json::json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": "{\"query\":\"x\"}"}
            }]
        })
    }

    fn tool_result_msg(id: &str, text: &str) -> Value {
        serde_json::json!({
            "role": "tool",
            "tool_call_id": id,
            "content": text
        })
    }

    #[test]
    fn ordinary_prompt_selects_no_tools() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({"role": "user", "content": "Help"})],
            &Some(serde_json::json!([
                {"type": "function", "function": {"name": "read"}},
                {"type": "function", "function": {"name": "web_search"}}
            ])),
        );
        assert_eq!(
            selected_tool_names_for_turn(&session, &[]),
            Vec::<String>::new()
        );
    }
}
