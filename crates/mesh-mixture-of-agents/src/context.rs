//! Context packing — tailor what each worker sees.
//!
//! Full context enters the gateway, but workers get role-shaped slices of
//! the REAL context — the agent's actual system prompt, messages, and tool
//! definitions.  The gateway does not replace the agent's prompt with a
//! synthetic "you are a worker" envelope.  It augments with a short preamble
//! and varies the depth per role:
//!
//! - Fast:       system prompt + last user msg + optional tool names
//! - Specialist: system prompt + last 4 msgs + optional tool summaries/schemas
//! - Strong:     system prompt + last 10 msgs + optional full tool schemas
//! - Reducer:    system prompt + worker outputs + optional full tool schemas

use crate::normalize::WorkerOutput;
use crate::session::Session;
use crate::worker::WorkerRole;
use serde_json::{Value, json};

const TOOL_RESULT_CONTEXT_WINDOW: usize = 10;
const TOOL_EVIDENCE_MAX_RESULTS: usize = 8;
const TOOL_EVIDENCE_MAX_RESULT_CHARS: usize = 800;
const TOOL_RESULT_RAW_MAX_CHARS: usize = 2_400;
const TOOL_RESULT_JSON_MAX_SCALARS: usize = 48;
const TOOL_RESULT_JSON_MAX_ARRAY_ITEMS: usize = 12;
const TOOL_RESULT_SCALAR_MAX_CHARS: usize = 180;

/// Packed context ready to send to a worker.
pub struct PackedContext {
    pub messages: Vec<Value>,
    pub max_tokens: u32,
    /// Tool definitions to forward (if any).  `None` means don't send tools.
    pub tools: Option<Value>,
}

/// Build a context packet for a worker based on its role.
///
/// Each worker gets a slice of the real conversation — the agent's actual
/// system prompt and messages — not a synthetic replacement.  The depth of
/// the slice and tool detail varies by role.
pub fn pack_for_worker(session: &Session, role: WorkerRole, has_tools: bool) -> PackedContext {
    pack_for_worker_selected(session, role, has_tools, &[])
}

/// Build a worker context with native tool schemas narrowed to
/// `selected_tool_names` when non-empty.
pub fn pack_for_worker_selected(
    session: &Session,
    role: WorkerRole,
    has_tools: bool,
    selected_tool_names: &[String],
) -> PackedContext {
    match role {
        WorkerRole::Fast => pack_fast(session, has_tools, selected_tool_names),
        WorkerRole::Specialist => pack_specialist(session, has_tools, selected_tool_names),
        WorkerRole::Strong | WorkerRole::Generalist | WorkerRole::Reducer => {
            pack_strong(session, has_tools, selected_tool_names)
        }
    }
}

// ── MoA preamble ─────────────────────────────────────────────────────
// A short addition to the system prompt.  Does NOT replace the agent's
// system prompt — it's prepended so the model still sees the original
// instructions.

const MOA_PREAMBLE: &str = "\
[Multiple models are analyzing this request in parallel. \
Respond with your best answer or tool call. Be direct.]";

/// Text-turn preamble.
///
/// The tool-turn wording ("your best answer **or tool call**. Be direct.") is
/// wrong on a text turn twice over: there is no tool to call, and "be direct"
/// pushes workers toward stubs. Since these drafts are the *input* to the
/// refinement round, brevity here compounds — measured end-to-end, MoA answers
/// ran ~3.3k chars against a ~4.1k-char solo baseline and lost on judged
/// quality. The study that showed the gain gave workers no such instruction.
const MOA_PREAMBLE_TEXT: &str = "\
[Multiple models are answering this request in parallel; the best parts of each \
will be combined. Give your most accurate and complete answer.]";

fn augmented_system_prompt_for_mode(session: &Session, include_tool_guidance: bool) -> String {
    let preamble = if include_tool_guidance {
        MOA_PREAMBLE
    } else {
        MOA_PREAMBLE_TEXT
    };
    match session.system_prompt() {
        Some(sp) => {
            let prompt = if include_tool_guidance {
                sp
            } else {
                strip_tool_guidance_sections(&sp)
            };
            format!("{preamble}\n\n{prompt}")
        }
        None => preamble.to_string(),
    }
}

fn strip_tool_guidance_sections(prompt: &str) -> String {
    const STRIPPED_HEADINGS: &[&str] = &["## Tooling", "## Tool Call Style"];

    let mut out = Vec::new();
    let mut skipping = false;
    for line in prompt.lines() {
        if line.starts_with("## ") {
            skipping = STRIPPED_HEADINGS
                .iter()
                .any(|heading| line.trim() == *heading);
        }
        if !skipping {
            out.push(line);
        }
    }

    out.join("\n").trim().to_string()
}

/// Augmented system prompt with a compact tool catalogue appended.
fn system_with_tool_names(
    session: &Session,
    has_tools: bool,
    selected_tool_names: &[String],
) -> String {
    let mut prompt = augmented_system_prompt_for_mode(session, has_tools);
    let tools = selected_tools(session, has_tools, selected_tool_names);
    let names = tool_names_from(tools.as_ref());
    if !names.is_empty() {
        prompt.push_str(&format!("\n\nAvailable tools: {}", names.join(", ")));
    }
    prompt
}

fn system_with_tool_summaries(
    session: &Session,
    has_tools: bool,
    selected_tool_names: &[String],
) -> String {
    let mut prompt = augmented_system_prompt_for_mode(session, has_tools);
    let tools = selected_tools(session, has_tools, selected_tool_names);
    let summaries = tool_summaries_from(tools.as_ref());
    if !summaries.is_empty() {
        prompt.push_str("\n\nAvailable tools:");
        for s in &summaries {
            prompt.push_str(&format!("\n  - {s}"));
        }
    }
    prompt
}

// ── Fast worker ──────────────────────────────────────────────────────
// System prompt + last user message + tool names only.
// Smallest context, quickest to respond.

fn pack_fast(session: &Session, has_tools: bool, selected_tool_names: &[String]) -> PackedContext {
    let system = system_with_tool_names(session, has_tools, selected_tool_names);
    let user_text = session.last_user_text();

    // Per-request sessions: the caller owns the multi-turn loop and
    // sends the full history each request. Continuation context lives
    // in `session.messages()`; this path intentionally trims to just
    // the last user message to keep the fast worker's context small.
    PackedContext {
        messages: vec![
            json!({"role": "system", "content": system}),
            json!({"role": "user", "content": user_text}),
        ],
        max_tokens: 256,
        tools: None, // Fast worker doesn't get tool schemas — just names
    }
}

// ── Specialist worker ────────────────────────────────────────────────
// System prompt + last 4 messages + tool name+description summaries.

fn pack_specialist(
    session: &Session,
    has_tools: bool,
    selected_tool_names: &[String],
) -> PackedContext {
    let system = system_with_tool_summaries(session, has_tools, selected_tool_names);

    let mut messages = vec![json!({"role": "system", "content": system})];

    // Recent messages — skip system (already included), skip raw tool results
    // (they'd confuse models that don't have the tool_call context)
    let recent = session.recent_messages(4);
    for msg in &recent {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role == "user" || (role == "assistant" && msg.get("tool_calls").is_none()) {
            messages.push(msg.clone());
        }
    }

    // Ensure the last message is the current user turn
    let user_text = session.last_user_text();
    if messages
        .last()
        .and_then(|m| m.get("content").and_then(|c| c.as_str()))
        != Some(&user_text)
    {
        messages.push(json!({"role": "user", "content": user_text}));
    }

    PackedContext {
        messages,
        max_tokens: 512,
        tools: selected_tools(session, has_tools, selected_tool_names),
    }
}

// ── Strong worker ────────────────────────────────────────────────────
// System prompt + last 10 messages + full tool schemas forwarded natively.
// This worker gets the deepest context and the actual tool definitions so
// it can produce native tool_calls if the backend supports it.

fn pack_strong(
    session: &Session,
    has_tools: bool,
    selected_tool_names: &[String],
) -> PackedContext {
    let system = augmented_system_prompt_for_mode(session, has_tools);

    let mut messages = vec![json!({"role": "system", "content": system})];

    // Deep recent history — include tool result messages too since this
    // worker gets full tool schemas and can understand the context
    let recent = session.recent_messages(10);
    for msg in &recent {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role != "system" && !role.is_empty() {
            messages.push(msg.clone());
        }
    }

    let user_text = session.last_user_text();
    if messages
        .last()
        .and_then(|m| m.get("content").and_then(|c| c.as_str()))
        != Some(&user_text)
    {
        messages.push(json!({"role": "user", "content": user_text}));
    }

    // Forward the real tool schemas — the strong worker can produce native
    // tool_calls through the OpenAI API
    let tools = selected_tools(session, has_tools, selected_tool_names);

    PackedContext {
        messages,
        max_tokens: 1024,
        tools,
    }
}

// ── Reducer / conflict resolution ────────────────────────────────────

/// Build context for the reducer when arbitration is needed.
///
/// The reducer gets: agent's system prompt + worker outputs + full tool
/// schemas.  It sees what the workers proposed and makes the final call.
pub fn pack_for_reducer(
    session: &Session,
    outputs: &[WorkerOutput],
    reason: &str,
    has_tools: bool,
) -> (Vec<Value>, Option<Value>) {
    pack_for_reducer_selected(session, outputs, reason, has_tools, &[])
}

/// Build reducer context with native tools narrowed to `selected_tool_names`
/// when non-empty.
pub fn pack_for_reducer_selected(
    session: &Session,
    outputs: &[WorkerOutput],
    reason: &str,
    has_tools: bool,
    selected_tool_names: &[String],
) -> (Vec<Value>, Option<Value>) {
    let user_text = session.last_user_text();

    // Synthesis framing adapted from Together's MoA aggregator prompt: tell the
    // model to synthesize (not relay) and warn that inputs may be wrong — the
    // second clause stops it averaging in confidently-wrong inputs. We add
    // per-worker attribution and per-payload length bounds (below), which
    // Together omits.
    // The reducer is the synthesizer, not one of the parallel answerers. Giving
    // it the *worker* preamble ("multiple models are answering in parallel;
    // give your most complete answer") on top of the synthesis instruction is
    // contradictory framing: it tells the model to both draft and aggregate. A
    // 32B reconciles it; an 8B reducer does not. So the reducer gets the
    // agent's own system prompt (tool guidance only when this is a tool turn)
    // plus the synthesis instruction — matching the harness configuration that
    // measured 12W/2L on a 6x8B pool.
    let mut system_parts: Vec<String> = Vec::new();
    if let Some(sp) = session.system_prompt() {
        system_parts.push(sp);
        system_parts.push(String::new());
    }
    if has_tools {
        system_parts.push(format!(
            "Multiple models analyzed this request. Reason for synthesis: {reason}"
        ));
    }
    system_parts.push(synthesis_instruction(has_tools));

    // Worker outputs
    system_parts.push(String::new());
    system_parts.push("## Worker outputs".to_string());
    let payload_budget = reducer_payload_budget(has_tools);
    for (i, output) in outputs.iter().enumerate() {
        // Anonymous on text turns, matching the measured configuration and
        // Hermes, which anonymizes reference outputs "to prevent aggregator
        // bias" — a named model invites deference to the name rather than the
        // content. Tool turns keep attribution: the reducer is arbitrating
        // between proposals and provenance is genuinely useful there, and it
        // is what the tool-path tests pin.
        if has_tools {
            system_parts.push(format!("\n[Worker {} — {}]:", i + 1, output.model));
        } else {
            system_parts.push(format!("\n[Response {}]:", i + 1));
        }
        let payload = if output.payload.len() > payload_budget {
            format!(
                "{}...",
                crate::worker::truncate_chars(&output.payload, payload_budget - 3)
            )
        } else {
            output.payload.clone()
        };
        system_parts.push(payload);
        // Truncated inputs are labelled so the reducer treats them as partial
        // material rather than copying a dangling sentence as a finished
        // answer. `is_usable_answer` already bars them from winning verbatim;
        // this is what lets them still contribute here.
        if output.truncated {
            system_parts
                .push("  → NOTE: cut off at the token limit — incomplete, do not copy".to_string());
        }
        if let Some(ref tool) = output.tool_name {
            system_parts.push(format!("  → Proposed tool: {tool}"));
            if let Some(ref args) = output.tool_arguments {
                system_parts.push(format!("  → Arguments: {args}"));
            }
        }
    }

    let tools = selected_tools(session, has_tools, selected_tool_names);

    (
        vec![
            json!({"role": "system", "content": system_parts.join("\n")}),
            json!({"role": "user", "content": user_text}),
        ],
        tools,
    )
}

/// Advisor framing for [`pack_for_reference`]. Deliberately does NOT ask for a
/// tool call: references hold no schemas, so requesting one yields tool-shaped
/// prose that can pull the actor off its own (better) choice.
const REFERENCE_PREAMBLE: &str = "\
You are advising another model that will decide and act on this request. \
You do not have tools and must not emit a tool call. Give a short, direct \
analysis: what the request is really asking, and what you would do. Be concise.";

/// Pack context for a **reference** (advisor), Hermes-style: only the
/// conversation's user/assistant text.
///
/// Three things are withheld on purpose:
/// * the agent's system prompt — an advisor told "you are a coding agent, run
///   the tests" role-plays the actor instead of advising it;
/// * the tool-call transcript — it anchors every advisor on the trajectory
///   already taken, collapsing the error-independence aggregation depends on;
/// * any instruction to emit a tool call (see [`REFERENCE_PREAMBLE`]).
///
/// The view is uniform across advisors (no per-role trimming) and is a stable
/// function of the history, so it caches across iterations.
pub fn pack_for_reference(session: &Session, max_messages: usize) -> PackedContext {
    let mut messages = vec![json!({"role": "system", "content": REFERENCE_PREAMBLE})];

    // User/assistant prose only: no system turn, no tool_calls, no tool results.
    let history: Vec<Value> = session
        .messages()
        .iter()
        .filter(|m| {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("");
            let is_prose = matches!(role, "user" | "assistant");
            let carries_tool_call = m.get("tool_calls").is_some();
            let has_text = m
                .get("content")
                .and_then(Value::as_str)
                .is_some_and(|s| !s.trim().is_empty());
            is_prose && !carries_tool_call && has_text
        })
        .cloned()
        .collect();

    let start = history.len().saturating_sub(max_messages);
    messages.extend_from_slice(&history[start..]);

    // Guarantee the current request is present even if it was filtered above.
    let user_text = session.last_user_text();
    let last_is_current = messages
        .last()
        .and_then(|m| m.get("content").and_then(Value::as_str))
        == Some(user_text.as_str());
    if !last_is_current && !user_text.is_empty() {
        messages.push(json!({"role": "user", "content": user_text}));
    }

    PackedContext {
        messages,
        max_tokens: 600, // Hermes caps advisors; the slowest advisor sets turn latency.
        tools: None,
    }
}

/// How much of each peer draft a refiner may see.
const REFINEMENT_DRAFT_BUDGET: usize = 4000;

/// What the reducer is asked to do with the worker outputs.
///
/// Text turns use the wording the committee study measured
/// (`evals/moa-openrouter/RESULTS.md`), which asks for a *well-structured*
/// synthesis. The previous text wording framed the turn as reconciling a
/// disagreement and ended with "Be concise" — measured end-to-end through
/// `handle_turn`, that produced ~2.0k-char answers against a ~4.1k-char solo
/// baseline and lost to it on judged quality. Terseness is not the goal on a
/// reasoning turn; accuracy and completeness are.
///
/// Tool turns keep the tight framing: the output there is an action, and the
/// reducer must be free to emit a tool call rather than prose.
fn synthesis_instruction(has_tools: bool) -> String {
    if has_tools {
        "You have been provided with their responses below. Synthesize them into ONE \
         final response — either a direct answer or a tool call. Critically evaluate \
         what they say: some of it may be biased or incorrect, and agreement between \
         workers is not proof of correctness. Do not simply copy the longest or most \
         confident response; produce the most accurate reply to the request. Be concise."
            .to_string()
    } else {
        "You have been given a user request and several candidate responses from other \
         models. Synthesize them into one high-quality response. Critically evaluate them — \
         some may be biased or incorrect, and agreement is not proof of correctness. Do not \
         merely copy the longest or most confident; produce the most accurate, \
         well-structured reply. Be direct."
            .to_string()
    }
}

/// How much of each worker payload the reducer may see.
///
/// Tool turns keep the tight bound: the reducer is choosing an action, the
/// signal is the proposal itself, and long prose crowds out the tool schemas.
///
/// Text turns need far more. Measured refined answers average ~3.8k chars
/// (`evals/moa-openrouter/RESULTS.md`), so a 500-char bound would hand the
/// reducer ~13% of each answer and discard exactly the content refinement just
/// produced — the measured gain could not survive it. Together's aggregator
/// passes references unbounded; we keep a bound so a pathological worker can't
/// blow the context, just a realistic one.
fn reducer_payload_budget(has_tools: bool) -> usize {
    if has_tools { 500 } else { 4000 }
}

/// Pack context for a worker in the cross-peer refinement round.
///
/// The worker sees every round-1 draft (its own included, anonymized) and
/// rewrites its answer. Anonymizing keeps the worker from deferring to a name
/// it recognizes, and the framing asks for an improved answer rather than a
/// critique — the reducer still does the final synthesis.
pub fn pack_for_refinement(session: &Session, drafts: &[String]) -> PackedContext {
    // Wording deliberately matches the eval that measured the +0.250 gain
    // (`evals/moa-openrouter/RESULTS.md`), which in turn matches Together's
    // `advanced-moa.py` — it reuses the aggregator prompt for refinement
    // layers. A refinement-specific wording may well read better, but this is
    // the configuration with evidence behind it; changing it should be a
    // measured change, not an assumed improvement.
    let mut system = String::from(
        "You have been given a user request and several candidate responses from \
         other models. Synthesize them into one high-quality response. Critically \
         evaluate them — some may be biased or incorrect, and agreement is not proof \
         of correctness. Do not merely copy the longest or most confident; produce \
         the most accurate, well-structured reply. Be direct.\n\nCandidate responses:",
    );
    for (i, d) in drafts.iter().enumerate() {
        // Same reasoning as `reducer_payload_budget`: measured drafts average
        // ~3.8k chars, so a tight bound would hand each refiner a fraction of
        // what its peers actually said — the input the round exists to use.
        let bounded = if d.len() > REFINEMENT_DRAFT_BUDGET {
            format!(
                "{}...",
                crate::worker::truncate_chars(d, REFINEMENT_DRAFT_BUDGET - 3)
            )
        } else {
            d.clone()
        };
        system.push_str(&format!("\n[Response {}]:\n{bounded}\n", i + 1));
    }

    PackedContext {
        messages: vec![
            json!({"role": "system", "content": system}),
            json!({"role": "user", "content": session.last_user_text()}),
        ],
        max_tokens: 1024,
        tools: None,
    }
}

/// Pack context for the actor in the asymmetric tool path: "here is advice, now
/// you act" (not the reducer's "you disagreed, reconcile"). Advice is prose,
/// per-model length-bounded and truncation-labelled; `has_tools` /
/// `selected_tool_names` attach the real tools the advisors never saw.
pub fn pack_for_actor(
    session: &Session,
    references: &[WorkerOutput],
    has_tools: bool,
    selected_tool_names: &[String],
) -> (Vec<Value>, Option<Value>) {
    let user_text = session.last_user_text();

    let mut system_parts = vec![
        augmented_system_prompt_for_mode(session, has_tools),
        String::new(),
        "Other models were asked to advise on this request. They did not have \
         access to tools; you do. Use their advice as input, but you decide the \
         action. Critically evaluate what they say — some of it may be biased or \
         incorrect, and agreement between them is not proof of correctness. \
         Respond with the single best action: a direct answer, or the appropriate \
         tool call. Be concise."
            .to_string(),
    ];

    if references.is_empty() {
        // No advice in time (slow/absent peers): actor proceeds alone.
        system_parts.push(String::new());
        system_parts
            .push("(No advice from other models arrived in time — proceed on your own.)".into());
    } else {
        system_parts.push(String::new());
        system_parts.push("## Advice from other models".to_string());
        for (i, r) in references.iter().enumerate() {
            system_parts.push(format!("\n[Advisor {} — {}]:", i + 1, r.model));
            let payload = if r.payload.len() > 500 {
                format!("{}...", crate::worker::truncate_chars(&r.payload, 497))
            } else {
                r.payload.clone()
            };
            system_parts.push(payload);
            if r.truncated {
                system_parts.push(
                    "  → NOTE: cut off at the token limit — incomplete, treat as partial".into(),
                );
            }
        }
    }

    let tools = selected_tools(session, has_tools, selected_tool_names);

    (
        vec![
            json!({"role": "system", "content": system_parts.join("\n")}),
            json!({"role": "user", "content": user_text}),
        ],
        tools,
    )
}

fn selected_tools(
    session: &Session,
    has_tools: bool,
    selected_tool_names: &[String],
) -> Option<Value> {
    if !has_tools {
        return None;
    }

    let tools = session.tools()?;
    if selected_tool_names.is_empty() {
        return Some(tools.clone());
    }

    let selected: std::collections::HashSet<String> = selected_tool_names
        .iter()
        .map(|name| name.to_ascii_lowercase())
        .collect();
    let filtered: Vec<Value> = tools
        .as_array()
        .into_iter()
        .flatten()
        .filter(|tool| {
            tool.pointer("/function/name")
                .and_then(Value::as_str)
                .map(|name| selected.contains(&name.to_ascii_lowercase()))
                .unwrap_or(false)
        })
        .cloned()
        .collect();

    if filtered.is_empty() {
        Some(tools.clone())
    } else {
        Some(Value::Array(filtered))
    }
}

fn tool_names_from(tools: Option<&Value>) -> Vec<String> {
    tools
        .and_then(Value::as_array)
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| {
                    tool.pointer("/function/name")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
                .collect()
        })
        .unwrap_or_default()
}

fn tool_summaries_from(tools: Option<&Value>) -> Vec<String> {
    tools
        .and_then(Value::as_array)
        .map(|tools| {
            tools
                .iter()
                .filter_map(|tool| {
                    let name = tool.pointer("/function/name")?.as_str()?;
                    let desc = tool
                        .pointer("/function/description")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    let first_line = desc.lines().next().unwrap_or(desc);
                    let truncated = if first_line.len() > 80 {
                        format!("{}...", crate::worker::truncate_chars(first_line, 77))
                    } else {
                        first_line.to_string()
                    };
                    Some(format!("{name}: {truncated}"))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Build context for a tool-result turn (reducer only, not full fan-out).
///
/// The reducer gets: agent's system prompt + the original conversation
/// including assistant tool_call messages and the corresponding tool result
/// messages, plus full tool schemas so it can propose the next call.
///
/// We forward the raw message sequence rather than summarizing, because
/// the reducer model needs to see the tool_call → tool result pairs in
/// their native OpenAI format to reason about what happened and decide
/// what to do next.
pub fn pack_for_tool_result_turn(
    session: &Session,
    has_tools: bool,
) -> (Vec<Value>, Option<Value>) {
    pack_for_tool_result_turn_selected(session, has_tools, &[])
}

/// Build context for a tool-result turn with native tools narrowed to
/// `selected_tool_names` when non-empty.
pub fn pack_for_tool_result_turn_selected(
    session: &Session,
    has_tools: bool,
    selected_tool_names: &[String],
) -> (Vec<Value>, Option<Value>) {
    let mut system = augmented_system_prompt_for_mode(session, has_tools);
    if let Some(evidence) = tool_evidence_text(session) {
        system.push_str("\n\n");
        system.push_str(&evidence);
    }

    // Qwen3.5 and other strict chat templates allow a system message only in
    // the first position. Keep deterministic tool evidence inside that first
    // system message instead of emitting a second adjacent system message.
    let mut messages = vec![json!({"role": "system", "content": system})];

    // Forward the tail of the conversation that includes the current user turn,
    // assistant tool_call messages, and their tool results. Tool-call chains
    // can span multiple assistant/tool pairs; starting at the message before
    // the last assistant tool_call can leave a leading `tool` message, which
    // many chat templates reject.
    let all = session.all_messages();
    let mut start_idx = all.len().saturating_sub(TOOL_RESULT_CONTEXT_WINDOW);

    // Prefer the nearest user message before the latest tool result so the
    // reducer sees a valid user -> assistant(tool_calls) -> tool chain.
    let latest_tool_user_idx = all
        .iter()
        .rposition(|msg| message_role(msg) == "tool")
        .and_then(|last_tool_idx| {
            all[..=last_tool_idx]
                .iter()
                .rposition(|msg| message_role(msg) == "user")
        });

    if latest_tool_user_idx.is_none() {
        // Fall back to the last assistant tool_call message. This keeps the
        // message sequence syntactically valid even if no user message is
        // present in malformed input.
        for (i, msg) in all.iter().enumerate().rev() {
            if message_role(msg) == "assistant" && msg.get("tool_calls").is_some() {
                start_idx = i;
                break;
            }
        }
    }

    start_idx = valid_tool_result_start_idx(&all, start_idx);
    let prefix_user_idx = latest_tool_user_idx
        .filter(|user_idx| *user_idx < start_idx)
        .filter(|_| {
            !all[start_idx..]
                .iter()
                .any(|msg| message_role(msg) == "user")
        });

    if let Some(user_idx) = prefix_user_idx {
        messages.push(all[user_idx].clone());
    }

    for msg in &all[start_idx..] {
        let role = message_role(msg);
        if role != "system" && !role.is_empty() {
            messages.push(compact_tool_message(msg));
        }
    }

    let tools = selected_tools(session, has_tools, selected_tool_names);

    (messages, tools)
}

fn tool_evidence_text(session: &Session) -> Option<String> {
    let results = session.recent_tool_results();
    if results.is_empty() {
        return None;
    }

    let mut lines = vec![
        "Completed tool results. Preserve exact short values from these results when the user asks to include, recall, or return tool facts."
            .to_string(),
    ];
    for (idx, (name, result)) in results
        .iter()
        .rev()
        .take(TOOL_EVIDENCE_MAX_RESULTS)
        .enumerate()
    {
        let compacted = compact_tool_result_text(result);
        let result = if compacted.len() > TOOL_EVIDENCE_MAX_RESULT_CHARS {
            format!(
                "{}...",
                crate::worker::truncate_chars(&compacted, TOOL_EVIDENCE_MAX_RESULT_CHARS - 3)
            )
        } else {
            compacted
        };
        lines.push(format!("{}. {name}: {result}", idx + 1));
    }

    Some(lines.join("\n"))
}

fn compact_tool_message(msg: &Value) -> Value {
    if message_role(msg) != "tool" {
        return msg.clone();
    }

    let Some(content) = msg.get("content").and_then(Value::as_str) else {
        return msg.clone();
    };
    let compacted = compact_tool_result_text(content);
    if compacted == content {
        return msg.clone();
    }

    let mut compact = msg.clone();
    if let Some(obj) = compact.as_object_mut() {
        obj.insert("content".to_string(), Value::String(compacted));
    }
    compact
}

fn compact_tool_result_text(result: &str) -> String {
    if result.len() <= TOOL_RESULT_RAW_MAX_CHARS {
        return result.to_string();
    }

    if let Ok(json) = serde_json::from_str::<Value>(result) {
        return compact_json_tool_result(result.len(), &json);
    }

    format!(
        "Tool result compacted from {} chars; original was plain text.\n\
         Text preview:\n{}...",
        result.len(),
        crate::worker::truncate_chars(result, TOOL_RESULT_RAW_MAX_CHARS - 96)
    )
}

fn compact_json_tool_result(original_len: usize, value: &Value) -> String {
    let mut lines = vec![format!(
        "Tool result compacted from {original_len} chars; original was JSON."
    )];
    append_json_shape(value, &mut lines);
    let mut scalars = Vec::new();
    collect_json_scalars(value, "$", &mut scalars, 0);

    if scalars.is_empty() {
        lines.push("No compact scalar fields found.".to_string());
    } else {
        lines.push("Key scalar fields:".to_string());
        lines.extend(scalars.into_iter().map(|line| format!("- {line}")));
    }

    lines.join("\n")
}

fn append_json_shape(value: &Value, lines: &mut Vec<String>) {
    match value {
        Value::Array(items) => {
            lines.push(format!("JSON array with {} item(s).", items.len()));
        }
        Value::Object(map) => {
            lines.push(format!("JSON object with {} top-level key(s).", map.len()));
        }
        _ => {}
    }
}

fn collect_json_scalars(value: &Value, path: &str, out: &mut Vec<String>, depth: usize) {
    if out.len() >= TOOL_RESULT_JSON_MAX_SCALARS || depth > 6 {
        return;
    }

    match value {
        Value::Array(items) => collect_array_scalars(items, path, out, depth),
        Value::Object(map) => collect_object_scalars(map, path, out, depth),
        _ => push_scalar(path, value, out),
    }
}

fn collect_array_scalars(items: &[Value], path: &str, out: &mut Vec<String>, depth: usize) {
    for (idx, item) in items
        .iter()
        .take(TOOL_RESULT_JSON_MAX_ARRAY_ITEMS)
        .enumerate()
    {
        if out.len() >= TOOL_RESULT_JSON_MAX_SCALARS {
            break;
        }

        let item_path = format!("{path}[{idx}]");
        if let Some(row) = compact_object_row(item, &item_path) {
            out.push(row);
        } else {
            collect_json_scalars(item, &item_path, out, depth + 1);
        }
    }
}

fn collect_object_scalars(
    map: &serde_json::Map<String, Value>,
    path: &str,
    out: &mut Vec<String>,
    depth: usize,
) {
    for key in PREFERRED_JSON_KEYS {
        if out.len() >= TOOL_RESULT_JSON_MAX_SCALARS {
            return;
        }
        let Some(value) = map.get(*key) else {
            continue;
        };
        if is_scalar(value) {
            push_scalar(&format!("{path}.{key}"), value, out);
        }
    }

    for (key, value) in map {
        if out.len() >= TOOL_RESULT_JSON_MAX_SCALARS {
            return;
        }
        if PREFERRED_JSON_KEYS.contains(&key.as_str()) {
            continue;
        }
        let child_path = format!("{path}.{key}");
        collect_json_scalars(value, &child_path, out, depth + 1);
    }
}

fn compact_object_row(value: &Value, path: &str) -> Option<String> {
    let map = value.as_object()?;
    let mut fields = Vec::new();
    for key in PREFERRED_JSON_KEYS {
        let Some(value) = map.get(*key) else {
            continue;
        };
        if let Some(scalar) = scalar_to_string(value) {
            fields.push(format!("{key}={scalar}"));
        }
        if fields.len() >= 6 {
            break;
        }
    }

    (!fields.is_empty()).then(|| format!("{path}: {}", fields.join(", ")))
}

fn push_scalar(path: &str, value: &Value, out: &mut Vec<String>) {
    let Some(scalar) = scalar_to_string(value) else {
        return;
    };
    out.push(format!("{path}: {scalar}"));
}

fn scalar_to_string(value: &Value) -> Option<String> {
    match value {
        Value::String(text) => Some(format!(
            "\"{}\"",
            crate::worker::truncate_chars(text, TOOL_RESULT_SCALAR_MAX_CHARS)
        )),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        Value::Null | Value::Array(_) | Value::Object(_) => None,
    }
}

fn is_scalar(value: &Value) -> bool {
    matches!(value, Value::String(_) | Value::Number(_) | Value::Bool(_))
}

const PREFERRED_JSON_KEYS: &[&str] = &[
    "number",
    "title",
    "name",
    "full_name",
    "state",
    "status",
    "html_url",
    "url",
    "path",
    "file",
    "value",
    "fact",
    "result",
    "answer",
    "summary",
    "message",
    "stdout",
    "stderr",
    "description",
];

fn message_role(msg: &Value) -> &str {
    msg.get("role").and_then(|r| r.as_str()).unwrap_or("")
}

fn valid_tool_result_start_idx(all: &[Value], start_idx: usize) -> usize {
    let Some(first_non_system_idx) = all
        .iter()
        .enumerate()
        .skip(start_idx)
        .find_map(|(idx, msg)| (message_role(msg) != "system").then_some(idx))
    else {
        return start_idx;
    };

    if message_role(&all[first_non_system_idx]) != "tool" {
        return start_idx;
    }

    all[..first_non_system_idx]
        .iter()
        .enumerate()
        .rev()
        .find_map(|(idx, msg)| {
            (message_role(msg) == "assistant" && msg.get("tool_calls").is_some()).then_some(idx)
        })
        .unwrap_or(start_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::{OutputKind, WorkerOutput};
    use serde_json::json;

    fn user_msg(text: &str) -> Value {
        json!({"role": "user", "content": text})
    }
    fn system_msg(text: &str) -> Value {
        json!({"role": "system", "content": text})
    }
    fn assistant_msg(text: &str) -> Value {
        json!({"role": "assistant", "content": text})
    }
    fn assistant_tool_msg(id: &str, name: &str, arguments: Value) -> Value {
        json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments.to_string(),
                },
            }],
        })
    }
    fn tool_result_msg(id: &str, text: &str) -> Value {
        json!({"role": "tool", "tool_call_id": id, "content": text})
    }
    fn tools_two() -> Value {
        json!([
            {"type": "function", "function": {
                "name": "read_file",
                "description": "Read a file from disk",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
            }},
            {"type": "function", "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}
            }},
        ])
    }

    fn read_write_tools() -> Value {
        json!([
            {"type": "function", "function": {
                "name": "read_file",
                "description": "Read a file from disk",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
            }},
            {"type": "function", "function": {
                "name": "write_file",
                "description": "Write a file to disk",
                "parameters": {"type": "object", "properties": {
                    "path": {"type": "string"}, "content": {"type": "string"}
                }}
            }},
        ])
    }
    fn weather_tools() -> Value {
        json!([
            {"type": "function", "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }},
            {"type": "function", "function": {
                "name": "web_fetch",
                "description": "Fetch a URL",
                "parameters": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"]
                }
            }},
        ])
    }

    fn session_with(messages: &[Value], tools: Option<Value>) -> Session {
        let mut s = Session::new();
        s.ingest(messages, &tools);
        s
    }

    /// The reducer must actually see the answers it is synthesizing.
    ///
    /// Measured refined answers average ~3.8k chars. The old flat 500-char
    /// bound handed the reducer ~13% of each one, discarding exactly the
    /// content the refinement round produces — the measured gain could not
    /// have survived it. Tool turns keep the tight bound (the signal is the
    /// proposal, and prose crowds out schemas).
    #[test]
    fn text_reducer_sees_realistic_answer_lengths() {
        let long_answer = "x".repeat(3800);
        let outputs = vec![
            WorkerOutput {
                kind: OutputKind::Answer,
                confidence: 0.5,
                tool_name: None,
                tool_arguments: None,
                payload: long_answer.clone(),
                model: "a".into(),
                role: WorkerRole::Generalist,
                elapsed_ms: 0,
                truncated: false,
            },
            WorkerOutput {
                kind: OutputKind::Answer,
                confidence: 0.5,
                tool_name: None,
                tool_arguments: None,
                payload: long_answer,
                model: "b".into(),
                role: WorkerRole::Generalist,
                elapsed_ms: 0,
                truncated: false,
            },
        ];
        let session = session_with(&[json!({"role": "user", "content": "explain"})], None);

        let (messages, _) =
            pack_for_reducer_selected(&session, &outputs, "no agreement", false, &[]);
        let sys = system_text(&messages);
        // Each 3800-char answer must survive largely intact (2 answers).
        assert!(
            sys.len() > 7000,
            "text reducer truncated the answers it must synthesize: {} chars",
            sys.len()
        );
    }

    /// Tool turns keep the tight payload bound.
    #[test]
    fn tool_reducer_keeps_the_tight_payload_bound() {
        let outputs = vec![WorkerOutput {
            kind: OutputKind::Answer,
            confidence: 0.5,
            tool_name: None,
            tool_arguments: None,
            payload: "x".repeat(3800),
            model: "a".into(),
            role: WorkerRole::Generalist,
            elapsed_ms: 0,
            truncated: false,
        }];
        let session = session_with(&[json!({"role": "user", "content": "read a file"})], None);

        let (messages, _) = pack_for_reducer_selected(&session, &outputs, "conflict", true, &[]);
        let sys = system_text(&messages);
        assert!(
            !sys.contains(&"x".repeat(1000)),
            "tool reducer must keep payloads tight so schemas aren't crowded out"
        );
    }

    /// Refiners must see what their peers actually said, for the same reason.
    #[test]
    fn refiners_see_realistic_peer_draft_lengths() {
        let session = session_with(&[json!({"role": "user", "content": "explain"})], None);
        let drafts = vec!["y".repeat(3800), "z".repeat(3800)];

        let packed = pack_for_refinement(&session, &drafts);
        let sys = system_text(&packed.messages);
        assert!(
            sys.len() > 7000,
            "refiners were handed a fraction of their peers' drafts: {} chars",
            sys.len()
        );
    }

    /// Advisors must not be told to emit a tool call. Asking a schema-less
    /// model for one yields tool-shaped prose, which measurably pulled the
    /// actor off its own better choice.
    #[test]
    fn reference_packing_never_requests_a_tool_call() {
        let s = session_with(&[json!({"role": "user", "content": "list src"})], None);
        let packed = pack_for_reference(&s, 6);
        let sys = system_text(&packed.messages).to_lowercase();
        assert!(sys.contains("must not emit a tool call"));
        assert!(packed.tools.is_none(), "advisors never receive schemas");
    }

    /// The agent's system prompt is withheld: an advisor handed "you are a
    /// coding agent, run the tests" role-plays the actor instead of advising.
    #[test]
    fn reference_packing_strips_the_agent_system_prompt() {
        let s = session_with(
            &[
                json!({"role": "system", "content": "You are a coding agent. SECRET_MARKER."}),
                json!({"role": "user", "content": "what is failing?"}),
            ],
            None,
        );
        let packed = pack_for_reference(&s, 6);
        let all = serde_json::to_string(&packed.messages).unwrap();
        assert!(
            !all.contains("SECRET_MARKER"),
            "agent system prompt must not reach advisors: {all}"
        );
    }

    /// The tool transcript is withheld so advisors stay independent of the
    /// trajectory already taken — error independence is what makes
    /// aggregation worth anything.
    #[test]
    fn reference_packing_strips_the_tool_transcript() {
        let s = session_with(
            &[
                json!({"role": "user", "content": "find the bug"}),
                json!({"role": "assistant", "content": Value::Null,
                       "tool_calls": [{"id": "1", "type": "function",
                           "function": {"name": "list_dir", "arguments": "{\"path\":\"TRAJECTORY\"}"}}]}),
                json!({"role": "tool", "tool_call_id": "1", "content": "TOOL_RESULT_MARKER"}),
                json!({"role": "user", "content": "and now?"}),
            ],
            None,
        );
        let packed = pack_for_reference(&s, 6);
        let all = serde_json::to_string(&packed.messages).unwrap();
        assert!(!all.contains("TOOL_RESULT_MARKER"), "tool results leaked");
        assert!(!all.contains("TRAJECTORY"), "prior tool_calls leaked");
        assert!(all.contains("and now?"), "current request must survive");
    }

    /// Uniform view regardless of role: every advisor sees the same prose,
    /// so the packing is a stable function of history (and caches).
    #[test]
    fn reference_packing_keeps_user_assistant_prose() {
        let s = session_with(
            &[
                json!({"role": "user", "content": "first question"}),
                json!({"role": "assistant", "content": "first answer"}),
                json!({"role": "user", "content": "second question"}),
            ],
            None,
        );
        let packed = pack_for_reference(&s, 6);
        let all = serde_json::to_string(&packed.messages).unwrap();
        assert!(all.contains("first question"));
        assert!(all.contains("first answer"));
        assert!(all.contains("second question"));
    }

    /// Helper: extract the system message content from a packed message vec.
    fn system_text(messages: &[Value]) -> String {
        messages
            .iter()
            .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
            .and_then(|m| m.get("content").and_then(|c| c.as_str()))
            .unwrap_or("")
            .to_string()
    }

    // ── pack_for_worker: shape contract per role ─────────────────────

    #[test]
    fn fast_worker_has_system_user_only_no_tools() {
        let s = session_with(
            &[
                system_msg("You are a helpful assistant."),
                user_msg("first"),
                assistant_msg("first reply"),
                user_msg("second"),
            ],
            Some(tools_two()),
        );
        let packed = pack_for_worker(&s, WorkerRole::Fast, true);

        assert_eq!(packed.max_tokens, 256, "fast worker token budget");
        assert!(
            packed.tools.is_none(),
            "fast worker must not receive tool schemas"
        );
        assert_eq!(packed.messages.len(), 2, "fast = system + last user only");
        assert_eq!(
            packed.messages[0].get("role").and_then(|r| r.as_str()),
            Some("system"),
        );
        assert_eq!(
            packed.messages[1].get("role").and_then(|r| r.as_str()),
            Some("user"),
        );
        assert_eq!(
            packed.messages[1].get("content").and_then(|c| c.as_str()),
            Some("second"),
            "fast worker sees only the LAST user message",
        );

        // Tool *names* appear in system prompt; full schemas do not.
        let sys = system_text(&packed.messages);
        assert!(
            sys.contains("read_file"),
            "tool names present in system: {sys}"
        );
        assert!(
            sys.contains("web_search"),
            "tool names present in system: {sys}"
        );
        assert!(
            !sys.contains("\"parameters\""),
            "fast worker system must not contain JSON Schema fragments: {sys}",
        );
    }

    #[test]
    fn specialist_worker_has_summaries_and_native_tools() {
        let s = session_with(
            &[
                system_msg("Agent SP."),
                user_msg("m1"),
                assistant_msg("r1"),
                user_msg("m2"),
                assistant_msg("r2"),
                user_msg("m3"),
            ],
            Some(tools_two()),
        );
        let packed = pack_for_worker(&s, WorkerRole::Specialist, true);

        assert_eq!(packed.max_tokens, 512, "specialist token budget");
        assert!(
            packed.tools.is_some(),
            "specialist must receive full native tool schemas",
        );
        // Tool *summaries* (name + description) must be in the system prompt.
        let sys = system_text(&packed.messages);
        assert!(sys.contains("read_file"));
        assert!(
            sys.contains("Read a file"),
            "specialist system should include tool descriptions: {sys}",
        );

        // Last message is the latest user turn ("m3").
        let last = packed.messages.last().unwrap();
        assert_eq!(last.get("role").and_then(|r| r.as_str()), Some("user"));
        assert_eq!(last.get("content").and_then(|c| c.as_str()), Some("m3"));
    }

    #[test]
    fn ordinary_chat_omits_tool_summaries_and_native_tools() {
        let s = session_with(
            &[system_msg("Agent SP."), user_msg("What can you help with?")],
            Some(tools_two()),
        );
        let specialist = pack_for_worker(&s, WorkerRole::Specialist, false);
        let strong = pack_for_worker(&s, WorkerRole::Strong, false);

        assert!(specialist.tools.is_none());
        assert!(strong.tools.is_none());
        assert!(!system_text(&specialist.messages).contains("read_file"));
        assert!(!system_text(&strong.messages).contains("read_file"));
    }

    #[test]
    fn tool_selection_filters_native_tool_schemas() {
        let s = session_with(&[user_msg("Read the file")], Some(tools_two()));
        let selected = vec!["read_file".to_string()];
        let packed = pack_for_worker_selected(&s, WorkerRole::Strong, true, &selected);
        let tools = packed
            .tools
            .as_ref()
            .and_then(Value::as_array)
            .expect("selected tools array");

        assert_eq!(tools.len(), 1);
        assert_eq!(
            tools[0].pointer("/function/name").and_then(Value::as_str),
            Some("read_file")
        );
    }

    #[test]
    fn strong_worker_has_deep_history_and_native_tools() {
        // Build a session with many turns so we can verify depth.
        let mut msgs = vec![system_msg("Agent ST.")];
        for i in 0..8 {
            msgs.push(user_msg(&format!("u{i}")));
            msgs.push(assistant_msg(&format!("a{i}")));
        }
        msgs.push(user_msg("final"));
        let s = session_with(&msgs, Some(tools_two()));

        let packed = pack_for_worker(&s, WorkerRole::Strong, true);

        assert_eq!(packed.max_tokens, 1024, "strong token budget");
        assert!(
            packed.tools.is_some(),
            "strong must receive full native tool schemas",
        );
        // Strong gets up to last 10 messages on top of the system prompt,
        // so it should see deeper history than the specialist's 4-message window.
        assert!(
            packed.messages.len() >= 6,
            "strong worker should retain deep history, got {} messages",
            packed.messages.len(),
        );
        let last = packed.messages.last().unwrap();
        assert_eq!(last.get("content").and_then(|c| c.as_str()), Some("final"));
    }

    #[test]
    fn tool_result_reducer_context_keeps_chained_tool_messages_valid() {
        let s = session_with(
            &[
                user_msg("What is the weather today?"),
                assistant_tool_msg(
                    "call_search",
                    "web_search",
                    json!({"query": "weather Sydney today"}),
                ),
                tool_result_msg("call_search", "Search results include BOM and Weatherzone."),
                assistant_tool_msg(
                    "call_fetch",
                    "web_fetch",
                    json!({"url": "https://www.bom.gov.au/location/sydney"}),
                ),
                tool_result_msg("call_fetch", "BOM page content..."),
            ],
            Some(weather_tools()),
        );

        let (messages, tools) = pack_for_tool_result_turn(&s, true);
        let roles: Vec<&str> = messages
            .iter()
            .filter_map(|m| m.get("role").and_then(|r| r.as_str()))
            .collect();

        assert_eq!(
            roles,
            vec!["system", "user", "assistant", "tool", "assistant", "tool"],
            "tool-result reducer context must have one leading system message and must not start with a bare tool message",
        );
        assert_eq!(
            messages[1].get("content").and_then(|c| c.as_str()),
            Some("What is the weather today?"),
        );
        assert!(
            messages[0]
                .get("content")
                .and_then(Value::as_str)
                .is_some_and(|content| content.contains("Completed tool results.")),
            "tool evidence must be merged into the first system message",
        );
        assert!(
            messages[2].get("tool_calls").is_some(),
            "first tool result must retain its preceding assistant tool_call",
        );
        assert!(
            messages[4].get("tool_calls").is_some(),
            "latest tool result must retain its preceding assistant tool_call",
        );
        assert!(
            tools.is_some(),
            "tool-result reducer should still receive native tool schemas",
        );
    }

    #[test]
    fn strict_tool_result_reducer_uses_single_leading_system_message() {
        let s = session_with(
            &[
                system_msg("You are an agent. Use tools and return the result."),
                user_msg("Inspect the current task and report the result."),
                assistant_tool_msg(
                    "call_inspect",
                    "shell",
                    json!({"command": "inspect-task --current"}),
                ),
                tool_result_msg("call_inspect", "{\"status\":\"ready\"}"),
            ],
            Some(json!([{
                "type": "function",
                "function": {
                    "name": "shell",
                    "description": "Run a shell command",
                    "parameters": {"type": "object"}
                }
            }])),
        );

        let (messages, tools) = pack_for_tool_result_turn(&s, true);
        let roles: Vec<&str> = messages
            .iter()
            .filter_map(|message| message.get("role").and_then(Value::as_str))
            .collect();

        assert_eq!(roles, vec!["system", "user", "assistant", "tool"]);
        assert_eq!(roles.iter().filter(|role| **role == "system").count(), 1);
        assert!(
            messages[0]
                .get("content")
                .and_then(Value::as_str)
                .is_some_and(|content| content.contains("Completed tool results.")),
        );
        assert!(tools.is_some());
    }

    #[test]
    fn tool_result_reducer_keeps_read_and_write_schemas_across_the_chain() {
        let s = session_with(
            &[
                user_msg("Read both inputs, calculate the totals, then write the report."),
                assistant_tool_msg("call_read_a", "read_file", json!({"path": "/tmp/a"})),
                tool_result_msg("call_read_a", "north: 174"),
                assistant_tool_msg("call_read_b", "read_file", json!({"path": "/tmp/b"})),
                tool_result_msg("call_read_b", "south: 174"),
            ],
            Some(read_write_tools()),
        );
        let selected = vec!["read_file".to_string(), "write_file".to_string()];
        let (_messages, tools) = pack_for_tool_result_turn_selected(&s, true, &selected);
        let names = tool_names_from(tools.as_ref());

        assert_eq!(names, vec!["read_file", "write_file"]);
    }

    #[test]
    fn small_tool_result_content_is_preserved_exactly() {
        let s = session_with(
            &[
                user_msg("Read /tmp/a"),
                assistant_tool_msg("call_read", "read_file", json!({"path": "/tmp/a"})),
                tool_result_msg("call_read", "short exact result"),
            ],
            Some(tools_two()),
        );

        let (messages, _tools) = pack_for_tool_result_turn(&s, true);
        let tool = messages
            .iter()
            .find(|msg| msg.get("role").and_then(Value::as_str) == Some("tool"))
            .expect("tool message");

        assert_eq!(
            tool.get("content").and_then(Value::as_str),
            Some("short exact result")
        );
    }

    #[test]
    fn large_json_tool_result_is_compacted_for_reducer() {
        let noisy_body = "x".repeat(8_000);
        let result = json!([
            {
                "number": 801,
                "title": "Batch Skippy decode across concurrent requests",
                "html_url": "https://github.com/Mesh-LLM/mesh-llm/pull/801",
                "body": noisy_body,
                "user": {"login": "i386"}
            },
            {
                "number": 800,
                "title": "Reuse Skippy forwarded decode frames",
                "html_url": "https://github.com/Mesh-LLM/mesh-llm/issues/800",
                "body": "y".repeat(8_000),
                "user": {"login": "i386"}
            },
            {
                "number": 799,
                "title": "Reuse Skippy decode wire messages",
                "html_url": "https://github.com/Mesh-LLM/mesh-llm/issues/799",
                "body": "z".repeat(8_000),
                "user": {"login": "i386"}
            }
        ])
        .to_string();
        assert!(result.len() > TOOL_RESULT_RAW_MAX_CHARS);

        let s = session_with(
            &[
                user_msg("Summarize the issues"),
                assistant_tool_msg(
                    "call_exec",
                    "exec",
                    json!({"command": "curl https://api.github.com/repos/Mesh-LLM/mesh-llm/issues"}),
                ),
                tool_result_msg("call_exec", &result),
            ],
            Some(tools_two()),
        );

        let (messages, _tools) = pack_for_tool_result_turn(&s, true);
        let tool = messages
            .iter()
            .find(|msg| msg.get("role").and_then(Value::as_str) == Some("tool"))
            .expect("tool message");
        let content = tool
            .get("content")
            .and_then(Value::as_str)
            .expect("compacted content");

        assert!(content.contains("Tool result compacted from"));
        assert!(content.contains("$[0]: number=801"));
        assert!(content.contains("Batch Skippy decode across concurrent requests"));
        assert!(content.contains("$[1]: number=800"));
        assert!(content.contains("Reuse Skippy forwarded decode frames"));
        assert!(content.contains("$[2]: number=799"));
        assert!(content.contains("Reuse Skippy decode wire messages"));
        assert!(
            content.len() < 2_000,
            "compacted tool content should be small, got {} chars:\n{content}",
            content.len()
        );
        assert!(
            !content.contains(&"x".repeat(512)),
            "large noisy fields should not be forwarded raw"
        );
    }

    #[test]
    fn tool_result_reducer_strips_tool_guidance_when_tools_disabled() {
        let s = session_with(
            &[
                system_msg(
                    "You are helpful.\n## Tooling\ntool list goes here\n## Tool Call Style\ncall policy",
                ),
                user_msg("Answer without tools"),
                assistant_tool_msg("call_read", "read_file", json!({"path": "/tmp/a"})),
                tool_result_msg("call_read", "done"),
            ],
            Some(tools_two()),
        );

        let (messages, tools) = pack_for_tool_result_turn(&s, false);
        let system = messages[0]
            .get("content")
            .and_then(Value::as_str)
            .expect("system content");

        assert!(tools.is_none());
        assert!(system.contains("You are helpful."));
        assert!(!system.contains("tool list goes here"));
        assert!(!system.contains("call policy"));
    }

    #[test]
    fn tool_result_reducer_context_keeps_long_tool_chains_bounded() {
        let mut messages = vec![user_msg("Run the tool chain")];
        for idx in 0..12 {
            let id = format!("call_{idx}");
            messages.push(assistant_tool_msg(
                &id,
                "web_fetch",
                json!({"url": format!("https://example.com/{idx}")}),
            ));
            messages.push(tool_result_msg(&id, &format!("result {idx}")));
        }

        let s = session_with(&messages, Some(weather_tools()));
        let (packed, _tools) = pack_for_tool_result_turn(&s, true);
        let roles: Vec<&str> = packed
            .iter()
            .filter_map(|m| m.get("role").and_then(|r| r.as_str()))
            .collect();

        assert_eq!(
            roles[0], "system",
            "packed context should keep the MoA system preamble",
        );
        assert_eq!(
            roles[1], "user",
            "long bounded context should still include the original user query",
        );
        assert!(
            packed.len() <= TOOL_RESULT_CONTEXT_WINDOW + 2,
            "expected one system message + user prefix + bounded recent tail, got {} messages",
            packed.len(),
        );
        assert_ne!(
            roles[2], "tool",
            "bounded recent tail must not start with a bare tool message",
        );
    }

    #[test]
    fn generalist_and_reducer_roles_use_strong_shape() {
        let s = session_with(&[system_msg("Agent."), user_msg("hi")], Some(tools_two()));
        let g = pack_for_worker(&s, WorkerRole::Generalist, true);
        let r = pack_for_worker(&s, WorkerRole::Reducer, true);
        assert_eq!(g.max_tokens, 1024);
        assert_eq!(r.max_tokens, 1024);
        assert!(g.tools.is_some());
        assert!(r.tools.is_some());
    }

    // ── MoA preamble: augment, don't replace ─────────────────────────

    #[test]
    fn preamble_augments_existing_system_prompt() {
        let s = session_with(
            &[
                system_msg("CUSTOM_AGENT_INSTRUCTIONS_MARKER"),
                user_msg("hi"),
            ],
            None,
        );
        let packed = pack_for_worker(&s, WorkerRole::Strong, false);
        let sys = system_text(&packed.messages);
        assert!(
            sys.contains("CUSTOM_AGENT_INSTRUCTIONS_MARKER"),
            "agent's original system prompt must survive: {sys}",
        );
        assert!(
            sys.contains("Multiple models"),
            "MoA preamble must be present: {sys}",
        );
    }

    #[test]
    fn preamble_only_when_no_system_prompt() {
        let s = session_with(&[user_msg("hi")], None);
        let packed = pack_for_worker(&s, WorkerRole::Strong, false);
        let sys = system_text(&packed.messages);
        assert!(
            !sys.is_empty(),
            "should synthesize a system prompt from preamble"
        );
        assert!(sys.contains("Multiple models"));
    }

    #[test]
    fn ordinary_chat_strips_openclaw_tool_guidance_sections() {
        let prompt = "\
You are helpful.
## Tooling
tool list goes here
## Tool Call Style
tool-call policy goes here
## Safety
keep this";
        let stripped = strip_tool_guidance_sections(prompt);
        assert!(stripped.contains("You are helpful."));
        assert!(stripped.contains("## Safety"));
        assert!(stripped.contains("keep this"));
        assert!(!stripped.contains("tool list goes here"));
        assert!(!stripped.contains("tool-call policy goes here"));
    }

    // ── pack_for_reducer: includes reason + worker outputs ───────────

    fn worker_out(model: &str, payload: &str) -> WorkerOutput {
        WorkerOutput {
            kind: OutputKind::Answer,
            confidence: 0.6,
            tool_name: None,
            tool_arguments: None,
            payload: payload.to_string(),
            model: model.to_string(),
            role: WorkerRole::Strong,
            elapsed_ms: 0,
            truncated: false,
        }
    }

    #[test]
    fn reducer_context_includes_reason_and_worker_payloads() {
        let s = session_with(
            &[
                system_msg("Agent R."),
                user_msg("which is bigger, 7^3 or 350?"),
            ],
            Some(tools_two()),
        );
        let outputs = vec![
            worker_out("alpha", "It's 7^3 = 343, smaller than 350."),
            worker_out("beta", "350 is bigger."),
        ];
        let (messages, tools) = pack_for_reducer(&s, &outputs, "tie between answers", true);

        let sys = system_text(&messages);
        assert!(
            sys.contains("tie between answers"),
            "reason must appear in reducer system: {sys}",
        );
        assert!(sys.contains("alpha"), "worker model labels must appear");
        assert!(sys.contains("beta"));
        assert!(sys.contains("7^3 = 343"));
        assert!(sys.contains("350 is bigger"));
        assert!(
            tools.is_some(),
            "reducer should still have native tool schemas",
        );

        // Last message should be the user's actual query.
        let last = messages.last().unwrap();
        assert_eq!(last.get("role").and_then(|r| r.as_str()), Some("user"));
        assert_eq!(
            last.get("content").and_then(|c| c.as_str()),
            Some("which is bigger, 7^3 or 350?"),
        );
    }

    #[test]
    fn ordinary_chat_reducer_omits_native_tools() {
        let s = session_with(
            &[system_msg("Agent R."), user_msg("What can you help with?")],
            Some(tools_two()),
        );
        let outputs = vec![worker_out("alpha", "I can help with coding.")];
        let (_messages, tools) = pack_for_reducer(&s, &outputs, "ordinary answer", false);
        assert!(
            tools.is_none(),
            "ordinary chat reducer should not receive native tool schemas"
        );
    }

    #[test]
    fn reducer_truncates_long_worker_payloads() {
        let s = session_with(&[user_msg("go")], None);
        // Above the text-turn budget: realistic answers (~3.8k chars) must pass
        // through intact — see `text_reducer_sees_realistic_answer_lengths` —
        // but a pathological payload still gets bounded.
        let big = "x".repeat(9000);
        let outputs = vec![worker_out("alpha", &big)];

        let (messages, _tools) = pack_for_reducer(&s, &outputs, "conflict", false);
        let sys = system_text(&messages);

        assert!(
            !sys.contains(&big),
            "reducer must bound pathological worker payloads to keep context sane",
        );
        assert!(
            sys.contains("..."),
            "truncated payloads should be marked with an ellipsis",
        );
    }
}
