//! Context packing — tailor what each worker sees.
//!
//! Full context enters the gateway, but workers get role-shaped slices of
//! the REAL context — the agent's actual system prompt, messages, and tool
//! definitions.  The gateway does not replace the agent's prompt with a
//! synthetic "you are a worker" envelope.  It augments with a short preamble
//! and varies the depth per role:
//!
//! - Fast:       system prompt + recent dialog + optional tool names
//! - Specialist: system prompt + recent dialog + optional tool summaries/schemas
//! - Strong:     system prompt + deeper recent dialog + optional full tool schemas
//! - Reducer:    system prompt + recent transcript + worker outputs + optional full tool schemas

use crate::normalize::WorkerOutput;
use crate::session::Session;
use crate::worker::WorkerRole;
use serde_json::{Value, json};

const TOOL_RESULT_CONTEXT_WINDOW: usize = 10;
const FAST_TOOL_CONTEXT_WINDOW: usize = 4;
const FAST_PLAIN_CONTEXT_WINDOW: usize = 8;
const SPECIALIST_TOOL_CONTEXT_WINDOW: usize = 4;
const SPECIALIST_PLAIN_CONTEXT_WINDOW: usize = 12;
const STRONG_TOOL_CONTEXT_WINDOW: usize = 10;
const STRONG_PLAIN_CONTEXT_WINDOW: usize = 32;
const REDUCER_TOOL_CONTEXT_WINDOW: usize = 8;
const REDUCER_PLAIN_CONTEXT_WINDOW: usize = 32;
const FAST_PLAIN_CONTEXT_MAX_BYTES: usize = 8 * 1024;
const SPECIALIST_PLAIN_CONTEXT_MAX_BYTES: usize = 12 * 1024;
const STRONG_PLAIN_CONTEXT_MAX_BYTES: usize = 32 * 1024;
const REDUCER_PLAIN_CONTEXT_MAX_BYTES: usize = 32 * 1024;
const TOOL_EVIDENCE_MAX_RESULTS: usize = 8;
const TOOL_EVIDENCE_MAX_RESULT_CHARS: usize = 800;
const TOOL_RESULT_RAW_MAX_CHARS: usize = 2_400;
const TOOL_RESULT_JSON_MAX_SCALARS: usize = 48;
const TOOL_RESULT_JSON_MAX_ARRAY_ITEMS: usize = 12;
const TOOL_RESULT_SCALAR_MAX_CHARS: usize = 180;
const TOOL_RESULT_TEXT_PREVIEW_CHARS: usize = 1_600;
const TOOL_RESULT_WEB_SEARCH_MAX_RESULTS: usize = 6;
const TOOL_RESULT_WEB_SEARCH_TITLE_MAX_CHARS: usize = 120;
const TOOL_RESULT_WEB_SEARCH_SNIPPET_MAX_CHARS: usize = 200;
const TOOL_RESULT_WEB_SEARCH_URL_MAX_CHARS: usize = 120;
const TOOL_RESULT_WEB_SEARCH_ROW_MAX_CHARS: usize = 320;
const TOOL_TASK_ANCHOR_MAX_CHARS: usize = 2_000;

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

fn augmented_system_prompt_for_mode(session: &Session, include_tool_guidance: bool) -> String {
    match session.system_prompt() {
        Some(sp) => {
            let prompt = strip_silent_reply_sections(&sp);
            let prompt = if include_tool_guidance {
                prompt
            } else {
                strip_tool_guidance_sections(&prompt)
            };
            format!("{MOA_PREAMBLE}\n\n{prompt}")
        }
        None => MOA_PREAMBLE.to_string(),
    }
}

fn strip_silent_reply_sections(prompt: &str) -> String {
    strip_markdown_sections(prompt, &["## Silent Replies"])
}

fn strip_tool_guidance_sections(prompt: &str) -> String {
    strip_markdown_sections(prompt, &["## Tooling", "## Tool Call Style"])
}

fn strip_markdown_sections(prompt: &str, stripped_headings: &[&str]) -> String {
    let mut out = Vec::new();
    let mut skipping = false;
    for line in prompt.lines() {
        if line.starts_with("## ") {
            skipping = stripped_headings
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
// System prompt + recent dialog + tool names only.
// Smallest context, quickest to respond.

fn pack_fast(session: &Session, has_tools: bool, selected_tool_names: &[String]) -> PackedContext {
    let system = system_with_tool_names(session, has_tools, selected_tool_names);
    let mut messages = vec![json!({"role": "system", "content": system})];
    let (window, max_bytes) = if has_tools {
        (FAST_TOOL_CONTEXT_WINDOW, None)
    } else {
        (
            FAST_PLAIN_CONTEXT_WINDOW,
            Some(FAST_PLAIN_CONTEXT_MAX_BYTES),
        )
    };
    append_recent_dialog_messages(&mut messages, session, window, max_bytes, has_tools);

    // Per-request sessions: the caller owns the multi-turn loop and
    // sends the full history each request. Continuation context lives
    // in `session.messages()`. Keep this context small, but do not
    // isolate follow-up questions from the visible same-chat history.
    PackedContext {
        messages,
        max_tokens: 256,
        tools: None, // Fast worker doesn't get tool schemas — just names
    }
}

// ── Specialist worker ────────────────────────────────────────────────
// System prompt + recent messages + tool name+description summaries.

fn pack_specialist(
    session: &Session,
    has_tools: bool,
    selected_tool_names: &[String],
) -> PackedContext {
    let system = system_with_tool_summaries(session, has_tools, selected_tool_names);

    let mut messages = vec![json!({"role": "system", "content": system})];

    let (window, max_bytes) = if has_tools {
        (SPECIALIST_TOOL_CONTEXT_WINDOW, None)
    } else {
        (
            SPECIALIST_PLAIN_CONTEXT_WINDOW,
            Some(SPECIALIST_PLAIN_CONTEXT_MAX_BYTES),
        )
    };
    append_recent_dialog_messages(&mut messages, session, window, max_bytes, has_tools);

    PackedContext {
        messages,
        max_tokens: 512,
        tools: selected_tools(session, has_tools, selected_tool_names),
    }
}

fn append_recent_dialog_messages(
    messages: &mut Vec<Value>,
    session: &Session,
    window: usize,
    max_bytes: Option<usize>,
    include_tool_task_anchor: bool,
) {
    let recent = bounded_recent_messages(session.recent_messages(window), max_bytes, true);
    if include_tool_task_anchor {
        append_tool_task_anchor_if_missing(messages, session, &recent);
    }
    for msg in &recent {
        if plain_dialog_message_for_compact_context(msg) {
            messages.push(msg.clone());
        }
    }
    append_current_user_if_missing(messages, session);
}

fn plain_dialog_message_for_compact_context(msg: &Value) -> bool {
    match message_role(msg) {
        "user" => true,
        "assistant" => msg.get("tool_calls").is_none(),
        _ => false,
    }
}

fn append_current_user_if_missing(messages: &mut Vec<Value>, session: &Session) {
    let Some(last_user) = session.last_user_message() else {
        return;
    };
    if messages.last() != Some(last_user) {
        messages.push(last_user.clone());
    }
}

// ── Strong worker ────────────────────────────────────────────────────
// System prompt + deeper recent history + full tool schemas forwarded natively.
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
    let (window, max_bytes) = if has_tools {
        (STRONG_TOOL_CONTEXT_WINDOW, None)
    } else {
        (
            STRONG_PLAIN_CONTEXT_WINDOW,
            Some(STRONG_PLAIN_CONTEXT_MAX_BYTES),
        )
    };
    let recent = bounded_recent_messages(session.recent_messages(window), max_bytes, true);
    if has_tools {
        append_tool_task_anchor_if_missing(&mut messages, session, &recent);
    }
    for msg in &recent {
        let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if role != "system" && !role.is_empty() {
            messages.push(msg.clone());
        }
    }

    append_current_user_if_missing(&mut messages, session);

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
    let mut system_parts = vec![
        augmented_system_prompt_for_mode(session, has_tools),
        String::new(),
        format!("Multiple models analyzed this request and disagreed. Reason: {reason}"),
        "Review their outputs below and produce ONE final response — either a direct answer \
         or a tool call. Be concise."
            .to_string(),
    ];

    // Worker outputs
    system_parts.push(String::new());
    system_parts.push("## Worker outputs".to_string());
    for (i, output) in outputs.iter().enumerate() {
        system_parts.push(format!("\n[Worker {} — {}]:", i + 1, output.model,));
        let payload = if output.payload.len() > 500 {
            format!("{}...", crate::worker::truncate_chars(&output.payload, 497))
        } else {
            output.payload.clone()
        };
        system_parts.push(payload);
        if let Some(ref tool) = output.tool_name {
            system_parts.push(format!("  → Proposed tool: {tool}"));
            if let Some(ref args) = output.tool_arguments {
                system_parts.push(format!("  → Arguments: {args}"));
            }
        }
    }

    let tools = selected_tools(session, has_tools, selected_tool_names);

    let mut messages = vec![json!({"role": "system", "content": system_parts.join("\n")})];
    messages.extend(reducer_recent_transcript_messages(session, has_tools));
    append_current_user_if_missing(&mut messages, session);

    (messages, tools)
}

fn reducer_recent_transcript_messages(session: &Session, has_tools: bool) -> Vec<Value> {
    let all = session.messages();
    let Some(last_user_idx) = all.iter().rposition(|msg| message_role(msg) == "user") else {
        return Vec::new();
    };

    let (window, max_bytes) = if has_tools {
        (REDUCER_TOOL_CONTEXT_WINDOW, None)
    } else {
        (
            REDUCER_PLAIN_CONTEXT_WINDOW,
            Some(REDUCER_PLAIN_CONTEXT_MAX_BYTES),
        )
    };
    let start_idx = last_user_idx.saturating_sub(window);
    let recent = bounded_recent_messages(all[start_idx..last_user_idx].to_vec(), max_bytes, false);
    recent
        .iter()
        .filter(|msg| reducer_can_replay_message(msg))
        .cloned()
        .collect()
}

fn bounded_recent_messages(
    messages: Vec<Value>,
    max_bytes: Option<usize>,
    keep_latest_when_oversized: bool,
) -> Vec<Value> {
    let Some(max_bytes) = max_bytes else {
        return messages;
    };
    let mut selected_rev = Vec::new();
    let mut used = 0usize;
    for message in messages.into_iter().rev() {
        let cost = estimated_message_context_bytes(&message);
        let can_keep_oversized_latest = keep_latest_when_oversized && selected_rev.is_empty();
        if !can_keep_oversized_latest && used.saturating_add(cost) > max_bytes {
            break;
        }
        used = used.saturating_add(cost);
        selected_rev.push(message);
    }
    selected_rev.reverse();
    selected_rev
}

fn estimated_message_context_bytes(message: &Value) -> usize {
    serde_json::to_string(message)
        .map(|encoded| encoded.len())
        .unwrap_or(0)
}

fn reducer_can_replay_message(msg: &Value) -> bool {
    match message_role(msg) {
        "user" => true,
        "assistant" => msg.get("tool_calls").is_none(),
        _ => false,
    }
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
    let system = augmented_system_prompt_for_mode(session, has_tools);

    let mut messages = vec![json!({"role": "system", "content": system})];
    if let Some(evidence) = tool_evidence_message(session) {
        messages.push(evidence);
    }

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

    let mut transcript = Vec::new();
    if let Some(user_idx) = prefix_user_idx {
        transcript.push(all[user_idx].clone());
    }

    for msg in &all[start_idx..] {
        let role = message_role(msg);
        if role != "system" && !role.is_empty() {
            transcript.push(compact_tool_message(msg));
        }
    }
    append_tool_task_anchor_if_missing(&mut messages, session, &transcript);
    messages.extend(transcript);

    let tools = selected_tools(session, has_tools, selected_tool_names);

    (messages, tools)
}

fn append_tool_task_anchor_if_missing(
    messages: &mut Vec<Value>,
    session: &Session,
    visible_messages: &[Value],
) {
    let Some(task) = first_user_task_text(session) else {
        return;
    };
    if messages_contain_text(visible_messages, &task) {
        return;
    }

    let task = truncate_with_ellipsis(&task, TOOL_TASK_ANCHOR_MAX_CHARS);
    messages.push(json!({
        "role": "system",
        "content": format!("Original user task for this tool loop:\n{task}"),
    }));
}

fn first_user_task_text(session: &Session) -> Option<String> {
    session
        .messages()
        .iter()
        .rev()
        .find_map(|message| {
            (message_role(message) == "user"
                && !user_message_looks_like_tool_response(message)
                && !user_message_looks_like_runtime_info(message))
            .then(|| message_text_content(message))
            .flatten()
        })
        .map(|text| text.trim().to_string())
        .filter(|text| !text.is_empty())
}

fn user_message_looks_like_runtime_info(message: &Value) -> bool {
    message_text_content(message).is_some_and(|text| {
        let trimmed = text.trim_start();
        trimmed.starts_with("<info-msg>")
            || trimmed.starts_with("<environment_context>")
            || trimmed.starts_with("<system-reminder>")
    })
}

fn user_message_looks_like_tool_response(message: &Value) -> bool {
    message
        .get("content")
        .and_then(Value::as_array)
        .is_some_and(|parts| parts.iter().all(content_part_looks_like_tool_response))
}

fn content_part_looks_like_tool_response(part: &Value) -> bool {
    matches!(
        part.get("type").and_then(Value::as_str),
        Some("toolResponse" | "tool_result" | "tool_call_output" | "function_call_output")
    ) || part.get("toolResult").is_some()
        || part.get("tool_result").is_some()
        || part.get("tool_call_id").is_some()
        || part.get("call_id").is_some()
}

fn messages_contain_text(messages: &[Value], needle: &str) -> bool {
    messages
        .iter()
        .filter_map(message_text_content)
        .any(|text| text.contains(needle))
}

fn message_text_content(message: &Value) -> Option<String> {
    let content = message.get("content")?;
    if let Some(text) = content.as_str() {
        return Some(text.to_string());
    }
    let parts = content.as_array()?;
    let text = parts
        .iter()
        .filter_map(|part| {
            part.get("text")
                .and_then(Value::as_str)
                .or_else(|| part.get("input_text").and_then(Value::as_str))
                .or_else(|| part.get("output_text").and_then(Value::as_str))
        })
        .collect::<Vec<_>>()
        .join("\n");
    (!text.is_empty()).then_some(text)
}

fn truncate_with_ellipsis(text: &str, max_bytes: usize) -> String {
    let truncated = crate::worker::truncate_chars(text, max_bytes);
    if truncated.len() == text.len() {
        truncated.to_string()
    } else {
        format!("{truncated}...")
    }
}

fn tool_evidence_message(session: &Session) -> Option<Value> {
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

    Some(json!({"role": "system", "content": lines.join("\n")}))
}

fn compact_tool_message(msg: &Value) -> Value {
    match message_role(msg) {
        "tool" => compact_role_tool_message(msg),
        "user" if user_message_looks_like_tool_response(msg) => {
            compact_user_tool_response_message(msg)
        }
        _ => msg.clone(),
    }
}

fn compact_role_tool_message(msg: &Value) -> Value {
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

fn compact_user_tool_response_message(msg: &Value) -> Value {
    let Some(parts) = msg.get("content").and_then(Value::as_array) else {
        return msg.clone();
    };

    let mut changed = false;
    let compact_parts = parts
        .iter()
        .map(|part| {
            if !content_part_looks_like_tool_response(part) {
                return part.clone();
            }
            let Some(text) = tool_response_part_text(part) else {
                return part.clone();
            };
            changed = true;
            json!({
                "type": "text",
                "text": format!("Tool result:\n{}", compact_tool_result_text(&text)),
            })
        })
        .collect::<Vec<_>>();

    if !changed {
        return msg.clone();
    }

    let mut compact = msg.clone();
    if let Some(obj) = compact.as_object_mut() {
        obj.insert("content".to_string(), Value::Array(compact_parts));
    }
    compact
}

fn tool_response_part_text(part: &Value) -> Option<String> {
    content_text_fields(part)
        .or_else(|| part.get("output").and_then(value_text_content))
        .or_else(|| {
            part.pointer("/toolResult/value/content")
                .and_then(content_array_text)
        })
        .or_else(|| {
            part.pointer("/toolResult/value/structuredContent/stdout")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .or_else(|| {
            part.pointer("/toolResult/value/structuredContent/stderr")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .or_else(|| {
            part.pointer("/toolResult/value")
                .and_then(value_text_content)
        })
        .or_else(|| {
            part.pointer("/tool_result/content")
                .and_then(value_text_content)
        })
}

fn value_text_content(value: &Value) -> Option<String> {
    if let Some(text) = value.as_str() {
        return Some(text.to_string());
    }
    if let Some(text) = content_array_text(value) {
        return Some(text);
    }
    value.as_object()?;
    serde_json::to_string(value).ok()
}

fn content_array_text(value: &Value) -> Option<String> {
    let parts = value.as_array()?;
    let text = parts
        .iter()
        .filter_map(content_text_fields)
        .collect::<Vec<_>>()
        .join("\n");
    (!text.is_empty()).then_some(text)
}

fn content_text_fields(part: &Value) -> Option<String> {
    part.get("text")
        .and_then(Value::as_str)
        .or_else(|| part.get("input_text").and_then(Value::as_str))
        .or_else(|| part.get("output_text").and_then(Value::as_str))
        .map(str::to_string)
}

fn compact_tool_result_text(result: &str) -> String {
    if let Ok(json) = serde_json::from_str::<Value>(result) {
        if let Some(compacted) = compact_external_json_tool_result(result.len(), &json) {
            return compacted;
        }
        if result.len() > TOOL_RESULT_RAW_MAX_CHARS {
            return compact_json_tool_result(result.len(), &json);
        }
    }

    if result.len() <= TOOL_RESULT_RAW_MAX_CHARS {
        return result.to_string();
    }

    format!(
        "Tool result compacted from {} chars; original was plain text.\n\
         Text preview:\n{}...",
        result.len(),
        crate::worker::truncate_chars(result, TOOL_RESULT_RAW_MAX_CHARS - 96)
    )
}

fn compact_external_json_tool_result(original_len: usize, value: &Value) -> Option<String> {
    let map = value.as_object()?;
    let source = map
        .get("externalContent")
        .and_then(|external| external.get("source"))
        .and_then(Value::as_str)?;

    match source {
        "web_fetch" => compact_web_fetch_result(original_len, map),
        "web_search" => compact_web_search_result(original_len, map),
        _ => None,
    }
}

fn compact_web_fetch_result(
    original_len: usize,
    map: &serde_json::Map<String, Value>,
) -> Option<String> {
    let text = map.get("text").and_then(Value::as_str)?;
    let mut lines = vec![format!(
        "Tool result compacted from {original_len} chars; original was web_fetch JSON."
    )];
    push_clean_json_field("Title", "title", map, &mut lines);
    push_clean_json_field("URL", "url", map, &mut lines);
    push_clean_json_field("Status", "status", map, &mut lines);

    let preview = clean_external_tool_text(text);
    if preview.is_empty() {
        return None;
    }
    lines.push("Fetched content preview:".to_string());
    lines.push(crate::worker::truncate_chars(&preview, TOOL_RESULT_TEXT_PREVIEW_CHARS).to_string());
    Some(lines.join("\n"))
}

fn compact_web_search_result(
    original_len: usize,
    map: &serde_json::Map<String, Value>,
) -> Option<String> {
    let results = map.get("results").and_then(Value::as_array)?;
    let mut lines = vec![format!(
        "Tool result compacted from {original_len} chars; original was web_search JSON."
    )];
    push_clean_json_field("Query", "query", map, &mut lines);
    lines.push("Search results:".to_string());

    for (idx, result) in results
        .iter()
        .take(TOOL_RESULT_WEB_SEARCH_MAX_RESULTS)
        .enumerate()
    {
        let Some(result) = result.as_object() else {
            continue;
        };
        let title = clean_json_string_field(result, "title")
            .map(|value| truncate_tool_result_field(&value, TOOL_RESULT_WEB_SEARCH_TITLE_MAX_CHARS))
            .unwrap_or_else(|| "Untitled".into());
        let url = clean_json_string_field(result, "url")
            .map(|value| truncate_tool_result_field(&value, TOOL_RESULT_WEB_SEARCH_URL_MAX_CHARS))
            .unwrap_or_default();
        let snippet = clean_json_string_field(result, "snippet")
            .map(|value| {
                truncate_tool_result_field(&value, TOOL_RESULT_WEB_SEARCH_SNIPPET_MAX_CHARS)
            })
            .unwrap_or_default();
        let mut row = format!("{}. {title}", idx + 1);
        if !snippet.is_empty() {
            row.push_str(": ");
            row.push_str(&snippet);
        }
        if !url.is_empty() {
            row.push_str(" (");
            row.push_str(&url);
            row.push(')');
        }
        lines.push(truncate_tool_result_field(
            &row,
            TOOL_RESULT_WEB_SEARCH_ROW_MAX_CHARS,
        ));
    }

    (lines.len() > 3).then(|| lines.join("\n"))
}

fn truncate_tool_result_field(value: &str, max_bytes: usize) -> String {
    let truncated = crate::worker::truncate_chars(value, max_bytes);
    if truncated.len() == value.len() {
        truncated.to_string()
    } else {
        format!("{truncated}...")
    }
}

fn push_clean_json_field(
    label: &str,
    key: &str,
    map: &serde_json::Map<String, Value>,
    lines: &mut Vec<String>,
) {
    let Some(value) = clean_json_string_field(map, key) else {
        return;
    };
    if !value.is_empty() {
        lines.push(format!("{label}: {value}"));
    }
}

fn clean_json_string_field(map: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    let value = map.get(key)?;
    value
        .as_str()
        .map(clean_external_tool_text)
        .or_else(|| scalar_to_string(value).map(|scalar| scalar.trim_matches('"').to_string()))
}

fn clean_external_tool_text(text: &str) -> String {
    let body = unwrap_external_content_body(text).unwrap_or(text);
    body.trim().to_string()
}

fn unwrap_external_content_body(text: &str) -> Option<&str> {
    let marker_start = text.rfind("<<<EXTERNAL_UNTRUSTED_CONTENT")?;
    let after_marker = &text[marker_start..];
    let separator = after_marker.find("---")?;
    let mut body = after_marker.get(separator + 3..)?.trim_start_matches('\n');
    if let Some(end) = body.find("<<<END_EXTERNAL_UNTRUSTED_CONTENT") {
        body = &body[..end];
    }
    Some(body.trim())
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
    "text",
    "content",
    "snippet",
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
    fn user_tool_response_msg(id: &str, text: &str) -> Value {
        json!({
            "role": "user",
            "content": [{
                "type": "toolResponse",
                "id": id,
                "toolResult": {
                    "status": "success",
                    "value": {
                        "content": [{"type": "text", "text": text}],
                        "isError": false,
                    },
                },
            }],
        })
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

    /// Helper: extract the system message content from a packed message vec.
    fn system_text(messages: &[Value]) -> String {
        messages
            .iter()
            .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
            .and_then(|m| m.get("content").and_then(|c| c.as_str()))
            .unwrap_or("")
            .to_string()
    }

    fn serialized_messages(messages: &[Value]) -> String {
        serde_json::to_string(messages).unwrap()
    }

    // ── pack_for_worker: shape contract per role ─────────────────────

    #[test]
    fn fast_worker_has_recent_dialog_no_native_tools() {
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
        assert_eq!(
            packed.messages[0].get("role").and_then(|r| r.as_str()),
            Some("system"),
        );
        let body = serde_json::to_string(&packed.messages).unwrap();
        assert!(
            body.contains("first") && body.contains("first reply") && body.contains("second"),
            "fast worker should see a compact recent dialog, got {body}",
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
    fn worker_context_preserves_raw_current_user_message() {
        let current_user = json!({
            "role": "user",
            "content": [
                {"type": "input_text", "text": "use this structured prompt"},
                {"type": "text", "text": "without rebuilding it"},
            ],
        });
        let s = session_with(
            &[
                system_msg("Agent SP."),
                assistant_msg("ready"),
                current_user.clone(),
            ],
            None,
        );

        for role in [WorkerRole::Fast, WorkerRole::Specialist, WorkerRole::Strong] {
            let packed = pack_for_worker(&s, role, false);
            let user_messages = packed
                .messages
                .iter()
                .filter(|msg| msg.get("role").and_then(Value::as_str) == Some("user"))
                .collect::<Vec<_>>();
            assert_eq!(
                user_messages.len(),
                1,
                "{role:?} should not append a stringified duplicate current user message: {:?}",
                packed.messages,
            );
            assert_eq!(user_messages[0], &current_user);
        }
    }

    #[test]
    fn reducer_context_preserves_raw_current_user_message() {
        let current_user = json!({
            "role": "user",
            "content": [{"type": "input_text", "text": "structured final request"}],
        });
        let s = session_with(
            &[
                user_msg("earlier"),
                assistant_msg("earlier reply"),
                current_user.clone(),
            ],
            None,
        );
        let outputs = vec![WorkerOutput {
            model: "worker-a".into(),
            kind: OutputKind::Answer,
            payload: "candidate answer".into(),
            tool_name: None,
            tool_arguments: None,
            confidence: 0.4,
            role: WorkerRole::Fast,
            elapsed_ms: 10,
        }];

        let (messages, _tools) = pack_for_reducer(&s, &outputs, "conflict", false);
        assert_eq!(messages.last(), Some(&current_user));
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
    fn plain_chat_strong_worker_keeps_earlier_same_chat_topic() {
        let mut msgs = vec![
            system_msg("Agent ST."),
            user_msg("How do I run Windows commands over Tailscale from my Mac?"),
            assistant_msg("We discussed using a reachable Windows host over Tailscale."),
        ];
        for i in 0..12 {
            msgs.push(user_msg(&format!("follow-up topic {i}")));
            msgs.push(assistant_msg(&format!("follow-up answer {i}")));
        }
        msgs.push(user_msg("What did we discuss earlier in this chat?"));
        let s = session_with(&msgs, None);

        let packed = pack_for_worker(&s, WorkerRole::Strong, false);
        let body = serialized_messages(&packed.messages);

        assert!(
            body.contains("Windows commands over Tailscale"),
            "plain-chat strong worker should retain earlier same-chat topics, got {body}",
        );
        assert!(
            body.contains("What did we discuss earlier"),
            "current user turn must still be present, got {body}",
        );
    }

    #[test]
    fn plain_chat_strong_worker_drops_oversized_prior_blob_before_current() {
        let oversized_prior = format!(
            "OVERSIZED_STRONG_PRIOR_{}",
            "x".repeat(STRONG_PLAIN_CONTEXT_MAX_BYTES + 1024)
        );
        let s = session_with(
            &[
                system_msg("Agent ST."),
                user_msg("Read this large pasted result."),
                assistant_msg(&oversized_prior),
                user_msg("What should I do next?"),
            ],
            None,
        );

        let packed = pack_for_worker(&s, WorkerRole::Strong, false);
        let body = serialized_messages(&packed.messages);

        assert!(
            body.contains("What should I do next?"),
            "current user turn must still be present, got {body}",
        );
        assert!(
            !body.contains("OVERSIZED_STRONG_PRIOR_"),
            "plain-chat strong context should not carry an oversized prior blob, got {body}",
        );
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
            vec![
                "system",
                "system",
                "user",
                "assistant",
                "tool",
                "assistant",
                "tool"
            ],
            "tool-result reducer context must not start with a bare tool message",
        );
        assert_eq!(
            messages[2].get("content").and_then(|c| c.as_str()),
            Some("What is the weather today?"),
        );
        assert!(
            messages[3].get("tool_calls").is_some(),
            "first tool result must retain its preceding assistant tool_call",
        );
        assert!(
            messages[5].get("tool_calls").is_some(),
            "latest tool result must retain its preceding assistant tool_call",
        );
        assert!(
            tools.is_some(),
            "tool-result reducer should still receive native tool schemas",
        );
    }

    #[test]
    fn tool_result_reducer_filters_native_tool_schemas() {
        let s = session_with(
            &[
                user_msg("Read /tmp/a"),
                assistant_tool_msg("call_read", "read_file", json!({"path": "/tmp/a"})),
                tool_result_msg("call_read", "done"),
            ],
            Some(tools_two()),
        );
        let selected = vec!["read_file".to_string()];
        let (_messages, tools) = pack_for_tool_result_turn_selected(&s, true, &selected);
        let tools = tools
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
    fn web_fetch_tool_result_prefers_fetched_text_over_wrapper_keys() {
        let result = json!({
            "url": "https://www.smh.com.au",
            "finalUrl": "https://www.smh.com.au",
            "status": 200,
            "title": "\n<<<EXTERNAL_UNTRUSTED_CONTENT id=\"title\">>>\nSource: Web Fetch\n---\nAustralian Breaking News Headlines & World News Online\n<<<END_EXTERNAL_UNTRUSTED_CONTENT id=\"title\">>>",
            "externalContent": {
                "untrusted": true,
                "source": "web_fetch",
                "wrapped": true
            },
            "text": "SECURITY NOTICE: external content.\n\n<<<EXTERNAL_UNTRUSTED_CONTENT id=\"body\">>>\nSource: Web Fetch\n---\n### World Cup of chaos\nThe World Cup returns to North America.\n\n### Modi wants more of Australia's uranium\nAustralia and India struck a historic deal.\n<<<END_EXTERNAL_UNTRUSTED_CONTENT id=\"body\">>>"
        })
        .to_string();

        let compacted = compact_tool_result_text(&result);

        assert!(compacted.contains("original was web_fetch JSON"));
        assert!(compacted.contains("Fetched content preview:"));
        assert!(compacted.contains("World Cup of chaos"));
        assert!(compacted.contains("Modi wants more of Australia's uranium"));
        assert!(
            !compacted.contains("\"finalUrl\""),
            "wrapper JSON keys should not dominate reducer context: {compacted}"
        );
    }

    #[test]
    fn web_search_tool_result_compacts_to_result_rows() {
        let result = json!({
            "query": "headlines from www.smh.com.au",
            "provider": "duckduckgo",
            "externalContent": {
                "untrusted": true,
                "source": "web_search",
                "provider": "duckduckgo",
                "wrapped": true
            },
            "results": [
                {
                    "title": "\n<<<EXTERNAL_UNTRUSTED_CONTENT id=\"title\">>>\nSource: Web Search\n---\nLatest and Breaking News - The Sydney Morning Herald\n<<<END_EXTERNAL_UNTRUSTED_CONTENT id=\"title\">>>",
                    "url": "https://www.smh.com.au/breaking-news",
                    "snippet": format!(
                        "\n<<<EXTERNAL_UNTRUSTED_CONTENT id=\"snippet\">>>\nSource: Web Search\n---\nThe future of Australian swimming has arrived. {}\n<<<END_EXTERNAL_UNTRUSTED_CONTENT id=\"snippet\">>>",
                        "noise ".repeat(200)
                    )
                }
            ]
        })
        .to_string();

        let compacted = compact_tool_result_text(&result);

        assert!(compacted.contains("original was web_search JSON"));
        assert!(compacted.contains("Latest and Breaking News"));
        assert!(compacted.contains("Australian swimming"));
        assert!(
            compacted.find("Search results:").unwrap() < compacted.find("Latest").unwrap(),
            "result rows should be explicit and easy for the reducer to read: {compacted}"
        );
        assert!(
            compacted.lines().all(|line| line.len() <= 360),
            "web_search result rows should stay bounded: {compacted}"
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
            roles[2], "user",
            "long bounded context should still include the original user query",
        );
        assert!(
            packed.len() <= TOOL_RESULT_CONTEXT_WINDOW + 3,
            "expected system + evidence + user prefix + bounded recent tail, got {} messages",
            packed.len(),
        );
        assert_ne!(
            roles[3], "tool",
            "bounded recent tail must not start with a bare tool message",
        );
    }

    #[test]
    fn tool_result_reducer_anchors_original_task_when_latest_user_is_tool_wrapper() {
        let mut messages = vec![
            user_msg("Old chat about Windows and Tailscale."),
            user_msg("Implement src/smoke_calc.py so the tests pass."),
        ];
        for idx in 0..8 {
            let id = format!("call_tree_{idx}");
            messages.push(assistant_tool_msg(
                &id,
                "tree",
                json!({"path": format!("dir{idx}")}),
            ));
            messages.push(user_tool_response_msg(
                &id,
                &format!("tree result {idx}: src/smoke_calc.py"),
            ));
        }
        messages.push(assistant_tool_msg(
            "call_read",
            "read_file",
            json!({"path": "src/smoke_calc.py"}),
        ));
        messages.push(tool_result_msg(
            "call_read",
            "raise NotImplementedError(\"ci smoke fixture\")",
        ));
        let s = session_with(&messages, Some(tools_two()));

        let recent_without_anchor =
            s.recent_messages(TOOL_RESULT_CONTEXT_WINDOW)
                .iter()
                .any(|msg| {
                    message_text_content(msg)
                        .is_some_and(|text| text.contains("Implement src/smoke_calc.py"))
                });
        assert!(
            !recent_without_anchor,
            "test must push the original task outside the bounded recent tail",
        );

        let (messages, _tools) = pack_for_tool_result_turn(&s, true);
        let body = serialized_messages(&messages);

        assert!(
            body.contains("Original user task for this tool loop"),
            "tool-result context should add a bounded task anchor, got {body}",
        );
        assert!(
            body.contains("Implement src/smoke_calc.py so the tests pass."),
            "task anchor should preserve the real agent task, got {body}",
        );
        assert!(
            !body.contains("Old chat about Windows and Tailscale"),
            "task anchor should prefer the active task over older chat, got {body}",
        );
        assert!(
            body.contains("NotImplementedError"),
            "latest tool result should remain visible, got {body}",
        );
    }

    #[test]
    fn tool_worker_context_anchors_original_task_after_user_tool_wrappers() {
        let mut messages = vec![
            user_msg("Old chat about Windows and Tailscale."),
            user_msg("Implement src/smoke_calc.py so the tests pass."),
            user_msg("<info-msg>\nWorking directory: /tmp/project\n</info-msg>"),
        ];
        for idx in 0..8 {
            let id = format!("call_{idx}");
            messages.push(assistant_tool_msg(
                &id,
                "tree",
                json!({"path": format!("dir{idx}")}),
            ));
            messages.push(user_tool_response_msg(
                &id,
                &format!("tree result {idx}: src/smoke_calc.py"),
            ));
        }
        let s = session_with(&messages, Some(tools_two()));

        let packed = pack_for_worker(&s, WorkerRole::Strong, true);
        let body = serialized_messages(&packed.messages);

        assert!(
            body.contains("Original user task for this tool loop"),
            "tool worker context should anchor the original task, got {body}",
        );
        assert!(
            body.contains("Implement src/smoke_calc.py so the tests pass."),
            "tool worker task anchor should preserve the real task, got {body}",
        );
        assert!(
            !body.contains("Old chat about Windows and Tailscale"),
            "tool worker anchor should prefer the active task over older chat, got {body}",
        );
    }

    #[test]
    fn user_wrapped_tool_result_is_compacted_for_reducer() {
        let huge_result = "x".repeat(TOOL_RESULT_RAW_MAX_CHARS + 500);
        let messages = vec![
            user_msg("Read the large output."),
            assistant_tool_msg("call_read", "read", json!({"path": "large.txt"})),
            user_tool_response_msg("call_read", &huge_result),
        ];
        let s = session_with(&messages, Some(tools_two()));

        let (messages, _tools) = pack_for_tool_result_turn(&s, true);
        let body = serialized_messages(&messages);

        assert!(
            body.contains("Tool result compacted"),
            "user-wrapped tool result should be compacted, got {body}",
        );
        assert!(
            !body.contains(&huge_result),
            "full nested user-wrapped tool result should not leak, got {body}",
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

    #[test]
    fn tool_context_strips_silent_reply_sections() {
        let s = session_with(
            &[
                system_msg(
                    "\
You are helpful.
## Silent Replies
When you have nothing to say, respond with ONLY: NO_REPLY
## Safety
keep this",
                ),
                user_msg("Run the requested command."),
            ],
            Some(tools_two()),
        );
        let packed = pack_for_worker(&s, WorkerRole::Strong, true);
        let sys = system_text(&packed.messages);

        assert!(sys.contains("You are helpful."));
        assert!(sys.contains("## Safety"));
        assert!(sys.contains("keep this"));
        assert!(!sys.contains("NO_REPLY"), "{sys}");
        assert!(!sys.contains("Silent Replies"), "{sys}");
        assert!(
            packed.tools.is_some(),
            "tool schemas should still be present"
        );
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
    fn reducer_context_preserves_same_chat_recent_history_before_current_turn() {
        let s = session_with(
            &[
                system_msg("Agent R."),
                user_msg("How do I run commands on Windows via Tailscale from a Mac?"),
                assistant_msg(
                    "Use a reachable Windows host and run the command over the mesh or SSH tunnel.",
                ),
                user_msg("What were the two questions above in this same conversation?"),
            ],
            None,
        );
        let outputs = vec![
            worker_out("fast", "Only two questions are visible in this request."),
            worker_out("strong", "Earlier history mentioned Windows via Tailscale."),
        ];
        let (messages, _tools) = pack_for_reducer(&s, &outputs, "history disagreement", false);

        let transcript = serde_json::to_string(&messages).unwrap();
        assert!(
            transcript.contains("Windows via Tailscale"),
            "reducer should see recent same-chat history, got {transcript}",
        );
        assert!(
            transcript.contains("What were the two questions above"),
            "reducer should still receive the current user turn, got {transcript}",
        );
        assert_eq!(
            messages
                .last()
                .and_then(|message| message.get("content"))
                .and_then(Value::as_str),
            Some("What were the two questions above in this same conversation?"),
        );
    }

    #[test]
    fn plain_chat_reducer_keeps_earlier_same_chat_topic_beyond_short_tail() {
        let mut msgs = vec![
            system_msg("Agent R."),
            user_msg("How do I run Windows commands over Tailscale from my Mac?"),
            assistant_msg("We discussed using a reachable Windows host over Tailscale."),
        ];
        for i in 0..12 {
            msgs.push(user_msg(&format!("follow-up topic {i}")));
            msgs.push(assistant_msg(&format!("follow-up answer {i}")));
        }
        msgs.push(user_msg("What did we discuss earlier in this chat?"));
        let s = session_with(&msgs, None);
        let outputs = vec![
            worker_out("fast", "I only see the latest question."),
            worker_out("strong", "Earlier history mentioned Windows and Tailscale."),
        ];

        let (messages, _tools) = pack_for_reducer(&s, &outputs, "history disagreement", false);
        let body = serialized_messages(&messages);

        assert!(
            body.contains("Windows commands over Tailscale"),
            "plain-chat reducer should retain earlier same-chat topics, got {body}",
        );
        assert!(
            body.contains("What did we discuss earlier"),
            "current user turn must still be present, got {body}",
        );
    }

    #[test]
    fn plain_chat_reducer_drops_oversized_prior_blob_before_current() {
        let oversized_prior = format!(
            "OVERSIZED_REDUCER_PRIOR_{}",
            "x".repeat(REDUCER_PLAIN_CONTEXT_MAX_BYTES + 1024)
        );
        let s = session_with(
            &[
                system_msg("Agent R."),
                user_msg("Read this large pasted result."),
                assistant_msg(&oversized_prior),
                user_msg("What should I do next?"),
            ],
            None,
        );
        let outputs = vec![
            worker_out("fast", "Answer the current question only."),
            worker_out("strong", "Do not replay the huge prior blob."),
        ];

        let (messages, _tools) = pack_for_reducer(&s, &outputs, "history disagreement", false);
        let body = serialized_messages(&messages);

        assert!(
            body.contains("What should I do next?"),
            "current user turn must still be present, got {body}",
        );
        assert!(
            !body.contains("OVERSIZED_REDUCER_PRIOR_"),
            "plain-chat reducer context should not carry an oversized prior blob, got {body}",
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
        let big = "x".repeat(2000);
        let outputs = vec![worker_out("alpha", &big)];

        let (messages, _tools) = pack_for_reducer(&s, &outputs, "conflict", false);
        let sys = system_text(&messages);

        // Long payloads must be truncated (cap is ~500 chars + ellipsis).
        // The full 2000-char string must NOT appear verbatim.
        assert!(
            !sys.contains(&big),
            "reducer must truncate long worker payloads to keep context bounded",
        );
        assert!(
            sys.contains("..."),
            "truncated payloads should be marked with an ellipsis",
        );
    }
}
