use super::*;

// ─── Post-hoc tool-call correction study ─────────────────────────────
//
// The mesh scenario neither Hermes nor Together handles: a WEAK tool-caller
// with no strong peer to defer to. Pre-hoc advice was inert-to-harmful (both
// prose and structured). This tests correction of the *concrete drafted call*
// instead — there's a real artifact to judge, not open-ended dilution.
//
// One weak finalizer drafts a call, then three arms (reuse A/B/C JSONL):
//   A draft-alone         — baseline, no correction
//   B deterministic       — schema-validate the draft; on structural failure
//                           re-prompt the finalizer with the specific error
//   C semantic            — a different-family peer critiques the concrete
//                           drafted call; the finalizer revises once
//
// analyze_ablation.py then gives B-vs-A (deterministic rescue), C-vs-A
// (semantic rescue), and differential B-C.

pub(crate) fn correction_finalizer() -> String {
    std::env::var("MOA_CORRECTION_FINALIZER").unwrap_or_else(|_| "qwen/qwen3-8b".to_string())
}

pub(crate) fn correction_critic() -> String {
    std::env::var("MOA_CORRECTION_CRITIC")
        .unwrap_or_else(|_| "mistralai/mistral-small-3.2-24b-instruct".to_string())
}

/// Structural validity of a drafted call: unknown tool, unparseable args, or a
/// missing/empty required field. Returns an error message, or None if valid.
/// This is the deterministic check a mesh can do with zero extra model calls.
pub(crate) fn validate_tool_call(name: &str, args_json: &str) -> Option<String> {
    let required: &[&str] = match name {
        "list_dir" | "read_file" => &["path"],
        "search" => &["pattern", "path"],
        "run_command" => &["cmd"],
        other => return Some(format!("unknown tool '{other}'")),
    };
    let parsed: Value = match serde_json::from_str(args_json) {
        Ok(v) => v,
        Err(e) => return Some(format!("arguments are not valid JSON: {e}")),
    };
    let Some(obj) = parsed.as_object() else {
        return Some("arguments must be a JSON object".to_string());
    };
    for field in required {
        match obj.get(*field).and_then(Value::as_str) {
            Some(s) if !s.trim().is_empty() => {}
            _ => return Some(format!("missing or empty required field '{field}'")),
        }
    }
    None
}

/// One tool-drafting call against the finalizer with the given extra messages
/// appended after the task prompt (for correction re-prompts).
pub(crate) async fn draft_call(
    backend: &OpenRouterBackend,
    finalizer: &str,
    task: &AblationTask,
    extra: &[Value],
) -> Result<Option<(String, String)>, String> {
    let mut msgs = vec![json!({"role": "user", "content": task.prompt})];
    msgs.extend_from_slice(extra);
    let body = backend
        .chat_completion_retrying(
            finalizer,
            &msgs,
            Some(&agent_tools()),
            2048,
            SamplingParams::reducer().with_thinking(Some(false)),
        )
        .await?;
    Ok(response_tool_calls(&body).into_iter().next())
}

/// Deterministic arm: draft, validate, and on structural failure re-prompt with
/// the concrete error (up to 2 corrections). No extra model beyond the retries.
pub(crate) async fn deterministic_correction(
    backend: &OpenRouterBackend,
    finalizer: &str,
    task: &AblationTask,
) -> ArmOutcome {
    let mut extra: Vec<Value> = Vec::new();
    for correction_index in 0..3 {
        let draft = match draft_call(backend, finalizer, task, &extra).await {
            Ok(d) => d,
            Err(e) if e.starts_with("INFRA:") => return ArmOutcome::Infra,
            Err(_) => return ArmOutcome::Fail,
        };
        let Some((name, args)) = draft else {
            // No tool call — acceptable only if the task wanted none.
            return if task.accept_tools.is_empty() {
                ArmOutcome::Pass
            } else {
                ArmOutcome::Fail
            };
        };
        match validate_tool_call(&name, &args) {
            None => return outcome_for(&[(name, args)], task),
            Some(err) => {
                let tool_call_id = format!("correction-{correction_index}");
                extra.push(json!({"role": "assistant", "content": Value::Null,
                    "tool_calls": [{"id": tool_call_id, "type": "function",
                        "function": {"name": name, "arguments": args}}]}));
                extra.push(json!({"role": "tool", "tool_call_id": tool_call_id,
                    "content": format!("Tool call validation failed: {err}")}));
                extra.push(json!({"role": "user",
                    "content": format!("That tool call is invalid: {err}. Emit a corrected tool call.")}));
            }
        }
    }
    ArmOutcome::Fail
}

/// Semantic arm: draft, have a different-family critic review the CONCRETE call,
/// then let the finalizer revise once given the critique.
pub(crate) async fn semantic_correction(
    backend: &OpenRouterBackend,
    finalizer: &str,
    critic: &str,
    task: &AblationTask,
) -> ArmOutcome {
    let draft = match draft_call(backend, finalizer, task, &[]).await {
        Ok(Some(d)) => d,
        Ok(None) => {
            return if task.accept_tools.is_empty() {
                ArmOutcome::Pass
            } else {
                ArmOutcome::Fail
            };
        }
        Err(e) if e.starts_with("INFRA:") => return ArmOutcome::Infra,
        Err(_) => return ArmOutcome::Fail,
    };

    let critique_prompt = format!(
        "Task: {}\n\nA model proposed this tool call:\n  {}({})\n\nTools available: \
         list_dir(path), read_file(path), search(pattern,path), run_command(cmd).\n\
         Is this the right tool and arguments for the task? If good, reply exactly OK. \
         If not, briefly say what's wrong.",
        task.prompt, draft.0, draft.1
    );
    let critique = match backend
        .chat_completion_retrying(
            critic,
            &[json!({"role": "user", "content": critique_prompt})],
            None,
            512,
            SamplingParams::worker().with_thinking(Some(false)),
        )
        .await
    {
        Ok(body) => response_text(&body),
        Err(e) if e.starts_with("INFRA:") => return ArmOutcome::Infra,
        Err(_) => return outcome_for(&[draft], task), // critic down → keep draft
    };

    if critique.trim().eq_ignore_ascii_case("OK") || critique.trim().is_empty() {
        return outcome_for(&[draft], task);
    }

    // Revise once given the concrete critique.
    let extra = vec![
        json!({"role": "assistant", "content": Value::Null,
            "tool_calls": [{"id": "c", "type": "function",
                "function": {"name": draft.0, "arguments": draft.1}}]}),
        json!({"role": "user",
            "content": format!("A reviewer said about your tool call: {}. Emit the corrected tool call.", critique.trim())}),
    ];
    match draft_call(backend, finalizer, task, &extra).await {
        Ok(Some(revised)) => outcome_for(&[revised], task),
        Ok(None) => ArmOutcome::Fail,
        Err(e) if e.starts_with("INFRA:") => ArmOutcome::Infra,
        Err(_) => ArmOutcome::Fail,
    }
}

pub(crate) fn outcome_for(tools: &[(String, String)], task: &AblationTask) -> ArmOutcome {
    if scores_ablation(tools, task) {
        ArmOutcome::Pass
    } else {
        ArmOutcome::Fail
    }
}
