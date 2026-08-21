//! OpenAI-compatible MoA response builders.

use crate::MOA_ERR_NO_USABLE_ANSWER;
use crate::VIRTUAL_MODEL_NAME;
use crate::normalize::{self, WorkerOutput};
use mesh_llm_guardrails::tool_arguments_wire_string;
use serde_json::{Value, json};

// ─── Response builders ───────────────────────────────────────────────

pub(crate) fn best_answer(outputs: &[WorkerOutput]) -> String {
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

pub(crate) fn fallback_worker_response(outputs: &[WorkerOutput]) -> Value {
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

pub(crate) fn tool_proposal_response(output: &WorkerOutput, has_tools: bool) -> Value {
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
pub(crate) fn error_response(message: &str, code: &str) -> Value {
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

pub(crate) fn chat_response(content: &str) -> Value {
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

pub(crate) fn tool_call_response(name: &str, arguments: &Value) -> Value {
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
    use crate::MOA_ERR_ALL_WORKERS_FAILED;
    use crate::arbiter;
    use crate::config::GatewayConfig;
    use crate::fanout::GraceMode;
    use crate::gateway::{
        forced_tool_choice, grace_mode_for_turn, infer_tool_arguments_from_prompt,
        tool_names_for_turn,
    };
    use crate::normalize::{OutputKind, WorkerOutput};
    use crate::resolve::resolve_decision;
    use crate::session::Session;
    use crate::tool_result::{
        handle_tool_result, repair_tool_result_answer, repeated_identical_tool_results,
    };
    use crate::turn::{DecisionResolution, ForcedToolChoice};
    use crate::worker::WorkerRole;
    use std::time::{Duration, Instant};

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
            truncated: false,
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
            truncated: false,
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

        assert_eq!(args, json!({"key": "Primary"}));

        let path_args = infer_tool_arguments_from_prompt(
            "lookup_probe_fact",
            tools.as_ref(),
            "Use key=README.md.",
        );
        assert_eq!(path_args, json!({"key": "README.md"}));
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
            strong_patience: Duration::ZERO,
            enable_thinking: Some(false),
            actor_candidates: Vec::new(),
            reference_policy: Default::default(),
            refinement_policy: Default::default(),
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
    fn tool_turn_preserves_all_caller_tools() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({
                "role": "user",
                "content": "Read both data files, calculate the totals, then write the report."
            })],
            &Some(serde_json::json!([
                {"type": "function", "function": {"name": "read_file"}},
                {"type": "function", "function": {"name": "write_file"}}
            ])),
        );

        assert_eq!(
            tool_names_for_turn(&session, &[]),
            vec!["read_file".to_string(), "write_file".to_string()]
        );
    }

    #[test]
    fn explicit_allowed_tools_remain_an_authoritative_subset() {
        let mut session = Session::new();
        session.ingest(
            &[serde_json::json!({"role": "user", "content": "Do the task"})],
            &Some(serde_json::json!([
                {"type": "function", "function": {"name": "read_file"}},
                {"type": "function", "function": {"name": "write_file"}}
            ])),
        );

        assert_eq!(
            tool_names_for_turn(&session, &["read_file".to_string()]),
            vec!["read_file".to_string()]
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

        assert_eq!(repeated_identical_tool_results(&session), None);
    }

    #[test]
    fn three_identical_tool_results_force_answer() {
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
            repeated_identical_tool_results(&session),
            Some(("web_search".to_string(), 3))
        );
    }

    #[test]
    fn three_same_tool_results_with_different_arguments_do_not_force_answer() {
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": "inspect the project"}),
                tool_call_msg_with_arguments("call_1", "tree", r#"{"path":"."}"#),
                tool_result_msg("call_1", "facts src tests"),
                tool_call_msg_with_arguments("call_2", "tree", r#"{"path":"facts"}"#),
                tool_result_msg("call_2", "signal.md"),
                tool_call_msg_with_arguments("call_3", "tree", r#"{"path":"src"}"#),
                tool_result_msg("call_3", "smoke_calc.py"),
            ],
            &Some(serde_json::json!([
                {"type": "function", "function": {"name": "tree"}}
            ])),
        );

        assert_eq!(repeated_identical_tool_results(&session), None);
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
    fn repair_tool_result_answer_preserves_structured_result_after_forced_tool_call() {
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({
                    "role": "user",
                    "content": "Call the lookup_fixture_fact tool with key=codeword. Do not answer directly before the tool call."
                }),
                tool_call_msg("call_fixture", "lookup_fixture_fact"),
                tool_result_msg(
                    "call_fixture",
                    r#"{"key":"codeword","value":"signal-7429"}"#,
                ),
            ],
            &None,
        );

        let repaired = repair_tool_result_answer(&session, "Done.");

        assert!(repaired.contains("signal-7429"));
    }

    #[test]
    fn repair_tool_result_answer_descends_into_nested_evidence_objects() {
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": "lookup"}),
                tool_call_msg("call_nested", "lookup_fixture_fact"),
                tool_result_msg(
                    "call_nested",
                    r#"{"data":{"result":{"value":"signal-7429"}}}"#,
                ),
            ],
            &None,
        );

        let repaired = repair_tool_result_answer(&session, "Done.");

        assert!(repaired.contains("signal-7429"));
    }

    #[tokio::test]
    async fn failed_tool_result_reducer_is_not_reported_as_used() {
        let config = GatewayConfig {
            backends: Vec::new(),
            models: Vec::new(),
            worker_timeout: Duration::from_secs(1),
            reducer_timeout: Duration::from_secs(1),
            hedge_delay: Duration::ZERO,
            first_answer_grace: Duration::ZERO,
            strong_patience: Duration::ZERO,
            enable_thinking: Some(false),
            actor_candidates: Vec::new(),
            reference_policy: Default::default(),
            refinement_policy: Default::default(),
        };
        let mut session = Session::new();
        session.ingest(
            &[
                serde_json::json!({"role": "user", "content": "lookup"}),
                tool_call_msg("call_failed", "lookup_fixture_fact"),
                tool_result_msg("call_failed", r#"{"value":"signal-7429"}"#),
            ],
            &None,
        );

        let result = handle_tool_result(&config, &session, false, &[], Instant::now()).await;

        assert!(!result.reducer_used);
        assert_eq!(
            result
                .response_body
                .pointer("/error/code")
                .and_then(Value::as_str),
            Some(crate::MOA_ERR_ALL_REDUCERS_FAILED)
        );
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
        tool_call_msg_with_arguments(id, name, r#"{"query":"x"}"#)
    }

    fn tool_call_msg_with_arguments(id: &str, name: &str, arguments: &str) -> Value {
        serde_json::json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": arguments}
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
}
