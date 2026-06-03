//! Allowed-tool enforcement.
//!
//! Workers sometimes hallucinate tool names — e.g. proposing
//! `execute_typescript` when only `shell` was declared on the request.
//! This module demotes those proposals to `Uncertainty` before they
//! reach arbitration, so a hallucinated name can't win consensus or
//! get emitted as a `tool_call` to the client.

use crate::normalize::{OutputKind, WorkerOutput};
use mesh_llm_guardrails::sanitize_tool_arguments_for_tool;
use serde_json::Value;

/// Enforce the caller's declared tool contract before arbitration.
///
/// Unknown tool names are demoted to `Uncertainty`, and known tool calls have
/// their arguments normalized against the declared JSON schema. If cleanup
/// removes a required argument, the proposal is demoted rather than emitted as
/// a broken OpenAI `tool_call`.
pub(crate) fn enforce_tool_call_contract(
    output: &mut WorkerOutput,
    allowed_tools: &[String],
    tools: Option<&Value>,
    model: &str,
) {
    if output.kind != OutputKind::ToolProposal {
        return;
    }
    let Some(ref name) = output.tool_name else {
        return;
    };

    if !allowed_tools.is_empty() && !allowed_tools.iter().any(|t| t == name) {
        tracing::warn!(
            "moa: worker {model} proposed unknown tool {name:?}, demoting to uncertainty \
             (allowed: {allowed_tools:?})"
        );
        demote_tool_proposal(output);
        return;
    }

    let args = output.tool_arguments.clone().unwrap_or(Value::Null);
    match sanitize_tool_arguments_for_tool(name, &args, tools) {
        Ok(cleaned) => output.tool_arguments = Some(cleaned),
        Err(err) => {
            tracing::warn!(
                "moa: worker {model} proposed invalid arguments for tool {name:?}: {err}; \
                 demoting to uncertainty"
            );
            demote_tool_proposal(output);
        }
    }
}

fn demote_tool_proposal(output: &mut WorkerOutput) {
    output.kind = OutputKind::Uncertainty;
    output.tool_name = None;
    output.tool_arguments = None;
    // Drop confidence so this proposal doesn't outrank real ones in any
    // tie-breaking path that still inspects it.
    output.confidence = 0.0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::WorkerRole;
    use serde_json::{Value, json};

    fn proposal(tool: &str) -> WorkerOutput {
        WorkerOutput {
            kind: OutputKind::ToolProposal,
            confidence: 0.9,
            tool_name: Some(tool.to_string()),
            tool_arguments: Some(json!({"path": "README.md"})),
            payload: format!("calling {tool}"),
            model: "alpha".into(),
            role: WorkerRole::Strong,
            elapsed_ms: 0,
        }
    }

    fn tools() -> Value {
        json!([{
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"],
                    "additionalProperties": false
                }
            }
        }])
    }

    #[test]
    fn allowed_tool_passes_through() {
        let mut out = proposal("read_file");
        enforce_tool_call_contract(&mut out, &["read_file".into()], Some(&tools()), "alpha");
        assert_eq!(out.kind, OutputKind::ToolProposal);
        assert_eq!(out.tool_name.as_deref(), Some("read_file"));
        assert!(out.tool_arguments.is_some());
        assert!((out.confidence - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn unknown_tool_is_demoted() {
        let mut out = proposal("execute_typescript");
        enforce_tool_call_contract(&mut out, &["shell".into()], None, "alpha");
        assert_eq!(out.kind, OutputKind::Uncertainty);
        assert!(out.tool_name.is_none());
        assert!(out.tool_arguments.is_none());
        assert_eq!(
            out.confidence, 0.0,
            "demoted proposals must drop confidence so they don't outrank real answers",
        );
    }

    #[test]
    fn empty_allowed_list_is_noop() {
        // No tools declared on the request — don't second-guess the worker
        // here; the reducer applies the same policy downstream.
        let mut out = proposal("anything");
        enforce_tool_call_contract(&mut out, &[], None, "alpha");
        assert_eq!(out.kind, OutputKind::ToolProposal);
        assert_eq!(out.tool_name.as_deref(), Some("anything"));
    }

    #[test]
    fn invalid_schema_arguments_are_demoted() {
        let mut out = proposal("read_file");
        out.tool_arguments = Some(json!({"path": 42, "ignored": true}));

        enforce_tool_call_contract(&mut out, &["read_file".into()], Some(&tools()), "alpha");

        assert_eq!(out.kind, OutputKind::Uncertainty);
        assert!(out.tool_name.is_none());
        assert!(out.tool_arguments.is_none());
        assert_eq!(out.confidence, 0.0);
    }

    #[test]
    fn extra_schema_arguments_are_stripped() {
        let mut out = proposal("read_file");
        out.tool_arguments = Some(json!({"path": "README.md", "ignored": true}));

        enforce_tool_call_contract(&mut out, &["read_file".into()], Some(&tools()), "alpha");

        assert_eq!(out.kind, OutputKind::ToolProposal);
        assert_eq!(out.tool_arguments, Some(json!({"path": "README.md"})));
    }

    #[test]
    fn non_proposal_outputs_untouched() {
        let mut out = WorkerOutput {
            kind: OutputKind::Answer,
            confidence: 0.7,
            tool_name: None,
            tool_arguments: None,
            payload: "Tokyo.".into(),
            model: "beta".into(),
            role: WorkerRole::Fast,
            elapsed_ms: 0,
        };
        enforce_tool_call_contract(&mut out, &["read_file".into()], None, "beta");
        assert_eq!(out.kind, OutputKind::Answer);
        assert!((out.confidence - 0.7).abs() < f32::EPSILON);
    }
}
