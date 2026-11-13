//! One Responses request from the ingress that received it to its terminal
//! response: continuation restore, the response it reports, and the snapshot
//! it leaves for a later `previous_response_id`.
//!
//! The ingress prepares a [`ResponsesTurn`] from the client's raw body
//! *before* the body is translated onto Chat Completions, so the translated
//! request already carries the restored history and can be routed like any
//! other chat request. Remote workers only ever see that expanded history.
//! The ingress then builds every JSON body and stream event from the same
//! turn, and commits the snapshot only from a completed terminal response.

use std::{collections::BTreeSet, sync::Arc};

use serde_json::{Map, Value, json};

use super::resource::{
    ResponseEnvelope, ResponseError, ResponseSnapshot, ResponseStatus, ResponsesUsage,
    new_response_id,
};
use super::terminal::{chat_completion_output, malformed_tool_call};
use super::tools::{ToolOutputViolation, ToolPolicy};
use super::{continuation::ResponseContinuation, store::StoredResponse};
use crate::errors::InferenceError;

#[derive(Debug, Clone)]
enum TurnKind {
    Response,
    Compaction {
        user_messages: Vec<Value>,
        pending_calls: Vec<Value>,
    },
}

/// A Responses request being served.
#[derive(Debug, Clone)]
pub struct ResponsesTurn {
    id: String,
    created_at: i64,
    model: String,
    envelope: ResponseEnvelope,
    instructions: Option<String>,
    configuration: Map<String, Value>,
    tool_policy: ToolPolicy,
    input: Vec<Value>,
    parent: Option<Arc<StoredResponse>>,
    continuation: ResponseContinuation,
    kind: TurnKind,
}

/// A validated terminal response that has not changed continuation state yet.
///
/// Buffered SSE replay and quarantined tool streams use this staging point so
/// their entire terminal event sequence can be admitted before either the
/// shared store or a connection-local session observes the response.
pub(super) struct StagedTurnFinish<'a> {
    turn: &'a ResponsesTurn,
    status: ResponseStatus,
    output: Vec<Value>,
    response: Value,
}

impl StagedTurnFinish<'_> {
    pub(super) fn response(&self) -> &Value {
        &self.response
    }

    /// Commit a completed response and report the value the resource's
    /// `store` field must expose. Incomplete and failed responses never commit.
    pub(super) fn commit_storage(&self) -> bool {
        self.status == ResponseStatus::Completed && self.turn.commit(&self.output)
    }

    fn finalize(mut self) -> Value {
        self.response["store"] = Value::Bool(self.commit_storage());
        self.response
    }
}

impl ResponsesTurn {
    /// Prepare a raw `/v1/responses` body: reject what this server cannot
    /// honour, capture the echoed configuration, and replace a
    /// `previous_response_id` with the history it names.
    pub fn prepare(
        body: &mut Value,
        continuation: ResponseContinuation,
    ) -> Result<Self, InferenceError> {
        let object = body
            .as_object_mut()
            .ok_or_else(|| InferenceError::invalid_request("request body must be a JSON object"))?;
        reject_unsupported_request_fields(object)?;
        let explicit_model = object
            .get("model")
            .and_then(Value::as_str)
            .is_some_and(|model| !model.is_empty());
        let previous_response_id = take_previous_response_id(object)?;
        let parent = previous_response_id
            .as_deref()
            .map(|id| continuation.lookup(id))
            .transpose()?;

        // Continuations inherit the requested model when they omit it,
        // keeping `auto`/`mesh` directives rather than a serving peer. They
        // also inherit effective generation configuration unless the child
        // explicitly supplies that field. Instructions intentionally do not
        // inherit, per the Responses schema.
        if let Some(parent) = &parent {
            if object.get("model").is_none_or(Value::is_null) {
                object.insert("model".into(), Value::String(parent.model.clone()));
            }
            for (key, value) in &parent.configuration {
                object.entry(key.clone()).or_insert_with(|| value.clone());
            }
        }
        // The original ingress owns authorization. Capture the effective
        // policy after continuation inheritance and before Chat normalization
        // reduces an allowed-tools choice to its backend selection mode.
        let tool_policy = ToolPolicy::capture(object)?;
        let mut echoed = object.clone();
        if let Some(id) = &previous_response_id {
            echoed.insert("previous_response_id".into(), Value::String(id.clone()));
        }
        let model = object
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let mut envelope = ResponseEnvelope::from_request(&echoed);
        envelope.set_store(envelope.store() && continuation.available().is_some());
        let configuration = inheritable_configuration(object);
        // This is a Responses ingress policy, not a Chat Completions field.
        // The captured policy, envelope, and continuation configuration remain
        // authoritative after the backend body is normalized or transformed.
        object.remove("max_tool_calls");

        let input = effective_input_items(object)?;
        if let Some(parent) = &parent {
            let mut history = parent.transcript();
            history.extend(input.iter().cloned());
            object.insert("input".into(), Value::Array(history));
            // A chain shares one prompt-cache key, so its prefix stays warm.
            object
                .entry("prompt_cache_key")
                .or_insert_with(|| Value::String(parent.root_id().to_string()));
        }
        super::compaction::restore_input(object, &model, explicit_model)?;
        super::lineage::validate(&effective_input_items(object)?)?;
        let instructions = object
            .get("instructions")
            .and_then(Value::as_str)
            .map(ToString::to_string);
        Ok(Self {
            id: new_response_id(),
            created_at: crate::responses::now_unix_secs_i64(),
            model,
            envelope,
            instructions,
            configuration,
            tool_policy,
            input,
            parent,
            continuation,
            kind: TurnKind::Response,
        })
    }

    /// Prepare `/v1/responses/compact` as one non-streaming model summary.
    /// The original transcript is serialized inside a data-only prompt so its
    /// instructions and tool calls are never executed by the compaction pass.
    pub fn prepare_compact(
        body: &mut Value,
        continuation: ResponseContinuation,
    ) -> Result<Self, InferenceError> {
        let prepared = super::compaction::prepare(body, &continuation)?;
        Ok(Self {
            id: prepared.id,
            created_at: prepared.created_at,
            model: prepared.model,
            envelope: ResponseEnvelope::default(),
            instructions: None,
            configuration: Map::new(),
            tool_policy: ToolPolicy::capture(&Map::new())
                .expect("the default tool policy is valid"),
            input: prepared.input,
            parent: prepared.parent,
            continuation: ResponseContinuation::disabled(),
            kind: TurnKind::Compaction {
                user_messages: prepared.user_messages,
                pending_calls: prepared.pending_calls,
            },
        })
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn created_at(&self) -> i64 {
        self.created_at
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// The `ResponseResource` for this turn in the given state.
    pub fn response(
        &self,
        status: ResponseStatus,
        output: &[Value],
        usage: Option<ResponsesUsage>,
        error: Option<ResponseError<'_>>,
    ) -> Value {
        self.envelope.response(ResponseSnapshot {
            id: &self.id,
            created_at: self.created_at,
            model: &self.model,
            status,
            output,
            usage,
            error,
        })
    }

    /// Validate and build a terminal response without changing continuation
    /// state. The caller admits any required output before committing it.
    pub(super) fn stage_finish(
        &self,
        status: ResponseStatus,
        output: Vec<Value>,
        usage: Option<ResponsesUsage>,
    ) -> Result<StagedTurnFinish<'_>, ToolOutputViolation> {
        self.validate_output(status, &output)?;
        let mut envelope = self.envelope.clone();
        // `false` is conservative for replay admission (`false` serializes one
        // byte larger than `true`) and prevents a candidate from claiming it
        // was stored before the admission boundary has accepted every event.
        envelope.set_store(false);
        let response = envelope.response(ResponseSnapshot {
            id: &self.id,
            created_at: self.created_at,
            model: &self.model,
            status,
            output: &output,
            usage,
            error: None,
        });
        Ok(StagedTurnFinish {
            turn: self,
            status,
            output,
            response,
        })
    }

    /// Finish a failed turn without retaining its partial output.
    pub fn fail(
        &self,
        output: &[Value],
        usage: Option<ResponsesUsage>,
        error: ResponseError<'_>,
    ) -> Value {
        let mut envelope = self.envelope.clone();
        envelope.set_store(false);
        envelope.response(ResponseSnapshot {
            id: &self.id,
            created_at: self.created_at,
            model: &self.model,
            status: ResponseStatus::Failed,
            output,
            usage,
            error: Some(error),
        })
    }

    /// Translate a whole Chat Completions body into this turn's terminal
    /// response, and save it when it completed.
    pub fn finish_chat_completion(&self, chat: &Value) -> Value {
        self.try_finish_chat_completion(chat)
            .unwrap_or_else(|error| {
                serde_json::to_value(error.body()).expect("InferenceError body serializes")
            })
    }

    /// Fallibly translate a terminal Chat Completions body.
    ///
    /// Ordinary Responses failures remain response resources for compatibility.
    /// Compaction instead rejects an invalid or truncated backend result so an
    /// HTTP owner can return a non-success status and mark its lifecycle failed.
    pub fn try_finish_chat_completion(&self, chat: &Value) -> Result<Value, InferenceError> {
        if let TurnKind::Compaction {
            user_messages,
            pending_calls,
        } = &self.kind
        {
            return self.finish_compaction(chat, user_messages, pending_calls);
        }
        Ok(self.stage_chat_completion(chat).finalize())
    }

    pub(super) fn stage_chat_completion(&self, chat: &Value) -> StagedTurnFinish<'_> {
        if let Some(error) = chat.get("error") {
            return self.stage_failure(
                None,
                ResponseError {
                    code: error
                        .get("code")
                        .and_then(Value::as_str)
                        .unwrap_or("server_error"),
                    message: error
                        .get("message")
                        .and_then(Value::as_str)
                        .or_else(|| error.as_str())
                        .unwrap_or("upstream response failed"),
                },
            );
        }
        let choice = chat
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|choices| choices.first());
        let finish_reason = choice
            .and_then(|choice| choice.get("finish_reason"))
            .and_then(Value::as_str);
        if choice.is_none() || finish_reason.is_none() || finish_reason == Some("error") {
            return self.stage_failure(
                chat.get("usage").and_then(ResponsesUsage::from_chat_value),
                ResponseError {
                    code: "upstream_response_invalid",
                    message: "upstream response did not contain a successful terminal choice",
                },
            );
        }
        let (output, status) = match chat_completion_output(chat) {
            Ok(output) => output,
            Err(violation) => {
                return self.stage_failure(
                    chat.get("usage").and_then(ResponsesUsage::from_chat_value),
                    ResponseError {
                        code: violation.code,
                        message: violation.message,
                    },
                );
            }
        };
        let usage = chat.get("usage").and_then(ResponsesUsage::from_chat_value);
        match self.stage_finish(status, output, usage) {
            Ok(staged) => staged,
            Err(violation) => self.stage_failure(
                usage,
                ResponseError {
                    code: violation.code,
                    message: violation.message,
                },
            ),
        }
    }

    fn stage_failure(
        &self,
        usage: Option<ResponsesUsage>,
        error: ResponseError<'_>,
    ) -> StagedTurnFinish<'_> {
        StagedTurnFinish {
            turn: self,
            status: ResponseStatus::Failed,
            output: Vec::new(),
            response: self.fail(&[], usage, error),
        }
    }

    fn finish_compaction(
        &self,
        chat: &Value,
        user_messages: &[Value],
        pending_calls: &[Value],
    ) -> Result<Value, InferenceError> {
        super::compaction::finish(
            &self.id,
            self.created_at,
            &self.model,
            user_messages,
            pending_calls,
            chat,
        )
    }

    fn commit(&self, output: &[Value]) -> bool {
        let snapshot = || {
            StoredResponse::new(
                self.id.clone(),
                self.model.clone(),
                self.parent.clone(),
                self.instructions.clone(),
                self.configuration.clone(),
                self.input.clone(),
                output.to_vec(),
            )
        };
        if let Some(session) = &self.continuation.session {
            session.replace(snapshot());
        }
        if !self.envelope.store() {
            return false;
        }
        let Some((store, scope)) = self.continuation.available() else {
            return false;
        };
        store.insert(scope.clone(), snapshot())
    }

    fn validate_output(
        &self,
        status: ResponseStatus,
        output: &[Value],
    ) -> Result<(), ToolOutputViolation> {
        let calls = output
            .iter()
            .filter(|item| item.get("type").and_then(Value::as_str) == Some("function_call"))
            .map(validated_output_call)
            .collect::<Result<Vec<_>, _>>()?;
        let mut call_ids = BTreeSet::new();
        if calls.iter().any(|(id, _)| !call_ids.insert(*id)) {
            return Err(malformed_tool_call());
        }
        let names = calls.iter().map(|(_, name)| *name).collect::<Vec<_>>();
        self.tool_policy
            .validate_calls(status, !names.is_empty(), names.iter().copied())
    }

    pub(super) fn validate_stream_tool_calls<'a>(
        &self,
        status: ResponseStatus,
        had_call_fragments: bool,
        names: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), ToolOutputViolation> {
        self.tool_policy
            .validate_calls(status, had_call_fragments, names)
    }
}

fn validated_output_call(item: &Value) -> Result<(&str, &str), ToolOutputViolation> {
    let valid_string = |field: &str| {
        item.get(field)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                ToolOutputViolation::new(
                    "upstream_tool_call_invalid",
                    "upstream response contained a malformed function call",
                )
            })
    };
    let call_id = valid_string("call_id")?;
    item.get("arguments")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ToolOutputViolation::new(
                "upstream_tool_call_invalid",
                "upstream response contained non-string function arguments",
            )
        })?;
    Ok((call_id, valid_string("name")?))
}

/// Request controls that remain effective on a continuation unless the child
/// replaces them. State/lifecycle fields and instructions are per response.
const INHERITABLE_CONFIGURATION_FIELDS: &[&str] = &[
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "text",
    "reasoning",
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "top_logprobs",
    "max_output_tokens",
    "max_tool_calls",
];

fn inheritable_configuration(request: &Map<String, Value>) -> Map<String, Value> {
    INHERITABLE_CONFIGURATION_FIELDS
        .iter()
        .filter_map(|key| {
            request
                .get(*key)
                .map(|value| ((*key).to_string(), value.clone()))
        })
        .collect()
}

pub(super) fn take_previous_response_id(
    object: &mut Map<String, Value>,
) -> Result<Option<String>, InferenceError> {
    match object.remove("previous_response_id") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(id)) if !id.is_empty() => Ok(Some(id)),
        Some(_) => Err(InferenceError::invalid_request(
            "previous_response_id must be a non-empty string",
        )
        .with_param("previous_response_id")),
    }
}

/// Fields this server reads but cannot honour. They are rejected rather than
/// silently ignored.
pub(super) fn reject_unsupported_request_fields(
    object: &Map<String, Value>,
) -> Result<(), InferenceError> {
    if object.get("background").and_then(Value::as_bool) == Some(true) {
        return Err(
            InferenceError::unsupported("background responses are not supported")
                .with_param("background"),
        );
    }
    match object.get("truncation") {
        None | Some(Value::Null) => {}
        Some(Value::String(mode)) if mode == "disabled" => {}
        Some(Value::String(mode)) if mode == "auto" => {
            return Err(
                InferenceError::unsupported("automatic input truncation is not supported")
                    .with_param("truncation"),
            );
        }
        Some(_) => {
            return Err(InferenceError::invalid_request(
                "truncation must be \"disabled\" or \"auto\"",
            )
            .with_param("truncation"));
        }
    }
    if object
        .get("text")
        .and_then(|text| text.get("verbosity"))
        .is_some_and(|value| !value.is_null())
    {
        return Err(
            InferenceError::unsupported("text verbosity is not supported")
                .with_param("text.verbosity"),
        );
    }
    if let Some(tier) = object.get("service_tier").filter(|value| !value.is_null())
        && !matches!(tier.as_str(), Some("auto" | "default"))
    {
        return Err(
            InferenceError::unsupported("only the default service tier is supported")
                .with_param("service_tier"),
        );
    }
    Ok(())
}

/// The structured items a request's `input` adds to the conversation.
pub(super) fn input_items(input: &Value) -> Result<Vec<Value>, InferenceError> {
    let user_message =
        |content: Value| json!({"type": "message", "role": "user", "content": content});
    match input {
        Value::Null => Ok(Vec::new()),
        Value::String(text) => Ok(vec![user_message(
            json!([{"type": "input_text", "text": text}]),
        )]),
        Value::Array(items) if items.iter().all(is_conversation_item) => Ok(items.clone()),
        // A bare list of content parts is one user message.
        Value::Array(_) => Ok(vec![user_message(input.clone())]),
        Value::Object(_) if is_conversation_item(input) => Ok(vec![input.clone()]),
        Value::Object(_) => Ok(vec![user_message(json!([input]))]),
        _ => Err(
            InferenceError::unsupported("unsupported /v1/responses input shape")
                .with_param("input"),
        ),
    }
}

fn effective_input_items(object: &Map<String, Value>) -> Result<Vec<Value>, InferenceError> {
    object
        .get("input")
        .or_else(|| object.get("messages"))
        .map(input_items)
        .transpose()
        .map(|items| items.unwrap_or_default())
}

pub(super) fn take_effective_input(object: &mut Map<String, Value>) -> Option<Value> {
    if object.contains_key("input") {
        let input = object.remove("input");
        object.remove("messages");
        input
    } else {
        object.remove("messages")
    }
}

fn is_conversation_item(item: &Value) -> bool {
    item.get("role").is_some()
        || matches!(
            item.get("type").and_then(Value::as_str),
            Some("message" | "function_call" | "function_call_output" | "reasoning" | "compaction")
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses::store::{CallerScope, ResponseStore, ResponseStoreLimits};

    fn store() -> Arc<ResponseStore> {
        Arc::new(ResponseStore::new(ResponseStoreLimits::default()))
    }

    fn local(store: &Arc<ResponseStore>) -> ResponseContinuation {
        ResponseContinuation::new(Some(store.clone()), Some(CallerScope::LocalHost))
    }

    fn remote(store: &Arc<ResponseStore>, id: &str) -> ResponseContinuation {
        ResponseContinuation::new(
            Some(store.clone()),
            Some(CallerScope::RemoteEndpoint(id.into())),
        )
    }

    fn chat(text: &str) -> Value {
        json!({
            "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        })
    }

    fn run(body: Value, continuation: ResponseContinuation, answer: &str) -> (Value, Value) {
        let mut body = body;
        let turn = ResponsesTurn::prepare(&mut body, continuation).unwrap();
        let response = turn.finish_chat_completion(&chat(answer));
        (body, response)
    }

    #[test]
    fn continuation_restores_prior_input_and_output_before_new_input() {
        let store = store();
        let (_, first) = run(
            json!({"model": "m", "input": "my name is Ada", "instructions": "first"}),
            local(&store),
            "hello Ada",
        );
        let mut body = json!({
            "previous_response_id": first["id"],
            "input": "what is my name?",
            "instructions": "second",
        });
        let turn = ResponsesTurn::prepare(&mut body, local(&store)).unwrap();
        let input = body["input"].as_array().unwrap();
        assert_eq!(input.len(), 3);
        assert_eq!(input[0]["content"][0]["text"], "my name is Ada");
        assert_eq!(input[1]["content"][0]["text"], "hello Ada");
        assert_eq!(input[2]["content"][0]["text"], "what is my name?");
        // Only the new request's instructions apply.
        assert_eq!(body["instructions"], "second");
        assert_eq!(body["model"], "m");
        assert!(body.get("previous_response_id").is_none());
        let response = turn.response(ResponseStatus::InProgress, &[], None, None);
        assert_eq!(response["previous_response_id"], first["id"]);
        assert_ne!(response["id"], first["id"]);
    }

    #[test]
    fn function_call_round_trip_keeps_its_call_id() {
        let store = store();
        let mut body = json!({"model": "m", "input": "weather?", "tools": [{"type": "function", "name": "weather"}]});
        let turn = ResponsesTurn::prepare(&mut body, local(&store)).unwrap();
        let call = turn.finish_chat_completion(&json!({
            "choices": [{"message": {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_7", "type": "function", "function": {"name": "weather", "arguments": "{}"}}
            ]}, "finish_reason": "tool_calls"}],
        }));
        assert_eq!(call["output"][0]["type"], "function_call");
        assert_eq!(call["output"][0]["call_id"], "call_7");
        let mut body = json!({
            "previous_response_id": call["id"],
            "input": [{"type": "function_call_output", "call_id": "call_7", "output": "sunny"}],
        });
        ResponsesTurn::prepare(&mut body, local(&store)).unwrap();
        let input = body["input"].as_array().unwrap();
        assert_eq!(input[1]["call_id"], "call_7");
        assert_eq!(input[2]["call_id"], "call_7");
    }

    #[test]
    fn continuation_restores_typed_messages_alias_and_keeps_only_child_snapshot() {
        let store = store();
        let mut parent_body = json!({
            "model":"m",
            "input":"weather?",
            "tools":[{"type":"function","name":"weather"}]
        });
        let parent_turn = ResponsesTurn::prepare(&mut parent_body, local(&store)).unwrap();
        let parent = parent_turn.finish_chat_completion(&json!({
            "choices":[{"message":{"content":null,"tool_calls":[{
                "id":"call_weather","type":"function",
                "function":{"name":"weather","arguments":"{}"}
            }]},"finish_reason":"tool_calls"}]
        }));

        let mut child_body = json!({
            "previous_response_id":parent["id"],
            "messages":{"type":"function_call_output","call_id":"call_weather","output":"sunny"}
        });
        let child_turn = ResponsesTurn::prepare(&mut child_body, local(&store)).unwrap();
        crate::responses::normalize_openai_compat_request("/v1/responses", &mut child_body)
            .unwrap();
        assert_eq!(child_body["messages"][2]["role"], "tool");
        assert_eq!(child_body["messages"][2]["tool_call_id"], "call_weather");
        let child = child_turn.finish_chat_completion(&chat("done"));

        let mut grandchild = json!({
            "previous_response_id":child["id"],
            "input":"next"
        });
        ResponsesTurn::prepare(&mut grandchild, local(&store)).unwrap();
        let transcript = grandchild["input"].as_array().unwrap();
        assert_eq!(
            transcript
                .iter()
                .filter(|item| item["call_id"] == "call_weather")
                .count(),
            2,
            "the child stores only its original output item, not expanded parent history"
        );
    }

    #[test]
    fn continuation_inherits_effective_tool_configuration_with_child_overrides() {
        let store = store();
        let mut first = json!({
            "model": "m",
            "input": "weather?",
            "tools": [{"type": "function", "name": "weather"}],
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "temperature": 0.2,
            "text": {"format": {"type": "json_object"}},
        });
        let turn = ResponsesTurn::prepare(&mut first, local(&store)).unwrap();
        let response = turn.finish_chat_completion(&chat("first"));

        let mut child = json!({
            "previous_response_id": response["id"],
            "input": "again",
            "temperature": 0.7,
        });
        ResponsesTurn::prepare(&mut child, local(&store)).unwrap();
        assert_eq!(child["tools"][0]["name"], "weather");
        assert_eq!(child["tool_choice"], "auto");
        assert_eq!(child["parallel_tool_calls"], false);
        assert_eq!(child["text"]["format"]["type"], "json_object");
        assert_eq!(child["temperature"], 0.7);
        assert!(child.get("instructions").is_none());
    }

    #[test]
    fn nullable_input_and_tools_are_accepted_as_omitted() {
        let mut body = json!({"model": "m", "input": null, "tools": null});
        let turn = ResponsesTurn::prepare(&mut body, ResponseContinuation::disabled()).unwrap();
        assert!(turn.input.is_empty());
        crate::responses::normalize_openai_compat_request("/v1/responses", &mut body).unwrap();
        assert!(body["messages"].as_array().is_none_or(Vec::is_empty));
        assert!(body.get("tools").is_none_or(Value::is_null));
    }

    #[test]
    fn scopes_cannot_continue_each_other() {
        let store = store();
        let (_, first) = run(
            json!({"model": "m", "input": "hi"}),
            remote(&store, "a"),
            "x",
        );
        for continuation in [
            local(&store),
            remote(&store, "b"),
            ResponseContinuation::new(Some(store.clone()), None),
            ResponseContinuation::disabled(),
        ] {
            let mut body = json!({"previous_response_id": first["id"], "input": "again"});
            let error = ResponsesTurn::prepare(&mut body, continuation).unwrap_err();
            assert_eq!(error.status(), axum::http::StatusCode::NOT_FOUND);
        }
        let mut body = json!({"previous_response_id": first["id"], "input": "again"});
        assert!(ResponsesTurn::prepare(&mut body, remote(&store, "a")).is_ok());
    }

    #[test]
    fn store_false_child_is_not_kept_but_its_parent_continues() {
        let store = store();
        let (_, parent) = run(json!({"model": "m", "input": "hi"}), local(&store), "a");
        let (_, child) = run(
            json!({"previous_response_id": parent["id"], "input": "more", "store": false}),
            local(&store),
            "b",
        );
        assert_eq!(child["store"], false);
        let mut body = json!({"previous_response_id": child["id"], "input": "x"});
        let missing = ResponsesTurn::prepare(&mut body, local(&store)).unwrap_err();
        let mut body = json!({"previous_response_id": "resp_unknown", "input": "x"});
        let unknown = ResponsesTurn::prepare(&mut body, local(&store)).unwrap_err();
        assert_eq!(missing.status(), unknown.status());
        assert_eq!(missing.body().error.code, unknown.body().error.code);
        let mut body = json!({"previous_response_id": parent["id"], "input": "x"});
        assert!(ResponsesTurn::prepare(&mut body, local(&store)).is_ok());
    }

    #[test]
    fn branches_from_one_parent_are_independent() {
        let store = store();
        let (_, parent) = run(json!({"model": "m", "input": "root"}), local(&store), "r");
        let (_, left) = run(
            json!({"previous_response_id": parent["id"], "input": "left"}),
            local(&store),
            "L",
        );
        let (_, right) = run(
            json!({"previous_response_id": parent["id"], "input": "right"}),
            local(&store),
            "R",
        );
        let mut body = json!({"previous_response_id": left["id"], "input": "next"});
        ResponsesTurn::prepare(&mut body, local(&store)).unwrap();
        let text = body["input"].to_string();
        assert!(text.contains("left") && !text.contains("right"));
        let mut body = json!({"previous_response_id": right["id"], "input": "next"});
        ResponsesTurn::prepare(&mut body, local(&store)).unwrap();
        let text = body["input"].to_string();
        assert!(text.contains("right") && !text.contains("left"));
    }

    #[test]
    fn only_completed_responses_are_saved() {
        let store = store();
        let mut body = json!({"model": "m", "input": "hi"});
        let turn = ResponsesTurn::prepare(&mut body, local(&store)).unwrap();
        let failed = turn.response(
            ResponseStatus::Failed,
            &[],
            None,
            Some(ResponseError {
                code: "server_error",
                message: "boom",
            }),
        );
        let truncated = turn.finish_chat_completion(&json!({
            "choices": [{"message": {"content": "cut"}, "finish_reason": "length"}],
        }));
        assert_eq!(truncated["status"], "incomplete");
        assert_eq!(truncated["output"][0]["status"], "incomplete");
        for response in [failed, truncated] {
            let mut body = json!({"previous_response_id": response["id"], "input": "x"});
            assert!(ResponsesTurn::prepare(&mut body, local(&store)).is_err());
        }
    }

    #[test]
    fn malformed_and_error_chat_responses_fail_without_being_saved() {
        let store = store();
        for chat in [
            json!({"choices": []}),
            json!({"choices": [{"message": {"content": "partial"}}]}),
            json!({"choices": [{"message": {}, "finish_reason": "error"}]}),
            json!({"error": {"code": "backend_down", "message": "unavailable"}}),
        ] {
            let mut body = json!({"model": "m", "input": "hi"});
            let turn = ResponsesTurn::prepare(&mut body, local(&store)).unwrap();
            let response = turn.finish_chat_completion(&chat);
            assert_eq!(response["status"], "failed");
            assert_eq!(response["store"], false);
            let mut next = json!({"previous_response_id": response["id"], "input": "x"});
            assert!(ResponsesTurn::prepare(&mut next, local(&store)).is_err());
        }
    }

    #[test]
    fn disabled_storage_reports_store_false_and_saves_nothing() {
        let (_, response) = run(
            json!({"model": "m", "input": "hi"}),
            ResponseContinuation::disabled(),
            "a",
        );
        assert_eq!(response["store"], false);
    }

    #[test]
    fn unsupported_request_fields_are_rejected() {
        for body in [
            json!({"input": "x", "background": true}),
            json!({"input": "x", "truncation": "middle"}),
            json!({"input": "x", "truncation": "auto"}),
            json!({"input": "x", "text": {"verbosity": "high"}}),
            json!({"input": "x", "service_tier": "priority"}),
            json!({"input": "x", "previous_response_id": 7}),
        ] {
            let mut body = body;
            assert!(ResponsesTurn::prepare(&mut body, ResponseContinuation::disabled()).is_err());
        }
        let mut body = json!({"input": "x", "background": false, "truncation": "disabled"});
        assert!(ResponsesTurn::prepare(&mut body, ResponseContinuation::disabled()).is_ok());
    }

    #[test]
    fn compaction_uses_model_summary_usage_and_preserves_user_messages() {
        let mut body = json!({
            "model": "m",
            "input": [
                {"type":"message","role":"user","content":"launch Tuesday"},
                {"type":"message","role":"assistant","content":"support first"}
            ],
            "prompt_cache_key": "compact-test"
        });
        let turn =
            ResponsesTurn::prepare_compact(&mut body, ResponseContinuation::disabled()).unwrap();
        assert_eq!(body["tool_choice"], "none");
        assert!(
            body["messages"][1]["content"]
                .as_str()
                .unwrap()
                .contains("launch Tuesday")
        );
        let compacted = turn.finish_chat_completion(&json!({
            "choices": [{"message":{"content":"Launch Tuesday; notify support first."},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":23,"completion_tokens":7,"total_tokens":30}
        }));
        assert_eq!(compacted["object"], "response.compaction");
        assert_eq!(compacted["usage"]["input_tokens"], 23);
        assert_eq!(compacted["usage"]["output_tokens"], 7);
        assert_eq!(compacted["output"][0]["role"], "user");
        assert_eq!(compacted["output"][1]["type"], "compaction");
        assert_ne!(
            compacted["output"][1]["encrypted_content"],
            "Launch Tuesday; notify support first."
        );
    }

    #[test]
    fn compact_previous_response_resolves_history_but_does_not_store_a_response() {
        let store = store();
        let (_, parent) = run(
            json!({"model":"m","input":"remember slate"}),
            local(&store),
            "remembered",
        );
        let mut body = json!({
            "model":"m",
            "previous_response_id":parent["id"],
            "input":"compact now"
        });
        let compact = ResponsesTurn::prepare_compact(&mut body, local(&store)).unwrap();
        let data = body["messages"][1]["content"].as_str().unwrap();
        assert!(data.contains("remember slate") && data.contains("remembered"));
        let compacted = compact.finish_chat_completion(&chat("slate was remembered"));
        let mut bogus = json!({"previous_response_id":compacted["id"],"input":"continue"});
        assert!(ResponsesTurn::prepare(&mut bogus, local(&store)).is_err());
    }

    #[test]
    fn compaction_item_requires_explicit_matching_model_and_restores_summary() {
        let mut compact_body = json!({"model":"m","input":"remember slate"});
        let compact =
            ResponsesTurn::prepare_compact(&mut compact_body, ResponseContinuation::disabled())
                .unwrap()
                .finish_chat_completion(&chat("the code word is slate"));
        let item = compact["output"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()
            .clone();

        let mut missing = json!({"input":[item.clone()]});
        assert!(ResponsesTurn::prepare(&mut missing, ResponseContinuation::disabled()).is_err());
        let mut wrong = json!({"model":"other","input":[item.clone()]});
        assert!(ResponsesTurn::prepare(&mut wrong, ResponseContinuation::disabled()).is_err());
        let mut valid = json!({"model":"m","input":[item]});
        ResponsesTurn::prepare(&mut valid, ResponseContinuation::disabled()).unwrap();
        assert!(
            valid["input"][0]["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("the code word is slate")
        );
    }

    #[test]
    fn compaction_rejects_unknown_fields_and_missing_model() {
        for mut body in [
            json!({"input":"x"}),
            json!({"model":"m","stream":false,"input":"x"}),
            json!({"model":"m","prompt_cache_key":"x".repeat(65)}),
        ] {
            assert!(
                ResponsesTurn::prepare_compact(&mut body, ResponseContinuation::disabled())
                    .is_err()
            );
        }
    }

    #[test]
    fn json_allowed_tools_rejects_unauthorized_calls_without_storage() {
        let store = store();
        let mut body = json!({
            "model":"m", "input":"hi",
            "tools":[
                {"type":"function","name":"safe","parameters":{"type":"object"}},
                {"type":"function","name":"admin","description":"must stay visible"}
            ],
            "tool_choice":{"type":"allowed_tools","tools":[
                {"type":"function","name":"safe"}
            ]}
        });
        let turn = ResponsesTurn::prepare(&mut body, local(&store)).unwrap();
        let id = turn.id().to_string();
        crate::responses::normalize_openai_compat_request("/v1/responses", &mut body).unwrap();
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["tools"].as_array().unwrap().len(), 2);
        assert_eq!(
            body["tools"][1]["function"]["description"],
            "must stay visible"
        );
        let response = turn.finish_chat_completion(&json!({
            "choices":[{"message":{"content":null,"tool_calls":[{
                "id":"call_admin","type":"function",
                "function":{"name":"admin","arguments":"{}"}
            }]},"finish_reason":"tool_calls"}]
        }));
        assert_eq!(response["status"], "failed");
        assert_eq!(response["error"]["code"], "tool_choice_violation");
        assert_eq!(response["output"], json!([]));
        assert_eq!(response["store"], false);
        assert_eq!(response["tool_choice"]["mode"], "auto");
        assert_eq!(response["tools"].as_array().unwrap().len(), 2);
        assert!(store.get(&CallerScope::LocalHost, &id).is_err());
    }

    #[test]
    fn inherited_original_allowed_policy_remains_authoritative() {
        let store = store();
        let mut parent = json!({
            "model":"m", "input":"hi",
            "tools":[
                {"type":"function","name":"safe"},
                {"type":"function","name":"admin"}
            ],
            "tool_choice":{"type":"allowed_tools","tools":[
                {"type":"function","name":"safe"}
            ],"mode":"required"}
        });
        let parent_turn = ResponsesTurn::prepare(&mut parent, local(&store)).unwrap();
        let parent_response = parent_turn.finish_chat_completion(&json!({
            "choices":[{"message":{"content":null,"tool_calls":[{
                "id":"c1","type":"function","function":{"name":"safe","arguments":"{}"}
            }]},"finish_reason":"tool_calls"}]
        }));
        assert_eq!(parent_response["status"], "completed");

        let mut child = json!({
            "previous_response_id":parent_response["id"], "input":"again"
        });
        let child_turn = ResponsesTurn::prepare(&mut child, local(&store)).unwrap();
        assert_eq!(child["tool_choice"]["type"], "allowed_tools");
        let rejected = child_turn.finish_chat_completion(&json!({
            "choices":[{"message":{"content":null,"tool_calls":[{
                "id":"c2","type":"function","function":{"name":"admin","arguments":"{}"}
            }]},"finish_reason":"tool_calls"}]
        }));
        assert_eq!(rejected["status"], "failed");
        assert_eq!(rejected["error"]["code"], "tool_choice_violation");
    }

    #[test]
    fn json_required_allows_incomplete_without_a_call_but_not_completed() {
        for (finish_reason, status) in [("length", "incomplete"), ("stop", "failed")] {
            let mut body = json!({
                "model":"m", "input":"hi", "tool_choice":"required",
                "tools":[{"type":"function","name":"safe"}]
            });
            let turn = ResponsesTurn::prepare(&mut body, ResponseContinuation::disabled()).unwrap();
            let response = turn.finish_chat_completion(&json!({
                "choices":[{"message":{"content":"partial"},"finish_reason":finish_reason}]
            }));
            assert_eq!(response["status"], status);
        }
    }
}
