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

use std::sync::Arc;

use serde_json::{Map, Value, json};

use super::resource::{
    ResponseEnvelope, ResponseError, ResponseSnapshot, ResponseStatus, ResponsesUsage,
    function_call_item, message_item, new_item_id, new_response_id, reasoning_item,
};
use super::store::{CallerScope, ResponseStore, StoredResponse, previous_response_not_found};
use crate::errors::InferenceError;

/// Where a request may look up and save earlier responses.
///
/// Storage is a startup setting: when it is off no store exists, nothing is
/// saved, and every `previous_response_id` is not found. A request without a
/// caller scope (for example one a reverse proxy relayed) is treated the
/// same way, so continuation fails closed.
#[derive(Debug, Clone, Default)]
pub struct ResponseContinuation {
    store: Option<Arc<ResponseStore>>,
    scope: Option<CallerScope>,
}

impl ResponseContinuation {
    pub fn new(store: Option<Arc<ResponseStore>>, scope: Option<CallerScope>) -> Self {
        Self { store, scope }
    }

    /// No lookup and no save.
    pub fn disabled() -> Self {
        Self::default()
    }

    fn available(&self) -> Option<(&ResponseStore, &CallerScope)> {
        Some((self.store.as_deref()?, self.scope.as_ref()?))
    }

    fn lookup(&self, id: &str) -> Result<Arc<StoredResponse>, InferenceError> {
        let (store, scope) = self
            .available()
            .ok_or_else(|| previous_response_not_found(id))?;
        store.get(scope, id)
    }
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
    input: Vec<Value>,
    parent: Option<Arc<StoredResponse>>,
    continuation: ResponseContinuation,
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
        let previous_response_id = match object.remove("previous_response_id") {
            None | Some(Value::Null) => None,
            Some(Value::String(id)) => Some(id),
            Some(_) => {
                return Err(InferenceError::invalid_request(
                    "previous_response_id must be a string",
                )
                .with_param("previous_response_id"));
            }
        };
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

        let input = object
            .get("input")
            .map(input_items)
            .transpose()?
            .unwrap_or_default();
        if let Some(parent) = &parent {
            let mut history = parent.transcript();
            history.extend(input.iter().cloned());
            object.insert("input".into(), Value::Array(history));
            // A chain shares one prompt-cache key, so its prefix stays warm.
            object
                .entry("prompt_cache_key")
                .or_insert_with(|| Value::String(parent.root_id().to_string()));
        }
        let instructions = object
            .get("instructions")
            .and_then(Value::as_str)
            .map(ToString::to_string);
        let configuration = inheritable_configuration(object);

        Ok(Self {
            id: new_response_id(),
            created_at: crate::responses::now_unix_secs_i64(),
            model,
            envelope,
            instructions,
            configuration,
            input,
            parent,
            continuation,
        })
    }

    /// A turn for a chat completion translated without its request, such as
    /// a virtual model's buffered answer: default configuration, no storage.
    pub fn detached(model: &str) -> Self {
        Self {
            id: new_response_id(),
            created_at: crate::responses::now_unix_secs_i64(),
            model: model.to_string(),
            envelope: ResponseEnvelope::default(),
            instructions: None,
            configuration: Map::new(),
            input: Vec::new(),
            parent: None,
            continuation: ResponseContinuation::disabled(),
        }
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

    /// Finish the turn from its terminal state, saving it for continuation
    /// when it completed and the client asked for it to be stored.
    pub fn finish(
        &self,
        status: ResponseStatus,
        output: &[Value],
        usage: Option<ResponsesUsage>,
    ) -> Value {
        let stored = status == ResponseStatus::Completed && self.commit(output);
        let mut envelope = self.envelope.clone();
        envelope.set_store(stored);
        envelope.response(ResponseSnapshot {
            id: &self.id,
            created_at: self.created_at,
            model: &self.model,
            status,
            output,
            usage,
            error: None,
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
        if let Some(error) = chat.get("error") {
            return self.fail(
                &[],
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
            return self.fail(
                &[],
                chat.get("usage").and_then(ResponsesUsage::from_chat_value),
                ResponseError {
                    code: "upstream_response_invalid",
                    message: "upstream response did not contain a successful terminal choice",
                },
            );
        }
        let (output, status) = chat_completion_output(chat);
        let usage = chat.get("usage").and_then(ResponsesUsage::from_chat_value);
        self.finish(status, &output, usage)
    }

    fn commit(&self, output: &[Value]) -> bool {
        if !self.envelope.store() {
            return false;
        }
        let Some((store, scope)) = self.continuation.available() else {
            return false;
        };
        store.insert(
            scope.clone(),
            StoredResponse::new(
                self.id.clone(),
                self.model.clone(),
                self.parent.clone(),
                self.instructions.clone(),
                self.configuration.clone(),
                self.input.clone(),
                output.to_vec(),
            ),
        )
    }
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
        .get("max_tool_calls")
        .is_some_and(|value| !value.is_null())
    {
        return Err(
            InferenceError::unsupported("max_tool_calls is not supported")
                .with_param("max_tool_calls"),
        );
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
fn input_items(input: &Value) -> Result<Vec<Value>, InferenceError> {
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

fn is_conversation_item(item: &Value) -> bool {
    item.get("role").is_some()
        || matches!(
            item.get("type").and_then(Value::as_str),
            Some("message" | "function_call" | "function_call_output" | "reasoning")
        )
}

/// The output items and status of a whole Chat Completions body.
pub(crate) fn chat_completion_output(chat: &Value) -> (Vec<Value>, ResponseStatus) {
    let choice = chat
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first());
    let message = choice.and_then(|choice| choice.get("message"));
    let status = ResponseStatus::from_finish_reason(
        choice
            .and_then(|choice| choice.get("finish_reason"))
            .and_then(Value::as_str),
    );
    let mut output = Vec::new();
    let reasoning = message
        .and_then(|message| message.get("reasoning_content"))
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty());
    if let Some(reasoning) = reasoning {
        output.push(reasoning_item(&new_item_id("rs"), reasoning));
    }
    let text = message.map(chat_message_text).unwrap_or_default();
    let calls = message
        .and_then(|message| message.get("tool_calls"))
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    if !text.is_empty() || calls.is_empty() {
        let logprobs = choice
            .and_then(|choice| choice.get("logprobs"))
            .filter(|logprobs| !logprobs.is_null())
            .cloned();
        // Text followed by calls was finished before them.
        let text_status = if calls.is_empty() {
            status.item_status()
        } else {
            "completed"
        };
        output.push(message_item(
            &new_item_id("msg"),
            text_status,
            &text,
            logprobs,
            message
                .and_then(|message| message.get("phase"))
                .and_then(Value::as_str),
        ));
    }
    for (position, call) in calls.iter().enumerate() {
        // Only the call still being generated when the response ended is cut.
        let call_status = if position + 1 == calls.len() {
            status.item_status()
        } else {
            "completed"
        };
        let function = call.get("function");
        let field = |key: &str| {
            function
                .and_then(|function| function.get(key))
                .or_else(|| call.get(key))
        };
        let arguments = match field("arguments") {
            Some(Value::String(arguments)) => arguments.clone(),
            Some(Value::Null) | None => String::new(),
            Some(other) => other.to_string(),
        };
        let call_id = call
            .get("id")
            .and_then(Value::as_str)
            .map(ToString::to_string)
            .unwrap_or_else(|| new_item_id("call"));
        output.push(function_call_item(
            &new_item_id("fc"),
            call_status,
            &call_id,
            field("name").and_then(Value::as_str).unwrap_or("tool"),
            &arguments,
        ));
    }
    (output, status)
}

fn chat_message_text(message: &Value) -> String {
    match message.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::responses::store::ResponseStoreLimits;

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
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
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
            json!({"input": "x", "max_tool_calls": 2}),
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
    fn chat_output_orders_reasoning_text_and_calls() {
        let (output, status) = chat_completion_output(&json!({
            "choices": [{"message": {
                "reasoning_content": "think",
                "content": "say",
                "tool_calls": [{"id": "c1", "function": {"name": "f", "arguments": "{}"}}],
            }, "finish_reason": "tool_calls"}],
        }));
        assert_eq!(status, ResponseStatus::Completed);
        let kinds: Vec<_> = output.iter().map(|item| item["type"].clone()).collect();
        assert_eq!(kinds, ["reasoning", "message", "function_call"]);
        assert_eq!(output[0]["content"][0]["type"], "reasoning_text");
    }

    #[test]
    fn chat_output_preserves_valid_assistant_phase() {
        let (output, _) = chat_completion_output(&json!({
            "choices": [{"message": {
                "content": "done",
                "phase": "final_answer"
            }, "finish_reason": "stop"}],
        }));
        assert_eq!(output[0]["phase"], "final_answer");
    }
}
