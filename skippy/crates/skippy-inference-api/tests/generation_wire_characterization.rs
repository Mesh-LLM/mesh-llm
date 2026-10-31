//! HTTP characterization captured on main before the #2232 boundary migration.
//! Keep the golden file unchanged while migrating backend implementations.
//! Independent upstream fixes are recorded in fixtures/README.md.
//! Only frontend-generated IDs/timestamps are normalized; scripted backend IDs,
//! tool IDs, payloads, errors, optional fields and SSE ordering remain exact.

use skippy_inference_api::generation::{
    GeneratedMessage, GenerationChoice, GenerationEvent, GenerationEventStream, GenerationRequest,
    GenerationResponse,
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use axum::{body::Body, http::Request};
use futures_util::stream;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use skippy_inference_api::{
    FinishReason, InferenceBackend, InferenceError, InferenceFrontendConfig,
    InferenceRequestContext, InferenceResult, ModelObject, SystemOneRequest, SystemOneResponse,
    Usage, router_for_with_config,
};
use tower::ServiceExt;

const REQUEST_ID: &str = "11111111-2222-4333-8444-555555555555";

struct ScriptedBackend {
    script: &'static str,
    requests: Mutex<Vec<Value>>,
}

impl ScriptedBackend {
    fn record(&self, request: &GenerationRequest, context: Option<&InferenceRequestContext>) {
        self.requests.lock().unwrap().push(json!({
            "request": skippy_inference_api::chat::ChatCompletionRequest::from(request.clone()),
            "context": context.map(|context| json!({
                "request_id": context.request_id().map(|id| id.as_uuid().to_string()),
                "trusted_session": context.has_trusted_agent_session(),
                "observe_usage": context.observes_stream_usage(),
                "cancelled": context.is_cancelled(),
            })),
        }));
    }

    fn response(&self) -> InferenceResult<GenerationResponse> {
        if self.script == "reject" {
            return Err(InferenceError::invalid_request("scripted rejection").with_param("seed"));
        }
        let tools = self.script == "tools";
        Ok(GenerationResponse::from_parts(
            "scripted-generation", 42, "fixture-model",
            vec![GenerationChoice {
                index: 0,
                message: GeneratedMessage {
                    role: "assistant",
                    content: (!tools).then(|| "answer".into()),
                    reasoning_content: Some("reason".into()),
                    tool_calls: tools.then(|| json!([
                        {"id":"call_alpha","type":"function","function":{"name":"alpha","arguments":"{\"x\":1}"}},
                        {"id":"call_beta","type":"function","function":{"name":"beta","arguments":"{}"}}
                    ])),
                },
                logprobs: Some(json!({"content":[]})),
                finish_reason: Some(if tools { FinishReason::ToolCalls } else { FinishReason::Stop }),
            }], Usage::new(21, 7), Some(BTreeMap::from([("decode_ms".into(), json!(12.5))])),
        ))
    }

    fn chunks(&self) -> Vec<InferenceResult<GenerationEvent>> {
        let mut first = GenerationEvent::delta("fixture-model", "an");
        first.choices[0].delta.reasoning_content = Some("reason".into());
        let mut second = GenerationEvent::delta("fixture-model", "swer");
        let reason = if self.script == "tools" {
            first.choices[0].delta.content = None;
            first.choices[0].delta.tool_calls = Some(json!([
                {"index":0,"id":"call_alpha","type":"function","function":{"name":"alpha","arguments":"{\"x\":"}},
                {"index":1,"id":"call_beta","type":"function","function":{"name":"beta","arguments":"{"}}
            ]));
            second.choices[0].delta.content = None;
            second.choices[0].delta.tool_calls = Some(json!([
                {"index":0,"function":{"arguments":"1}"}},
                {"index":1,"function":{"arguments":"}"}}
            ]));
            FinishReason::ToolCalls
        } else {
            FinishReason::Stop
        };
        let mut chunks = vec![
            first,
            second,
            GenerationEvent::done_with_reason("fixture-model", reason),
            GenerationEvent::usage("fixture-model", Usage::new(21, 7)),
        ];
        for chunk in &mut chunks {
            chunk.id = "scripted-generation".into();
            chunk.created = 42;
        }
        let mut result: Vec<_> = chunks.into_iter().map(Ok).collect();
        if self.script == "stream-error" {
            result.insert(1, Err(InferenceError::internal("scripted stream failure")));
        }
        result
    }
}

#[async_trait]
impl InferenceBackend for ScriptedBackend {
    async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("fixture-model")])
    }

    async fn count_chat_tokens(&self, request: GenerationRequest) -> InferenceResult<u32> {
        self.record(&request, None);
        Ok(21)
    }

    async fn chat_completion(
        &self,
        request: GenerationRequest,
    ) -> InferenceResult<GenerationResponse> {
        self.record(&request, None);
        self.response()
    }

    async fn chat_completion_with_context(
        &self,
        request: GenerationRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<GenerationResponse> {
        self.record(&request, Some(&context));
        self.response()
    }

    async fn chat_completion_stream(
        &self,
        request: GenerationRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<GenerationEventStream> {
        self.record(&request, Some(&context));
        if self.script == "reject" {
            return self.response().map(|_| unreachable!());
        }
        Ok(Box::pin(stream::iter(self.chunks())))
    }

    async fn system_one(&self, request: SystemOneRequest) -> InferenceResult<SystemOneResponse> {
        self.requests
            .lock()
            .unwrap()
            .push(json!({"system_one":request}));
        Err(InferenceError::unsupported("scripted SystemOne backend"))
    }
}

struct Case {
    name: &'static str,
    path: &'static str,
    body: Value,
    script: &'static str,
}

fn chat_cases() -> Vec<Case> {
    let base = json!({"model":"fixture-model","messages":[{"role":"user","content":"hello"}]});
    let variants = [
        ("plain", json!({}), "plain"),
        ("stream", json!({"stream":true}), "plain"),
        (
            "stream-usage",
            json!({"stream":true,"stream_options":{"include_usage":true}}),
            "plain",
        ),
        ("tools", json!({}), "tools"),
        ("stream-tools", json!({"stream":true}), "tools"),
        ("reject", json!({}), "reject"),
        ("stream-reject", json!({"stream":true}), "reject"),
        ("stream-error", json!({"stream":true}), "stream-error"),
        (
            "controls",
            json!({"max_tokens":9,"max_completion_tokens":7,"temperature":0.2,"top_p":0.9,"seed":6,"presence_penalty":0.1,"frequency_penalty":0.3,"logit_bias":{"7":-1},"logprobs":true,"top_logprobs":2,"reasoning_effort":"high","prompt_cache_key":"cache-a","prompt_cache_retention":"in_memory","user":"person","stop":["END"],"n":1}),
            "plain",
        ),
        (
            "ignored-fields",
            json!({"future_option":{"a":1},"messages":[{"role":"future-role","content":"hello","future_field":null}]}),
            "plain",
        ),
        (
            "null-content",
            json!({"messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"call_alpha","type":"function","function":{"name":"alpha","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_alpha","content":"result"}]}),
            "plain",
        ),
        (
            "ordered-parts",
            json!({"messages":[{"role":"user","content":[{"type":"text","text":"before"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}},{"type":"text","text":"after"}]}]}),
            "plain",
        ),
        (
            "tools-input",
            json!({"tools":[{"type":"function","function":{"name":"alpha","parameters":{"type":"object"},"future":1},"future":2}],"tool_choice":{"type":"function","function":{"name":"alpha"}},"parallel_tool_calls":false}),
            "plain",
        ),
        ("empty-messages", json!({"messages":[]}), "plain"),
        (
            "empty-prompt",
            json!({"messages":[{"role":"user","content":""}]}),
            "plain",
        ),
        (
            "invalid-tools",
            json!({"tools":[{"type":"function","name":"flat"}]}),
            "plain",
        ),
        (
            "invalid-format",
            json!({"response_format":{"type":4}}),
            "plain",
        ),
        (
            "zero-limit",
            json!({"max_tokens":0,"max_completion_tokens":7}),
            "plain",
        ),
        ("many-choices", json!({"n":2}), "plain"),
        ("logprobs-without-flag", json!({"top_logprobs":2}), "plain"),
    ];
    variants
        .into_iter()
        .map(|(name, changes, script)| {
            let mut body = base.clone();
            body.as_object_mut()
                .unwrap()
                .extend(changes.as_object().unwrap().clone());
            Case {
                name,
                path: "/v1/chat/completions",
                body,
                script,
            }
        })
        .collect()
}

fn anthropic_cases() -> Vec<Case> {
    let base = json!({"model":"fixture-model","max_tokens":16,"messages":[{"role":"user","content":"hello"}]});
    let variants = [
        ("messages-plain", json!({}), "plain"),
        ("messages-stream", json!({"stream":true}), "plain"),
        ("messages-tools", json!({}), "tools"),
        ("messages-stream-tools", json!({"stream":true}), "tools"),
        ("messages-reject", json!({}), "reject"),
        (
            "messages-stream-error",
            json!({"stream":true}),
            "stream-error",
        ),
        (
            "messages-unknown-field",
            json!({"future_option":true,"context_management":{"edits":[]},"metadata":{"user_id":"person"}}),
            "plain",
        ),
        (
            "messages-compatible-defaults",
            json!({"thinking":{"type":"adaptive","display":"omitted"},"output_config":{"effort":"high"},"context_management":{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]},"system":[{"type":"text","text":"system","cache_control":{"type":"ephemeral"}}]}),
            "plain",
        ),
        (
            "messages-inline-system",
            json!({"system":"first","messages":[{"role":"user","content":"hello"},{"role":"system","content":"inline"}]}),
            "plain",
        ),
        (
            "messages-tool-result-order",
            json!({"messages":[{"role":"assistant","content":[{"type":"tool_use","id":"call_alpha","name":"alpha","input":{"x":1}}]},{"role":"user","content":[{"type":"text","text":"before"},{"type":"tool_result","tool_use_id":"call_alpha","content":"result"},{"type":"text","text":"after"}]}]}),
            "plain",
        ),
        (
            "messages-bad-role",
            json!({"messages":[{"role":"developer","content":"hello"}]}),
            "plain",
        ),
        ("messages-empty", json!({"messages":[]}), "plain"),
    ];
    variants
        .into_iter()
        .map(|(name, changes, script)| {
            let mut body = base.clone();
            body.as_object_mut()
                .unwrap()
                .extend(changes.as_object().unwrap().clone());
            Case {
                name,
                path: "/v1/messages",
                body,
                script,
            }
        })
        .collect()
}

fn other_cases() -> Vec<Case> {
    vec![
        Case {
            name: "responses",
            path: "/v1/responses",
            body: json!({"model":"fixture-model","input":"hello"}),
            script: "plain",
        },
        Case {
            name: "responses-stream",
            path: "/v1/responses",
            body: json!({"model":"fixture-model","input":"hello","stream":true}),
            script: "plain",
        },
        Case {
            name: "responses-tools",
            path: "/v1/responses",
            body: json!({"model":"fixture-model","input":"hello","tools":[{"type":"function","name":"alpha","parameters":{"type":"object"}}]}),
            script: "tools",
        },
        Case {
            name: "messages-count",
            path: "/v1/messages/count_tokens",
            body: json!({"model":"fixture-model","messages":[{"role":"user","content":"hello"}]}),
            script: "plain",
        },
        Case {
            name: "messages-missing-limit",
            path: "/v1/messages",
            body: json!({"model":"fixture-model","messages":[{"role":"user","content":"hello"}]}),
            script: "plain",
        },
        Case {
            name: "system-one",
            path: "/systemone",
            body: json!({"model":"fixture-model","state":{"z":1,"a":2},"questions":{"q":{"type":"noul"}}}),
            script: "plain",
        },
    ]
}

// Normalize only frontend-created identity/time, never tool-call IDs, content,
// backend IDs or the stable scripted timestamp (42).
fn normalize(value: &mut Value) {
    match value {
        Value::Array(values) => values.iter_mut().for_each(normalize),
        Value::Object(object) => {
            for (key, value) in object {
                if matches!(key.as_str(), "created" | "created_at" | "completed_at")
                    && value.as_u64().is_some_and(|v| v > 42)
                {
                    *value = json!("<frontend-time>");
                } else if matches!(key.as_str(), "id" | "item_id" | "response_id")
                    && value.as_str().is_some_and(frontend_id)
                {
                    let prefix = value.as_str().unwrap().split(['_', '-']).next().unwrap();
                    *value = json!(format!("<{prefix}-frontend-id>"));
                } else {
                    normalize(value);
                }
            }
        }
        _ => {}
    }
}

fn frontend_id(id: &str) -> bool {
    id.starts_with("chatcmpl-")
        || ["resp_", "msg_", "fc_"].iter().any(|prefix| {
            id.strip_prefix(prefix).is_some_and(|tail| {
                tail.split('_')
                    .next()
                    .is_some_and(|part| part.parse::<u64>().is_ok_and(|v| v > 42))
            })
        })
}

fn wire_body(body: &str) -> Value {
    if let Ok(mut value) = serde_json::from_str::<Value>(body) {
        normalize(&mut value);
        return value;
    }
    Value::Array(
        body.split("\n\n")
            .filter(|block| !block.is_empty())
            .map(|block| {
                Value::Array(
                    block
                        .lines()
                        .map(|line| {
                            if let Some(data) = line.strip_prefix("data: ")
                                && let Ok(mut value) = serde_json::from_str::<Value>(data)
                            {
                                normalize(&mut value);
                                return json!({"data":value});
                            }
                            json!(line)
                        })
                        .collect(),
                )
            })
            .collect(),
    )
}

async fn characterize(case: Case) -> Value {
    let backend = Arc::new(ScriptedBackend {
        script: case.script,
        requests: Mutex::default(),
    });
    let config = InferenceFrontendConfig::default()
        .with_agent_session_header("x-agent-session".parse().unwrap());
    let response = router_for_with_config(backend.clone(), config)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(case.path)
                .header("content-type", "application/json")
                .header("x-request-id", REQUEST_ID)
                .header("x-capsule-client-nonce", REQUEST_ID)
                .header("x-agent-session", "fixture-session")
                .body(Body::from(case.body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status().as_u16();
    let headers: BTreeMap<_, _> = response
        .headers()
        .iter()
        .filter(|(name, _)| name.as_str() != "content-length")
        .map(|(name, value)| (name.to_string(), value.to_str().unwrap().to_owned()))
        .collect();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    json!({"name":case.name,"status":status,"headers":headers,
        "body":wire_body(std::str::from_utf8(&bytes).unwrap()),"backend":*backend.requests.lock().unwrap()})
}

#[tokio::test]
async fn generation_wire_matches_main_baseline() {
    let mut results = Vec::new();
    for case in chat_cases()
        .into_iter()
        .chain(anthropic_cases())
        .chain(other_cases())
    {
        results.push(characterize(case).await);
    }
    let actual = Value::Array(results);
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/generation-wire-main.json");
    if std::env::var_os("UPDATE_GENERATION_WIRE_FIXTURES").is_some() {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(
            &path,
            format!("{}\n", serde_json::to_string_pretty(&actual).unwrap()),
        )
        .unwrap();
    }
    let expected: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    for (actual, expected) in actual
        .as_array()
        .unwrap()
        .iter()
        .zip(expected.as_array().unwrap())
    {
        assert_eq!(actual, expected, "wire case: {}", actual["name"]);
    }
    assert_eq!(
        actual.as_array().unwrap().len(),
        expected.as_array().unwrap().len()
    );
}
