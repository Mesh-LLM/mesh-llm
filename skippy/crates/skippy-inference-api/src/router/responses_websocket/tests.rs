use std::{net::SocketAddr, sync::Arc, time::Duration};

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt, stream};
use serde_json::{Value, json};
use tokio::sync::Notify;
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async, tungstenite::Message as ClientMessage,
};

use crate::{
    CompletionRequest, CompletionResponse, CompletionStream, FinishReason, InferenceBackend,
    InferenceFrontendConfig, InferenceLifecycleEvent, InferenceLifecycleObserver,
    InferenceRequestContext, InferenceResult, ModelObject, ResponsesSessionConfig, Usage,
    generation::{GenerationEvent, GenerationEventStream, GenerationRequest, GenerationResponse},
    messages_to_plain_prompt, router_for_with_config,
};

type Socket = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

#[derive(Default)]
struct SocketBackend {
    started: Arc<Notify>,
    cancelled: Arc<Notify>,
}

#[async_trait]
impl InferenceBackend for SocketBackend {
    async fn models(&self) -> InferenceResult<Vec<ModelObject>> {
        Ok(vec![ModelObject::new("echo")])
    }

    async fn chat_completion(
        &self,
        request: GenerationRequest,
    ) -> InferenceResult<GenerationResponse> {
        Ok(GenerationResponse::new(
            request.model,
            "compacted summary",
            Usage::new(4, 2),
        ))
    }

    async fn chat_completion_stream(
        &self,
        request: GenerationRequest,
        context: InferenceRequestContext,
    ) -> InferenceResult<GenerationEventStream> {
        match request.model.as_str() {
            "reject" => Err(crate::InferenceError::model_not_found("reject")),
            "stream-error" => Ok(Box::pin(stream::iter([Err(
                crate::InferenceError::backend("stream failed"),
            )]))),
            "pending" => {
                self.started.notify_one();
                let cancelled = self.cancelled.clone();
                tokio::spawn(async move {
                    context.cancellation_token().cancelled().await;
                    cancelled.notify_one();
                });
                Ok(Box::pin(stream::pending()))
            }
            "starting-pending" => {
                self.started.notify_one();
                let cancellation = context.cancellation_token();
                let cancelled = self.cancelled.clone();
                tokio::spawn(async move {
                    cancellation.cancelled().await;
                    cancelled.notify_one();
                });
                std::future::pending().await
            }
            "tool-call-pair" => {
                let model = request.model;
                Ok(Box::pin(stream::iter([
                    Ok(GenerationEvent {
                        id: "chatcmpl_socket_tool_pair".to_string(),
                        created: 123,
                        model: model.clone(),
                        choices: vec![crate::generation::GenerationEventChoice {
                            index: 0,
                            delta: crate::generation::GenerationDelta {
                                role: None,
                                content: None,
                                reasoning_content: None,
                                tool_calls: Some(json!([
                                    {"index":0,"id":"call_0","type":"function","function":{"name":"same","arguments":"{}"}},
                                    {"index":1,"id":"call_1","type":"function","function":{"name":"same","arguments":"{}"}}
                                ])),
                            },
                            logprobs: None,
                            finish_reason: None,
                        }],
                        usage: None,
                        timings: None,
                    }),
                    Ok(GenerationEvent::done_with_reason(
                        model,
                        crate::FinishReason::ToolCalls,
                    )),
                ])))
            }
            _ => {
                let model = request.model;
                let text = format!("echo: {}", messages_to_plain_prompt(&request.messages));
                Ok(Box::pin(stream::iter([
                    Ok(GenerationEvent::delta(model.clone(), text)),
                    Ok(GenerationEvent::usage(model.clone(), Usage::new(3, 2))),
                    Ok(GenerationEvent::done_with_reason(model, FinishReason::Stop)),
                ])))
            }
        }
    }

    async fn completion(&self, _request: CompletionRequest) -> InferenceResult<CompletionResponse> {
        unreachable!("Responses tests do not call completions")
    }

    async fn completion_stream(
        &self,
        _request: CompletionRequest,
        _context: InferenceRequestContext,
    ) -> InferenceResult<CompletionStream> {
        unreachable!("Responses tests do not call completions")
    }
}

#[derive(Default)]
struct RecordingObserver(std::sync::Mutex<Vec<InferenceLifecycleEvent>>);

impl InferenceLifecycleObserver for RecordingObserver {
    fn observe(&self, event: &InferenceLifecycleEvent) {
        self.0.lock().expect("observer lock").push(event.clone());
    }
}

struct Server {
    address: SocketAddr,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn serve(backend: Arc<SocketBackend>, config: InferenceFrontendConfig) -> Server {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(
            listener,
            router_for_with_config(backend, config)
                .into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .unwrap();
    });
    Server { address, task }
}

fn config(session: ResponsesSessionConfig) -> InferenceFrontendConfig {
    InferenceFrontendConfig::default()
        .without_backend_timeout()
        .with_response_store(Arc::new(crate::responses::ResponseStore::default()))
        .with_responses_websocket_config(session)
}

async fn connect(address: SocketAddr) -> Socket {
    connect_async(format!("ws://{address}/v1/responses"))
        .await
        .unwrap()
        .0
}

async fn send(socket: &mut Socket, event: Value) {
    socket
        .send(ClientMessage::Text(event.to_string().into()))
        .await
        .unwrap();
}

async fn next_event(socket: &mut Socket) -> Value {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match socket.next().await.unwrap().unwrap() {
                ClientMessage::Text(text) => return serde_json::from_str(&text).unwrap(),
                ClientMessage::Ping(payload) => {
                    socket.send(ClientMessage::Pong(payload)).await.unwrap();
                }
                _ => {}
            }
        }
    })
    .await
    .expect("server event deadline")
}

async fn terminal(socket: &mut Socket) -> Vec<Value> {
    let mut events = Vec::new();
    loop {
        let event = next_event(socket).await;
        let kind = event["type"].as_str().unwrap_or_default().to_string();
        let pre_stream_error = kind == "error" && event.get("status").is_some();
        events.push(event);
        if pre_stream_error
            || matches!(
                kind.as_str(),
                "response.completed" | "response.incomplete" | "response.failed"
            )
        {
            return events;
        }
    }
}

fn completed(events: &[Value]) -> &Value {
    events
        .iter()
        .find(|event| event["type"] == "response.completed")
        .expect("completed response")
}

#[tokio::test]
async fn websocket_quarantines_calls_over_max_tool_calls() {
    let server = serve(
        Arc::new(SocketBackend::default()),
        config(ResponsesSessionConfig::default()),
    )
    .await;
    let mut socket = connect(server.address).await;
    send(
        &mut socket,
        json!({
            "type":"response.create",
            "model":"tool-call-pair",
            "input":"hi",
            "max_tool_calls":1,
            "tools":[{"type":"function","name":"same"}]
        }),
    )
    .await;

    let events = terminal(&mut socket).await;
    for (sequence, event) in events.iter().enumerate() {
        assert_eq!(event["sequence_number"], sequence as u64);
    }
    let failed = events.last().unwrap();
    assert_eq!(failed["type"], "response.failed");
    assert_eq!(
        failed["response"]["error"]["code"],
        "tool_call_limit_exceeded"
    );
    assert_eq!(failed["response"]["output"], json!([]));
    assert!(events.iter().all(|event| {
        !matches!(
            event["type"].as_str(),
            Some(
                "response.output_item.added"
                    | "response.function_call_arguments.delta"
                    | "response.function_call_arguments.done"
            )
        )
    }));
}

#[tokio::test]
async fn websocket_quarantines_calls_when_parallel_tools_are_disabled() {
    let server = serve(
        Arc::new(SocketBackend::default()),
        config(ResponsesSessionConfig::default()),
    )
    .await;
    let mut socket = connect(server.address).await;
    send(
        &mut socket,
        json!({
            "type":"response.create",
            "model":"tool-call-pair",
            "input":"hi",
            "parallel_tool_calls":false,
            "tools":[{"type":"function","name":"same"}]
        }),
    )
    .await;

    let events = terminal(&mut socket).await;
    for (sequence, event) in events.iter().enumerate() {
        assert_eq!(event["sequence_number"], sequence as u64);
    }
    let failed = events.last().unwrap();
    assert_eq!(failed["type"], "response.failed");
    assert_eq!(
        failed["response"]["error"]["code"],
        "tool_call_limit_exceeded"
    );
    assert_eq!(failed["response"]["output"], json!([]));
    assert!(events.iter().all(|event| {
        !matches!(
            event["type"].as_str(),
            Some(
                "response.output_item.added"
                    | "response.function_call_arguments.delta"
                    | "response.function_call_arguments.done"
            )
        )
    }));
}

#[tokio::test]
async fn official_basic_sequential_and_same_socket_store_false_cases() {
    let server = serve(
        Arc::new(SocketBackend::default()),
        config(ResponsesSessionConfig::default()),
    )
    .await;
    let mut socket = connect(server.address).await;

    send(
        &mut socket,
        json!({"type":"response.create","model":"echo","store":false,"input":"one"}),
    )
    .await;
    send(
        &mut socket,
        json!({"type":"response.create","model":"echo","store":false,"input":"queued"}),
    )
    .await;
    let first = terminal(&mut socket).await;
    let first_completed = completed(&first);
    assert_eq!(first[0]["sequence_number"], 0);
    assert_eq!(first_completed["response"]["store"], false);
    let delta = first
        .iter()
        .find(|event| event["type"] == "response.output_text.delta")
        .expect("default WebSocket turn emits a text delta");
    assert!(delta["obfuscation"].as_str().is_some());
    assert_eq!(serde_json::to_vec(delta).unwrap().len() % 256, 0);
    let second = terminal(&mut socket).await;
    assert_eq!(second[0]["type"], "response.created");
    assert_eq!(second[0]["sequence_number"], 0);
    let second_id = completed(&second)["response"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    send(
        &mut socket,
        json!({
            "type":"response.create",
            "store":false,
            "previous_response_id":second_id,
            "input":"continue"
        }),
    )
    .await;
    let continued = terminal(&mut socket).await;
    assert!(
        completed(&continued)["response"]["output_text"]
            .as_str()
            .unwrap()
            .contains("continue")
    );
}

#[tokio::test]
async fn official_reconnect_missing_and_failed_continuation_eviction_cases() {
    let server = serve(
        Arc::new(SocketBackend::default()),
        config(ResponsesSessionConfig::default()),
    )
    .await;
    let mut first_socket = connect(server.address).await;
    send(
        &mut first_socket,
        json!({"type":"response.create","model":"echo","input":"persist me"}),
    )
    .await;
    let first = terminal(&mut first_socket).await;
    let stored_id = completed(&first)["response"]["id"]
        .as_str()
        .unwrap()
        .to_string();
    drop(first_socket);

    let mut socket = connect(server.address).await;
    send(
        &mut socket,
        json!({"type":"response.create","previous_response_id":stored_id,"input":"reconnected"}),
    )
    .await;
    assert_eq!(
        completed(&terminal(&mut socket).await)["type"],
        "response.completed"
    );

    send(
        &mut socket,
        json!({"type":"response.create","store":false,"previous_response_id":stored_id,"input":"no shared fallback"}),
    )
    .await;
    assert_eq!(
        terminal(&mut socket).await[0]["error"]["code"],
        "previous_response_not_found"
    );

    send(
        &mut socket,
        json!({"type":"response.create","store":false,"previous_response_id":"resp_missing","input":"x"}),
    )
    .await;
    let missing = terminal(&mut socket).await;
    assert_eq!(missing[0]["error"]["code"], "previous_response_not_found");

    send(
        &mut socket,
        json!({"type":"response.create","model":"echo","store":false,"input":"local parent"}),
    )
    .await;
    let local_id = completed(&terminal(&mut socket).await)["response"]["id"]
        .as_str()
        .unwrap()
        .to_string();
    send(
        &mut socket,
        json!({
            "type":"response.create",
            "model":"reject",
            "store":false,
            "previous_response_id":local_id,
            "input":"fail"
        }),
    )
    .await;
    assert_eq!(terminal(&mut socket).await[0]["status"], 404);
    send(
        &mut socket,
        json!({"type":"response.create","store":false,"previous_response_id":local_id,"input":"retry"}),
    )
    .await;
    assert_eq!(
        terminal(&mut socket).await[0]["error"]["code"],
        "previous_response_not_found"
    );

    send(
        &mut socket,
        json!({"type":"response.create","model":"echo","store":false,"input":"stream parent"}),
    )
    .await;
    let stream_parent = completed(&terminal(&mut socket).await)["response"]["id"]
        .as_str()
        .unwrap()
        .to_string();
    send(
        &mut socket,
        json!({
            "type":"response.create",
            "model":"stream-error",
            "store":false,
            "previous_response_id":stream_parent,
            "input":"fail after response.created"
        }),
    )
    .await;
    let failed = terminal(&mut socket).await;
    assert_eq!(failed.last().unwrap()["type"], "response.failed");
    send(
        &mut socket,
        json!({"type":"response.create","store":false,"previous_response_id":stream_parent,"input":"retry"}),
    )
    .await;
    assert_eq!(
        terminal(&mut socket).await[0]["error"]["code"],
        "previous_response_not_found"
    );
}

#[tokio::test]
async fn orphan_output_evicts_only_the_store_false_local_parent() {
    let server = serve(
        Arc::new(SocketBackend::default()),
        config(ResponsesSessionConfig::default()),
    )
    .await;
    let mut socket = connect(server.address).await;
    send(
        &mut socket,
        json!({
            "type":"response.create",
            "model":"echo",
            "store":false,
            "input":[
                {"type":"message","role":"user","content":"use the tool"},
                {"type":"function_call","call_id":"call_parent","name":"same","arguments":"{}"}
            ]
        }),
    )
    .await;
    let parent = completed(&terminal(&mut socket).await)["response"]["id"]
        .as_str()
        .unwrap()
        .to_string();

    send(
        &mut socket,
        json!({
            "type":"response.create",
            "store":false,
            "previous_response_id":parent,
            "input":[{"type":"function_call_output","call_id":"call_orphan","output":"x"}]
        }),
    )
    .await;
    let orphan = terminal(&mut socket).await;
    assert_eq!(orphan[0]["type"], "error");
    assert_eq!(orphan[0]["status"], 400);
    assert_eq!(orphan[0]["error"]["type"], "invalid_request_error");
    assert_eq!(orphan[0]["error"]["code"], "invalid_value");
    assert_eq!(orphan[0]["error"]["param"], "input");

    send(
        &mut socket,
        json!({
            "type":"response.create",
            "store":false,
            "previous_response_id":parent,
            "input":[{"type":"function_call_output","call_id":"call_parent","output":"ok"}]
        }),
    )
    .await;
    assert_eq!(
        terminal(&mut socket).await[0]["error"]["code"],
        "previous_response_not_found"
    );
}

#[tokio::test]
async fn official_compacted_output_starts_a_new_websocket_chain() {
    let server = serve(
        Arc::new(SocketBackend::default()),
        config(ResponsesSessionConfig::default()),
    )
    .await;
    let compacted = live_post(
        server.address,
        "/v1/responses/compact",
        json!({"model":"echo","input":"remember slate"}),
    )
    .await;
    assert_eq!(compacted["object"], "response.compaction");
    let mut input = compacted["output"].as_array().unwrap().clone();
    input.push(json!({"type":"message","role":"user","content":"what was it?"}));

    let mut socket = connect(server.address).await;
    send(
        &mut socket,
        json!({"type":"response.create","model":"echo","input":input}),
    )
    .await;
    let response = terminal(&mut socket).await;
    assert_eq!(
        completed(&response)["response"]["previous_response_id"],
        Value::Null
    );
}

#[tokio::test]
async fn forbidden_field_presence_evicts_the_referenced_local_response() {
    let server = serve(
        Arc::new(SocketBackend::default()),
        config(ResponsesSessionConfig::default()),
    )
    .await;
    let mut socket = connect(server.address).await;
    for (field, event) in [
        (
            "background",
            json!({"type":"response.create","model":"echo","background":false,"input":"bad"}),
        ),
        (
            "stream_options",
            json!({"type":"response.create","model":"echo","stream_options":null,"input":"bad"}),
        ),
    ] {
        send(&mut socket, event).await;
        assert_eq!(terminal(&mut socket).await[0]["error"]["param"], field);
    }
    send(
        &mut socket,
        json!({"type":"response.create","model":"echo","store":false,"input":"parent"}),
    )
    .await;
    let id = completed(&terminal(&mut socket).await)["response"]["id"]
        .as_str()
        .unwrap()
        .to_string();
    send(
        &mut socket,
        json!({"type":"response.create","store":false,"previous_response_id":id,"stream":null,"input":"bad"}),
    )
    .await;
    assert_eq!(terminal(&mut socket).await[0]["error"]["param"], "stream");
    send(
        &mut socket,
        json!({"type":"response.create","store":false,"previous_response_id":id,"input":"retry"}),
    )
    .await;
    assert_eq!(
        terminal(&mut socket).await[0]["error"]["code"],
        "previous_response_not_found"
    );
}

#[tokio::test]
async fn queue_and_transcript_memory_are_bounded() {
    let backend = Arc::new(SocketBackend::default());
    let started = backend.started.clone();
    let server = serve(
        backend,
        config(
            ResponsesSessionConfig::default()
                .with_max_pending_turns(1)
                .with_max_transcript_bytes(1),
        ),
    )
    .await;
    let mut socket = connect(server.address).await;
    send(
        &mut socket,
        json!({"type":"response.create","model":"pending","input":"hold"}),
    )
    .await;
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .expect("backend start deadline");
    send(
        &mut socket,
        json!({"type":"response.create","model":"echo","input":"queued"}),
    )
    .await;
    send(
        &mut socket,
        json!({"type":"response.create","model":"echo","input":"overflow"}),
    )
    .await;
    loop {
        let event = next_event(&mut socket).await;
        if event["error"]["code"] == "websocket_queue_full" {
            break;
        }
    }
    socket.send(ClientMessage::Close(None)).await.unwrap();

    let mut socket = connect(server.address).await;
    send(
        &mut socket,
        json!({"type":"response.create","model":"echo","store":false,"input":"too large to retain"}),
    )
    .await;
    let id = completed(&terminal(&mut socket).await)["response"]["id"]
        .as_str()
        .unwrap()
        .to_string();
    send(
        &mut socket,
        json!({"type":"response.create","store":false,"previous_response_id":id,"input":"next"}),
    )
    .await;
    assert_eq!(
        terminal(&mut socket).await[0]["error"]["code"],
        "previous_response_not_found"
    );
}

#[tokio::test]
async fn oversized_websocket_frames_are_closed_by_the_upgrade_limit() {
    let backend = Arc::new(SocketBackend::default());
    let server = serve(
        backend,
        config(ResponsesSessionConfig::default()).with_max_request_body_bytes(128),
    )
    .await;
    let mut socket = connect(server.address).await;
    let oversized = "x".repeat(256);
    let _ = socket.send(ClientMessage::Text(oversized.into())).await;
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match socket.next().await {
                None | Some(Err(_)) | Some(Ok(ClientMessage::Close(_))) => return,
                Some(Ok(_)) => {}
            }
        }
    })
    .await
    .expect("oversized frame closes the socket");
}

#[tokio::test]
async fn ping_and_close_frames_are_handled_without_starting_a_turn() {
    let server = serve(
        Arc::new(SocketBackend::default()),
        config(ResponsesSessionConfig::default()),
    )
    .await;
    let mut socket = connect(server.address).await;
    socket
        .send(ClientMessage::Ping(vec![1, 2, 3].into()))
        .await
        .unwrap();
    let pong = tokio::time::timeout(Duration::from_secs(5), socket.next())
        .await
        .expect("pong deadline")
        .unwrap()
        .unwrap();
    assert_eq!(pong, ClientMessage::Pong(vec![1, 2, 3].into()));

    socket.send(ClientMessage::Close(None)).await.unwrap();
    let close = tokio::time::timeout(Duration::from_secs(5), socket.next())
        .await
        .expect("close deadline");
    assert!(
        matches!(close, None | Some(Ok(ClientMessage::Close(_)))),
        "unexpected close frame: {close:?}"
    );
}

#[tokio::test]
async fn absolute_deadline_cancels_active_generation_and_terminalizes_once() {
    let backend = Arc::new(SocketBackend::default());
    let started = backend.started.clone();
    let cancelled = backend.cancelled.clone();
    let observer = Arc::new(RecordingObserver::default());
    let server = serve(
        backend,
        config(ResponsesSessionConfig::default().with_connection_lifetime(Duration::from_secs(30)))
            .with_lifecycle_observer(observer.clone()),
    )
    .await;
    let mut socket = connect(server.address).await;
    send(
        &mut socket,
        json!({"type":"response.create","model":"pending","input":"hold"}),
    )
    .await;
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .expect("backend start deadline");
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(30)).await;
    tokio::time::resume();
    loop {
        let event = next_event(&mut socket).await;
        if event["error"]["code"] == "websocket_connection_limit_reached" {
            assert_eq!(event["status"], 400);
            break;
        }
    }
    tokio::time::timeout(Duration::from_secs(5), cancelled.notified())
        .await
        .expect("backend cancellation deadline");
    let events = observer.0.lock().expect("observer lock");
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, InferenceLifecycleEvent::StreamDropped { .. }))
            .count(),
        1
    );
}

#[tokio::test]
async fn disconnect_cancels_generation_while_backend_start_is_pending() {
    let backend = Arc::new(SocketBackend::default());
    let started = backend.started.clone();
    let cancelled = backend.cancelled.clone();
    let server = serve(backend, config(ResponsesSessionConfig::default())).await;
    let mut socket = connect(server.address).await;
    send(
        &mut socket,
        json!({"type":"response.create","model":"starting-pending","input":"hold"}),
    )
    .await;
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .expect("backend start deadline");
    drop(socket);
    tokio::time::timeout(Duration::from_secs(5), cancelled.notified())
        .await
        .expect("backend-start cancellation");
}

#[tokio::test]
async fn disconnect_cancels_active_generation_without_polling_or_sleeping() {
    let backend = Arc::new(SocketBackend::default());
    let started = backend.started.clone();
    let cancelled = backend.cancelled.clone();
    let server = serve(backend, config(ResponsesSessionConfig::default())).await;
    let mut socket = connect(server.address).await;
    send(
        &mut socket,
        json!({"type":"response.create","model":"pending","input":"hold"}),
    )
    .await;
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .expect("backend start deadline");
    drop(socket);
    tokio::time::timeout(Duration::from_secs(5), cancelled.notified())
        .await
        .expect("backend cancellation");
}

async fn live_post(address: SocketAddr, path: &str, body: Value) -> Value {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut stream = tokio::net::TcpStream::connect(address).await.unwrap();
    let body = body.to_string();
    stream
        .write_all(
            format!(
                "POST {path} HTTP/1.1\r\nHost: {address}\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )
            .as_bytes(),
        )
        .await
        .unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    let body_start = response
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .unwrap()
        + 4;
    serde_json::from_slice(&response[body_start..]).unwrap()
}
