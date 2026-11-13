use std::{collections::HashMap, net::SocketAddr, time::Duration};

use futures_util::{SinkExt, StreamExt};
use serde_json::{Value, json};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::watch,
};
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async,
    tungstenite::{Message, protocol::Role},
};

use crate::{
    inference::election::{InferenceTarget, ModelTargets},
    mesh::{self, NodeRole},
    network::{affinity::AffinityRouter, openai::ingress::api_proxy},
};

type Socket = WebSocketStream<MaybeTlsStream<TcpStream>>;
const MODEL: &str = "echo";

struct Harness {
    address: SocketAddr,
    targets: watch::Sender<ModelTargets>,
    backend_ports: Vec<u16>,
    _proxy: tokio::task::JoinHandle<()>,
    _backends: Vec<tokio::task::JoinHandle<()>>,
}

impl Harness {
    async fn start(labels: &[&str]) -> Self {
        let mut ports = Vec::new();
        let mut backends = Vec::new();
        for label in labels {
            let (port, task) = fake_backend(label).await;
            ports.push(port);
            backends.push(task);
        }
        let node = mesh::Node::new_for_tests(NodeRole::Worker).await.unwrap();
        node.set_hosted_models(vec![MODEL.into(), "reject".into(), "stream-error".into()])
            .await;
        node.set_served_model_descriptors(
            [MODEL, "reject", "stream-error"]
                .into_iter()
                .map(|name| mesh::ServedModelDescriptor {
                    identity: mesh::ServedModelIdentity {
                        model_name: name.into(),
                        ..Default::default()
                    },
                    metadata: Some(mesh::ServedModelMetadata {
                        workload_class: Some(mesh::ModelWorkloadClass::CausalGeneration),
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .collect(),
        )
        .await;
        let initial = targets_for(ports[0]);
        let (targets, receiver) = watch::channel(initial);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let proxy = tokio::spawn(api_proxy(
            node,
            address.port(),
            receiver,
            Some(listener.into()),
            false,
            AffinityRouter::new(),
        ));
        Self {
            address,
            targets,
            backend_ports: ports,
            _proxy: proxy,
            _backends: backends,
        }
    }

    async fn connect(&self) -> Socket {
        connect_async(format!("ws://{}/v1/responses", self.address))
            .await
            .unwrap()
            .0
    }
}

fn targets_for(port: u16) -> ModelTargets {
    let mut targets = ModelTargets::default();
    targets.targets = [MODEL, "reject", "stream-error"]
        .into_iter()
        .map(|model| (model.into(), vec![InferenceTarget::Local(port)]))
        .collect::<HashMap<_, _>>();
    targets
}

async fn fake_backend(label: &str) -> (u16, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let label = label.to_string();
    let task = tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            let label = label.clone();
            tokio::spawn(async move {
                let request = read_request(&mut stream).await;
                let value: Value = serde_json::from_slice(request_body(&request)).unwrap();
                let model = value["model"].as_str().unwrap_or(MODEL);
                if model == "reject" {
                    write_json_error(&mut stream, 404, "model rejected").await;
                    return;
                }
                if model == "stream-error" {
                    write_sse(
                        &mut stream,
                        "data: {\"error\":{\"message\":\"stream failed\",\"type\":\"server_error\",\"code\":\"backend_error\"}}\n\ndata: [DONE]\n\n",
                    )
                    .await;
                    return;
                }
                let content = format!("{label}:{}", last_user_text(&value));
                if value["stream"].as_bool() == Some(true) {
                    write_chat_sse(&mut stream, model, &content).await;
                } else {
                    write_chat_json(&mut stream, model, &content).await;
                }
            });
        }
    });
    (port, task)
}

async fn read_request(stream: &mut TcpStream) -> Vec<u8> {
    let mut raw = Vec::new();
    loop {
        let mut chunk = [0; 8192];
        let count = stream.read(&mut chunk).await.unwrap();
        assert!(count > 0);
        raw.extend_from_slice(&chunk[..count]);
        let Some(header_end) = header_end(&raw) else {
            continue;
        };
        let headers = String::from_utf8_lossy(&raw[..header_end]);
        let length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())?
            })
            .unwrap_or(0);
        if raw.len() >= header_end + length {
            raw.truncate(header_end + length);
            return raw;
        }
    }
}

fn request_body(raw: &[u8]) -> &[u8] {
    &raw[header_end(raw).unwrap()..]
}

fn header_end(raw: &[u8]) -> Option<usize> {
    raw.windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|offset| offset + 4)
}

fn last_user_text(request: &Value) -> String {
    request["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .rev()
        .find(|message| message["role"] == "user")
        .and_then(|message| message["content"].as_str())
        .unwrap_or("continued")
        .to_string()
}

async fn write_chat_sse(stream: &mut TcpStream, model: &str, content: &str) {
    let first = json!({
        "id":"chatcmpl-mesh", "object":"chat.completion.chunk", "created":1,
        "model":model, "choices":[{"index":0,"delta":{"role":"assistant","content":content},"finish_reason":null}]
    });
    let terminal = json!({
        "id":"chatcmpl-mesh", "object":"chat.completion.chunk", "created":1,
        "model":model, "choices":[{"index":0,"delta":{},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}
    });
    write_sse(
        stream,
        &format!("data: {first}\n\ndata: {terminal}\n\ndata: [DONE]\n\n"),
    )
    .await;
}

async fn write_sse(stream: &mut TcpStream, body: &str) {
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n{}\r\n0\r\n\r\n",
        body.len(),
        body
    );
    stream.write_all(response.as_bytes()).await.unwrap();
}

async fn write_chat_json(stream: &mut TcpStream, model: &str, content: &str) {
    let body = json!({
        "id":"chatcmpl-mesh", "object":"chat.completion", "created":1, "model":model,
        "choices":[{"index":0,"message":{"role":"assistant","content":content},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}
    })
    .to_string();
    write_http_json(stream, 200, &body).await;
}

async fn write_json_error(stream: &mut TcpStream, status: u16, message: &str) {
    let body = json!({"error":{"message":message,"type":"not_found_error","param":null,"code":"model_not_found"}}).to_string();
    write_http_json(stream, status, &body).await;
}

async fn write_http_json(stream: &mut TcpStream, status: u16, body: &str) {
    let response = format!(
        "HTTP/1.1 {status} Test\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(response.as_bytes()).await.unwrap();
}

async fn send(socket: &mut Socket, value: Value) {
    socket
        .send(Message::Text(value.to_string().into()))
        .await
        .unwrap();
}

async fn next_event<S>(socket: &mut WebSocketStream<S>) -> Value
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    loop {
        let message = tokio::time::timeout(Duration::from_secs(5), socket.next())
            .await
            .expect("WebSocket event deadline")
            .expect("socket open")
            .expect("valid frame");
        if let Message::Text(text) = message {
            return serde_json::from_str(&text).unwrap();
        }
    }
}

async fn terminal<S>(socket: &mut WebSocketStream<S>) -> Vec<Value>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let mut events = Vec::new();
    loop {
        let event = next_event(socket).await;
        let kind = event["type"].as_str().unwrap_or_default().to_string();
        let terminal = (kind == "error" && event.get("status").is_some())
            || matches!(
                kind.as_str(),
                "response.completed" | "response.incomplete" | "response.failed"
            );
        events.push(event);
        if terminal {
            return events;
        }
    }
}

fn completed(events: &[Value]) -> &Value {
    events
        .iter()
        .find(|event| event["type"] == "response.completed")
        .unwrap()
}

#[tokio::test]
async fn official_basic_sequential_and_same_socket_store_false_cases_use_mesh_proxy() {
    let harness = Harness::start(&["mesh"]).await;
    let mut socket = harness.connect().await;
    send(
        &mut socket,
        json!({"type":"response.create","model":MODEL,"store":false,"input":"one"}),
    )
    .await;
    send(
        &mut socket,
        json!({"type":"response.create","model":MODEL,"store":false,"input":"queued"}),
    )
    .await;
    let first = terminal(&mut socket).await;
    assert_eq!(first[0]["type"], "response.created");
    assert_eq!(first[0]["sequence_number"], 0);
    let second = terminal(&mut socket).await;
    assert_eq!(second[0]["sequence_number"], 0);
    let second_id = completed(&second)["response"]["id"]
        .as_str()
        .unwrap()
        .to_string();
    send(
        &mut socket,
        json!({"type":"response.create","store":false,"previous_response_id":second_id,"input":"continue"}),
    )
    .await;
    assert!(
        completed(&terminal(&mut socket).await)["response"]["output_text"]
            .as_str()
            .unwrap()
            .contains("continue")
    );
}

#[tokio::test]
async fn official_reconnect_missing_and_failed_eviction_cases_use_mesh_proxy() {
    let harness = Harness::start(&["mesh"]).await;
    let mut first = harness.connect().await;
    send(
        &mut first,
        json!({"type":"response.create","model":MODEL,"input":"persist"}),
    )
    .await;
    let stored = completed(&terminal(&mut first).await)["response"]["id"]
        .as_str()
        .unwrap()
        .to_string();
    drop(first);

    let mut socket = harness.connect().await;
    send(
        &mut socket,
        json!({"type":"response.create","previous_response_id":stored,"input":"reconnected"}),
    )
    .await;
    assert_eq!(
        completed(&terminal(&mut socket).await)["type"],
        "response.completed"
    );
    send(
        &mut socket,
        json!({"type":"response.create","store":false,"previous_response_id":"resp_missing","input":"x"}),
    )
    .await;
    assert_eq!(
        terminal(&mut socket).await[0]["error"]["code"],
        "previous_response_not_found"
    );

    send(
        &mut socket,
        json!({"type":"response.create","model":MODEL,"store":false,"input":"local parent"}),
    )
    .await;
    let local = completed(&terminal(&mut socket).await)["response"]["id"]
        .as_str()
        .unwrap()
        .to_string();
    send(
        &mut socket,
        json!({"type":"response.create","model":"reject","store":false,"previous_response_id":local,"input":"fail"}),
    )
    .await;
    assert_eq!(terminal(&mut socket).await[0]["status"], 404);
    send(
        &mut socket,
        json!({"type":"response.create","store":false,"previous_response_id":local,"input":"retry"}),
    )
    .await;
    assert_eq!(
        terminal(&mut socket).await[0]["error"]["code"],
        "previous_response_not_found"
    );
}

#[tokio::test]
async fn official_compacted_output_starts_a_new_mesh_websocket_chain() {
    let harness = Harness::start(&["mesh"]).await;
    let compacted = post_json(
        harness.address,
        "/v1/responses/compact",
        json!({"model":MODEL,"input":"remember slate"}),
    )
    .await;
    assert_eq!(compacted["object"], "response.compaction");
    let mut input = compacted["output"].as_array().unwrap().clone();
    input.push(json!({"type":"message","role":"user","content":"what was it?"}));
    let mut socket = harness.connect().await;
    send(
        &mut socket,
        json!({"type":"response.create","model":MODEL,"input":input}),
    )
    .await;
    assert_eq!(
        completed(&terminal(&mut socket).await)["response"]["previous_response_id"],
        Value::Null
    );
}

#[tokio::test]
async fn every_turn_refreshes_mesh_routing_targets() {
    let harness = Harness::start(&["first", "second"]).await;
    let mut socket = harness.connect().await;
    send(
        &mut socket,
        json!({"type":"response.create","model":MODEL,"input":"one"}),
    )
    .await;
    assert!(
        completed(&terminal(&mut socket).await)["response"]["output_text"]
            .as_str()
            .unwrap()
            .starts_with("first:")
    );
    harness
        .targets
        .send(targets_for(harness.backend_ports[1]))
        .unwrap();
    send(
        &mut socket,
        json!({"type":"response.create","model":MODEL,"input":"two"}),
    )
    .await;
    assert!(
        completed(&terminal(&mut socket).await)["response"]["output_text"]
            .as_str()
            .unwrap()
            .starts_with("second:")
    );
}

#[tokio::test]
async fn a_stream_failure_after_created_has_one_protocol_terminal() {
    let harness = Harness::start(&["mesh"]).await;
    let mut socket = harness.connect().await;
    send(
        &mut socket,
        json!({"type":"response.create","model":"stream-error","input":"fail"}),
    )
    .await;

    let events = terminal(&mut socket).await;
    assert_eq!(events[0]["type"], "response.created");
    assert_eq!(events.last().unwrap()["type"], "response.failed");
    assert_eq!(events.last().unwrap()["response"]["status"], "failed");
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(
                event["type"].as_str(),
                Some("response.completed" | "response.incomplete" | "response.failed")
            ))
            .count(),
        1
    );
}

#[tokio::test]
async fn ping_gets_exactly_one_automatic_pong() {
    let harness = Harness::start(&["mesh"]).await;
    let mut socket = harness.connect().await;
    socket
        .send(Message::Ping(b"probe".to_vec().into()))
        .await
        .unwrap();
    let message = tokio::time::timeout(Duration::from_secs(2), socket.next())
        .await
        .expect("pong deadline")
        .expect("socket remains open")
        .expect("valid pong");
    assert_eq!(message, Message::Pong(b"probe".to_vec().into()));
    assert!(
        tokio::time::timeout(Duration::from_millis(100), socket.next())
            .await
            .is_err(),
        "a manual pong duplicated Tungstenite's automatic reply"
    );
}

#[tokio::test]
async fn peer_close_is_flushed_as_a_close_handshake() {
    let harness = Harness::start(&["mesh"]).await;
    let mut socket = harness.connect().await;
    socket.send(Message::Close(None)).await.unwrap();
    let reply = tokio::time::timeout(Duration::from_secs(2), socket.next())
        .await
        .expect("close handshake deadline")
        .expect("peer close reply arrives")
        .expect("close reply is valid");
    assert!(matches!(reply, Message::Close(_)));
}

#[tokio::test]
async fn an_early_response_create_frame_is_preserved_by_mesh_ingress() {
    let harness = Harness::start(&["mesh"]).await;
    let mut stream = TcpStream::connect(harness.address).await.unwrap();
    let event = json!({"type":"response.create","model":MODEL,"input":"early"}).to_string();
    let mut opening = format!(
        "GET /v1/responses HTTP/1.1\r\nHost: {}\r\nConnection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n",
        harness.address
    )
    .into_bytes();
    opening.extend(masked_text_frame(event.as_bytes()));
    stream.write_all(&opening).await.unwrap();
    let mut head = Vec::new();
    while header_end(&head).is_none() {
        let mut byte = [0];
        stream.read_exact(&mut byte).await.unwrap();
        head.push(byte[0]);
    }
    assert!(String::from_utf8_lossy(&head).starts_with("HTTP/1.1 101"));
    let mut socket = WebSocketStream::from_raw_socket(stream, Role::Client, None).await;
    assert_eq!(terminal(&mut socket).await[0]["type"], "response.created");
}

fn masked_text_frame(payload: &[u8]) -> Vec<u8> {
    let mask = [0x11, 0x22, 0x33, 0x44];
    let mut frame = vec![0x81];
    if payload.len() < 126 {
        frame.push(0x80 | payload.len() as u8);
    } else {
        frame.push(0x80 | 126);
        frame.extend_from_slice(&(payload.len() as u16).to_be_bytes());
    }
    frame.extend_from_slice(&mask);
    frame.extend(
        payload
            .iter()
            .enumerate()
            .map(|(index, byte)| byte ^ mask[index % mask.len()]),
    );
    frame
}

async fn post_json(address: SocketAddr, path: &str, value: Value) -> Value {
    let mut stream = TcpStream::connect(address).await.unwrap();
    let body = value.to_string();
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
    serde_json::from_slice(&response[header_end(&response).unwrap()..]).unwrap()
}
