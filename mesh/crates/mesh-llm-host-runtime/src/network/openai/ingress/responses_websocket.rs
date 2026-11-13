//! Persistent Responses WebSocket ingress for the Mesh public API.
//!
//! The socket and shared continuation state stay at the caller's ingress.
//! Every turn becomes a normal buffered request, so admission, transforms,
//! routing, retry, and the local/QUIC execution boundary remain unchanged.

use std::{collections::VecDeque, net::SocketAddr, sync::Arc, time::Duration};

use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::{io::AsyncWriteExt, sync::mpsc, task::JoinSet};
use tokio_tungstenite::{
    WebSocketStream,
    tungstenite::{
        Message,
        handshake::derive_accept_key,
        protocol::{Role, WebSocketConfig},
    },
};

use super::{IngressRouteContext, ProxyConnectionContext, handle_buffered_api_request};
use crate::{
    inference::election,
    mesh,
    network::{affinity, openai::client_stream::ClientStream},
    runtime::IngressType,
};
use skippy_inference_api::{
    InferenceError, InferenceErrorKind,
    responses::{
        FORWARDING_HEADERS, ResponsesSession, ResponsesSessionConfig, WebSocketErrorEvent,
        caller_scope_for_http_peer,
    },
};

const CONTROL_QUEUE: usize = 4;
const CAPTURE_BYTES: usize = 64 * 1024;
const DISCONNECT_GRACE: Duration = Duration::from_secs(2);
const WRITER_SHUTDOWN_GRACE: Duration = Duration::from_secs(1);

pub(in crate::network::openai) fn is_upgrade(
    request: &crate::network::openai::transport::BufferedHttpRequest,
) -> bool {
    request.method.eq_ignore_ascii_case("GET")
        && request.client_path.split('?').next() == Some("/v1/responses")
        && header_has_token(&request.raw, "connection", "upgrade")
        && header_has_token(&request.raw, "upgrade", "websocket")
}

pub(super) async fn serve(
    node: mesh::Node,
    client: ClientStream,
    handshake: crate::network::openai::transport::BufferedHttpRequest,
    affinity: affinity::AffinityRouter,
    target_rx: tokio::sync::watch::Receiver<election::ModelTargets>,
    scope: Option<mesh::PrivateScope>,
    source_addr: Option<SocketAddr>,
) {
    let forwarded = FORWARDING_HEADERS
        .iter()
        .any(|name| header_present(&handshake.raw, name));
    let caller_scope = caller_scope_for_http_peer(source_addr.map(|addr| addr.ip()), forwarded);
    let session = ResponsesSession::new(
        ResponsesSessionConfig::default(),
        affinity.response_store(),
        caller_scope,
    );
    let Some(socket) = accept_websocket(client, &handshake).await else {
        return;
    };
    Connection {
        node,
        affinity,
        target_rx,
        scope,
        source_addr,
        handshake: Arc::new(handshake),
        session,
    }
    .run(socket)
    .await;
}

struct Connection {
    node: mesh::Node,
    affinity: affinity::AffinityRouter,
    target_rx: tokio::sync::watch::Receiver<election::ModelTargets>,
    scope: Option<mesh::PrivateScope>,
    source_addr: Option<SocketAddr>,
    handshake: Arc<crate::network::openai::transport::BufferedHttpRequest>,
    session: ResponsesSession,
}

impl Connection {
    async fn run(self, socket: WebSocketStream<ClientStream>) {
        let deadline = tokio::time::sleep(self.session.config().connection_lifetime());
        tokio::pin!(deadline);
        let (mut sink, mut incoming) = socket.split();
        let (events_tx, mut events_rx) =
            mpsc::channel::<Value>(self.session.config().max_outbound_events());
        let (control_tx, mut control_rx) = mpsc::channel::<Message>(CONTROL_QUEUE);
        let mut writer = tokio::spawn(async move {
            let mut events_open = true;
            let mut control_open = true;
            loop {
                let message = tokio::select! {
                    biased;
                    message = control_rx.recv(), if control_open => match message {
                        Some(message) => message,
                        None => {
                            control_open = false;
                            continue;
                        }
                    },
                    value = events_rx.recv(), if events_open => match value {
                        Some(value) => Message::Text(value.to_string().into()),
                        None => {
                            events_open = false;
                            continue;
                        }
                    },
                    else => break,
                };
                let close = matches!(message, Message::Close(_));
                // Reading a peer Close queues Tungstenite's automatic reply.
                // Flush that reply before an explicit Close is attempted; the
                // protocol may reject the latter once it is already closing.
                if close && sink.flush().await.is_err() {
                    break;
                }
                if sink.send(message).await.is_err() || close {
                    break;
                }
            }
        });
        let _writer_abort = AbortTaskOnDrop(writer.abort_handle());
        let mut pending = VecDeque::new();
        let mut active = JoinSet::new();
        let (disconnect_tx, _) = tokio::sync::watch::channel(false);

        let exit = loop {
            tokio::select! {
                biased;
                () = &mut deadline => break ConnectionExit::Deadline,
                _ = &mut writer => break ConnectionExit::WriterStopped,
                _ = active.join_next(), if !active.is_empty() => {
                    if !start_pending(
                        &mut pending,
                        &mut active,
                        &self,
                        events_tx.clone(),
                        disconnect_tx.subscribe(),
                    ) {
                        break ConnectionExit::OutboundBackpressure;
                    }
                }
                message = incoming.next() => {
                    if let Some(exit) = self.handle_incoming(
                        message,
                        &mut pending,
                        &mut active,
                        &events_tx,
                        &control_tx,
                        disconnect_tx.subscribe(),
                    ) {
                        break exit;
                    }
                }
            }
        };

        if matches!(exit, ConnectionExit::Deadline) {
            let error = InferenceError::invalid_request(
                "Responses WebSocket connections are limited to 60 minutes",
            )
            .with_code("websocket_connection_limit_reached");
            let _ = control_tx.try_send(Message::Text(
                WebSocketErrorEvent::from_error(&error)
                    .into_value()
                    .to_string()
                    .into(),
            ));
            let _ = control_tx.try_send(Message::Close(None));
        } else if matches!(exit, ConnectionExit::OutboundBackpressure) {
            let _ = control_tx.try_send(Message::Close(None));
        }
        let _ = disconnect_tx.send(true);
        settle_active(&mut active).await;
        drop(events_tx);
        drop(control_tx);
        if !matches!(exit, ConnectionExit::WriterStopped) {
            tokio::select! {
                _ = &mut writer => {}
                () = tokio::time::sleep(WRITER_SHUTDOWN_GRACE) => writer.abort(),
            }
        }
    }

    fn handle_incoming(
        &self,
        message: Option<Result<Message, tokio_tungstenite::tungstenite::Error>>,
        pending: &mut VecDeque<Value>,
        active: &mut JoinSet<()>,
        events: &mpsc::Sender<Value>,
        control: &mpsc::Sender<Message>,
        disconnect: tokio::sync::watch::Receiver<bool>,
    ) -> Option<ConnectionExit> {
        match message {
            Some(Ok(Message::Text(text))) => (!self
                .handle_text(&text, pending, active, events, disconnect))
            .then_some(ConnectionExit::OutboundBackpressure),
            // Tungstenite queues and flushes the matching Pong while reading.
            // Sending one manually duplicates the automatic protocol reply.
            Some(Ok(Message::Ping(_))) => None,
            Some(Ok(Message::Close(frame))) => {
                let _ = control.try_send(Message::Close(frame));
                Some(ConnectionExit::PeerClosed)
            }
            Some(Ok(Message::Binary(_))) => {
                let error = InferenceError::invalid_request(
                    "Responses WebSocket messages must be JSON text",
                );
                try_send_error(events, &error)
                    .is_err()
                    .then_some(ConnectionExit::OutboundBackpressure)
            }
            Some(Ok(_)) => None,
            Some(Err(_)) | None => Some(ConnectionExit::PeerDisconnected),
        }
    }

    fn handle_text(
        &self,
        text: &str,
        pending: &mut VecDeque<Value>,
        active: &mut JoinSet<()>,
        events: &mpsc::Sender<Value>,
        disconnect: tokio::sync::watch::Receiver<bool>,
    ) -> bool {
        let value = match serde_json::from_str::<Value>(text) {
            Ok(value) => value,
            Err(error) => {
                let error =
                    InferenceError::invalid_request(format!("invalid WebSocket JSON: {error}"));
                return try_send_error(events, &error).is_ok();
            }
        };
        if active.is_empty() {
            pending.push_back(value);
            return start_pending(pending, active, self, events.clone(), disconnect);
        }
        if pending.len() < self.session.config().max_pending_turns() {
            pending.push_back(value);
            return true;
        }
        self.session
            .evict_referenced(value.get("previous_response_id").and_then(Value::as_str));
        let error = InferenceError::from_kind(
            http::StatusCode::TOO_MANY_REQUESTS,
            InferenceErrorKind::RateLimit,
            "Responses WebSocket pending-turn queue is full",
        )
        .with_code("websocket_queue_full");
        try_send_error(events, &error).is_ok()
    }
}

#[derive(Clone, Copy)]
enum ConnectionExit {
    Deadline,
    WriterStopped,
    PeerClosed,
    PeerDisconnected,
    OutboundBackpressure,
}

struct AbortTaskOnDrop(tokio::task::AbortHandle);

impl Drop for AbortTaskOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

fn start_pending(
    pending: &mut VecDeque<Value>,
    active: &mut JoinSet<()>,
    connection: &Connection,
    events: mpsc::Sender<Value>,
    disconnect: tokio::sync::watch::Receiver<bool>,
) -> bool {
    while active.is_empty() {
        let Some(value) = pending.pop_front() else {
            return true;
        };
        let prepared = match connection.session.prepare_create(value, None) {
            Ok(prepared) => prepared,
            Err(error) => {
                if try_send_error(&events, &error).is_err() {
                    return false;
                }
                continue;
            }
        };
        let turn = TurnExecution {
            node: connection.node.clone(),
            affinity: connection.affinity.clone(),
            target_rx: connection.target_rx.clone(),
            scope: connection.scope.clone(),
            source_addr: connection.source_addr,
            handshake: Arc::clone(&connection.handshake),
            session: connection.session.clone(),
            events: events.clone(),
            disconnect: disconnect.clone(),
        };
        active.spawn(async move {
            let scope = turn.scope.clone();
            // The ordinary routed dispatch future contains every admission,
            // routing and response-adapter arm. Keep it behind a heap seam so
            // the private-scope wrapper and JoinSet task retain only a pointer.
            mesh::with_private_scope(scope, Box::pin(turn.run(prepared))).await;
        });
    }
    true
}

struct TurnExecution {
    node: mesh::Node,
    affinity: affinity::AffinityRouter,
    target_rx: tokio::sync::watch::Receiver<election::ModelTargets>,
    scope: Option<mesh::PrivateScope>,
    source_addr: Option<SocketAddr>,
    handshake: Arc<crate::network::openai::transport::BufferedHttpRequest>,
    session: ResponsesSession,
    events: mpsc::Sender<Value>,
    disconnect: tokio::sync::watch::Receiver<bool>,
}

impl TurnExecution {
    async fn run(self, prepared: skippy_inference_api::responses::PreparedResponsesTurn) {
        let previous_response_id = prepared.previous_response_id().map(ToString::to_string);
        let plugin_manager = self.node.plugin_manager().await;
        let request = match crate::network::openai::request_parse::buffer_prepared_responses_turn(
            &self.handshake,
            prepared,
            plugin_manager.as_ref(),
        )
        .await
        {
            Ok(request) => request,
            Err(error) => {
                self.session
                    .evict_referenced(previous_response_id.as_deref());
                let error = error
                    .downcast_ref::<InferenceError>()
                    .cloned()
                    .unwrap_or_else(|| InferenceError::backend(error.to_string()));
                let _ = send_error(&self.events, &error).await;
                return;
            }
        };
        let mut targets = self.node.peer_blocks.without_blocked(
            &self.target_rx.borrow(),
            crate::network::peer_blocks::now_ms(),
        );
        if let Some(scope) = self.scope.as_ref() {
            targets = self.node.targets_in_private_scope(scope, &targets).await;
        }
        let (capture_writer, capture_reader) = ClientStream::capture_pair(CAPTURE_BYTES);
        let route = IngressRouteContext {
            node: &self.node,
            targets: &targets,
            affinity: &self.affinity,
            plugin_manager: plugin_manager.as_ref(),
            requested_by_node_id: None,
            automatic_retry: None,
            #[cfg(test)]
            exchange_channel: None,
        };
        let dispatch = handle_buffered_api_request(
            capture_writer,
            request,
            ProxyConnectionContext { route },
            self.source_addr,
            IngressType::LocalOpenAi,
        );
        let mut disconnect = self.disconnect;
        let capture = async {
            tokio::select! {
                outcome = crate::network::openai::response::capture_responses_events(
                    capture_reader,
                    self.events.clone(),
                ) => Some(outcome),
                _ = wait_for_disconnect(&mut disconnect) => None,
            }
        };
        let (_, outcome) = tokio::join!(dispatch, capture);
        let Some(outcome) = outcome else {
            return;
        };
        let failed = match outcome {
            Ok(crate::network::openai::response::CapturedResponsesOutcome::Completed) => false,
            Ok(crate::network::openai::response::CapturedResponsesOutcome::Failed) => true,
            Err(error) => {
                let _ = send_error(&self.events, &InferenceError::backend(error.to_string())).await;
                true
            }
        };
        if failed {
            self.session
                .evict_referenced(previous_response_id.as_deref());
        }
    }
}

async fn wait_for_disconnect(disconnect: &mut tokio::sync::watch::Receiver<bool>) {
    if *disconnect.borrow() {
        return;
    }
    let _ = disconnect.changed().await;
}

async fn settle_active(active: &mut JoinSet<()>) {
    let drain = async { while active.join_next().await.is_some() {} };
    if tokio::time::timeout(DISCONNECT_GRACE, drain).await.is_err() {
        active.shutdown().await;
    }
}

async fn accept_websocket(
    mut client: ClientStream,
    request: &crate::network::openai::transport::BufferedHttpRequest,
) -> Option<WebSocketStream<ClientStream>> {
    let key = single_header(&request.raw, "sec-websocket-key");
    let version = single_header(&request.raw, "sec-websocket-version");
    let (Some(key), Some(version)) = (key, version) else {
        let _ =
            crate::network::openai::response::send_400(client, "invalid WebSocket handshake").await;
        return None;
    };
    if version != "13" {
        let _ = crate::network::openai::response::send_400(client, "unsupported WebSocket version")
            .await;
        return None;
    }
    let Some(header_end) = request
        .raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .map(|offset| offset + 4)
    else {
        let _ =
            crate::network::openai::response::send_400(client, "invalid WebSocket handshake").await;
        return None;
    };
    let response = format!(
        "HTTP/1.1 101 Switching Protocols\r\nConnection: Upgrade\r\nUpgrade: websocket\r\nSec-WebSocket-Accept: {}\r\n\r\n",
        derive_accept_key(key.as_bytes())
    );
    if client.write_all(response.as_bytes()).await.is_err() {
        return None;
    }
    let client = client.with_read_prefix(request.raw[header_end..].to_vec());
    let max = crate::network::openai::request_parse::MAX_BODY_BYTES;
    let config = WebSocketConfig::default()
        .read_buffer_size(16 * 1024)
        .write_buffer_size(16 * 1024)
        .max_write_buffer_size(2 * 1024 * 1024)
        .max_message_size(Some(max))
        .max_frame_size(Some(max));
    Some(WebSocketStream::from_raw_socket(client, Role::Server, Some(config)).await)
}

fn try_send_error(sender: &mpsc::Sender<Value>, error: &InferenceError) -> Result<(), ()> {
    sender
        .try_send(WebSocketErrorEvent::from_error(error).into_value())
        .map_err(|_| ())
}

async fn send_error(sender: &mpsc::Sender<Value>, error: &InferenceError) -> Result<(), ()> {
    sender
        .send(WebSocketErrorEvent::from_error(error).into_value())
        .await
        .map_err(|_| ())
}

fn header_present(raw: &[u8], name: &str) -> bool {
    crate::network::openai::request_parse::header_values_from_parsed_raw(raw, name)
        .is_ok_and(|values| !values.is_empty())
}

fn header_has_token(raw: &[u8], name: &str, expected: &str) -> bool {
    crate::network::openai::request_parse::header_values_from_parsed_raw(raw, name)
        .unwrap_or_default()
        .iter()
        .flat_map(|value| value.split(','))
        .any(|value| value.trim().eq_ignore_ascii_case(expected))
}

fn single_header(raw: &[u8], name: &str) -> Option<String> {
    let values =
        crate::network::openai::request_parse::header_values_from_parsed_raw(raw, name).ok()?;
    let [value] = values.as_slice() else {
        return None;
    };
    Some(value.clone())
}

#[cfg(test)]
#[path = "responses_websocket/tests.rs"]
mod tests;
