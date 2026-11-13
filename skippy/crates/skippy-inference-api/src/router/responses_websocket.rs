//! Direct Axum adapter for Responses over WebSocket.

use std::{collections::VecDeque, net::SocketAddr, time::Duration};

use axum::{
    extract::{
        ConnectInfo, Extension, State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::{HeaderMap, StatusCode},
    response::Response,
};
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use tokio::{sync::mpsc, task::JoinSet};

use super::{FrontendState, agent_session_from_header};
use crate::{
    AgentSessionIdentity, InferenceError, InferenceErrorKind, InferenceFrontendRoute,
    InferenceLifecycleContext, InferenceRequestMethod,
    request_lifecycle::RequestLifecycle,
    responses::{
        ResponsesSession, ResponsesSessionExecutor, WebSocketErrorEvent, caller_scope_for_http_peer,
    },
};

const CONTROL_QUEUE: usize = 4;
const WRITER_SHUTDOWN_GRACE: Duration = Duration::from_secs(1);

pub(super) async fn responses_websocket(
    State(state): State<FrontendState>,
    connect_info: Option<Extension<ConnectInfo<SocketAddr>>>,
    headers: HeaderMap,
    upgrade: WebSocketUpgrade,
) -> Result<Response, InferenceError> {
    let header_session = agent_session_from_header(&state.config, &headers)?;
    let peer = connect_info.map(|Extension(ConnectInfo(addr))| addr.ip());
    let forwarded = crate::responses::FORWARDING_HEADERS
        .iter()
        .any(|name| headers.contains_key(*name));
    let caller_scope = caller_scope_for_http_peer(peer, forwarded);
    let max_message_bytes = state.config.max_request_body_bytes.max(1);
    let session_config = state.config.responses_websocket.clone();
    Ok(upgrade
        .max_message_size(max_message_bytes)
        .max_frame_size(max_message_bytes)
        .on_upgrade(move |socket| async move {
            let session = ResponsesSession::new(
                session_config,
                state.config.response_store.clone(),
                caller_scope,
            );
            let executor = ResponsesSessionExecutor::new(
                state.backend.clone(),
                state.config.backend_timeout,
                state.config.lifecycle_observer.clone(),
            );
            Connection::new(state, session, executor, header_session)
                .run(socket)
                .await;
        }))
}

struct Connection {
    state: FrontendState,
    session: ResponsesSession,
    executor: ResponsesSessionExecutor,
    header_session: Option<AgentSessionIdentity>,
}

impl Connection {
    fn new(
        state: FrontendState,
        session: ResponsesSession,
        executor: ResponsesSessionExecutor,
        header_session: Option<AgentSessionIdentity>,
    ) -> Self {
        Self {
            state,
            session,
            executor,
            header_session,
        }
    }

    async fn run(self, socket: WebSocket) {
        let max_pending = self.session.config().max_pending_turns();
        let connection_lifetime = self.session.config().connection_lifetime();
        let outbound_capacity = self.session.config().max_outbound_events();
        let deadline = tokio::time::sleep(connection_lifetime);
        tokio::pin!(deadline);
        let (mut sink, mut incoming) = socket.split();
        let (events_tx, mut events_rx) = mpsc::channel::<Message>(outbound_capacity);
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
                    message = events_rx.recv(), if events_open => match message {
                        Some(message) => message,
                        None => {
                            events_open = false;
                            continue;
                        }
                    },
                    else => break,
                };
                let close = matches!(message, Message::Close(_));
                // Reading a peer Close queues its automatic reply and marks
                // the protocol closing. Flush that reply before attempting an
                // explicit server Close (which may then be rejected as closed).
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

        let exit = loop {
            tokio::select! {
                biased;
                () = &mut deadline => {
                    break ConnectionExit::Deadline;
                }
                _ = &mut writer => {
                    break ConnectionExit::WriterStopped;
                }
                result = active.join_next(), if !active.is_empty() => {
                    let _ = result;
                    if !start_pending(
                        &mut pending,
                        &mut active,
                        &self,
                        events_tx.clone(),
                    ) {
                        break ConnectionExit::OutboundBackpressure;
                    }
                }
                message = incoming.next() => {
                    match message {
                        Some(Ok(message)) => {
                            if !self.handle_message(
                                message,
                                &mut pending,
                                &mut active,
                                max_pending,
                                &events_tx,
                                &control_tx,
                            ) {
                                break ConnectionExit::PeerClosed;
                            }
                        }
                        Some(Err(_)) | None => {
                            break ConnectionExit::PeerDisconnected;
                        }
                    }
                }
            }
        };

        active.abort_all();
        drop(active);
        drop(events_tx);
        if matches!(exit, ConnectionExit::Deadline) {
            let error = InferenceError::invalid_request(
                "Responses WebSocket connections are limited to 60 minutes",
            )
            .with_code("websocket_connection_limit_reached");
            let _ = try_send_error(&control_tx, &error);
            let _ = control_tx.try_send(Message::Close(None));
        } else if matches!(
            exit,
            ConnectionExit::PeerClosed | ConnectionExit::OutboundBackpressure
        ) {
            let _ = control_tx.try_send(Message::Close(None));
        }
        drop(control_tx);

        if !matches!(exit, ConnectionExit::WriterStopped) {
            finish_writer(&mut writer, exit.flushes_writer()).await;
        }
    }

    fn handle_message(
        &self,
        message: Message,
        pending: &mut VecDeque<Value>,
        active: &mut JoinSet<()>,
        max_pending: usize,
        events: &mpsc::Sender<Message>,
        control: &mpsc::Sender<Message>,
    ) -> bool {
        match message {
            Message::Text(text) => self.handle_text(&text, pending, active, max_pending, events),
            // Tungstenite queues and flushes the matching Pong while reading.
            // Sending a second Pong here duplicates the protocol response.
            Message::Ping(_) => true,
            Message::Pong(_) => true,
            Message::Binary(_) => {
                let error = InferenceError::invalid_request(
                    "Responses WebSocket messages must be JSON text",
                );
                try_send_error(events, &error).is_ok()
            }
            Message::Close(frame) => {
                let _ = control.try_send(Message::Close(frame));
                false
            }
        }
    }

    fn handle_text(
        &self,
        text: &str,
        pending: &mut VecDeque<Value>,
        active: &mut JoinSet<()>,
        max_pending: usize,
        events: &mpsc::Sender<Message>,
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
            return start_pending(pending, active, self, events.clone());
        }
        if pending.len() < max_pending {
            pending.push_back(value);
            return true;
        }
        self.session
            .evict_referenced(value.get("previous_response_id").and_then(Value::as_str));
        let error = InferenceError::from_kind(
            StatusCode::TOO_MANY_REQUESTS,
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

impl ConnectionExit {
    fn flushes_writer(self) -> bool {
        matches!(
            self,
            Self::Deadline | Self::PeerClosed | Self::OutboundBackpressure
        )
    }
}

fn start_pending(
    pending: &mut VecDeque<Value>,
    active: &mut JoinSet<()>,
    connection: &Connection,
    events: mpsc::Sender<Message>,
) -> bool {
    while active.is_empty() {
        let Some(value) = pending.pop_front() else {
            return true;
        };
        let context = InferenceLifecycleContext::new(
            crate::generate_request_id(),
            InferenceRequestMethod::Post,
            InferenceFrontendRoute::Responses,
        );
        let mut request_lifecycle = RequestLifecycle::admit(
            connection.state.config.lifecycle_observer.clone(),
            context.clone(),
        );
        let prepared = match connection
            .session
            .prepare_create(value, connection.header_session.clone())
        {
            Ok(prepared) => prepared,
            Err(error) => {
                request_lifecycle.finish_with_usage(error.status(), None, None);
                if try_send_error(&events, &error).is_err() {
                    return false;
                }
                continue;
            }
        };
        let executor = connection.executor.clone();
        let session = connection.session.clone();
        let events = events.clone();
        active.spawn(async move {
            run_turn(
                executor,
                session,
                prepared,
                context,
                request_lifecycle,
                events,
            )
            .await;
        });
    }
    true
}

async fn run_turn(
    executor: ResponsesSessionExecutor,
    session: ResponsesSession,
    prepared: crate::responses::PreparedResponsesTurn,
    context: InferenceLifecycleContext,
    mut request_lifecycle: RequestLifecycle,
    events: mpsc::Sender<Message>,
) {
    let previous_response_id = prepared.previous_response_id().map(ToString::to_string);
    let mut execution = match executor.start(prepared, context).await {
        Ok(execution) => execution,
        Err(error) => {
            session.evict_referenced(previous_response_id.as_deref());
            request_lifecycle.finish_with_usage(error.status(), None, None);
            let _ = send_error(&events, &error).await;
            return;
        }
    };
    request_lifecycle.transfer_to_stream();
    if send_events(&events, execution.start_events())
        .await
        .is_err()
    {
        return;
    }
    while let Some(batch) = execution.next_events().await {
        if send_events(&events, batch).await.is_err() {
            return;
        }
    }
    if execution.is_failed() {
        session.evict_referenced(previous_response_id.as_deref());
    }
}

async fn send_events(
    sender: &mpsc::Sender<Message>,
    events: Vec<crate::responses::ResponsesStreamEvent>,
) -> Result<(), ()> {
    for event in events {
        send_value(sender, event.data).await?;
    }
    Ok(())
}

async fn send_error(sender: &mpsc::Sender<Message>, error: &InferenceError) -> Result<(), ()> {
    send_value(sender, WebSocketErrorEvent::from_error(error).into_value()).await
}

fn try_send_error(sender: &mpsc::Sender<Message>, error: &InferenceError) -> Result<(), ()> {
    sender
        .try_send(Message::Text(
            WebSocketErrorEvent::from_error(error)
                .into_value()
                .to_string()
                .into(),
        ))
        .map_err(|_| ())
}

async fn send_value(sender: &mpsc::Sender<Message>, value: Value) -> Result<(), ()> {
    sender
        .send(Message::Text(value.to_string().into()))
        .await
        .map_err(|_| ())
}

async fn finish_writer(writer: &mut tokio::task::JoinHandle<()>, flush: bool) {
    if flush {
        tokio::select! {
            biased;
            _ = &mut *writer => return,
            () = tokio::time::sleep(WRITER_SHUTDOWN_GRACE) => {}
        }
    }
    writer.abort();
}

struct AbortTaskOnDrop(tokio::task::AbortHandle);

impl Drop for AbortTaskOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[cfg(test)]
#[path = "responses_websocket/tests.rs"]
mod tests;
