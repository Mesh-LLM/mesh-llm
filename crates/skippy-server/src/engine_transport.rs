//! Minimal engine-neutral server for the existing Skippy binary stage wire.
//!
//! The mature llama.cpp lane in [`crate::binary_transport`] still owns KV-page
//! caching, telemetry, batching, MTP, and OpenAI orchestration. This module is
//! intentionally the smaller compatibility seam: it proves a `StageEngine`
//! can participate in a real multi-process Skippy chain without introducing a
//! second wire protocol. Advanced operations stay capability-gated.

use std::{
    collections::HashSet,
    io::{self, Write},
    net::{Shutdown, SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

const ENGINE_STAGE_IO_TIMEOUT: Duration = Duration::from_secs(5 * 60);

use anyhow::{Context, Result, bail, ensure};
use skippy_engine::{
    StageActivation, StageEngine, StageExecutionKind, StageExecutionOutput, StageExecutionRequest,
};
use skippy_protocol::binary::{
    StageReply, StageWireMessage, WireActivationDType, WireMessageKind, WireReplyKind,
    encode_f32_activation_payload, read_stage_message, recv_ready, recv_reply, send_ready,
    send_reply_ack_with_stats, send_reply_predicted_tokens_with_stats,
    send_reply_predicted_with_tokens_and_stats, write_stage_message,
};

#[derive(Clone, Debug)]
pub struct EngineStageServerOptions {
    pub bind_addr: SocketAddr,
    pub downstream_addr: Option<SocketAddr>,
    pub wire_dtype: WireActivationDType,
}

pub fn serve_stage_engine(
    engine: Arc<dyn StageEngine>,
    options: EngineStageServerOptions,
) -> Result<()> {
    serve_stage_engine_until(engine, options, Arc::new(AtomicBool::new(false)))
}

pub fn serve_stage_engine_until(
    engine: Arc<dyn StageEngine>,
    options: EngineStageServerOptions,
    shutdown: Arc<AtomicBool>,
) -> Result<()> {
    let listener = prepare_stage_engine_listener(engine.as_ref(), &options)?;
    serve_prepared_stage_engine_until(engine, options, listener, shutdown)
}

pub(crate) fn prepare_stage_engine_listener(
    engine: &dyn StageEngine,
    options: &EngineStageServerOptions,
) -> Result<TcpListener> {
    engine.info().validate()?;
    validate_topology(engine, options)?;
    let listener = TcpListener::bind(options.bind_addr)
        .with_context(|| format!("bind engine stage at {}", options.bind_addr))?;
    listener.set_nonblocking(true)?;
    if let Some(downstream_addr) = options.downstream_addr {
        drop(
            connect_downstream(downstream_addr).with_context(|| {
                format!("preflight downstream engine stage at {downstream_addr}")
            })?,
        );
    }
    eprintln!(
        "skippy engine stage listening: engine={} model={} binary={} layers={}..{} width={} dtype={:?}",
        engine.info().engine,
        engine.info().model_id,
        listener.local_addr()?,
        engine.info().layer_start,
        engine.info().layer_end,
        engine.info().activation_width,
        options.wire_dtype,
    );
    Ok(listener)
}

pub(crate) fn serve_prepared_stage_engine_until(
    engine: Arc<dyn StageEngine>,
    options: EngineStageServerOptions,
    listener: TcpListener,
    shutdown: Arc<AtomicBool>,
) -> Result<()> {
    let mut connections = Vec::new();
    let result = (|| {
        while !shutdown.load(Ordering::SeqCst) {
            reap_finished_connections(&mut connections);
            let (upstream, peer_addr) = match listener.accept() {
                Ok(connection) => connection,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(25));
                    continue;
                }
                Err(error) => return Err(error).context("accept engine stage connection"),
            };
            upstream.set_nonblocking(false)?;
            upstream.set_nodelay(true).ok();
            let control = upstream.try_clone()?;
            let downstream_control = Arc::new(Mutex::new(None));
            let engine = engine.clone();
            let options = options.clone();
            let connection_downstream = Arc::clone(&downstream_control);
            let connection = thread::spawn(move || {
                if let Err(error) =
                    handle_connection(engine, options, upstream, connection_downstream)
                {
                    eprintln!("engine stage connection from {peer_addr} failed: {error:#}");
                }
            });
            connections.push(ActiveConnection {
                control,
                downstream_control,
                thread: connection,
            });
        }
        Ok(())
    })();
    stop_active_connections(connections);
    result
}

struct ActiveConnection {
    control: TcpStream,
    downstream_control: Arc<Mutex<Option<TcpStream>>>,
    thread: thread::JoinHandle<()>,
}

fn reap_finished_connections(connections: &mut Vec<ActiveConnection>) {
    let mut index = 0;
    while index < connections.len() {
        if connections[index].thread.is_finished() {
            let connection = connections.swap_remove(index);
            let _ = connection.thread.join();
        } else {
            index += 1;
        }
    }
}

fn stop_active_connections(connections: Vec<ActiveConnection>) {
    for connection in &connections {
        let _ = connection.control.shutdown(Shutdown::Both);
        if let Ok(downstream) = connection.downstream_control.lock()
            && let Some(downstream) = downstream.as_ref()
        {
            let _ = downstream.shutdown(Shutdown::Both);
        }
    }
    for connection in connections {
        let _ = connection.thread.join();
    }
}

fn validate_topology(engine: &dyn StageEngine, options: &EngineStageServerOptions) -> Result<()> {
    ensure!(
        engine.info().is_final() == options.downstream_addr.is_none(),
        "only the final stage may omit a downstream address"
    );
    Ok(())
}

fn handle_connection(
    engine: Arc<dyn StageEngine>,
    options: EngineStageServerOptions,
    mut upstream: TcpStream,
    downstream_control: Arc<Mutex<Option<TcpStream>>>,
) -> Result<()> {
    upstream.set_read_timeout(Some(ENGINE_STAGE_IO_TIMEOUT))?;
    upstream.set_write_timeout(Some(ENGINE_STAGE_IO_TIMEOUT))?;
    let mut downstream = options
        .downstream_addr
        .map(connect_downstream)
        .transpose()?;
    if let Some(stream) = downstream.as_ref() {
        *downstream_control
            .lock()
            .expect("downstream control lock poisoned") = Some(stream.try_clone()?);
    }
    send_ready(&mut upstream).context("send engine stage ready")?;
    upstream.flush().ok();
    let activation_width =
        i32::try_from(engine.info().activation_width).context("activation width exceeds i32")?;
    let mut active_sessions = HashSet::new();
    let result = handle_connection_messages(
        engine.as_ref(),
        &options,
        &mut upstream,
        downstream.as_mut(),
        activation_width,
        &mut active_sessions,
    );
    let cleanup = cleanup_connection_sessions(
        engine.as_ref(),
        downstream.as_mut(),
        options.wire_dtype,
        &active_sessions,
    );
    match (result, cleanup) {
        (Err(error), Err(cleanup_error)) => {
            eprintln!("engine stage session cleanup also failed: {cleanup_error:#}");
            Err(error)
        }
        (Err(error), _) => Err(error),
        (Ok(()), cleanup) => cleanup,
    }
}

fn handle_connection_messages(
    engine: &dyn StageEngine,
    options: &EngineStageServerOptions,
    upstream: &mut TcpStream,
    mut downstream: Option<&mut TcpStream>,
    activation_width: i32,
    active_sessions: &mut HashSet<u64>,
) -> Result<()> {
    loop {
        let message = match read_stage_message(&mut *upstream, activation_width) {
            Ok(message) => message,
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(error) => return Err(error).context("read engine stage message"),
        };
        if message.kind == WireMessageKind::Stop {
            engine.reset_session(message.session_id)?;
            let downstream_reply =
                forward_control(downstream.as_deref_mut(), &message, options.wire_dtype)?;
            send_ack(upstream, downstream_reply)?;
            active_sessions.remove(&message.session_id);
            continue;
        }
        if message.kind.is_session_control() {
            active_sessions.insert(message.session_id);
            execute_session_control(engine, &message)?;
            let downstream_reply =
                forward_control(downstream.as_deref_mut(), &message, options.wire_dtype)?;
            send_ack(upstream, downstream_reply)?;
            continue;
        }

        active_sessions.insert(message.session_id);
        let request = execution_request(&message, activation_width)?;
        let output = engine.execute(request)?;
        match downstream.as_deref_mut() {
            Some(downstream) => {
                let forwarded = forwarded_message(engine, &message, output, options.wire_dtype)?;
                write_stage_message(&mut *downstream, &forwarded, options.wire_dtype)
                    .context("forward engine stage message")?;
                downstream.flush().ok();
                let reply = recv_reply(&mut *downstream).context("receive downstream reply")?;
                send_reply(upstream, reply)?;
            }
            None => send_final_reply(upstream, &message, output)?,
        }
    }
}

fn cleanup_connection_sessions(
    engine: &dyn StageEngine,
    mut downstream: Option<&mut TcpStream>,
    wire_dtype: WireActivationDType,
    active_sessions: &HashSet<u64>,
) -> Result<()> {
    let mut first_error = None;
    for session_id in active_sessions {
        if let Err(error) = engine.reset_session(*session_id)
            && first_error.is_none()
        {
            first_error = Some(error.context(format!("reset abandoned session {session_id}")));
        }
        let stop = StageWireMessage::stop_with_identity(wire_dtype, *session_id, *session_id);
        let downstream_result = forward_control(downstream.as_deref_mut(), &stop, wire_dtype)
            .and_then(|reply| {
                ensure!(
                    reply.is_none_or(|reply| reply.kind == WireReplyKind::Ack),
                    "abandoned session stop expected downstream ACK"
                );
                Ok(())
            });
        if let Err(error) = downstream_result
            && first_error.is_none()
        {
            first_error =
                Some(error.context(format!("propagate abandoned session {session_id} stop")));
        }
    }
    first_error.map_or(Ok(()), Err)
}

fn connect_downstream(addr: SocketAddr) -> Result<TcpStream> {
    const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
    let mut stream = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)
        .with_context(|| format!("connect downstream engine stage at {addr}"))?;
    stream.set_nodelay(true).ok();
    stream.set_read_timeout(Some(CONNECT_TIMEOUT))?;
    stream.set_write_timeout(Some(CONNECT_TIMEOUT))?;
    recv_ready(&mut stream).context("downstream engine stage did not become ready")?;
    stream.set_read_timeout(Some(ENGINE_STAGE_IO_TIMEOUT))?;
    stream.set_write_timeout(Some(ENGINE_STAGE_IO_TIMEOUT))?;
    Ok(stream)
}

fn execute_session_control(engine: &dyn StageEngine, message: &StageWireMessage) -> Result<()> {
    match message.kind {
        WireMessageKind::CheckpointSession => engine.checkpoint_session(message.session_id),
        WireMessageKind::RestoreSession => engine.restore_session(message.session_id),
        WireMessageKind::TrimSession => {
            engine.trim_session(message.session_id, message.token_count.max(0) as u64)
        }
        _ => bail!("message is not session control"),
    }
}

fn forward_control(
    downstream: Option<&mut TcpStream>,
    message: &StageWireMessage,
    wire_dtype: WireActivationDType,
) -> Result<Option<StageReply>> {
    let Some(downstream) = downstream else {
        return Ok(None);
    };
    write_stage_message(&mut *downstream, message, wire_dtype)?;
    downstream.flush().ok();
    Ok(Some(recv_reply(&mut *downstream)?))
}

fn execution_request(
    message: &StageWireMessage,
    activation_width: i32,
) -> Result<StageExecutionRequest> {
    let kind = match message.kind {
        WireMessageKind::PrefillEmbd => StageExecutionKind::Prefill,
        WireMessageKind::PrefillFinalEmbd => StageExecutionKind::PrefillFinal,
        WireMessageKind::DecodeEmbd
        | WireMessageKind::DecodeReadout
        | WireMessageKind::DecodeLightCtx
        | WireMessageKind::DecodeReplayEmbd
        | WireMessageKind::DecodeReplayFinalEmbd => StageExecutionKind::Decode,
        WireMessageKind::VerifySpan => StageExecutionKind::Verify,
        other => bail!("engine stage does not execute {other:?}"),
    };
    let token_count = usize::try_from(message.token_count).context("negative token count")?;
    let token_ids = execution_tokens(message, kind, token_count)?;
    let input = if message.activation.is_empty() {
        None
    } else {
        let bytes = message
            .activation_f32_payload(activation_width)
            .context("decode input activation")?;
        Some(StageActivation::new(
            token_count,
            usize::try_from(activation_width)?,
            bytes,
        )?)
    };
    Ok(StageExecutionRequest {
        session_id: message.session_id,
        kind,
        token_ids,
        positions: message.positions.clone(),
        input,
        sampling: message.sampling.clone(),
    })
}

fn execution_tokens(
    message: &StageWireMessage,
    kind: StageExecutionKind,
    token_count: usize,
) -> Result<Vec<i32>> {
    if kind == StageExecutionKind::Decode {
        ensure!(token_count == 1, "decode requires one token");
        return Ok(vec![message.state.current_token]);
    }
    ensure!(
        message.tokens.len() == token_count,
        "token sideband length does not match token count"
    );
    Ok(message.tokens.clone())
}

fn forwarded_message(
    engine: &dyn StageEngine,
    incoming: &StageWireMessage,
    output: StageExecutionOutput,
    wire_dtype: WireActivationDType,
) -> Result<StageWireMessage> {
    let activation = output
        .activation
        .context("non-final engine stage returned no activation")?;
    ensure!(
        activation.width == engine.info().activation_width as usize,
        "engine output activation width mismatch"
    );
    let mut state = incoming.state;
    state.source_stage_index = i32::try_from(engine.info().stage_index)?;
    state.reserved = wire_dtype as i32;
    let activation = encode_f32_activation_payload(
        wire_dtype,
        incoming.token_count,
        i32::try_from(activation.width)?,
        &activation.f32_le_bytes,
    )?;
    Ok(StageWireMessage {
        kind: incoming.kind,
        pos_start: incoming.pos_start,
        token_count: incoming.token_count,
        state,
        request_id: incoming.request_id,
        session_id: incoming.session_id,
        sampling: incoming.sampling.clone(),
        chat_sampling_metadata: incoming.chat_sampling_metadata.clone(),
        tokens: incoming.tokens.clone(),
        positions: incoming.positions.clone(),
        activation,
        raw_bytes: Vec::new(),
    })
}

fn send_final_reply(
    upstream: &mut TcpStream,
    message: &StageWireMessage,
    output: StageExecutionOutput,
) -> Result<()> {
    if message.kind.requires_predicted_reply() {
        ensure!(
            !output.predicted_tokens.is_empty(),
            "final engine stage returned no prediction"
        );
        send_reply_predicted_with_tokens_and_stats(
            &mut *upstream,
            output.predicted().expect("checked non-empty"),
            &output.predicted_tokens,
            Default::default(),
        )?;
    } else {
        send_reply_ack_with_stats(&mut *upstream, Default::default())?;
    }
    upstream.flush().ok();
    Ok(())
}

fn send_ack(upstream: &mut TcpStream, downstream: Option<StageReply>) -> Result<()> {
    if let Some(reply) = downstream {
        ensure!(
            reply.kind == WireReplyKind::Ack,
            "control expected downstream ACK"
        );
        send_reply_ack_with_stats(&mut *upstream, reply.stats)?;
    } else {
        send_reply_ack_with_stats(&mut *upstream, Default::default())?;
    }
    upstream.flush().ok();
    Ok(())
}

fn send_reply(upstream: &mut TcpStream, reply: StageReply) -> Result<()> {
    match reply.kind {
        WireReplyKind::Ack => send_reply_ack_with_stats(&mut *upstream, reply.stats)?,
        WireReplyKind::PredictedToken => send_reply_predicted_with_tokens_and_stats(
            &mut *upstream,
            reply.predicted,
            &reply.predicted_tokens,
            reply.stats,
        )?,
        WireReplyKind::PredictedTokens => send_reply_predicted_tokens_with_stats(
            &mut *upstream,
            &reply.predicted_tokens,
            reply.stats,
        )?,
    }
    upstream.flush().ok();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::binary::StageStateHeader;

    struct RecordingEngine {
        info: skippy_engine::StageEngineInfo,
        resets: Mutex<Vec<u64>>,
    }

    impl RecordingEngine {
        fn new() -> Self {
            Self {
                info: skippy_engine::StageEngineInfo {
                    engine: "test".to_string(),
                    model_id: "test/model".to_string(),
                    stage_index: 0,
                    layer_start: 0,
                    layer_end: 1,
                    total_layers: 1,
                    activation_width: 4,
                },
                resets: Mutex::new(Vec::new()),
            }
        }
    }

    impl StageEngine for RecordingEngine {
        fn info(&self) -> &skippy_engine::StageEngineInfo {
            &self.info
        }

        fn execute(&self, _request: StageExecutionRequest) -> Result<StageExecutionOutput> {
            Ok(StageExecutionOutput::default())
        }

        fn reset_session(&self, session_id: u64) -> Result<()> {
            self.resets.lock().unwrap().push(session_id);
            Ok(())
        }
    }

    fn decode_message(tokens: Vec<i32>, current_token: i32) -> StageWireMessage {
        let kind = WireMessageKind::DecodeEmbd;
        let mut state = StageStateHeader::new(kind, WireActivationDType::F16);
        state.current_token = current_token;
        StageWireMessage {
            kind,
            pos_start: 0,
            token_count: 1,
            state,
            request_id: 1,
            session_id: 2,
            sampling: None,
            chat_sampling_metadata: None,
            tokens,
            positions: Vec::new(),
            activation: Vec::new(),
            raw_bytes: Vec::new(),
        }
    }

    #[test]
    fn decode_uses_current_token_not_prompt_sideband() {
        let request = execution_request(&decode_message(vec![1, 2, 3], 7), 4).unwrap();
        assert_eq!(request.token_ids, vec![7]);
        assert_eq!(request.kind, StageExecutionKind::Decode);
    }

    #[test]
    fn connection_cleanup_resets_every_abandoned_session() {
        let engine = RecordingEngine::new();
        let sessions = HashSet::from([7, 9]);

        cleanup_connection_sessions(&engine, None, WireActivationDType::F16, &sessions).unwrap();

        let mut resets = engine.resets.lock().unwrap().clone();
        resets.sort_unstable();
        assert_eq!(resets, vec![7, 9]);
    }
}
