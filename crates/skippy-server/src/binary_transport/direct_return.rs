use std::{
    collections::HashMap,
    io,
    net::{SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use skippy_protocol::{
    StageConfig, StageTopology,
    binary::{
        StageReply, StageStateHeader, StageWireMessage, WireActivationDType, WireMessageKind,
        WireReplyKind, read_stage_message, recv_ready, recv_reply, send_ready,
        send_reply_ack_with_stats, send_reply_predicted_tokens_with_stats,
        send_reply_predicted_with_stats, write_stage_message,
    },
};

use super::socket::{connect_downstream_socket, downstream_source_ip, resolve_downstream_endpoint};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) struct PredictionReturnKey {
    request_id: u64,
    session_id: u64,
}

impl PredictionReturnKey {
    pub(crate) fn new(request_id: u64, session_id: u64) -> Self {
        Self {
            request_id,
            session_id,
        }
    }
}

pub struct PredictionReturnHub {
    waiters: Mutex<HashMap<PredictionReturnKey, mpsc::Sender<Result<StageReply, String>>>>,
}

impl Default for PredictionReturnHub {
    fn default() -> Self {
        Self {
            waiters: Mutex::new(HashMap::new()),
        }
    }
}

pub struct PredictionReturnListener {
    shutdown: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    hub: Arc<PredictionReturnHub>,
}

impl PredictionReturnListener {
    pub fn start(bind_addr: SocketAddr) -> Result<Self> {
        let listener = TcpListener::bind(bind_addr)
            .with_context(|| format!("bind direct prediction return listener {bind_addr}"))?;
        listener
            .set_nonblocking(true)
            .context("set direct prediction return listener nonblocking")?;
        let shutdown = Arc::new(AtomicBool::new(false));
        let thread_shutdown = shutdown.clone();
        let hub = Arc::new(PredictionReturnHub::default());
        let thread_hub = hub.clone();
        let thread = thread::spawn(move || {
            while !thread_shutdown.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        if let Err(error) = stream.set_nonblocking(false) {
                            eprintln!(
                                "direct prediction return connection failed: set blocking: {error}"
                            );
                            continue;
                        }
                        let hub = thread_hub.clone();
                        thread::spawn(move || {
                            if let Err(error) = handle_prediction_return_connection(hub, stream) {
                                eprintln!("direct prediction return connection failed: {error:#}");
                            }
                        });
                    }
                    Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(50));
                    }
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
                    Err(error) => {
                        eprintln!("direct prediction return listener failed: {error}");
                        break;
                    }
                }
            }
        });
        Ok(Self {
            shutdown,
            thread: Some(thread),
            hub,
        })
    }

    pub fn hub(&self) -> Arc<PredictionReturnHub> {
        self.hub.clone()
    }
}

impl Drop for PredictionReturnListener {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn handle_prediction_return_connection(
    hub: Arc<PredictionReturnHub>,
    mut stream: TcpStream,
) -> Result<()> {
    send_ready(&mut stream).context("send direct prediction return ready")?;
    let open = read_stage_message(&mut stream, 0).context("read direct prediction return open")?;
    hub.handle_return_connection(open, stream)
}

impl PredictionReturnHub {
    pub(crate) fn register(
        self: &Arc<Self>,
        request_id: u64,
        session_id: u64,
    ) -> Result<PredictionReturnReceiver> {
        let key = PredictionReturnKey::new(request_id, session_id);
        let (sender, receiver) = mpsc::channel();
        self.waiters
            .lock()
            .map_err(|_| anyhow!("prediction return hub lock poisoned"))?
            .insert(key, sender);
        Ok(PredictionReturnReceiver {
            key,
            hub: self.clone(),
            receiver,
            timeout: Duration::from_secs(300),
        })
    }

    pub(crate) fn unregister(&self, key: PredictionReturnKey) {
        if let Ok(mut waiters) = self.waiters.lock() {
            waiters.remove(&key);
        }
    }

    pub(crate) fn handle_return_connection(
        &self,
        open: StageWireMessage,
        mut stream: TcpStream,
    ) -> Result<()> {
        if open.kind != WireMessageKind::PredictionReturnOpen {
            bail!("expected prediction return open message");
        }
        let key = PredictionReturnKey::new(open.request_id, open.session_id);
        let sender = self
            .waiters
            .lock()
            .map_err(|_| anyhow!("prediction return hub lock poisoned"))?
            .get(&key)
            .cloned()
            .ok_or_else(|| anyhow!("no prediction return waiter for request {}", key.request_id))?;
        loop {
            match recv_reply(&mut stream) {
                Ok(reply) => {
                    if sender.send(Ok(reply)).is_err() {
                        return Ok(());
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
                Err(error) => {
                    let _ = sender.send(Err(error.to_string()));
                    return Err(error).context("read direct prediction return");
                }
            }
        }
    }
}

pub(crate) struct PredictionReturnReceiver {
    key: PredictionReturnKey,
    hub: Arc<PredictionReturnHub>,
    receiver: mpsc::Receiver<Result<StageReply, String>>,
    timeout: Duration,
}

impl PredictionReturnReceiver {
    pub(crate) fn recv_expected(&self, expected: WireReplyKind) -> Result<StageReply> {
        let reply = self.recv()?;
        if reply.kind != expected {
            bail!(
                "expected {expected:?} direct prediction return, got {:?}",
                reply.kind
            );
        }
        Ok(reply)
    }

    pub(crate) fn recv(&self) -> Result<StageReply> {
        let reply = self
            .receiver
            .recv_timeout(self.timeout)
            .context("timed out waiting for direct prediction return")?
            .map_err(|error| anyhow!(error))?;
        Ok(reply)
    }
}

impl Drop for PredictionReturnReceiver {
    fn drop(&mut self) {
        self.hub.unregister(self.key);
    }
}

pub(crate) fn open_prediction_return_stream(
    config: &StageConfig,
    topology: Option<&StageTopology>,
    request_id: u64,
    session_id: u64,
    wire_dtype: WireActivationDType,
    timeout_secs: u64,
) -> Result<TcpStream> {
    let endpoint = driver_stage_endpoint(config, topology)?;
    let return_addr = resolve_downstream_endpoint(endpoint)?;
    let source_ip = downstream_source_ip(config)?;
    let attempts = timeout_secs.saturating_mul(2).max(1);
    let mut last_error = None;
    for _ in 0..attempts {
        match connect_downstream_socket(return_addr, source_ip, Duration::from_secs(2)) {
            Ok(mut stream) => {
                stream.set_nodelay(true).ok();
                recv_ready(&mut stream).context("prediction return sink did not become ready")?;
                write_stage_message(
                    &mut stream,
                    &prediction_return_open_message(request_id, session_id),
                    wire_dtype,
                )
                .context("open direct prediction return stream")?;
                return Ok(stream);
            }
            Err(error) => {
                last_error = Some(anyhow!(error));
                std::thread::sleep(Duration::from_millis(500));
            }
        }
    }
    Err(last_error
        .unwrap_or_else(|| anyhow!("timed out"))
        .context(format!(
            "connect direct prediction return sink at {endpoint}"
        )))
}

pub(crate) fn send_direct_prediction_return(
    stream: &mut TcpStream,
    reply: StageReply,
) -> Result<()> {
    match reply.kind {
        WireReplyKind::PredictedToken => {
            send_reply_predicted_with_stats(stream, reply.predicted, reply.stats)
                .context("send direct predicted-token return")
        }
        WireReplyKind::PredictedTokens => {
            send_reply_predicted_tokens_with_stats(stream, &reply.predicted_tokens, reply.stats)
                .context("send direct predicted-tokens return")
        }
        WireReplyKind::Ack => {
            send_reply_ack_with_stats(stream, reply.stats).context("send direct ACK return")
        }
    }
}

fn driver_stage_endpoint<'a>(
    config: &'a StageConfig,
    topology: Option<&'a StageTopology>,
) -> Result<&'a str> {
    if let Some(topology) = topology {
        return driver_stage_endpoint_from_topology(topology);
    }
    if let Some(upstream) = config
        .upstream
        .as_ref()
        .filter(|upstream| upstream.stage_index == 0)
    {
        return Ok(strip_tcp_prefix(&upstream.endpoint));
    }
    Err(anyhow!("direct prediction return requires topology"))
}

fn driver_stage_endpoint_from_topology(topology: &StageTopology) -> Result<&str> {
    topology
        .stages
        .iter()
        .find(|stage| stage.stage_index == 0)
        .map(|stage| strip_tcp_prefix(&stage.endpoint))
        .ok_or_else(|| anyhow!("topology does not contain driver-facing stage 0"))
}

fn strip_tcp_prefix(endpoint: &str) -> &str {
    endpoint.strip_prefix("tcp://").unwrap_or(endpoint)
}

fn prediction_return_open_message(request_id: u64, session_id: u64) -> StageWireMessage {
    StageWireMessage {
        kind: WireMessageKind::PredictionReturnOpen,
        pos_start: 0,
        token_count: 0,
        state: StageStateHeader::new(
            WireMessageKind::PredictionReturnOpen,
            WireActivationDType::F32,
        ),
        request_id,
        session_id,
        sampling: None,
        chat_sampling_metadata: None,
        tokens: Vec::new(),
        positions: Vec::new(),
        activation: Vec::new(),
        raw_bytes: Vec::new(),
    }
}
