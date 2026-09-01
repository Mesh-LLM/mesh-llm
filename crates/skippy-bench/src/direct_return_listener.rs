//! Closes the direct prediction return ring for the distributed bench driver.
//!
//! The deployment plan rewrites stage 0's topology endpoint to the driver
//! return endpoint, so the final stage forwards each prediction "downstream"
//! straight back to the driver. Nothing listened on that endpoint before this
//! module existed: the final stage's connect was refused, its upstream
//! fallback never reached the driver, and every distributed run deadlocked on
//! decode step 0 waiting for a reply.

use std::{
    collections::HashMap,
    io::ErrorKind,
    net::{IpAddr, SocketAddr, TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use skippy_protocol::binary::{
    READY_MAGIC, StageReply, WireMessageKind, read_stage_message, recv_ready, recv_reply,
    send_ready,
};

/// Upper bound on concurrently served return connections. Each distributed
/// run drives one prompt at a time, so legitimate traffic is a handful of
/// connections; the cap only exists so idle or hostile peers cannot pile up
/// blocking handler threads.
const MAX_RETURN_CONNECTIONS: usize = 16;

/// A peer must finish the ready/open handshake within this deadline.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// How long the optional client ready hello has to appear, and to arrive in
/// full once its first bytes have. Bounded so the other dialect — the stage
/// waits for our ready before sending anything — still moves on promptly.
const HELLO_PEEK_BUDGET: Duration = Duration::from_millis(250);

/// Pause between peeks while a partial hello completes.
const HELLO_PEEK_POLL: Duration = Duration::from_millis(1);

/// Framed replies must keep arriving within this deadline once the stream is
/// open. Matches the driver-side reply wait so a wedged stage releases the
/// handler thread instead of pinning it forever.
const REPLY_READ_TIMEOUT: Duration = Duration::from_secs(180);

type ReplyResult = std::result::Result<StageReply, String>;

#[derive(Default)]
struct Waiters {
    map: Mutex<HashMap<(u64, u64), mpsc::Sender<ReplyResult>>>,
}

pub(crate) struct DriverReturnListener {
    shutdown: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    waiters: Arc<Waiters>,
    #[cfg_attr(not(test), allow(dead_code))]
    local_addr: SocketAddr,
}

pub(crate) struct DriverReturnReceiver {
    key: (u64, u64),
    waiters: Arc<Waiters>,
    receiver: mpsc::Receiver<ReplyResult>,
}

struct ConnectionSlot {
    active: Arc<AtomicUsize>,
}

impl ConnectionSlot {
    fn acquire(active: &Arc<AtomicUsize>) -> Option<Self> {
        let mut current = active.load(Ordering::SeqCst);
        loop {
            if current >= MAX_RETURN_CONNECTIONS {
                return None;
            }
            match active.compare_exchange(current, current + 1, Ordering::SeqCst, Ordering::SeqCst)
            {
                Ok(_) => {
                    return Some(Self {
                        active: active.clone(),
                    });
                }
                Err(observed) => current = observed,
            }
        }
    }
}

impl Drop for ConnectionSlot {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::SeqCst);
    }
}

impl DriverReturnListener {
    /// Bind the return listener. `allowed_sources` is the set of peer
    /// addresses permitted to deliver replies (loopback is always allowed so
    /// same-host stages keep working); connections from any other source are
    /// dropped before the handshake.
    pub(crate) fn start(bind_addr: SocketAddr, allowed_sources: Vec<IpAddr>) -> Result<Self> {
        let listener = TcpListener::bind(bind_addr)
            .with_context(|| format!("bind driver prediction return listener {bind_addr}"))?;
        let local_addr = listener
            .local_addr()
            .context("read driver prediction return listener local addr")?;
        listener
            .set_nonblocking(true)
            .context("set driver prediction return listener nonblocking")?;
        let shutdown = Arc::new(AtomicBool::new(false));
        let waiters = Arc::new(Waiters::default());
        let active_connections = Arc::new(AtomicUsize::new(0));
        let thread_shutdown = shutdown.clone();
        let thread_waiters = waiters.clone();
        let thread = thread::spawn(move || {
            while !thread_shutdown.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, peer)) => {
                        if !source_allowed(peer.ip(), &allowed_sources) {
                            eprintln!(
                                "driver prediction return refused connection from unexpected \
                                 source {peer}"
                            );
                            continue;
                        }
                        let Some(slot) = ConnectionSlot::acquire(&active_connections) else {
                            eprintln!(
                                "driver prediction return refused connection from {peer}: \
                                 connection limit ({MAX_RETURN_CONNECTIONS}) reached"
                            );
                            continue;
                        };
                        // Accepted sockets inherit O_NONBLOCK from the listener
                        // on BSD/macOS (Linux clears it); restore blocking mode
                        // so the framed reads below don't fail with EAGAIN.
                        if let Err(error) = stream.set_nonblocking(false) {
                            eprintln!(
                                "driver prediction return accept failed to restore blocking mode: {error}"
                            );
                            continue;
                        }
                        let waiters = thread_waiters.clone();
                        thread::spawn(move || {
                            let _slot = slot;
                            if let Err(error) = handle_return_connection(&waiters, stream) {
                                eprintln!("driver prediction return connection failed: {error:#}");
                            }
                        });
                    }
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(50));
                    }
                    Err(error) if error.kind() == ErrorKind::Interrupted => {}
                    Err(error) => {
                        eprintln!("driver prediction return listener failed: {error}");
                        break;
                    }
                }
            }
        });
        Ok(Self {
            shutdown,
            thread: Some(thread),
            waiters,
            local_addr,
        })
    }

    #[cfg(test)]
    pub(crate) fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub(crate) fn register(
        &self,
        request_id: u64,
        session_id: u64,
    ) -> Result<DriverReturnReceiver> {
        let key = (request_id, session_id);
        let (sender, receiver) = mpsc::channel();
        self.waiters
            .map
            .lock()
            .map_err(|_| anyhow!("driver prediction return waiters lock poisoned"))?
            .insert(key, sender);
        Ok(DriverReturnReceiver {
            key,
            waiters: self.waiters.clone(),
            receiver,
        })
    }
}

fn source_allowed(source: IpAddr, allowed_sources: &[IpAddr]) -> bool {
    source.is_loopback() || allowed_sources.contains(&source)
}

impl Drop for DriverReturnListener {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for DriverReturnReceiver {
    fn drop(&mut self) {
        if let Ok(mut map) = self.waiters.map.lock() {
            map.remove(&self.key);
        }
    }
}

impl DriverReturnReceiver {
    /// Wait for the next direct-return reply. `Ok(None)` means nothing arrived
    /// within the timeout; the caller decides whether to fall back to the
    /// upstream reply path.
    pub(crate) fn recv_timeout(&self, timeout: Duration) -> Result<Option<StageReply>> {
        match self.receiver.recv_timeout(timeout) {
            Ok(Ok(reply)) => Ok(Some(reply)),
            Ok(Err(error)) => bail!("direct prediction return failed: {error}"),
            Err(mpsc::RecvTimeoutError::Timeout) => Ok(None),
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                bail!("direct prediction return listener stopped")
            }
        }
    }
}

fn handle_return_connection(waiters: &Waiters, mut stream: TcpStream) -> Result<()> {
    stream
        .set_read_timeout(Some(HANDSHAKE_TIMEOUT))
        .context("set driver prediction return handshake deadline")?;
    consume_optional_ready_hello(&mut stream)?;
    send_ready(&mut stream).context("send driver prediction return ready")?;
    let open = read_stage_message(&mut stream, 0).context("read driver prediction return open")?;
    if open.kind != WireMessageKind::PredictionReturnOpen {
        bail!(
            "expected prediction return open message, got {:?}",
            open.kind
        );
    }
    let sender = waiters
        .map
        .lock()
        .map_err(|_| anyhow!("driver prediction return waiters lock poisoned"))?
        .get(&(open.request_id, open.session_id))
        .cloned()
        .ok_or_else(|| {
            anyhow!(
                "no driver prediction return waiter for request {} session {}",
                open.request_id,
                open.session_id
            )
        })?;
    stream
        .set_read_timeout(Some(REPLY_READ_TIMEOUT))
        .context("set driver prediction return reply deadline")?;
    loop {
        match recv_reply(&mut stream) {
            Ok(reply) => {
                if sender.send(Ok(reply)).is_err() {
                    return Ok(());
                }
            }
            Err(error) if error.kind() == ErrorKind::UnexpectedEof => {
                let _ = sender.send(Err(format!(
                    "direct prediction return closed before the next reply: {error}"
                )));
                return Ok(());
            }
            Err(error) => {
                let _ = sender.send(Err(error.to_string()));
                return Err(error).context("read driver prediction return reply");
            }
        }
    }
}

/// The connecting stage may open with a client ready hello before it waits for
/// our ready. Mirror the stage server's optional peek so both dialects work.
fn consume_optional_ready_hello(stream: &mut TcpStream) -> Result<()> {
    let previous = stream
        .read_timeout()
        .context("read driver prediction return stream timeout")?;
    stream
        .set_read_timeout(Some(HELLO_PEEK_BUDGET))
        .context("set driver prediction return hello peek timeout")?;
    let peeked = peek_ready_hello(stream);
    stream
        .set_read_timeout(previous)
        .context("restore driver prediction return stream timeout")?;
    if peeked.context("peek driver prediction return hello")? {
        recv_ready(&mut *stream).context("consume driver prediction return client hello")?;
    }
    Ok(())
}

/// Reports whether the peer opened with a client ready hello, leaving the
/// stream unread either way. A short peek is not proof there is no hello: TCP
/// may split the four magic bytes, so keep peeking while what has arrived is
/// still a `READY_MAGIC` prefix. Anything else — a full non-matching word, a
/// byte that diverges from the magic, EOF, or the budget running out — means
/// no hello, and the caller reads the bytes as a stage message instead.
fn peek_ready_hello(stream: &mut TcpStream) -> std::io::Result<bool> {
    let magic = READY_MAGIC.to_le_bytes();
    let deadline = Instant::now() + HELLO_PEEK_BUDGET;
    loop {
        let mut bytes = [0_u8; 4];
        match stream.peek(&mut bytes) {
            Ok(4) => return Ok(bytes == magic),
            // Only a prefix so far; the rest may still be in flight.
            Ok(peeked) if peeked > 0 && bytes[..peeked] == magic[..peeked] => {}
            Ok(_) => return Ok(false),
            Err(error) if error.kind() == ErrorKind::Interrupted => {}
            Err(error) if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                return Ok(false);
            }
            Err(error) => return Err(error),
        }
        // The buffered prefix makes the next peek return immediately, so pause
        // rather than spin while the remaining bytes arrive.
        if Instant::now() >= deadline {
            return Ok(false);
        }
        thread::sleep(HELLO_PEEK_POLL);
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use skippy_protocol::binary::{
        StageStateHeader, StageWireMessage, send_reply_predicted_with_stats, write_stage_message,
    };

    fn open_message(request_id: u64, session_id: u64) -> StageWireMessage {
        StageWireMessage {
            kind: WireMessageKind::PredictionReturnOpen,
            pos_start: 0,
            token_count: 0,
            state: StageStateHeader::new(WireMessageKind::PredictionReturnOpen),
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

    fn loopback_listener() -> DriverReturnListener {
        DriverReturnListener::start(SocketAddr::from(([127, 0, 0, 1], 0)), Vec::new()).unwrap()
    }

    #[test]
    fn delivers_framed_reply_to_registered_waiter_after_handshake() {
        let listener = loopback_listener();
        let receiver = listener.register(17, 23).unwrap();

        let mut client = TcpStream::connect(listener.local_addr()).unwrap();
        send_ready(&mut client).unwrap();
        recv_ready(&mut client).unwrap();
        write_stage_message(&mut client, &open_message(17, 23)).unwrap();
        send_reply_predicted_with_stats(&mut client, 42, Default::default()).unwrap();

        let reply = receiver
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .expect("direct return reply");
        assert_eq!(reply.predicted, 42);
    }

    #[test]
    fn delivers_reply_when_the_client_hello_arrives_split() {
        let listener = loopback_listener();
        let receiver = listener.register(19, 27).unwrap();

        let mut client = TcpStream::connect(listener.local_addr()).unwrap();
        // TCP may split the four-byte hello; the peek must wait for the rest
        // instead of reading the tail as the start of a stage message.
        let magic = READY_MAGIC.to_le_bytes();
        client.set_nodelay(true).unwrap();
        client.write_all(&magic[..1]).unwrap();
        thread::sleep(Duration::from_millis(25));
        client.write_all(&magic[1..]).unwrap();

        recv_ready(&mut client).unwrap();
        write_stage_message(&mut client, &open_message(19, 27)).unwrap();
        send_reply_predicted_with_stats(&mut client, 55, Default::default()).unwrap();

        let reply = receiver
            .recv_timeout(Duration::from_secs(5))
            .unwrap()
            .expect("direct return reply");
        assert_eq!(reply.predicted, 55);
    }

    #[test]
    fn closed_connection_wakes_registered_waiter_with_error() {
        let listener = loopback_listener();
        let receiver = listener.register(29, 31).unwrap();

        let mut client = TcpStream::connect(listener.local_addr()).unwrap();
        recv_ready(&mut client).unwrap();
        write_stage_message(&mut client, &open_message(29, 31)).unwrap();
        drop(client);

        let error = receiver
            .recv_timeout(Duration::from_secs(5))
            .expect_err("closed direct return must wake the waiter");
        assert!(error.to_string().contains("closed before the next reply"));
    }

    #[test]
    fn unregistered_ids_never_reach_a_waiter() {
        let listener = loopback_listener();
        let receiver = listener.register(37, 41).unwrap();

        let mut client = TcpStream::connect(listener.local_addr()).unwrap();
        recv_ready(&mut client).unwrap();
        write_stage_message(&mut client, &open_message(999, 999)).unwrap();
        // The listener rejects the unregistered open immediately, so the
        // following write may either reach the socket buffer or observe the
        // peer reset. Both outcomes exercise the same rejection path.
        let _ = send_reply_predicted_with_stats(&mut client, 7, Default::default());

        assert!(
            receiver
                .recv_timeout(Duration::from_millis(300))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn source_allowlist_always_admits_loopback() {
        assert!(source_allowed("127.0.0.1".parse().unwrap(), &[]));
        assert!(source_allowed("::1".parse().unwrap(), &[]));
        let lan: IpAddr = "192.168.0.54".parse().unwrap();
        assert!(source_allowed(lan, &[lan]));
        assert!(!source_allowed("192.168.0.99".parse().unwrap(), &[lan]));
    }
}
