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
use skippy_protocol::binary::{
    READY_MAGIC, StageReply, WireMessageKind, read_stage_message, recv_ready, recv_reply,
    send_ready,
};

type ReplyResult = std::result::Result<StageReply, String>;

#[derive(Default)]
struct Waiters {
    map: Mutex<HashMap<(u64, u64), mpsc::Sender<ReplyResult>>>,
}

pub(crate) struct DriverReturnListener {
    shutdown: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    waiters: Arc<Waiters>,
}

pub(crate) struct DriverReturnReceiver {
    key: (u64, u64),
    waiters: Arc<Waiters>,
    receiver: mpsc::Receiver<ReplyResult>,
}

impl DriverReturnListener {
    pub(crate) fn start(bind_addr: SocketAddr) -> Result<Self> {
        let listener = TcpListener::bind(bind_addr)
            .with_context(|| format!("bind driver prediction return listener {bind_addr}"))?;
        listener
            .set_nonblocking(true)
            .context("set driver prediction return listener nonblocking")?;
        let shutdown = Arc::new(AtomicBool::new(false));
        let waiters = Arc::new(Waiters::default());
        let thread_shutdown = shutdown.clone();
        let thread_waiters = waiters.clone();
        let thread = thread::spawn(move || {
            while !thread_shutdown.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => {
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
        })
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
        .set_read_timeout(Some(Duration::from_millis(250)))
        .context("set driver prediction return hello peek timeout")?;
    let mut bytes = [0_u8; 4];
    let peeked = stream.peek(&mut bytes);
    stream
        .set_read_timeout(previous)
        .context("restore driver prediction return stream timeout")?;
    match peeked {
        Ok(4) if i32::from_le_bytes(bytes) == READY_MAGIC => {
            recv_ready(&mut *stream).context("consume driver prediction return client hello")?;
        }
        Ok(_) => {}
        Err(error) if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
        Err(error) => return Err(error).context("peek driver prediction return hello"),
    }
    Ok(())
}
