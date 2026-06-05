use std::{
    collections::HashMap,
    io,
    net::{SocketAddr, TcpListener, TcpStream},
    sync::{Arc, Mutex, mpsc},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use skippy_protocol::binary::{StageReply, WireMessageKind, WireReplyKind, recv_reply, send_ready};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct DirectReturnKey {
    request_id: u64,
    session_id: u64,
}

type DirectReturnResult = Result<StageReply, String>;
type DirectReturnSender = mpsc::Sender<DirectReturnResult>;
type DirectReturnWaiters = Arc<Mutex<HashMap<DirectReturnKey, DirectReturnSender>>>;

pub(crate) struct BenchDirectReturnServer {
    local_addr: SocketAddr,
    waiters: DirectReturnWaiters,
}

impl BenchDirectReturnServer {
    pub(crate) fn start(bind_addr: &str) -> Result<Self> {
        let listener = TcpListener::bind(bind_addr)
            .with_context(|| format!("bind benchmark direct-return listener {bind_addr}"))?;
        let local_addr = listener
            .local_addr()
            .context("read benchmark direct-return listener address")?;
        let waiters = Arc::new(Mutex::new(HashMap::new()));
        let thread_waiters = waiters.clone();
        thread::spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        let waiters = thread_waiters.clone();
                        thread::spawn(move || {
                            if let Err(error) =
                                handle_bench_direct_return_connection(waiters, stream)
                            {
                                eprintln!("benchmark direct-return connection failed: {error:#}");
                            }
                        });
                    }
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => {
                        eprintln!("benchmark direct-return listener failed: {error}");
                        break;
                    }
                }
            }
        });
        Ok(Self {
            local_addr,
            waiters,
        })
    }

    pub(crate) fn endpoint(&self) -> String {
        self.local_addr.to_string()
    }

    pub(crate) fn register(
        &self,
        request_id: u64,
        session_id: u64,
    ) -> Result<BenchDirectReturnReceiver> {
        let key = DirectReturnKey {
            request_id,
            session_id,
        };
        let (sender, receiver) = mpsc::channel();
        self.waiters
            .lock()
            .map_err(|_| anyhow!("benchmark direct-return hub lock poisoned"))?
            .insert(key, sender);
        Ok(BenchDirectReturnReceiver {
            key,
            waiters: self.waiters.clone(),
            receiver,
        })
    }
}

pub(crate) struct BenchDirectReturnReceiver {
    key: DirectReturnKey,
    waiters: DirectReturnWaiters,
    receiver: mpsc::Receiver<DirectReturnResult>,
}

impl BenchDirectReturnReceiver {
    pub(crate) fn recv_expected(&self, expected: WireReplyKind) -> Result<StageReply> {
        let reply = self
            .receiver
            .recv_timeout(Duration::from_secs(300))
            .context("timed out waiting for benchmark direct prediction return")?
            .map_err(|error| anyhow!(error))?;
        if reply.kind != expected {
            bail!(
                "expected {expected:?} direct prediction return, got {:?}",
                reply.kind
            );
        }
        Ok(reply)
    }
}

impl Drop for BenchDirectReturnReceiver {
    fn drop(&mut self) {
        if let Ok(mut waiters) = self.waiters.lock() {
            waiters.remove(&self.key);
        }
    }
}

fn handle_bench_direct_return_connection(
    waiters: DirectReturnWaiters,
    mut stream: TcpStream,
) -> Result<()> {
    send_ready(&mut stream).context("send benchmark direct-return ready")?;
    let open = skippy_protocol::binary::read_stage_message(&mut stream, 0)
        .context("read benchmark direct-return open")?;
    if open.kind != WireMessageKind::PredictionReturnOpen {
        bail!("expected prediction-return-open message");
    }
    let key = DirectReturnKey {
        request_id: open.request_id,
        session_id: open.session_id,
    };
    let sender = waiters
        .lock()
        .map_err(|_| anyhow!("benchmark direct-return hub lock poisoned"))?
        .get(&key)
        .cloned()
        .ok_or_else(|| {
            anyhow!(
                "no benchmark direct-return waiter for request {}",
                key.request_id
            )
        })?;
    loop {
        match recv_reply(&mut stream) {
            Ok(reply) => {
                if sender.send(Ok(reply)).is_err() {
                    return Ok(());
                }
            }
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(error) => {
                let _ = sender.send(Err(error.to_string()));
                return Err(error).context("read benchmark direct prediction return");
            }
        }
    }
}
