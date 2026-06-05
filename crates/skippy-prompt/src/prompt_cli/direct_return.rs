#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct PromptDirectReturnKey {
    request_id: u64,
    session_id: u64,
}

type PromptDirectReturnResult = Result<StageReply, String>;
type PromptDirectReturnSender = mpsc::Sender<PromptDirectReturnResult>;
type PromptDirectReturnWaiters =
    Arc<Mutex<HashMap<PromptDirectReturnKey, PromptDirectReturnSender>>>;

struct PromptDirectReturnServer {
    local_addr: SocketAddr,
    waiters: PromptDirectReturnWaiters,
}

impl PromptDirectReturnServer {
    fn start(bind_addr: SocketAddr) -> Result<Self> {
        let listener = TcpListener::bind(bind_addr)
            .with_context(|| format!("bind prompt direct-return listener {bind_addr}"))?;
        let local_addr = listener
            .local_addr()
            .context("read prompt direct-return listener address")?;
        let waiters = Arc::new(Mutex::new(HashMap::new()));
        let thread_waiters = waiters.clone();
        thread::spawn(move || {
            for stream in listener.incoming() {
                match stream {
                    Ok(stream) => {
                        let waiters = thread_waiters.clone();
                        thread::spawn(move || {
                            if let Err(error) =
                                handle_prompt_direct_return_connection(waiters, stream)
                            {
                                eprintln!("prompt direct-return connection failed: {error:#}");
                            }
                        });
                    }
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => {
                        eprintln!("prompt direct-return listener failed: {error}");
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

    fn endpoint(&self) -> SocketAddr {
        self.local_addr
    }

    fn register(
        &self,
        request_id: u64,
        session_id: u64,
        timeout: Duration,
    ) -> Result<PromptDirectReturnReceiver> {
        let key = PromptDirectReturnKey {
            request_id,
            session_id,
        };
        let (sender, receiver) = mpsc::channel();
        self.waiters
            .lock()
            .map_err(|_| anyhow!("prompt direct-return hub lock poisoned"))?
            .insert(key, sender);
        Ok(PromptDirectReturnReceiver {
            key,
            waiters: self.waiters.clone(),
            receiver,
            timeout,
        })
    }
}

struct PromptDirectReturnReceiver {
    key: PromptDirectReturnKey,
    waiters: PromptDirectReturnWaiters,
    receiver: mpsc::Receiver<PromptDirectReturnResult>,
    timeout: Duration,
}

impl PromptDirectReturnReceiver {
    fn recv_expected(&self, expected: WireReplyKind) -> Result<StageReply> {
        let reply = self
            .receiver
            .recv_timeout(self.timeout)
            .context("timed out waiting for prompt direct prediction return")?
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

impl Drop for PromptDirectReturnReceiver {
    fn drop(&mut self) {
        if let Ok(mut waiters) = self.waiters.lock() {
            waiters.remove(&self.key);
        }
    }
}

fn handle_prompt_direct_return_connection(
    waiters: PromptDirectReturnWaiters,
    mut stream: TcpStream,
) -> Result<()> {
    send_ready(&mut stream).context("send prompt direct-return ready")?;
    let open = read_stage_message(&mut stream, 0).context("read prompt direct-return open")?;
    if open.kind != WireMessageKind::PredictionReturnOpen {
        bail!("expected prediction-return-open message");
    }
    let key = PromptDirectReturnKey {
        request_id: open.request_id,
        session_id: open.session_id,
    };
    let sender = waiters
        .lock()
        .map_err(|_| anyhow!("prompt direct-return hub lock poisoned"))?
        .get(&key)
        .cloned()
        .ok_or_else(|| {
            anyhow!(
                "no prompt direct-return waiter for request {}",
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
                return Err(error).context("read prompt direct prediction return");
            }
        }
    }
}
