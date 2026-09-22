use skippy_protocol::binary::StageStream as TcpStream;
use std::{
    io,
    io::Write,
    net::{Shutdown, SocketAddr, TcpListener},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use skippy_protocol::binary::{StageReply, WireMessageKind, read_stage_message, recv_reply};

pub(super) struct PredictionReturnListener {
    bind_addr: SocketAddr,
    receiver: mpsc::Receiver<Result<StageReply, String>>,
    shutdown: Arc<AtomicBool>,
    connection: Arc<Mutex<Option<TcpStream>>>,
    thread: Option<JoinHandle<()>>,
}

impl PredictionReturnListener {
    pub(super) fn start() -> Result<Self> {
        let listener = TcpListener::bind("127.0.0.1:0")
            .context("bind correctness prediction return listener")?;
        let bind_addr = listener
            .local_addr()
            .context("read correctness prediction return listener address")?;
        listener
            .set_nonblocking(true)
            .context("set correctness prediction return listener nonblocking")?;
        let shutdown = Arc::new(AtomicBool::new(false));
        let thread_shutdown = shutdown.clone();
        let connection = Arc::new(Mutex::new(None));
        let thread_connection = connection.clone();
        let (sender, receiver) = mpsc::channel();
        let thread = thread::spawn(move || {
            let result =
                accept_prediction_return(listener, &thread_shutdown, &thread_connection, &sender);
            if let Err(error) = result {
                let _ = sender.send(Err(format!("{error:#}")));
            }
        });
        Ok(Self {
            bind_addr,
            receiver,
            shutdown,
            connection,
            thread: Some(thread),
        })
    }

    pub(super) fn endpoint(&self) -> String {
        format!("tcp://{}", self.bind_addr)
    }

    pub(super) fn receive(&self, timeout: Duration) -> Result<StageReply> {
        match self.receiver.recv_timeout(timeout) {
            Ok(Ok(reply)) => Ok(reply),
            Ok(Err(error)) => Err(anyhow!(error)),
            Err(mpsc::RecvTimeoutError::Timeout) => {
                bail!("timed out waiting for direct prediction return")
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                bail!("direct prediction return listener disconnected")
            }
        }
    }
}

impl Drop for PredictionReturnListener {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Ok(connection) = self.connection.lock()
            && let Some(stream) = connection.as_ref()
        {
            let _ = stream.shutdown(Shutdown::Both);
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn accept_prediction_return(
    listener: TcpListener,
    shutdown: &AtomicBool,
    connection: &Mutex<Option<TcpStream>>,
    sender: &mpsc::Sender<Result<StageReply, String>>,
) -> Result<()> {
    let stream = loop {
        match listener.accept() {
            Ok((stream, _)) => break stream,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                if shutdown.load(Ordering::SeqCst) {
                    return Ok(());
                }
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error).context("accept direct prediction return"),
        }
    };
    let mut stream = TcpStream::new(stream);
    stream
        .set_nonblocking(false)
        .context("set direct prediction return stream blocking")?;
    let shutdown_stream = stream
        .try_clone()
        .context("clone direct prediction return stream for shutdown")?;
    {
        let mut connection = connection
            .lock()
            .map_err(|_| anyhow!("direct prediction return connection lock poisoned"))?;
        if shutdown.load(Ordering::SeqCst) {
            return Ok(());
        }
        *connection = Some(shutdown_stream);
    }
    skippy_protocol::binary::server_setup(
        &mut stream,
        None,
        None,
        None,
        std::time::Instant::now() + Duration::from_secs(20),
        shutdown,
    )?;
    stream.flush().ok();
    let open =
        read_stage_message(&mut stream, 0).context("read direct prediction return open message")?;
    if open.kind != WireMessageKind::PredictionReturnOpen {
        bail!("expected direct prediction return open message");
    }
    loop {
        match recv_reply(&mut stream) {
            Ok(reply) => {
                if sender.send(Ok(reply)).is_err() {
                    return Ok(());
                }
            }
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => {
                let _ = sender.send(Err(format!(
                    "direct prediction return closed before a reply: {error}"
                )));
                return Ok(());
            }
            Err(error) => return Err(error).context("read direct prediction return reply"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_protocol::binary::{
        StageStateHeader, StageWireMessage, send_reply_predicted, write_stage_message,
    };

    #[test]
    fn receives_prediction_over_direct_return_endpoint() {
        let listener = PredictionReturnListener::start().unwrap();
        let endpoint = listener.endpoint();
        let address = endpoint.strip_prefix("tcp://").unwrap().to_string();
        let client = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            skippy_protocol::binary::client_setup(
                &mut stream,
                skippy_protocol::binary::ConnectionRole::PredictionReturn,
                None,
                None,
                std::time::Instant::now() + Duration::from_secs(2),
                &AtomicBool::new(false),
            )
            .unwrap();
            let kind = WireMessageKind::PredictionReturnOpen;
            let open = StageWireMessage {
                kind,
                pos_start: 0,
                token_count: 0,
                state: StageStateHeader::new(kind),
                request_id: 11,
                session_id: 13,
                sampling: None,
                chat_sampling_metadata: None,
                tokens: Vec::new(),
                positions: Vec::new(),
                activation: Vec::new(),
                raw_bytes: Vec::new(),
            };
            write_stage_message(&mut stream, &open).unwrap();
            send_reply_predicted(&mut stream, 674).unwrap();
        });

        let reply = listener.receive(Duration::from_secs(1)).unwrap();

        assert_eq!(reply.predicted, 674);
        client.join().unwrap();
    }

    #[test]
    fn drop_stops_when_connected_peer_stalls() {
        let listener = PredictionReturnListener::start().unwrap();
        let address = listener
            .endpoint()
            .strip_prefix("tcp://")
            .unwrap()
            .to_string();
        let mut stream = TcpStream::connect(address).unwrap();
        skippy_protocol::binary::client_setup(
            &mut stream,
            skippy_protocol::binary::ConnectionRole::PredictionReturn,
            None,
            None,
            std::time::Instant::now() + Duration::from_secs(2),
            &AtomicBool::new(false),
        )
        .unwrap();

        let started = std::time::Instant::now();
        drop(listener);

        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn closed_return_stream_reports_the_transport_failure() {
        let listener = PredictionReturnListener::start().unwrap();
        let address = listener
            .endpoint()
            .strip_prefix("tcp://")
            .unwrap()
            .to_string();
        let client = thread::spawn(move || {
            let mut stream = TcpStream::connect(address).unwrap();
            skippy_protocol::binary::client_setup(
                &mut stream,
                skippy_protocol::binary::ConnectionRole::PredictionReturn,
                None,
                None,
                std::time::Instant::now() + Duration::from_secs(2),
                &AtomicBool::new(false),
            )
            .unwrap();
            let kind = WireMessageKind::PredictionReturnOpen;
            let open = StageWireMessage {
                kind,
                pos_start: 0,
                token_count: 0,
                state: StageStateHeader::new(kind),
                request_id: 17,
                session_id: 19,
                sampling: None,
                chat_sampling_metadata: None,
                tokens: Vec::new(),
                positions: Vec::new(),
                activation: Vec::new(),
                raw_bytes: Vec::new(),
            };
            write_stage_message(&mut stream, &open).unwrap();
        });

        let error = listener
            .receive(Duration::from_secs(1))
            .expect_err("closed direct return must not look like a clean listener exit");

        assert!(
            error
                .to_string()
                .contains("direct prediction return closed before a reply")
        );
        client.join().unwrap();
    }
}
