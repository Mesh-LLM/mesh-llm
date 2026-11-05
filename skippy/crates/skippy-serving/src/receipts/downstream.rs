//! How a stage reaches the receipts of the stage after it: over that
//! stage's receipts HTTP API, or over the binary stage connection the
//! chain already runs on.
//!
//! On the stage connection, a stage opens a dedicated connection to the
//! next stage whose messages are all [`WireMessageKind::Receipts`]. It
//! carries no activations and holds no lane, and it stays open between
//! requests. The stage that accepts it commits or opens its own part,
//! forwards the request to the stage after it the same way, and replies
//! with every later stage's part appended. In a Mesh split, the connection
//! crosses peers like every other stage connection.

use std::io::Write;
use std::net::TcpStream;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use skippy_protocol::StageConfig;
use skippy_protocol::binary::{
    StageWireMessage, WireMessageKind, read_stage_message, recv_ready, recv_receipts_reply,
    send_receipts_reply, write_stage_message,
};
use skippy_receipts::attention::opening::{BlockChallenge, RowsOpening};
use skippy_receipts::wire::{Challenge, Opening, StageCommitment};

use super::{CommitRequest, SessionSequences, StageReceipts};

/// The receipts of the stages after this one.
pub(super) trait Downstream: Send + Sync {
    fn commit(&self, request: &CommitRequest) -> Result<Vec<StageCommitment>>;
    fn open(&self, challenge: &Challenge) -> Result<Opening>;
    fn rows(&self, challenge: &BlockChallenge) -> Result<RowsOpening>;
}

/// The body of a receipts request on the stage connection.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ChainRequest {
    Commit(CommitRequest),
    Open(Challenge),
    Rows(BlockChallenge),
}

/// The next stage's receipts HTTP API.
pub(super) struct HttpDownstream {
    url: String,
    client: reqwest::blocking::Client,
}

impl HttpDownstream {
    pub(super) fn new(url: &str, timeout: Duration) -> Result<Self> {
        Ok(Self {
            url: url.trim_end_matches('/').to_owned(),
            client: reqwest::blocking::Client::builder()
                .timeout(timeout)
                .build()
                .context("build receipts HTTP client")?,
        })
    }

    fn post<T: DeserializeOwned>(&self, route: &str, body: &impl Serialize) -> Result<T> {
        let response = self
            .client
            .post(format!("{}/receipts/{route}", self.url))
            .json(body)
            .send()?;
        let status = response.status();
        let body = response.text()?;
        if !status.is_success() {
            bail!("downstream receipts API returned {status}: {body}");
        }
        serde_json::from_str(&body).context("decode downstream receipts reply")
    }
}

impl Downstream for HttpDownstream {
    fn commit(&self, request: &CommitRequest) -> Result<Vec<StageCommitment>> {
        self.post("commit", request)
    }

    fn open(&self, challenge: &Challenge) -> Result<Opening> {
        self.post("open", challenge)
    }

    fn rows(&self, challenge: &BlockChallenge) -> Result<RowsOpening> {
        self.post("rows", challenge)
    }
}

/// The next stage over a receipts connection of the binary stage protocol.
pub(super) struct StageConnection {
    config: StageConfig,
    timeout: Duration,
    /// The open connection, if any; one request runs on it at a time.
    stream: Mutex<Option<TcpStream>>,
}

impl StageConnection {
    pub(super) fn new(config: &StageConfig, timeout: Duration) -> Self {
        Self {
            config: config.clone(),
            timeout,
            stream: Mutex::new(None),
        }
    }

    fn connect(&self) -> Result<TcpStream> {
        let mut stream =
            crate::binary_transport::connect_binary_downstream(&self.config, self.timeout)?
                .context("this stage has no stage after it")?;
        crate::binary_transport::send_client_ready_hello_if_enabled(&mut stream)?;
        stream.set_read_timeout(Some(self.timeout))?;
        stream.set_write_timeout(Some(self.timeout))?;
        recv_ready(&mut stream).context("the next stage did not get ready")?;
        Ok(stream)
    }

    fn request<T: DeserializeOwned>(&self, request: &ChainRequest) -> Result<T> {
        let body = serde_json::to_vec(request)?;
        let mut kept = self.stream.lock().unwrap_or_else(PoisonError::into_inner);
        // a kept connection may have closed since its last request
        let (mut stream, reused) = match kept.take() {
            Some(stream) => (stream, true),
            None => (self.connect()?, false),
        };
        let reply = match exchange(&mut stream, &body) {
            Err(_) if reused => {
                stream = self.connect()?;
                exchange(&mut stream, &body)?
            }
            reply => reply?,
        };
        *kept = Some(stream);
        match reply {
            Ok(bytes) => serde_json::from_slice(&bytes).context("decode the next stage's receipts"),
            Err(reason) => bail!("the next stage: {reason}"),
        }
    }
}

fn exchange(stream: &mut TcpStream, body: &[u8]) -> Result<Result<Vec<u8>, String>> {
    write_stage_message(&mut *stream, &StageWireMessage::receipts(body.to_vec()))?;
    stream.flush()?;
    Ok(recv_receipts_reply(&mut *stream)?)
}

impl Downstream for StageConnection {
    fn commit(&self, request: &CommitRequest) -> Result<Vec<StageCommitment>> {
        self.request(&ChainRequest::Commit(request.clone()))
    }

    fn open(&self, challenge: &Challenge) -> Result<Opening> {
        self.request(&ChainRequest::Open(challenge.clone()))
    }

    fn rows(&self, challenge: &BlockChallenge) -> Result<RowsOpening> {
        self.request(&ChainRequest::Rows(challenge.clone()))
    }
}

/// Serves a receipts connection that an earlier stage opened with `first`,
/// until that stage closes it.
pub(crate) fn serve_connection(
    receipts: Option<Arc<StageReceipts>>,
    sessions: &dyn SessionSequences,
    mut upstream: TcpStream,
    first: StageWireMessage,
) -> Result<()> {
    let mut message = first;
    loop {
        let reply = match &receipts {
            Some(receipts) => answer(receipts, sessions, &message).map_err(|e| format!("{e:#}")),
            None => Err("this stage issues no receipts".to_owned()),
        };
        send_receipts_reply(&mut upstream, reply.as_deref().map_err(String::as_str))?;
        upstream.flush()?;
        message = match read_stage_message(&mut upstream, 0) {
            Ok(message) => message,
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(error) => return Err(error.into()),
        };
    }
}

fn answer(
    receipts: &StageReceipts,
    sessions: &dyn SessionSequences,
    message: &StageWireMessage,
) -> Result<Vec<u8>> {
    if message.kind != WireMessageKind::Receipts {
        bail!("a receipts connection carries only receipts requests");
    }
    match serde_json::from_slice(&message.raw_bytes).context("decode a receipts request")? {
        ChainRequest::Commit(request) => Ok(serde_json::to_vec(
            &receipts.commit_session(sessions, &request)?,
        )?),
        ChainRequest::Open(challenge) => Ok(serde_json::to_vec(&receipts.open(&challenge)?)?),
        ChainRequest::Rows(challenge) => Ok(serde_json::to_vec(&receipts.rows(&challenge)?)?),
    }
}
