//! How a stage reaches the receipts of the stage after it: over that
//! stage's receipts HTTP API, or over the binary stage connection the
//! chain already runs on.
//!
//! On the stage connection, a stage opens dedicated connections to the
//! next stage whose messages are all [`WireMessageKind::Receipts`]. They
//! carry no activations and hold no lane, and they stay open between
//! requests. The stage that accepts one commits or opens its own part,
//! forwards the request to the stage after it the same way, and replies
//! with every later stage's part appended ([`super::chain_server`]). In a
//! Mesh split, the connections cross peers like every other stage
//! connection.
//!
//! A commit ends every turn's response, so it does not wait behind
//! another client's replay: commits have a connection of their own and a
//! short timeout, after which the turn has no receipt. A next stage that
//! issues no receipts, or that does not answer, is not asked again for a
//! while, so it costs later turns nothing.
//!
//! No reply is read past what its request can produce
//! ([`super::reply_limits`]).

use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use skippy_protocol::StageConfig;
use skippy_protocol::binary::{
    MAX_STAGE_RECEIPTS_REFUSAL_BYTES, ReceiptsReply, StageWireMessage, recv_ready,
    recv_receipts_reply, write_stage_message,
};
use skippy_receipts::attention::opening::{BlockChallenge, RowsOpening};
use skippy_receipts::wire::{Challenge, Opening, StageCommitment};

use super::refusal::{self, Unavailable};
use super::{OPENING_DIGEST_HEADER, TURN_ROOT_HEADER};

/// Bound on a downstream open, which may replay a whole turn.
pub(super) const OPEN_TIMEOUT: Duration = Duration::from_secs(120);
/// Bound on a downstream commit: every later stage hashes the turn's
/// leaves into a tree. Past it the turn has no receipt.
pub(super) const COMMIT_TIMEOUT: Duration = Duration::from_secs(10);
/// How long a next stage that says it issues no receipts is not asked to
/// commit.
const NO_RECEIPTS_BACKOFF: Duration = Duration::from_secs(600);
/// How long a next stage that closed the connection on a commit is not
/// asked again: one that predates receipts, or one that is restarting.
const CLOSED_BACKOFF: Duration = Duration::from_secs(60);
/// How long a next stage that did not answer a commit is not asked again.
const UNREACHABLE_BACKOFF: Duration = Duration::from_secs(30);

/// A commit of a session's turn, as one stage asks it of the next.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct CommitRequest {
    pub receipt_id: String,
    pub session: String,
    pub position_start: u32,
    pub position_end: u32,
    /// Hex VRF key of the client the receipt is bound to. A stage opens
    /// such a turn only for a challenge with the client's proof, and pins
    /// its cells for the challenge window. Absent for a turn no client can
    /// challenge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_key: Option<String>,
}

/// A challenge as one stage forwards it to the next.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct ForwardedOpen {
    pub challenge: Challenge,
    /// Hex turn root of a bound receipt, which the challenge's proof is
    /// over.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turn_root: Option<String>,
}

/// A block challenge as one stage forwards it to the next.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct ForwardedRows {
    pub blocks: BlockChallenge,
    /// Hex digest of the opening stage 0 signed for a bound receipt, which
    /// the rows' proof is over.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opening: Option<String>,
}

/// The body of a receipts request on the stage connection.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ChainRequest {
    Commit(CommitRequest),
    Open(ForwardedOpen),
    Rows(ForwardedRows),
}

/// The receipts of the stages after this one. Each answer may hold at most
/// `limit` bytes.
pub(super) trait Downstream: Send + Sync {
    fn commit(&self, request: &CommitRequest, limit: usize) -> Result<Vec<StageCommitment>>;
    fn open(&self, request: &ForwardedOpen, limit: usize) -> Result<Opening>;
    fn rows(&self, request: &ForwardedRows, limit: usize) -> Result<RowsOpening>;
}

/// Why an exchange with the next stage gave no reply.
#[derive(Debug)]
enum Failure {
    /// The next stage did not answer in time, or could not be reached.
    Unreachable(anyhow::Error),
    /// The next stage closed the connection on the request: it does not
    /// serve receipts, as a stage that predates them.
    NoReceipts(anyhow::Error),
    /// The next stage sent what this stage does not take for a reply: one
    /// longer than the request can produce, or no receipts reply at all.
    Invalid(anyhow::Error),
}

impl Failure {
    fn of(error: std::io::Error) -> Self {
        use std::io::ErrorKind::{InvalidData, TimedOut, WouldBlock};
        match error.kind() {
            TimedOut | WouldBlock => Self::Unreachable(
                anyhow::Error::new(error).context("the next stage did not answer"),
            ),
            InvalidData => {
                Self::Invalid(anyhow::Error::new(error).context("the next stage's reply"))
            }
            _ => Self::NoReceipts(
                anyhow::Error::new(error).context("the next stage answered no receipts request"),
            ),
        }
    }

    fn into_error(self) -> anyhow::Error {
        match self {
            Self::Unreachable(error) | Self::Invalid(error) => error,
            Self::NoReceipts(error) => anyhow::Error::new(Unavailable(format!("{error:#}"))),
        }
    }
}

/// When the next stage is asked to commit again, and why not before.
#[derive(Default)]
struct CommitBackoff(Mutex<Option<(Instant, String)>>);

impl CommitBackoff {
    fn lock(&self) -> std::sync::MutexGuard<'_, Option<(Instant, String)>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Fails at once while the next stage is not to be asked.
    fn check(&self, now: Instant) -> Result<()> {
        let mut state = self.lock();
        match &*state {
            Some((until, why)) if now < *until => bail!("the next stage is not asked: {why}"),
            _ => {
                *state = None;
                Ok(())
            }
        }
    }

    /// Notes what a commit came to: `wait` is how long its reply says not
    /// to ask again ([`commit_backoff`]), and `result` why.
    fn note<T>(&self, wait: Option<Duration>, result: &Result<T>, now: Instant) {
        if let (Some(wait), Err(error)) = (wait, result) {
            *self.lock() = Some((now + wait, format!("{error:#}")));
        }
    }
}

/// How long the next stage is not asked to commit after it answered a
/// commit with `reply`; `None` when it answered or refused the commit
/// itself, which says nothing about the next one.
fn commit_backoff(reply: &Result<ReceiptsReply, Failure>) -> Option<Duration> {
    match reply {
        Ok(ReceiptsReply::Unavailable(_)) => Some(NO_RECEIPTS_BACKOFF),
        Err(Failure::NoReceipts(_)) => Some(CLOSED_BACKOFF),
        Err(Failure::Unreachable(_)) => Some(UNREACHABLE_BACKOFF),
        _ => None,
    }
}

/// The next stage's receipts HTTP API.
pub(super) struct HttpDownstream {
    url: String,
    client: reqwest::blocking::Client,
    commit_timeout: Duration,
    open_timeout: Duration,
    backoff: CommitBackoff,
}

impl HttpDownstream {
    pub(super) fn new(url: &str, commit_timeout: Duration, open_timeout: Duration) -> Result<Self> {
        Ok(Self {
            url: url.trim_end_matches('/').to_owned(),
            client: reqwest::blocking::Client::builder()
                .build()
                .context("build receipts HTTP client")?,
            commit_timeout,
            open_timeout,
            backoff: CommitBackoff::default(),
        })
    }

    /// Posts `body` and reads the reply, an answer of at most `limit` bytes.
    fn post(
        &self,
        route: &str,
        body: &impl Serialize,
        header: Option<(&str, &str)>,
        timeout: Duration,
        limit: usize,
    ) -> Result<ReceiptsReply, Failure> {
        let mut request = self
            .client
            .post(format!("{}/receipts/{route}", self.url))
            .timeout(timeout)
            .json(body);
        if let Some((name, value)) = header {
            request = request.header(name, value);
        }
        let response = request.send().map_err(|error| {
            Failure::Unreachable(anyhow::Error::new(error).context("the next stage's receipts API"))
        })?;
        let status = response.status();
        let limit = if status.is_success() {
            limit
        } else {
            MAX_STAGE_RECEIPTS_REFUSAL_BYTES
        };
        let mut bytes = Vec::new();
        response
            .take(limit as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(Failure::of)?;
        if bytes.len() > limit {
            return Err(Failure::Invalid(anyhow::anyhow!(
                "the next stage's reply exceeds what its request can produce"
            )));
        }
        Ok(http_reply(status, bytes))
    }
}

/// The reply an HTTP status and body of the receipts API stand for.
fn http_reply(status: reqwest::StatusCode, body: Vec<u8>) -> ReceiptsReply {
    use reqwest::StatusCode;
    if status.is_success() {
        return ReceiptsReply::Answer(body);
    }
    let json: Option<serde_json::Value> = serde_json::from_slice(&body).ok();
    let field = |name: &str| json.as_ref().and_then(|json| json.get(name)).cloned();
    let reason = field("error")
        .and_then(|error| error.as_str().map(str::to_owned))
        .unwrap_or_else(|| String::from_utf8_lossy(&body).into_owned());
    let reason = format!("{status}: {reason}");
    match status {
        StatusCode::NOT_FOUND => ReceiptsReply::NotFound(reason),
        StatusCode::SERVICE_UNAVAILABLE => ReceiptsReply::Unavailable(reason),
        StatusCode::CONFLICT | StatusCode::GONE => field("cannot_open")
            .and_then(|cannot| serde_json::to_vec(&cannot).ok())
            .map_or(ReceiptsReply::Refused(reason), ReceiptsReply::CannotOpen),
        _ => ReceiptsReply::Refused(reason),
    }
}

/// The typed answer of a reply, or why there is none.
fn decode<T: DeserializeOwned>(reply: Result<ReceiptsReply, Failure>) -> Result<T> {
    let bytes = refusal::answer(reply.map_err(Failure::into_error)?)?;
    serde_json::from_slice(&bytes).context("decode the next stage's receipts")
}

impl Downstream for HttpDownstream {
    fn commit(&self, request: &CommitRequest, limit: usize) -> Result<Vec<StageCommitment>> {
        self.backoff.check(Instant::now())?;
        let reply = self.post("commit", request, None, self.commit_timeout, limit);
        let wait = commit_backoff(&reply);
        let result = decode(reply);
        self.backoff.note(wait, &result, Instant::now());
        result
    }

    fn open(&self, request: &ForwardedOpen, limit: usize) -> Result<Opening> {
        let root = request
            .turn_root
            .as_deref()
            .map(|root| (TURN_ROOT_HEADER, root));
        decode(self.post("open", &request.challenge, root, self.open_timeout, limit))
    }

    fn rows(&self, request: &ForwardedRows, limit: usize) -> Result<RowsOpening> {
        let opening = request
            .opening
            .as_deref()
            .map(|digest| (OPENING_DIGEST_HEADER, digest));
        decode(self.post("rows", &request.blocks, opening, self.open_timeout, limit))
    }
}

/// One kept connection to the next stage; one request runs on it at a time.
struct Lane {
    timeout: Duration,
    stream: Mutex<Option<TcpStream>>,
}

impl Lane {
    fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            stream: Mutex::new(None),
        }
    }
}

/// The next stage over receipts connections of the binary stage protocol:
/// one for commits and one for openings and rows.
pub(super) struct StageConnection {
    config: StageConfig,
    commits: Lane,
    openings: Lane,
    backoff: CommitBackoff,
}

impl StageConnection {
    pub(super) fn new(
        config: &StageConfig,
        commit_timeout: Duration,
        open_timeout: Duration,
    ) -> Self {
        Self {
            config: config.clone(),
            commits: Lane::new(commit_timeout),
            openings: Lane::new(open_timeout),
            backoff: CommitBackoff::default(),
        }
    }

    fn connect(&self, timeout: Duration) -> Result<TcpStream, Failure> {
        let connect = || -> Result<TcpStream> {
            let mut stream =
                crate::binary_transport::connect_binary_downstream(&self.config, timeout)?
                    .context("this stage has no stage after it")?;
            crate::binary_transport::send_client_ready_hello_if_enabled(&mut stream)?;
            stream.set_read_timeout(Some(timeout))?;
            stream.set_write_timeout(Some(timeout))?;
            recv_ready(&mut stream).context("the next stage did not get ready")?;
            Ok(stream)
        };
        connect().map_err(Failure::Unreachable)
    }

    fn request(
        &self,
        lane: &Lane,
        request: &ChainRequest,
        limit: usize,
    ) -> Result<ReceiptsReply, Failure> {
        let body = serde_json::to_vec(request)
            .map_err(|error| Failure::Unreachable(anyhow::Error::new(error)))?;
        let mut kept = lane.stream.lock().unwrap_or_else(PoisonError::into_inner);
        let (mut stream, reused) = match kept.take() {
            Some(stream) => (stream, true),
            None => (self.connect(lane.timeout)?, false),
        };
        let reply = match exchange(&mut stream, &body, limit) {
            // a kept connection the next stage closed since its last
            // request: once more, on a new one. A request that timed out is
            // not sent again.
            Err(Failure::NoReceipts(_)) if reused => {
                stream = self.connect(lane.timeout)?;
                exchange(&mut stream, &body, limit)?
            }
            reply => reply?,
        };
        *kept = Some(stream);
        Ok(reply)
    }
}

fn exchange(stream: &mut TcpStream, body: &[u8], limit: usize) -> Result<ReceiptsReply, Failure> {
    write_stage_message(&mut *stream, &StageWireMessage::receipts(body.to_vec()))
        .and_then(|()| stream.flush())
        .map_err(Failure::of)?;
    recv_receipts_reply(&mut *stream, limit).map_err(Failure::of)
}

impl Downstream for StageConnection {
    fn commit(&self, request: &CommitRequest, limit: usize) -> Result<Vec<StageCommitment>> {
        self.backoff.check(Instant::now())?;
        let request = ChainRequest::Commit(request.clone());
        let reply = self.request(&self.commits, &request, limit);
        let wait = commit_backoff(&reply);
        let result = decode(reply);
        self.backoff.note(wait, &result, Instant::now());
        result
    }

    fn open(&self, request: &ForwardedOpen, limit: usize) -> Result<Opening> {
        let request = ChainRequest::Open(request.clone());
        decode(self.request(&self.openings, &request, limit))
    }

    fn rows(&self, request: &ForwardedRows, limit: usize) -> Result<RowsOpening> {
        let request = ChainRequest::Rows(request.clone());
        decode(self.request(&self.openings, &request, limit))
    }
}

#[cfg(test)]
#[path = "downstream_tests.rs"]
mod tests;
