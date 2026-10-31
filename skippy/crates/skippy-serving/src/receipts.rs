//! Opt-in verifiable inference receipts for a serving stage.
//!
//! With receipts on, the stage's native receipt recorder feeds a
//! [`Recorder`] on a worker thread: leaf hashes per position, and a replay
//! log per sequence.
//! When stage 0 finishes a turn, it commits its own leaves, asks the next
//! stage to commit the same session and positions, and keeps the assembled
//! [`Receipt`] under the request's `x-request-id`. A challenge is opened by
//! replaying the turn in a scratch context of the stage model. Every stage
//! serves a small HTTP API on its receipts address:
//!
//! - `GET /receipts/{id}`: the receipt (stage 0);
//! - `POST /receipts/open`: open a [`Challenge`], forwarded down the chain
//!   to the stage that holds the challenged layer;
//! - `POST /receipts/commit`: commit a session's positions (called by the
//!   previous stage).
//!
//! Openings carry activations derived from the prompt, so the API binds a
//! loopback address unless the operator allows a remote one.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use skippy_protocol::StageConfig;
use skippy_receipts::recorder::{Recorder, Replayer};
use skippy_receipts::wire::{Challenge, Opening, RECEIPT_VERSION, Receipt, StageCommitment};

use crate::runtime_state::RuntimeState;
use crate::runtime_state::panic_recovery::lock_runtime;

/// Receipts issued (or refused) by stage 0 that stay retrievable.
const ISSUED_RECEIPTS: usize = 256;
/// Bound on a downstream commit or open, which may replay a whole turn.
const DOWNSTREAM_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Clone)]
pub struct ReceiptsOptions {
    /// Address of this stage's receipts HTTP API.
    pub listen: SocketAddr,
    /// Permit a non-loopback `listen` address.
    pub allow_remote: bool,
    /// Receipts API of the next stage, for example `http://127.0.0.1:9502`.
    pub downstream: Option<String>,
    /// The model the operator claims to serve, named in every receipt.
    pub claimed_model: String,
}

/// Resolves a session id to the native sequence that computed it.
pub trait SessionSequences: Send + Sync {
    fn sequence(&self, session: &str) -> Option<i32>;
}

impl SessionSequences for Mutex<RuntimeState> {
    fn sequence(&self, session: &str) -> Option<i32> {
        lock_runtime(self).receipt_sequence(session)
    }
}

pub struct StageReceipts {
    reports: report_queue::ReportQueue,
    replayer: OnceLock<Box<dyn Replayer + Send + Sync>>,
    downstream: Option<String>,
    claimed_model: String,
    /// By receipt id: the receipt, or why the turn got none.
    issued: Mutex<VecDeque<(String, Result<Receipt, String>)>>,
    client: reqwest::blocking::Client,
}

#[derive(Debug, Serialize, Deserialize)]
struct CommitRequest {
    receipt_id: String,
    session: String,
    position_start: u32,
    position_end: u32,
}

#[derive(Clone)]
struct ApiState {
    receipts: Arc<StageReceipts>,
    sessions: Arc<dyn SessionSequences>,
}

/// Installs the receipt recorder, attaches receipts to the runtime, and
/// serves the receipts API.
pub fn start(
    options: ReceiptsOptions,
    config: &StageConfig,
    runtime: Arc<Mutex<RuntimeState>>,
) -> Result<()> {
    check_listen(&options)?;
    let backend = skippy_runtime::backend_devices()
        .ok()
        .and_then(|devices| devices.into_iter().next())
        .map_or_else(|| "unknown".to_string(), |device| device.name);
    let receipts = Arc::new(StageReceipts::new(
        config.stage_index,
        config.layer_start..config.layer_end,
        backend,
        options.downstream,
        options.claimed_model,
    )?);
    {
        let mut state = lock_runtime(&runtime);
        let replayer = native::NativeReplayer::new(state.model.receipt_replayer());
        receipts.set_replayer(Box::new(replayer));
        state
            .model
            .set_receipt_recorder(Arc::new(native::RecorderSink(receipts.clone())))
            .context("install the receipt recorder")?;
        state.set_receipts(receipts.clone());
    }
    let listener = std::net::TcpListener::bind(options.listen)
        .with_context(|| format!("bind receipts API on {}", options.listen))?;
    spawn_api(
        listener,
        ApiState {
            receipts,
            sessions: runtime,
        },
    )?;
    tracing::info!(
        "receipts on: api={} stage={} layers={}..{}",
        options.listen,
        config.stage_index,
        config.layer_start,
        config.layer_end
    );
    Ok(())
}

/// Openings carry prompt-derived activations; refuse a public bind unless
/// the operator asked for it.
fn check_listen(options: &ReceiptsOptions) -> Result<()> {
    if !options.listen.ip().is_loopback() && !options.allow_remote {
        bail!(
            "receipts API on {} would expose opened activations to the network; \
             bind a loopback address or pass --receipts-allow-remote",
            options.listen
        );
    }
    Ok(())
}

fn spawn_api(listener: std::net::TcpListener, state: ApiState) -> Result<()> {
    listener.set_nonblocking(true)?;
    std::thread::Builder::new()
        .name("skippy-receipts-api".into())
        .spawn(move || serve(listener, state))
        .context("spawn receipts API thread")?;
    Ok(())
}

fn serve(listener: std::net::TcpListener, state: ApiState) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            tracing::error!("receipts API runtime: {error}");
            return;
        }
    };
    runtime.block_on(async move {
        let listener = match tokio::net::TcpListener::from_std(listener) {
            Ok(listener) => listener,
            Err(error) => {
                tracing::error!("receipts API listener: {error}");
                return;
            }
        };
        if let Err(error) = axum::serve(listener, router(state)).await {
            tracing::error!("receipts API stopped: {error}");
        }
    });
}

fn router(state: ApiState) -> Router {
    Router::new()
        .route("/receipts/commit", post(commit_handler))
        .route("/receipts/open", post(open_handler))
        .route("/receipts/{id}", get(receipt_handler))
        .with_state(state)
}

struct ApiError(StatusCode, anyhow::Error);

impl From<anyhow::Error> for ApiError {
    fn from(error: anyhow::Error) -> Self {
        Self(StatusCode::BAD_REQUEST, error)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.0,
            Json(serde_json::json!({ "error": format!("{:#}", self.1) })),
        )
            .into_response()
    }
}

async fn blocking<T: Send + 'static>(
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T, ApiError> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|error| ApiError::from(anyhow!("receipts worker failed: {error}")))?
        .map_err(ApiError::from)
}

async fn commit_handler(
    State(state): State<ApiState>,
    Json(request): Json<CommitRequest>,
) -> Result<Json<Vec<StageCommitment>>, ApiError> {
    blocking(move || {
        let seq = state
            .sessions
            .sequence(&request.session)
            .with_context(|| format!("no native sequence for session {}", request.session))?;
        state.receipts.commit_chain(&request, seq)
    })
    .await
    .map(Json)
}

async fn open_handler(
    State(state): State<ApiState>,
    Json(challenge): Json<Challenge>,
) -> Result<Json<Opening>, ApiError> {
    blocking(move || state.receipts.open(&challenge))
        .await
        .map(Json)
}

async fn receipt_handler(
    State(state): State<ApiState>,
    Path(id): Path<String>,
) -> Result<Json<Receipt>, ApiError> {
    // stage 0 commits a turn after its response is sent, so a client may
    // ask before the receipt exists
    match state.receipts.receipt(&id) {
        Some(Ok(receipt)) => Ok(Json(receipt)),
        Some(Err(reason)) => Err(ApiError(
            StatusCode::BAD_REQUEST,
            anyhow!("turn {id} got no receipt: {reason}"),
        )),
        None => Err(ApiError(StatusCode::NOT_FOUND, anyhow!("no receipt {id}"))),
    }
}

impl StageReceipts {
    fn new(
        stage_index: u32,
        layers: std::ops::Range<u32>,
        backend: String,
        downstream: Option<String>,
        claimed_model: String,
    ) -> Result<Self> {
        Ok(Self {
            reports: report_queue::ReportQueue::new(Arc::new(Recorder::new(
                stage_index,
                layers,
                backend,
            ))),
            replayer: OnceLock::new(),
            downstream: downstream.map(|url| url.trim_end_matches('/').to_string()),
            claimed_model,
            issued: Mutex::new(VecDeque::new()),
            client: reqwest::blocking::Client::builder()
                .timeout(DOWNSTREAM_TIMEOUT)
                .build()
                .context("build receipts HTTP client")?,
        })
    }

    fn set_replayer(&self, replayer: Box<dyn Replayer + Send + Sync>) {
        let _ = self.replayer.set(replayer);
    }

    /// Commits this stage and every later one.
    fn commit_chain(&self, request: &CommitRequest, seq: i32) -> Result<Vec<StageCommitment>> {
        self.reports.wait_applied();
        let mut stages = vec![self.reports.recorder().commit(
            &request.receipt_id,
            seq,
            request.position_start..request.position_end,
        )?];
        if let Some(url) = &self.downstream {
            let response = self
                .client
                .post(format!("{url}/receipts/commit"))
                .json(request)
                .send()?;
            stages.extend(decode::<Vec<StageCommitment>>(response)?);
        }
        Ok(stages)
    }

    /// Stage 0, after a turn: commits the whole chain and keeps the
    /// receipt, or why there is none.
    fn finish_turn(&self, receipt_id: &str, session: &str, seq: i32) -> Result<Receipt> {
        let result = self.commit_turn(receipt_id, session, seq);
        self.keep(
            receipt_id,
            result
                .as_ref()
                .map(Receipt::clone)
                .map_err(|e| format!("{e:#}")),
        );
        result
    }

    fn keep(&self, receipt_id: &str, result: Result<Receipt, String>) {
        let mut issued = self
            .issued
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        issued.push_back((receipt_id.to_owned(), result));
        while issued.len() > ISSUED_RECEIPTS {
            issued.pop_front();
        }
    }

    fn commit_turn(&self, receipt_id: &str, session: &str, seq: i32) -> Result<Receipt> {
        self.reports.wait_applied();
        let recorder = self.reports.recorder();
        let positions =
            recorder
                .recorded_range(seq)
                .with_context(|| match recorder.last_error() {
                    Some(reason) => format!("the turn recorded nothing: {reason}"),
                    None => "the turn recorded nothing".to_owned(),
                })?;
        let request = CommitRequest {
            receipt_id: receipt_id.to_string(),
            session: session.to_string(),
            position_start: positions.start,
            position_end: positions.end,
        };
        let stages = self.commit_chain(&request, seq)?;
        Ok(Receipt {
            version: RECEIPT_VERSION,
            receipt_id: receipt_id.to_string(),
            model: self.claimed_model.clone(),
            position_start: positions.start,
            position_end: positions.end,
            stages,
        })
    }

    /// The receipt of a turn, or why it got none; `None` for an unknown id.
    fn receipt(&self, id: &str) -> Option<Result<Receipt, String>> {
        self.issued
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .iter()
            .rev()
            .find(|(receipt_id, _)| receipt_id == id)
            .map(|(_, result)| result.clone())
    }

    fn open(&self, challenge: &Challenge) -> Result<Opening> {
        let recorder = self.reports.recorder();
        if recorder.layers().contains(&challenge.layer) {
            let replayer = self
                .replayer
                .get()
                .context("this stage cannot replay turns")?;
            self.reports.wait_applied();
            return recorder.open(challenge, replayer.as_ref());
        }
        let url = self
            .downstream
            .as_ref()
            .with_context(|| format!("no stage holds layer {}", challenge.layer))?;
        let response = self
            .client
            .post(format!("{url}/receipts/open"))
            .json(challenge)
            .send()?;
        decode(response)
    }
}

fn decode<T: serde::de::DeserializeOwned>(response: reqwest::blocking::Response) -> Result<T> {
    let status = response.status();
    let body = response.text()?;
    if !status.is_success() {
        bail!("downstream receipts API returned {status}: {body}");
    }
    serde_json::from_str(&body).context("decode downstream receipts reply")
}

/// Stage 0 hook at the end of a turn. Failures are logged; they never fail
/// the user's request.
pub(crate) fn finish_stage_zero_turn(
    runtime: &Mutex<RuntimeState>,
    frontend_request_id: Option<[u8; 16]>,
    request_id: u64,
    session: &str,
) {
    let (receipts, seq) = {
        let state = lock_runtime(runtime);
        let Some(receipts) = state.receipts().cloned() else {
            return;
        };
        (receipts, state.receipt_sequence(session))
    };
    let receipt_id = frontend_request_id
        .map(|bytes| uuid::Uuid::from_bytes(bytes).hyphenated().to_string())
        .unwrap_or_else(|| format!("{request_id:016x}"));
    let result = seq
        .with_context(|| format!("no native sequence for session {session}"))
        .and_then(|seq| receipts.finish_turn(&receipt_id, session, seq));
    match result {
        Ok(receipt) => tracing::info!(
            "receipt {} committed: positions {}..{}, {} stages",
            receipt.receipt_id,
            receipt.position_start,
            receipt.position_end,
            receipt.stages.len()
        ),
        Err(error) => tracing::warn!("receipt {receipt_id} not committed: {error:#}"),
    }
}

mod native;
mod report_queue;

#[cfg(test)]
#[path = "receipts/tests.rs"]
mod tests;
