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
//! - `POST /receipts/open`: open a [`Challenge`] on this stage and every
//!   later one; each stage opens its picked layers, the input of layer 0
//!   if it holds layer 0, and the residuals at its boundaries;
//! - `POST /receipts/rows`: open the K and V rows of the attention blocks a
//!   client picked after the opening ([`BlockChallenge`]) on this stage and
//!   every later one;
//! - `POST /receipts/commit`: commit a session's positions (called by the
//!   previous stage).
//!
//! A request that names a client VRF key (`x-receipt-client-key`) gets a
//! bound receipt: the key, the parent turn and the runtime go into the turn
//! root, which the stage signs with its node key. Stage 0 then opens one
//! challenge per bound receipt, and only with a VRF proof of the turn root
//! under that key, so the client cannot choose what it challenges.
//!
//! Openings carry activations derived from the prompt, so the API binds a
//! loopback address unless the operator allows a remote one.

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use skippy_inference_api::ReceiptRequest;
use skippy_protocol::StageConfig;
use skippy_receipts::attention::opening::{BlockChallenge, RowsOpening};
use skippy_receipts::commitment::{Hash, hex_array};
use skippy_receipts::recorder::{KvStreams, Recorder, Replayer};
use skippy_receipts::turn::{self, BindingRequest, TurnBinding};
use skippy_receipts::vrf;
use skippy_receipts::wire::{Challenge, Opening, RECEIPT_VERSION, Receipt, StageCommitment};

/// What stage 0 knows about a turn's tokens beyond what it recorded.
#[derive(Clone, Debug)]
pub(crate) struct TurnTokens {
    /// The first position whose token was sampled: the prompt length.
    pub generation_start: u32,
    /// The last sampled token, which no ubatch computed.
    pub final_token: Option<i32>,
    /// How many tokens the turn sampled, the final one included.
    pub sampled: u32,
    /// How the node chose the tokens, as the receipt states it
    /// ([`sampler_statement`]).
    pub sampler: String,
}

use crate::runtime_state::RuntimeState;
use crate::runtime_state::panic_recovery::lock_runtime;

/// Receipts issued (or refused) by stage 0 that stay retrievable.
const ISSUED_RECEIPTS: usize = 256;
/// Bound on a downstream commit or open, which may replay a whole turn.
const DOWNSTREAM_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug, Clone)]
pub struct ReceiptsOptions {
    /// Address of this stage's receipts HTTP API. `None` serves none: a
    /// single stage whose embedder carries [`client_router`] on its OpenAI
    /// surface instead.
    pub listen: Option<SocketAddr>,
    /// Permit a non-loopback `listen` address.
    pub allow_remote: bool,
    /// Receipts API of the next stage, for example `http://127.0.0.1:9502`.
    /// `None` reaches the next stage, if there is one, over the binary stage
    /// connection ([`downstream`]).
    pub downstream: Option<String>,
    /// The model the operator claims to serve, named in every receipt.
    pub claimed_model: String,
    /// Ed25519 seed of the key that signs turn roots. A stage without one
    /// signs with a key made at startup.
    pub signing_key: Option<[u8; 32]>,
}

impl ReceiptsOptions {
    /// A stage embedded in a host: it serves no receipts API of its own,
    /// since the host carries [`client_router`], and reaches a later stage
    /// over the binary stage connection.
    pub fn embedded(claimed_model: String, signing_key: Option<[u8; 32]>) -> Self {
        Self {
            listen: None,
            allow_remote: false,
            downstream: None,
            claimed_model,
            signing_key,
        }
    }
}

/// The runtime a receipt names.
fn runtime_identity() -> String {
    turn::runtime_identity(
        skippy_runtime::ABI_VERSION_MAJOR,
        skippy_runtime::ABI_VERSION_MINOR,
        skippy_runtime::ABI_VERSION_PATCH,
        &skippy_runtime::receipt_runtime_source(),
    )
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
    /// The receipts of the stages after this one; `None` on the last stage.
    downstream: Option<Box<dyn downstream::Downstream>>,
    claimed_model: String,
    signing_key: ed25519_dalek::SigningKey,
    /// By receipt id: the receipt, or why the turn got none.
    issued: Mutex<VecDeque<(String, Result<Receipt, String>)>>,
    /// By bound receipt id: the one challenge it may be opened for.
    challenged: Mutex<HashMap<String, Challenge>>,
    /// By bound receipt id: the digest of the opening this stage signed,
    /// whose VRF proof under the client key authorizes reading the rows.
    opened: Mutex<HashMap<String, Hash>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CommitRequest {
    receipt_id: String,
    session: String,
    position_start: u32,
    position_end: u32,
    /// The receipt is bound to a client key, so the client may challenge
    /// it and the stages pin its cells ([`Recorder::pinned_cells`]).
    challengeable: bool,
}

#[derive(Clone)]
struct ApiState {
    receipts: Arc<StageReceipts>,
    sessions: Arc<dyn SessionSequences>,
}

/// How this stage reaches the receipts of the next one: over `url`, its
/// receipts API, when one is given, else over the stage connection when
/// there is a next stage.
fn next_stage(
    url: Option<&str>,
    config: &StageConfig,
) -> Result<Option<Box<dyn downstream::Downstream>>> {
    Ok(match url {
        Some(url) => Some(Box::new(downstream::HttpDownstream::new(
            url,
            DOWNSTREAM_TIMEOUT,
        )?)),
        None if config.downstream.is_some() => Some(Box::new(downstream::StageConnection::new(
            config,
            DOWNSTREAM_TIMEOUT,
        ))),
        None => None,
    })
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
    let seed = options.signing_key.unwrap_or_else(|| {
        let mut seed = [0u8; 32];
        rand::Rng::fill_bytes(&mut rand::rng(), &mut seed);
        seed
    });
    let signing_key = ed25519_dalek::SigningKey::from_bytes(&seed);
    tracing::info!(
        "receipts are signed by node key {}",
        hex::encode(signing_key.verifying_key().as_bytes())
    );
    let receipts = Arc::new(StageReceipts::new(
        config.stage_index,
        config.layer_start..config.layer_end,
        backend,
        next_stage(options.downstream.as_deref(), config)?,
        options.claimed_model,
        signing_key,
    )?);
    {
        let mut state = lock_runtime(&runtime);
        let replayer = native::NativeReplayer::new(
            state.model.receipt_replayer(),
            Arc::downgrade(&runtime),
            Arc::downgrade(&receipts),
        );
        receipts.set_replayer(Box::new(replayer));
        receipts.reports.set_native(state.model.receipt_flusher());
        let streams = state
            .model
            .receipt_replayer()
            .kv_streams()
            .context("read how the KV cache lays out its cells")?;
        receipts.reports.recorder().set_kv_streams(if streams > 1 {
            KvStreams::PerSequence
        } else {
            KvStreams::Shared
        });
        let recurrent = state
            .model
            .set_receipt_recorder(Arc::new(native::RecorderSink(receipts.clone())))
            .context("install the receipt recorder")?;
        receipts.reports.recorder().set_recurrent(recurrent);
        state.set_receipts(receipts.clone());
    }
    if let Some(listen) = options.listen {
        let listener = std::net::TcpListener::bind(listen)
            .with_context(|| format!("bind receipts API on {listen}"))?;
        spawn_api(
            listener,
            router(ApiState {
                receipts,
                sessions: runtime,
            }),
        )?;
    }
    tracing::info!(
        "receipts on: api={} stage={} layers={}..{}",
        options
            .listen
            .map_or_else(|| "none".to_owned(), |listen| listen.to_string()),
        config.stage_index,
        config.layer_start,
        config.layer_end
    );
    Ok(())
}

/// The receipts route a node serves on its OpenAI surface, for clients that
/// reach it only there, such as a Mesh peer: `POST /v1/receipts/open`.
/// It opens only bound receipts, for the one challenge the client's VRF
/// draws, so nobody but the client reads a turn's activations. `None`
/// when the runtime issues no receipts.
pub fn client_router(runtime: &Arc<Mutex<RuntimeState>>) -> Option<Router> {
    let receipts = lock_runtime(runtime).receipts().cloned()?;
    Some(client_routes(ApiState {
        receipts,
        sessions: runtime.clone(),
    }))
}

fn client_routes(state: ApiState) -> Router {
    Router::new()
        .route("/v1/receipts/open", post(bound_open_handler))
        .route("/v1/receipts/rows", post(bound_rows_handler))
        .with_state(state)
}

/// Openings carry prompt-derived activations; refuse a public bind unless
/// the operator asked for it.
fn check_listen(options: &ReceiptsOptions) -> Result<()> {
    if let Some(listen) = options.listen
        && !listen.ip().is_loopback()
        && !options.allow_remote
    {
        bail!(
            "receipts API on {listen} would expose opened activations to the network; \
             bind a loopback address or pass --receipts-allow-remote"
        );
    }
    Ok(())
}

fn spawn_api(listener: std::net::TcpListener, router: Router) -> Result<()> {
    listener.set_nonblocking(true)?;
    std::thread::Builder::new()
        .name("skippy-receipts-api".into())
        .spawn(move || serve(listener, router))
        .context("spawn receipts API thread")?;
    Ok(())
}

fn serve(listener: std::net::TcpListener, router: Router) {
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
        if let Err(error) = axum::serve(listener, router).await {
            tracing::error!("receipts API stopped: {error}");
        }
    });
}

fn router(state: ApiState) -> Router {
    Router::new()
        .route("/receipts/commit", post(commit_handler))
        .route("/receipts/open", post(open_handler))
        .route("/receipts/rows", post(rows_handler))
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
        state
            .receipts
            .commit_session(state.sessions.as_ref(), &request)
    })
    .await
    .map(Json)
}

/// Answers with the opening's bytes and, for a bound receipt, the node's
/// signature of exactly those bytes.
async fn open_handler(
    State(state): State<ApiState>,
    Json(challenge): Json<Challenge>,
) -> Result<Response, ApiError> {
    opening_response(state, challenge).await
}

/// [`open_handler`] for a bound receipt only. The body may also name the
/// `model`, which a Mesh node routes the request by.
async fn bound_open_handler(
    State(state): State<ApiState>,
    Json(challenge): Json<Challenge>,
) -> Result<Response, ApiError> {
    match state.receipts.receipt(&challenge.receipt_id) {
        Some(Ok(receipt)) if receipt.binding.is_some() => opening_response(state, challenge).await,
        Some(Ok(_)) => Err(ApiError(
            StatusCode::FORBIDDEN,
            anyhow!(
                "receipt {} is not bound to a client key",
                challenge.receipt_id
            ),
        )),
        Some(Err(reason)) => Err(ApiError(
            StatusCode::BAD_REQUEST,
            anyhow!("turn {} got no receipt: {reason}", challenge.receipt_id),
        )),
        None => Err(ApiError(
            StatusCode::NOT_FOUND,
            anyhow!("no receipt {}", challenge.receipt_id),
        )),
    }
}

async fn opening_response(state: ApiState, challenge: Challenge) -> Result<Response, ApiError> {
    let (bytes, signature) = blocking(move || {
        let opening = state.receipts.open(&challenge)?;
        let bytes = serde_json::to_vec(&opening)?;
        let signature = state.receipts.sign_opening(&challenge.receipt_id, &bytes)?;
        Ok((bytes, signature))
    })
    .await?;
    signed_response(bytes, signature)
}

/// A JSON body with the node's signature of it, when it signed it.
fn signed_response(bytes: Vec<u8>, signature: Option<String>) -> Result<Response, ApiError> {
    let mut response = (
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        bytes,
    )
        .into_response();
    if let Some(signature) = signature {
        response.headers_mut().insert(
            turn::OPENING_SIGNATURE_HEADER,
            signature
                .parse()
                .context("a hex signature is a header value")?,
        );
    }
    Ok(response)
}

/// Answers with the rows' bytes and, for a bound receipt, the node's
/// signature of exactly those bytes.
async fn rows_handler(
    State(state): State<ApiState>,
    Json(challenge): Json<BlockChallenge>,
) -> Result<Response, ApiError> {
    rows_response(state, challenge).await
}

/// [`rows_handler`] for a bound receipt that was challenged.
async fn bound_rows_handler(
    State(state): State<ApiState>,
    Json(challenge): Json<BlockChallenge>,
) -> Result<Response, ApiError> {
    match state.receipts.receipt(&challenge.receipt_id) {
        Some(Ok(receipt)) if receipt.binding.is_some() => rows_response(state, challenge).await,
        _ => Err(ApiError(
            StatusCode::FORBIDDEN,
            anyhow!(
                "receipt {} is not bound to a client key",
                challenge.receipt_id
            ),
        )),
    }
}

async fn rows_response(state: ApiState, challenge: BlockChallenge) -> Result<Response, ApiError> {
    let (bytes, signature) = blocking(move || {
        let rows = state.receipts.rows(&challenge)?;
        let bytes = serde_json::to_vec(&rows)?;
        let signature = state.receipts.sign_rows(&challenge.receipt_id, &bytes)?;
        Ok((bytes, signature))
    })
    .await?;
    signed_response(bytes, signature)
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

/// The positions a turn receipts: what it recorded, up to the last token it
/// fed. A verify window computes drafts past that when the turn stops inside
/// it, and the turn rejected them.
fn turn_positions(recorded: std::ops::Range<u32>, turn: &TurnTokens) -> std::ops::Range<u32> {
    let fed_end = turn
        .sampled
        .checked_sub(1)
        .map(|fed| turn.generation_start.saturating_add(fed));
    match fed_end {
        Some(end) if recorded.start < end && end < recorded.end => recorded.start..end,
        _ => recorded,
    }
}

impl StageReceipts {
    fn new(
        stage_index: u32,
        layers: std::ops::Range<u32>,
        backend: String,
        downstream: Option<Box<dyn downstream::Downstream>>,
        claimed_model: String,
        signing_key: ed25519_dalek::SigningKey,
    ) -> Result<Self> {
        Ok(Self {
            // the last stage, with no stage after it, holds the final layer
            reports: report_queue::ReportQueue::new(Arc::new(
                Recorder::new(stage_index, layers, backend).holding_output(downstream.is_none()),
            )),
            replayer: OnceLock::new(),
            downstream,
            claimed_model,
            signing_key,
            issued: Mutex::new(VecDeque::new()),
            challenged: Mutex::new(HashMap::new()),
            opened: Mutex::new(HashMap::new()),
        })
    }

    fn set_replayer(&self, replayer: Box<dyn Replayer + Send + Sync>) {
        let _ = self.replayer.set(replayer);
    }

    /// Pins the cells of this stage's recent turns, before the session
    /// that wrote them frees them, so their attention rows stay readable for
    /// the challenge window. A stage that cannot pin still opens a turn
    /// whose cells nothing wrote since.
    fn pin_recent_turns(&self) {
        let Some(replayer) = self.replayer.get() else {
            return;
        };
        let cells = self.reports.recorder().pinned_cells(Instant::now());
        if let Err(error) = replayer.pin_cells(&cells) {
            tracing::debug!("receipts: cannot pin the cells of recent turns: {error:#}");
        }
    }

    /// [`Self::commit_chain`] for the native sequence of the request's session.
    fn commit_session(
        &self,
        sessions: &dyn SessionSequences,
        request: &CommitRequest,
    ) -> Result<Vec<StageCommitment>> {
        let seq = sessions
            .sequence(&request.session)
            .with_context(|| format!("no native sequence for session {}", request.session))?;
        self.commit_chain(request, seq)
    }

    /// Commits this stage and every later one.
    fn commit_chain(&self, request: &CommitRequest, seq: i32) -> Result<Vec<StageCommitment>> {
        self.reports.wait_applied();
        let committed = self.reports.recorder().commit(
            &request.receipt_id,
            seq,
            request.position_start..request.position_end,
        );
        // the sequence's next ubatch starts its next turn
        self.reports.arm_state(seq);
        let mut stages = vec![committed?];
        if !request.challengeable {
            self.reports.recorder().release_pin(&request.receipt_id);
        }
        self.pin_recent_turns();
        if let Some(downstream) = &self.downstream {
            stages.extend(downstream.commit(request)?);
        }
        Ok(stages)
    }

    /// Stage 0, after a turn: commits the whole chain and keeps the
    /// receipt, or why there is none.
    fn finish_turn(
        &self,
        receipt_id: &str,
        session: &str,
        seq: i32,
        turn: TurnTokens,
        request: Option<ReceiptRequest>,
    ) -> Result<Receipt> {
        let sampler = turn.sampler.clone();
        let result = self
            .commit_turn(receipt_id, session, seq, turn, request.is_some())
            .map(|mut receipt| {
                if let Some(request) = request {
                    let binding = BindingRequest {
                        client_key: request.client_key,
                        parent: request.parent,
                        runtime: runtime_identity(),
                        sampler,
                    };
                    turn::bind(&mut receipt, &binding, &self.signing_key);
                }
                receipt
            });
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
        let mut issued = lock(&self.issued);
        issued.push_back((receipt_id.to_owned(), result));
        while issued.len() > ISSUED_RECEIPTS {
            if let Some((old, _)) = issued.pop_front() {
                lock(&self.challenged).remove(&old);
                lock(&self.opened).remove(&old);
            }
        }
    }

    fn commit_turn(
        &self,
        receipt_id: &str,
        session: &str,
        seq: i32,
        turn: TurnTokens,
        challengeable: bool,
    ) -> Result<Receipt> {
        self.reports.wait_applied();
        let recorder = self.reports.recorder();
        let Some(recorded) = recorder.recorded_range(seq) else {
            // the sequence's next ubatch still starts its next turn
            self.reports.arm_state(seq);
            match recorder.last_error() {
                Some(reason) => bail!("the turn recorded nothing: {reason}"),
                None => bail!("the turn recorded nothing"),
            }
        };
        let positions = turn_positions(recorded, &turn);
        let request = CommitRequest {
            receipt_id: receipt_id.to_string(),
            session: session.to_string(),
            position_start: positions.start,
            position_end: positions.end,
            challengeable,
        };
        let stages = self.commit_chain(&request, seq)?;
        // a turn that continues a cached prefix receipts only what it computed
        let generation_start = turn.generation_start.clamp(positions.start, positions.end);
        Ok(Receipt {
            version: RECEIPT_VERSION,
            receipt_id: receipt_id.to_string(),
            model: self.claimed_model.clone(),
            position_start: positions.start,
            position_end: positions.end,
            tokens: recorder.turn_tokens(receipt_id)?,
            generation_start,
            final_token: turn.final_token,
            stages,
            binding: None,
        })
    }

    /// The receipt of a turn, or why it got none; `None` for an unknown id.
    fn receipt(&self, id: &str) -> Option<Result<Receipt, String>> {
        lock(&self.issued)
            .iter()
            .rev()
            .find(|(receipt_id, _)| receipt_id == id)
            .map(|(_, result)| result.clone())
    }

    /// The binding of a bound receipt this stage issued.
    fn binding(&self, id: &str) -> Option<TurnBinding> {
        lock(&self.issued)
            .iter()
            .rev()
            .find(|(receipt_id, _)| receipt_id == id)
            .and_then(|(_, result)| result.as_ref().ok()?.binding.clone())
    }

    /// Stage 0: a bound receipt is opened for one challenge, drawn from the
    /// VRF of its turn root under the client's key. Later stages trust the
    /// stage that forwards to them.
    fn admit_challenge(&self, challenge: &Challenge) -> Result<()> {
        let Some(binding) = self.binding(&challenge.receipt_id) else {
            return Ok(());
        };
        let proof = challenge
            .proof
            .as_deref()
            .context("the receipt is bound to a client key; the challenge needs its VRF proof")?;
        let root = hex_array(&binding.turn_root, "the turn root")?;
        let client_key = vrf::key_from_hex(&binding.client_key)?;
        vrf::verify(
            &client_key,
            &turn::challenge_input(&root),
            &vrf::proof_from_hex(proof)?,
        )
        .context("the challenge's VRF proof")?;
        let mut challenged = lock(&self.challenged);
        match challenged.get(&challenge.receipt_id) {
            Some(first) if first != challenge => {
                bail!("receipt {} was already challenged", challenge.receipt_id)
            }
            Some(_) => {}
            None => {
                challenged.insert(challenge.receipt_id.clone(), challenge.clone());
            }
        }
        Ok(())
    }

    /// Stage 0's signature of an opening of a bound receipt it issued.
    fn sign_opening(&self, receipt_id: &str, bytes: &[u8]) -> Result<Option<String>> {
        let Some(binding) = self.binding(receipt_id) else {
            return Ok(None);
        };
        let root = hex_array(&binding.turn_root, "the turn root")?;
        lock(&self.opened).insert(receipt_id.to_owned(), *blake3::hash(bytes).as_bytes());
        Ok(Some(turn::sign_opening(&self.signing_key, &root, bytes)))
    }

    /// Stage 0's signature of attention rows of a bound receipt it issued.
    fn sign_rows(&self, receipt_id: &str, bytes: &[u8]) -> Result<Option<String>> {
        let Some(binding) = self.binding(receipt_id) else {
            return Ok(None);
        };
        let root = hex_array(&binding.turn_root, "the turn root")?;
        Ok(Some(turn::sign_rows(&self.signing_key, &root, bytes)))
    }

    /// Stage 0: a bound receipt's rows are read only with the client's VRF
    /// proof of the opening this stage signed. Later stages trust the stage
    /// that forwards to them.
    fn admit_rows(&self, challenge: &BlockChallenge) -> Result<()> {
        let Some(binding) = self.binding(&challenge.receipt_id) else {
            return Ok(());
        };
        let opened = lock(&self.opened)
            .get(&challenge.receipt_id)
            .copied()
            .with_context(|| format!("receipt {} was not opened", challenge.receipt_id))?;
        let proof = challenge
            .proof
            .as_deref()
            .context("the receipt is bound to a client key; the rows need its VRF proof")?;
        let client_key = vrf::key_from_hex(&binding.client_key)?;
        vrf::verify(
            &client_key,
            &turn::rows_input_of_digest(&opened),
            &vrf::proof_from_hex(proof)?,
        )
        .context("the rows' VRF proof")?;
        Ok(())
    }

    /// Opens the attention rows on this stage and every later one. A bound
    /// receipt's rows are opened only after its challenge was.
    fn rows(&self, challenge: &BlockChallenge) -> Result<RowsOpening> {
        self.admit_rows(challenge)?;
        let replayer = self
            .replayer
            .get()
            .context("this stage cannot replay turns")?;
        self.reports.wait_applied();
        let mut stages = vec![
            self.reports
                .recorder()
                .open_rows(challenge, replayer.as_ref())?,
        ];
        if let Some(downstream) = &self.downstream {
            stages.extend(downstream.rows(challenge)?.stages);
        }
        Ok(RowsOpening {
            receipt_id: challenge.receipt_id.clone(),
            stages,
        })
    }

    /// Opens the challenge on this stage and every later one.
    fn open(&self, challenge: &Challenge) -> Result<Opening> {
        self.admit_challenge(challenge)?;
        let recorder = self.reports.recorder();
        if self.downstream.is_none()
            && let Some(pick) = challenge
                .picks
                .iter()
                .find(|p| p.layer >= recorder.layers().end)
        {
            bail!("no stage holds layer {}", pick.layer);
        }
        let replayer = self
            .replayer
            .get()
            .context("this stage cannot replay turns")?;
        self.reports.wait_applied();
        let mut stages = vec![recorder.open(challenge, replayer.as_ref())?];
        if let Some(downstream) = &self.downstream {
            stages.extend(downstream.open(challenge)?.stages);
        }
        Ok(Opening {
            receipt_id: challenge.receipt_id.clone(),
            stages,
        })
    }
}

/// The grammar a turn's chat metadata constrains it with, if any.
fn grammar(chat_metadata: Option<&str>) -> Option<String> {
    let metadata: serde_json::Value = serde_json::from_str(chat_metadata?).ok()?;
    let grammar = metadata.get("grammar")?.as_str()?;
    (!grammar.is_empty()).then(|| grammar.to_owned())
}

/// Whether a turn took the best logit at every token: no temperature,
/// penalty, logit bias or grammar changed the choice. The receipt says so,
/// and a client checks greedy turns against the output head.
pub(crate) fn greedy_sampling(
    sampling: &skippy_runtime::SamplingConfig,
    chat_metadata: Option<&str>,
) -> bool {
    let unconstrained = grammar(chat_metadata).is_none();
    let best_logit = !sampling.enabled
        || (sampling.temperature <= 0.0
            && sampling.presence_penalty == 0.0
            && sampling.frequency_penalty == 0.0
            && sampling.repeat_penalty == 1.0
            && sampling.dry.multiplier == 0.0
            && sampling.logit_bias.is_empty());
    unconstrained && best_logit && reasoning_budget_tokens(sampling.reasoning_budget).is_none()
}

/// How a turn chose its tokens, as its receipt states it: [`turn::GREEDY`]
/// when it took the best logit at every token, else the sampler's
/// parameters and seed, so a node that samples otherwise than it says signs
/// a false statement. A grammar is named by its BLAKE3 hash.
pub(crate) fn sampler_statement(
    sampling: &skippy_runtime::SamplingConfig,
    chat_metadata: Option<&str>,
) -> String {
    if greedy_sampling(sampling, chat_metadata) {
        return turn::GREEDY.to_owned();
    }
    let grammar =
        grammar(chat_metadata).map(|grammar| blake3::hash(grammar.as_bytes()).to_hex().to_string());
    let s = sampling;
    turn::sampled(&serde_json::json!({
        "seed": s.seed,
        "samplers": s.samplers,
        "temperature": s.temperature,
        "dynatemp_range": s.dynatemp_range,
        "dynatemp_exponent": s.dynatemp_exponent,
        "top_k": s.top_k,
        "top_p": s.top_p,
        "min_p": s.min_p,
        "typical_p": s.typical_p,
        "top_nsigma": s.top_nsigma,
        "presence_penalty": s.presence_penalty,
        "frequency_penalty": s.frequency_penalty,
        "repeat_penalty": s.repeat_penalty,
        "penalty_last_n": s.penalty_last_n,
        "dry": {
            "multiplier": s.dry.multiplier,
            "base": s.dry.base,
            "allowed_length": s.dry.allowed_length,
            "penalty_last_n": s.dry.penalty_last_n,
            "sequence_breakers": s.dry.sequence_breakers,
        },
        "xtc": { "probability": s.xtc.probability, "threshold": s.xtc.threshold },
        "mirostat": {
            "mode": s.mirostat_mode,
            "entropy": s.mirostat_entropy,
            "learning_rate": s.mirostat_learning_rate,
        },
        "logit_bias": s.logit_bias.iter().map(|b| (b.token_id, b.bias)).collect::<Vec<_>>(),
        "ignore_eos": s.ignore_eos,
        "grammar": grammar,
        "reasoning_budget": reasoning_budget_tokens(s.reasoning_budget),
    }))
}

/// The tokens a reasoning budget allows, `None` when it is unrestricted. A
/// budget can force the end of reasoning, so it is part of the statement.
fn reasoning_budget_tokens(budget: skippy_runtime::ReasoningBudget) -> Option<i64> {
    use skippy_runtime::ReasoningBudget;
    match budget {
        ReasoningBudget::Unrestricted => None,
        ReasoningBudget::Explicit(tokens) | ReasoningBudget::Capped(tokens) => {
            Some(i64::from(tokens))
        }
        ReasoningBudget::Resolved(tokens) => (tokens >= 0).then_some(i64::from(tokens)),
    }
}

/// Stage 0 hook at the end of a turn. Failures are logged; they never fail
/// the user's request. Returns the receipt when the request asked for a bound
/// one, for the response to carry.
pub(crate) fn finish_stage_zero_turn(
    runtime: &Mutex<RuntimeState>,
    frontend_request_id: Option<[u8; 16]>,
    receipt_request: Option<ReceiptRequest>,
    request_id: u64,
    session: &str,
    turn: TurnTokens,
) -> Option<serde_json::Value> {
    let (receipts, seq) = {
        let state = lock_runtime(runtime);
        let receipts = state.receipts().cloned()?;
        (receipts, state.receipt_sequence(session))
    };
    let receipt_id = frontend_request_id
        .map(|bytes| uuid::Uuid::from_bytes(bytes).hyphenated().to_string())
        .unwrap_or_else(|| format!("{request_id:016x}"));
    let bound = receipt_request.is_some();
    let result = seq
        .with_context(|| format!("no native sequence for session {session}"))
        .and_then(|seq| receipts.finish_turn(&receipt_id, session, seq, turn, receipt_request));
    match result {
        Ok(receipt) => {
            tracing::info!(
                "receipt {} committed: positions {}..{}, {} stages",
                receipt.receipt_id,
                receipt.position_start,
                receipt.position_end,
                receipt.stages.len()
            );
            bound.then(|| serde_json::to_value(&receipt).ok()).flatten()
        }
        Err(error) => {
            tracing::warn!("receipt {receipt_id} not committed: {error:#}");
            None
        }
    }
}

mod downstream;
mod native;
mod report_queue;

pub(crate) use downstream::serve_connection;

#[cfg(test)]
#[path = "receipts/tests.rs"]
mod tests;

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
