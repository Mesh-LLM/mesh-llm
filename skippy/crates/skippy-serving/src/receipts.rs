//! Opt-in verifiable inference receipts for a serving stage.
//!
//! With receipts on, the stage's native receipt recorder feeds a
//! [`Recorder`] on a worker thread: leaf hashes per position, and a replay
//! log per turn of each sequence.
//! When stage 0 finishes a turn, it commits its own leaves, asks the next
//! stage to commit the same session and positions, and keeps the assembled
//! [`Receipt`] under the request's `x-request-id`. A challenge is opened by
//! replaying the turn in a scratch context of the stage model. Every stage
//! serves a small HTTP API on its receipts address ([`http_api`]):
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
//! root, which the stage signs with its node key. Every stage then opens
//! one challenge per bound receipt, and only with a VRF proof of the turn
//! root under that key, so the client cannot choose what it challenges
//! ([`admission`]).
//!
//! Openings carry activations derived from the prompt, so the API binds a
//! loopback address unless the operator allows a remote one.
//!
//! The stage owns its receipts: the runtime state holds them, and they go
//! with it when the model is unloaded, with their threads. The native
//! recorder, the HTTP routes and the stage connections only point at them.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use axum::Router;
use skippy_inference_api::ReceiptRequest;
use skippy_protocol::StageConfig;
use skippy_receipts::attention::opening::{BlockChallenge, RowsOpening};
use skippy_receipts::commitment::hex_array;
use skippy_receipts::recorder::{KvStreams, Recorder, Replayer};
use skippy_receipts::turn::{self, BindingRequest};
use skippy_receipts::vrf;
use skippy_receipts::wire::{Challenge, Opening, RECEIPT_VERSION, Receipt, StageCommitment};

use crate::runtime_state::RuntimeState;
pub use crate::runtime_state::SessionTurn;
use crate::runtime_state::panic_recovery::lock_runtime;

mod admission;
mod audited_backend;
mod audited_weights;
mod chain_server;
mod downstream;
mod http_api;
mod kept_turns;
mod kv_cache;
mod native;
mod opening_queue;
mod refusal;
mod reply_limits;
mod report_queue;
mod sampler_statement;
mod upkeep;

use downstream::{CommitRequest, ForwardedOpen, ForwardedRows};
use kept_turns::{KeptTurns, LaterStages, Outcome, Reservation};
use reply_limits::ReplyLimits;
pub(crate) use sampler_statement::{
    draws_from_the_chain, make_draw_replayable, prefer_receipt_seed, sampler_statement,
};

/// Request header of `POST /receipts/open` between stages: the hex turn
/// root of a bound receipt, which the challenge's proof is over.
pub const TURN_ROOT_HEADER: &str = "x-receipt-turn-root";
/// Request header of `POST /receipts/rows` between stages: the hex digest
/// of the opening stage 0 signed, which the rows' proof is over.
pub const OPENING_DIGEST_HEADER: &str = "x-receipt-opening-digest";

/// Checks that a turn computed the end of its prompt. A stage that issues
/// receipts does not reuse generated tokens, so a cached prefix holds only
/// tokens a request fed: the audit checks no sampled token below the turn.
/// And the prefix lookup leaves out the last prompt token (`prompt[..len -
/// 1]` in `backend.rs` and `token_generation.rs`), so the turn samples its
/// first token from a position it computed. A client refuses a receipt that
/// does not (`skippy_receipts::binding::verify_shape`).
fn check_turn_start(generation_start: u32, positions: &std::ops::Range<u32>) -> Result<()> {
    if generation_start < positions.start {
        bail!(
            "the turn's cached prefix of {} positions is longer than its prompt of {} tokens",
            positions.start,
            generation_start
        );
    }
    if generation_start == positions.start && !positions.is_empty() {
        bail!(
            "the turn's cached prefix holds all of its prompt of {generation_start} tokens, so it \
             samples its first token from a position no receipt covers"
        );
    }
    Ok(())
}

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

#[derive(Clone)]
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

impl std::fmt::Debug for ReceiptsOptions {
    /// The signing key is the node's secret: it is never written out.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let signing_key = self.signing_key.map(|_| "<redacted>");
        f.debug_struct("ReceiptsOptions")
            .field("listen", &self.listen)
            .field("allow_remote", &self.allow_remote)
            .field("downstream", &self.downstream)
            .field("claimed_model", &self.claimed_model)
            .field("signing_key", &signing_key)
            .finish()
    }
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

/// Resolves a session id to the turn it computes, or computed last.
pub trait SessionTurns: Send + Sync {
    fn turn(&self, session: &str) -> Option<SessionTurn>;
}

impl SessionTurns for Mutex<RuntimeState> {
    fn turn(&self, session: &str) -> Option<SessionTurn> {
        lock_runtime(self).receipt_turn(session)
    }
}

/// What a stage replays turns with, and what it holds for that between
/// replays.
trait StageReplayer: Replayer + Send + Sync {
    /// Frees what the replayer holds for replays once it has been idle,
    /// and returns when to ask again; `None` while it holds nothing.
    fn release_idle(&self, _now: Instant) -> Option<Instant> {
        None
    }
}

pub struct StageReceipts {
    /// The backend the stage computes on, as its receipts name it.
    backend: String,
    reports: report_queue::ReportQueue,
    replayer: OnceLock<Box<dyn StageReplayer>>,
    /// The receipts of the stages after this one; `None` on the last stage.
    downstream: Option<Box<dyn downstream::Downstream>>,
    limits: ReplyLimits,
    claimed_model: String,
    signing_key: ed25519_dalek::SigningKey,
    /// By receipt id: what became of the turn, and what admitting a
    /// challenge of it takes.
    kept: Mutex<KeptTurns>,
    /// The receipts connections this stage serves.
    connections: Arc<chain_server::ConnectionSlots>,
    /// When the next pinned turn's pin runs out, so cells are pinned anew
    /// then.
    pins_due: Mutex<Option<Instant>>,
    /// The openings and rows requests waiting for the stage.
    queue: opening_queue::OpeningQueue,
    upkeep: OnceLock<upkeep::Upkeep>,
    /// The stage's receipts API, which stops with the stage.
    api: OnceLock<http_api::ApiHandle>,
}

/// How this stage reaches the receipts of the next one: over `url`, its
/// receipts API, when one is given, else over the stage connection when
/// there is a next stage.
fn next_stage(
    url: Option<&str>,
    config: &StageConfig,
) -> Result<Option<Box<dyn downstream::Downstream>>> {
    use downstream::{COMMIT_TIMEOUT, OPEN_TIMEOUT};
    Ok(match url {
        Some(url) => Some(Box::new(downstream::HttpDownstream::new(
            url,
            COMMIT_TIMEOUT,
            OPEN_TIMEOUT,
        )?)),
        None if config.downstream.is_some() => Some(Box::new(downstream::StageConnection::new(
            config,
            COMMIT_TIMEOUT,
            OPEN_TIMEOUT,
        ))),
        None => None,
    })
}

/// Installs the receipt recorder, attaches receipts to the runtime, and
/// serves the receipts API.
///
/// A stage no audit covers gets none of that: one with a quantized KV
/// cache ([`kv_cache`]), on a backend without rounding profiles, such as
/// Vulkan or Metal ([`audited_backend`]), or with weights of a type that
/// has no rounding profile on its backend, such as f16 on CUDA
/// ([`audited_weights`]). It issues no receipts and serves as if they were
/// off. The reason is logged here, once.
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
    let unaudited = kv_cache::unaudited(config)
        .or_else(|| audited_backend::unaudited(&backend))
        .or_else(|| audited_weights::unaudited(config, &backend));
    if let Some(reason) = unaudited {
        tracing::warn!(
            stage = config.stage_index,
            "receipts stay off for this stage, which serves without them: {reason}"
        );
        // an operator who asked for receipts sees why the stage issues none
        let _ = skippy_events::diagnostics::emit(
            skippy_events::diagnostics::ServingDiagnostic::Warning {
                message: "Receipts stay off for this model stage".to_string(),
                context: Some(format!(
                    "stage_id={} model_id={} reason={reason}",
                    config.stage_id, config.model_id
                )),
            },
        );
        return Ok(());
    }
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
    // the residual width, which every stage of the chain shares
    let width = lock_runtime(&runtime)
        .output_activation_boundary()
        .and_then(|boundary| boundary.primary_token_width("output").ok())
        .and_then(|width| usize::try_from(width).ok());
    let receipts = Arc::new(StageReceipts::new(
        config.stage_index,
        config.layer_start..config.layer_end,
        backend,
        next_stage(options.downstream.as_deref(), config)?,
        options.claimed_model,
        signing_key,
        ReplyLimits::for_width(width),
    ));
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
        // pins for challenges are a share of a stream's cells
        receipts
            .reports
            .recorder()
            .set_pin_budget(config.ctx_size / streams.max(1));
        let recurrent = state
            .model
            .set_receipt_recorder(Arc::new(native::RecorderSink::new(&receipts)))
            .context("install the receipt recorder")?;
        receipts.reports.recorder().set_recurrent(recurrent);
        // the runtime state owns the receipts from here
        state.set_receipts(receipts.clone());
    }
    receipts.start_upkeep()?;
    if let Some(listen) = options.listen {
        let listener = std::net::TcpListener::bind(listen)
            .with_context(|| format!("bind receipts API on {listen}"))?;
        let state = http_api::ApiState::of(&receipts, &runtime);
        let api = http_api::spawn_api(listener, http_api::router(state))?;
        let _ = receipts.api.set(api);
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
/// when the runtime issues no receipts. The routes do not keep the runtime
/// loaded.
pub fn client_router(runtime: &Arc<Mutex<RuntimeState>>) -> Option<Router> {
    let receipts = lock_runtime(runtime).receipts().cloned()?;
    Some(http_api::client_routes(http_api::ApiState::of(
        &receipts, runtime,
    )))
}

/// The backend the runtime's receipts name, such as `CPU` or `CUDA0`: the
/// one a client picks rounding profiles by. `None` when the runtime issues
/// no receipts.
pub fn backend_name(runtime: &Arc<Mutex<RuntimeState>>) -> Option<String> {
    Some(lock_runtime(runtime).receipts()?.backend.clone())
}

/// Whether the runtime issues receipts and its stage keeps recurrent state
/// (a hybrid model). Such a stage issues no receipt for a turn that
/// continues a cached prefix, so a caller that owes a receipt for every
/// turn must not offer them.
pub fn keeps_recurrent_state(runtime: &Arc<Mutex<RuntimeState>>) -> bool {
    lock_runtime(runtime)
        .receipts()
        .is_some_and(|receipts| receipts.reports.recorder().is_recurrent())
}

/// Serves a receipts connection of the stage's binary listener, which an
/// earlier stage opened with `first`, with the receipts the runtime holds
/// now; a runtime that issues none says so once ([`chain_server`]).
pub(crate) fn serve_stage_connection(
    runtime: &Mutex<RuntimeState>,
    upstream: std::net::TcpStream,
    first: skippy_protocol::binary::StageWireMessage,
) -> Result<()> {
    let receipts = lock_runtime(runtime).receipts().cloned();
    chain_server::serve_connection(receipts, runtime, upstream, first)
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

/// The positions a turn receipts: what it recorded, up to the last token it
/// fed. A verify window computes drafts past that when the turn stops inside
/// it, and the turn rejected them.
///
/// A turn that fed a position it did not record computed a ubatch the
/// recorder could not receipt at its end, such as one of embeddings, and
/// gets no receipt: one that stopped short of the position its last token
/// was sampled from would not show how the node chose it.
fn turn_positions(
    recorded: std::ops::Range<u32>,
    turn: &TurnTokens,
) -> Result<std::ops::Range<u32>> {
    let fed_end = turn
        .sampled
        .checked_sub(1)
        .map(|fed| turn.generation_start.saturating_add(fed));
    Ok(match fed_end {
        Some(end) if end > recorded.end => bail!(
            "the turn fed positions up to {end}, and the stage recorded them up to {} only",
            recorded.end
        ),
        Some(end) if recorded.start < end && end < recorded.end => recorded.start..end,
        _ => recorded,
    })
}

impl StageReceipts {
    fn new(
        stage_index: u32,
        layers: std::ops::Range<u32>,
        backend: String,
        downstream: Option<Box<dyn downstream::Downstream>>,
        claimed_model: String,
        signing_key: ed25519_dalek::SigningKey,
        limits: ReplyLimits,
    ) -> Self {
        Self {
            // the last stage, with no stage after it, holds the final layer
            reports: report_queue::ReportQueue::new(Arc::new(
                Recorder::new(stage_index, layers, backend.clone())
                    .holding_output(downstream.is_none()),
            )),
            backend,
            replayer: OnceLock::new(),
            downstream,
            limits,
            claimed_model,
            signing_key,
            kept: Mutex::new(KeptTurns::default()),
            connections: Arc::default(),
            pins_due: Mutex::new(None),
            queue: opening_queue::OpeningQueue::default(),
            upkeep: OnceLock::new(),
            api: OnceLock::new(),
        }
    }

    /// Receipts of a stage with no native recorder, for tests of the code
    /// that serves beside them.
    #[cfg(test)]
    pub(crate) fn for_test(stage_index: u32) -> Self {
        Self::new(
            stage_index,
            0..1,
            "test".into(),
            None,
            "model.gguf".into(),
            ed25519_dalek::SigningKey::from_bytes(&[7; 32]),
            ReplyLimits::for_width(None),
        )
    }

    fn set_replayer(&self, replayer: Box<dyn StageReplayer>) {
        let _ = self.replayer.set(replayer);
    }

    /// Starts the thread of the stage's timed chores, which ends when the
    /// receipts are dropped.
    fn start_upkeep(self: &Arc<Self>) -> Result<()> {
        let receipts = Arc::downgrade(self);
        let upkeep = upkeep::Upkeep::start(move |now| Some(receipts.upgrade()?.run_upkeep(now)))
            .context("spawn receipts upkeep thread")?;
        let _ = self.upkeep.set(upkeep);
        Ok(())
    }

    fn poke_upkeep(&self) {
        if let Some(upkeep) = self.upkeep.get() {
            upkeep.poke();
        }
    }

    /// The stage's timed chores at `now`; returns when they are next due.
    fn run_upkeep(&self, now: Instant) -> Option<Instant> {
        if lock(&self.pins_due).is_some_and(|due| due <= now) {
            self.pin_turns(now);
        }
        let pins = *lock(&self.pins_due);
        let scratch = self
            .replayer
            .get()
            .and_then(|replayer| replayer.release_idle(now));
        [pins, scratch].into_iter().flatten().min()
    }

    /// A session starts a turn: the recorder is told before the session's
    /// first ubatch. The caller holds the stage idle.
    pub(crate) fn begin_turn(&self, turn: SessionTurn) {
        self.reports.begin_turn(turn.seq, turn.turn);
    }

    /// Pins the cells of the turns a client may still challenge at `now`,
    /// in place of those pinned before, and notes when the next pin runs
    /// out. A stage that cannot pin still opens a turn whose cells nothing
    /// wrote since.
    fn pin_turns(&self, now: Instant) {
        let Some(replayer) = self.replayer.get() else {
            return;
        };
        let recorder = self.reports.recorder();
        let cells = recorder.pinned_cells(now);
        match replayer.pin_cells(&cells) {
            Ok(()) => tracing::debug!("receipts: {} KV cells pinned", cells.len()),
            Err(error) => {
                recorder.pins_not_applied();
                tracing::debug!("receipts: cannot pin the cells of recent turns: {error:#}");
            }
        }
        *lock(&self.pins_due) = recorder.next_pin_expiry(now);
    }

    /// Pins the cells of this stage's recent turns after a pin was made or
    /// given up, before the session that wrote them frees them, so their
    /// attention rows stay readable for the challenge window and no longer.
    fn pin_recent_turns(&self) {
        self.pin_turns(Instant::now());
        // the upkeep pins anew when the next pin runs out
        self.poke_upkeep();
    }

    /// Commits a session's turn on this stage and every later one, for the
    /// stage before this one, and keeps a record of it to admit challenges
    /// by.
    fn commit_session(
        &self,
        sessions: &dyn SessionTurns,
        request: &CommitRequest,
    ) -> Result<Vec<StageCommitment>> {
        if request.receipt_id.len() > admission::MAX_RECEIPT_ID_BYTES {
            bail!("the receipt id is longer than a stage keeps");
        }
        let turn = sessions
            .turn(&request.session)
            .with_context(|| format!("no native sequence for session {}", request.session))?;
        let client_key = request
            .client_key
            .as_deref()
            .map(vrf::key_from_hex)
            .transpose()
            .context("the client key of the commit")?;
        let id = &request.receipt_id;
        let reservation = lock(&self.kept).reserve(id, client_key, Instant::now())?;
        let committed = self.commit_chain(request, turn);
        let outcome = match &committed {
            Ok(_) => Outcome::Committed,
            Err(error) => Outcome::Refused(format!("{error:#}")),
        };
        let later = later_stages(request, committed.as_deref().unwrap_or_default());
        self.settle(id, reservation, outcome, None, later)?;
        committed
    }

    /// Settles the record of a turn. A turn whose record cannot be kept has
    /// no receipt, so what the stage committed for it is dropped.
    fn settle(
        &self,
        id: &str,
        reservation: Reservation,
        outcome: Outcome,
        turn_root: Option<[u8; 32]>,
        later: LaterStages,
    ) -> Result<()> {
        let settled =
            lock(&self.kept).settle(id, reservation, outcome, turn_root, later, Instant::now());
        if let Err(kept_turns::Unsettled::Full(_)) = &settled {
            self.forget_turn(id);
        }
        Ok(settled?)
    }

    /// Drops what this stage committed for a turn that got no receipt.
    fn forget_turn(&self, receipt_id: &str) {
        self.reports.recorder().discard(receipt_id);
        self.pin_recent_turns();
    }

    /// Commits this stage and every later one. The later stages'
    /// commitments are kept only if they continue this stage's.
    fn commit_chain(
        &self,
        request: &CommitRequest,
        turn: SessionTurn,
    ) -> Result<Vec<StageCommitment>> {
        self.reports.wait_applied();
        let positions = request.position_start..request.position_end;
        let committed = self.reports.recorder().commit(
            &request.receipt_id,
            turn.seq,
            turn.turn,
            positions.clone(),
        );
        let mut stages = vec![committed?];
        // the sequence's next ubatch starts its next turn; a commit that
        // failed, such as one forged on the stage port, arms nothing
        self.reports.arm_state(turn.seq);
        let pinned = self.admit_pin(request);
        self.pin_recent_turns();
        pinned?;
        if let Some(downstream) = &self.downstream {
            let limit = ReplyLimits::commit(positions.end - positions.start);
            let later = downstream.commit(request, limit).and_then(|later| {
                reply_limits::check_commitments(&stages[0], &later, &positions)?;
                Ok(later)
            });
            match later {
                Ok(later) => stages.extend(later),
                Err(error) => {
                    // no receipt names the turn: nothing will open it
                    self.forget_turn(&request.receipt_id);
                    return Err(error);
                }
            }
        }
        Ok(stages)
    }

    /// Pins the committed turn of `request` while its client can challenge
    /// it. Older pins give way when the turn takes its client key or all
    /// keys past their share of the cache. A turn nobody can challenge
    /// needs no pin.
    fn admit_pin(&self, request: &CommitRequest) -> Result<()> {
        let recorder = self.reports.recorder();
        let Some(key) = request.client_key.as_deref() else {
            recorder.release_pin(&request.receipt_id);
            return Ok(());
        };
        let key = vrf::key_from_hex(key).context("the client key of the commit")?;
        recorder.admit_pin(&request.receipt_id, key, Instant::now())
    }

    /// Stage 0, after a turn: commits the whole chain and keeps the
    /// receipt, or why there is none.
    fn finish_turn(
        &self,
        receipt_id: &str,
        session: &str,
        session_turn: SessionTurn,
        tokens: TurnTokens,
        request: Option<ReceiptRequest>,
    ) -> Result<Receipt> {
        let client_key = request.as_ref().map(|request| request.client_key);
        let reservation = match lock(&self.kept).reserve(receipt_id, client_key, Instant::now()) {
            Ok(reservation) => reservation,
            Err(error) => {
                // the turn commits nothing; its sequence's next ubatch
                // still starts its next turn
                self.reports.arm_state(session_turn.seq);
                return Err(error);
            }
        };
        let sampler = tokens.sampler.clone();
        let result = self
            .commit_turn(receipt_id, session, session_turn, tokens, client_key)
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
        let (outcome, root, later) = match &result {
            Ok(receipt) => {
                let root = receipt
                    .binding
                    .as_ref()
                    .and_then(|binding| hex_array(&binding.turn_root, "the turn root").ok());
                let later = LaterStages {
                    position_start: receipt.position_start,
                    position_end: receipt.position_end,
                    commitments: receipt.stages.get(1..).unwrap_or_default().into(),
                };
                (Outcome::Issued(Arc::new(receipt.clone())), root, later)
            }
            Err(error) => (
                Outcome::Refused(format!("{error:#}")),
                None,
                LaterStages::default(),
            ),
        };
        self.settle(receipt_id, reservation, outcome, root, later)?;
        result
    }

    fn commit_turn(
        &self,
        receipt_id: &str,
        session: &str,
        session_turn: SessionTurn,
        tokens: TurnTokens,
        client_key: Option<[u8; 32]>,
    ) -> Result<Receipt> {
        self.reports.wait_applied();
        let recorder = self.reports.recorder();
        let Some(recorded) = recorder.recorded_range(session_turn.seq, session_turn.turn) else {
            // the sequence's next ubatch still starts its next turn
            self.reports.arm_state(session_turn.seq);
            match recorder.last_error() {
                Some(reason) => bail!("the turn recorded nothing: {reason}"),
                None => bail!("the turn recorded nothing"),
            }
        };
        let positions = match turn_positions(recorded, &tokens) {
            Ok(positions) => positions,
            Err(error) => {
                self.reports.arm_state(session_turn.seq);
                return Err(error);
            }
        };
        check_turn_start(tokens.generation_start, &positions)?;
        let request = CommitRequest {
            receipt_id: receipt_id.to_string(),
            session: session.to_string(),
            position_start: positions.start,
            position_end: positions.end,
            client_key: client_key.map(hex::encode),
        };
        // a turn that commits nothing still starts its sequence's next turn
        let stages = self
            .commit_chain(&request, session_turn)
            .inspect_err(|_| self.reports.arm_state(session_turn.seq))?;
        // a turn that continues a cached prefix receipts only what it computed
        let generation_start = tokens
            .generation_start
            .clamp(positions.start, positions.end);
        Ok(Receipt {
            version: RECEIPT_VERSION,
            receipt_id: receipt_id.to_string(),
            model: self.claimed_model.clone(),
            position_start: positions.start,
            position_end: positions.end,
            tokens: recorder.turn_tokens(receipt_id)?,
            generation_start,
            final_token: tokens.final_token,
            stages,
            binding: None,
        })
    }

    /// An error of the later stages, as this stage passes it on
    /// ([`refusal::from_later_stages`]).
    fn passed_on(&self, error: anyhow::Error, receipt_id: &str) -> anyhow::Error {
        let stage_index = self.reports.recorder().stage_index();
        refusal::from_later_stages(error, receipt_id, stage_index)
    }

    fn replayer(&self) -> Result<&dyn Replayer> {
        let replayer: &dyn StageReplayer = self
            .replayer
            .get()
            .context("this stage cannot replay turns")?
            .as_ref();
        Ok(replayer)
    }

    /// Opens the attention rows on this stage and every later one, once the
    /// request has its turn at the stage ([`opening_queue`]).
    fn rows(
        &self,
        blocks: &BlockChallenge,
        forwarded_opening: Option<[u8; 32]>,
    ) -> Result<RowsOpening> {
        let _slot = self.queue.enter(self.queue_key(&blocks.receipt_id)?)?;
        self.rows_now(blocks, forwarded_opening)
    }

    /// [`Self::rows`] without waiting for a turn, once the request is
    /// admitted ([`Self::admit_rows`]).
    fn rows_now(
        &self,
        blocks: &BlockChallenge,
        forwarded_opening: Option<[u8; 32]>,
    ) -> Result<RowsOpening> {
        let admitted = self.admit_rows(blocks, forwarded_opening)?;
        let replayer = self.replayer()?;
        self.reports.wait_applied();
        let own = self.reports.recorder().open_rows(blocks, replayer);
        // the rows are read: the turn's cells need no pin any more
        self.pin_recent_turns();
        let mut stages = vec![own?];
        if let Some(downstream) = &self.downstream {
            let request = ForwardedRows {
                blocks: blocks.clone(),
                opening: admitted.opening.map(hex::encode),
            };
            let later = downstream
                .rows(&request, self.limits.rows(blocks))
                .map_err(|e| self.passed_on(e, &blocks.receipt_id))?;
            if later.receipt_id != blocks.receipt_id {
                bail!("the next stage opened rows of another receipt");
            }
            let answered = later.stages.iter().map(|stage| stage.stage_index);
            reply_limits::check_stages(&admitted.later, answered)?;
            stages.extend(later.stages);
        }
        Ok(RowsOpening {
            receipt_id: blocks.receipt_id.clone(),
            stages,
        })
    }

    /// [`Self::rows`] for the stage before this one over the stage
    /// connection. The request had its turn at the first stage, so a later
    /// stage does not queue it or turn it away.
    fn rows_forwarded(&self, request: &ForwardedRows) -> Result<RowsOpening> {
        let opening = request
            .opening
            .as_deref()
            .map(|digest| hex_array(digest, "the opening digest"))
            .transpose()?;
        self.rows_now(&request.blocks, opening)
    }

    /// Opens the challenge on this stage and every later one, once it has
    /// its turn at the stage ([`opening_queue`]). A challenge turned away
    /// is not admitted, so the client can send it again.
    fn open(&self, challenge: &Challenge, forwarded_root: Option<[u8; 32]>) -> Result<Opening> {
        let key = self.queue_key(&challenge.receipt_id)?;
        self.pinned_while_opening(&challenge.receipt_id, || {
            let _slot = self.queue.enter(key)?;
            self.open_now(challenge, forwarded_root)
        })
    }

    /// Runs `open`, an opening of the turn `receipt_id`, with the turn's
    /// cells pinned from now until it ends ([`Recorder::hold_for_opening`]).
    /// The opening reads the cells after the replay, which takes seconds
    /// for a long prompt, and the challenge may wait in the queue first: a
    /// pin that ran out meanwhile would let a later request write the cells,
    /// and the stage could no longer open a turn it owes an opening.
    fn pinned_while_opening(
        &self,
        receipt_id: &str,
        open: impl FnOnce() -> Result<Opening>,
    ) -> Result<Opening> {
        let recorder = self.reports.recorder();
        recorder.hold_for_opening(receipt_id, Instant::now());
        self.pin_recent_turns();
        let opened = open();
        recorder.end_opening(receipt_id);
        self.pin_recent_turns();
        opened
    }

    /// [`Self::open`] without waiting for a turn, once the challenge is
    /// admitted ([`Self::admit_challenge`]): only an admitted challenge is
    /// replayed, and counts toward the replays a receipt gets.
    fn open_now(&self, challenge: &Challenge, forwarded_root: Option<[u8; 32]>) -> Result<Opening> {
        let admitted = self.admit_challenge(challenge, forwarded_root)?;
        let replayer = self.replayer()?;
        self.reports.wait_applied();
        let own = self.reports.recorder().open(challenge, replayer);
        // an opened turn is pinned for its rows, or not at all
        self.pin_recent_turns();
        let mut stages = vec![own?];
        self.mark_opened(&challenge.receipt_id);
        if let Some(downstream) = &self.downstream {
            let request = ForwardedOpen {
                challenge: challenge.clone(),
                turn_root: admitted.turn_root.map(hex::encode),
            };
            let limit = self.limits.opening(challenge, &admitted.later);
            let later = downstream
                .open(&request, limit)
                .map_err(|e| self.passed_on(e, &challenge.receipt_id))?;
            if later.receipt_id != challenge.receipt_id {
                bail!("the next stage opened another receipt");
            }
            let answered = later.stages.iter().map(|stage| stage.stage_index);
            reply_limits::check_stages(&admitted.later, answered)?;
            stages.extend(later.stages);
        }
        Ok(Opening {
            receipt_id: challenge.receipt_id.clone(),
            stages,
        })
    }

    /// [`Self::open`] for the stage before this one over the stage
    /// connection. The challenge had its turn at the first stage, so a
    /// later stage does not queue it or turn it away.
    fn open_forwarded(&self, request: &ForwardedOpen) -> Result<Opening> {
        let root = request
            .turn_root
            .as_deref()
            .map(|root| hex_array(root, "the turn root"))
            .transpose()?;
        self.pinned_while_opening(&request.challenge.receipt_id, || {
            self.open_now(&request.challenge, root)
        })
    }
}

/// The later stages of a committed chain, as this stage keeps them for a
/// request's turn.
fn later_stages(request: &CommitRequest, stages: &[StageCommitment]) -> LaterStages {
    LaterStages {
        position_start: request.position_start,
        position_end: request.position_end,
        commitments: stages.get(1..).unwrap_or_default().into(),
    }
}

/// The id a turn's receipt is issued under: the one the client asked for
/// (`x-receipt-id`), else the request's id, else the backend's own.
fn issued_id(asked: Option<uuid::Uuid>, frontend: Option<[u8; 16]>, request_id: u64) -> String {
    asked
        .or(frontend.map(uuid::Uuid::from_bytes))
        .map(|id| id.hyphenated().to_string())
        .unwrap_or_else(|| format!("{request_id:016x}"))
}

/// Stage 0 hook at the end of a turn, called while the turn's session still
/// holds its sequence. Failures are logged; they never fail the user's
/// request. Returns the receipt when the request asked for a bound one, for
/// the response to carry.
pub(crate) fn finish_stage_zero_turn(
    runtime: &Mutex<RuntimeState>,
    frontend_request_id: Option<[u8; 16]>,
    receipt_request: Option<ReceiptRequest>,
    request_id: u64,
    session: &str,
    tokens: TurnTokens,
) -> Option<serde_json::Value> {
    let (receipts, session_turn) = {
        let state = lock_runtime(runtime);
        let receipts = state.receipts().cloned()?;
        (receipts, state.receipt_turn(session))
    };
    let receipt_id = issued_id(
        receipt_request.and_then(|request| request.id),
        frontend_request_id,
        request_id,
    );
    let bound = receipt_request.is_some();
    let result = session_turn
        .with_context(|| format!("no native sequence for session {session}"))
        .and_then(|session_turn| {
            receipts.finish_turn(&receipt_id, session, session_turn, tokens, receipt_request)
        });
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

#[cfg(test)]
#[path = "receipts/tests.rs"]
mod tests;

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
