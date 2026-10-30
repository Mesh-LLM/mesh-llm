//! Safe API over skippy_api's llama.cpp backend.
//!
//! Ownership follows the native layer:
//!   [`Runtime`]  owns every model and session (one native runtime behind its serialiser)
//!   [`Model`]    borrows its `Runtime`; dropping it unloads the model
//!   [`Session`]  borrows its `Model`; dropping it releases the native context
//!
//! The borrows make the native rules static: a session cannot outlive its
//! model, a model cannot outlive the runtime, and `ModelInUse` cannot happen.
//!
//! Threading: the native runtime declares itself *not thread-affine; serialise
//! calls* (`Bridge` is `Send`, not `Sync`), so the consumer that shares it owns
//! the serialiser. That is [`Runtime::bridge`]: a `Mutex` held for exactly one
//! bridge call, never longer, which is what lets every handle here be
//! `Send + Sync`. It is the one lock in the facade and it guards a call
//! sequence, not shared state — the C++ side stays lock-free (TYPE-RULES §3.2).
//! It is held across the native call by design: today concurrency across
//! sessions of one runtime is a type-system fact, not a runtime one.

use std::marker::PhantomData;
use std::sync::{Mutex, MutexGuard, PoisonError};

use skippy_sys as sys;
use sys::cxx::UniquePtr;

/// A vocabulary token id; layout-compatible with llama's `int32_t`.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Token(pub i32);

#[derive(Debug, Clone, PartialEq, Eq, Hash, thiserror::Error)]
pub enum RuntimeError {
    #[error("model handle does not resolve")]
    InvalidModel,
    #[error("session handle does not resolve")]
    InvalidSession,
    #[error("model still has live sessions")]
    ModelInUse,
    #[error("native allocation failed")]
    AllocationFailed,
    #[error("model could not be loaded: {message}")]
    LoadFailed {
        /// The loader's own reason (the inventory parser's text, say); empty
        /// when the native load failed before producing one.
        message: String,
    },
    #[error("native context could not be created")]
    SessionCreationFailed,
    #[error("session configuration not supported by the backend")]
    UnsupportedConfiguration,
    #[error("runtime capacity exhausted")]
    CapacityExceeded,
    #[error("backend returned an invalid resource")]
    InvalidBackendResult,
    #[error("native layer failed unexpectedly")]
    Internal,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, thiserror::Error)]
pub enum WorkloadError {
    #[error("session handle does not resolve")]
    InvalidSession,
    #[error("model does not support this workload")]
    UnsupportedWorkload,
    #[error("rerank needs exactly one classifier output")]
    UnsupportedClassifierCount,
    #[error("input is empty")]
    EmptyInput,
    #[error("input exceeds the session's max_tokens")]
    TooManyTokens,
    #[error("more inputs than the session was created for")]
    TooManySequences,
    #[error("text could not be tokenized")]
    TokenizationFailed,
    #[error("model advertises no embedding width")]
    NoEmbeddingDimensions,
    #[error("output holds fewer than {required} values")]
    OutputTooSmall { required: usize },
    #[error("native allocation failed")]
    AllocationFailed,
    #[error("native decode failed: {message}")]
    DecodeFailed {
        /// The backend's own failure text (its exception's what()), empty when
        /// even copying it failed under OOM.
        message: String,
    },
    #[error("native encode failed: {message}")]
    EncodeFailed {
        /// The backend's own failure text (its exception's what()), empty when
        /// even copying it failed under OOM.
        message: String,
    },
    #[error("model has no decoder start token")]
    NoDecoderStartToken,
    #[error("decode produced no output")]
    NoOutput,
    #[error("native output width differs from the advertised width")]
    DimensionMismatch,
    #[error("native layer failed unexpectedly")]
    Internal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadKind {
    CausalGeneration,
    Embedding,
    Rerank,
    EncoderDecoder,
}

/// Pooling modes the embedding path can drive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pooling {
    None,
    Mean,
    Cls,
    Last,
}

/// Pooling as llama reports it. `Other` is a value from a newer llama.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NativePooling {
    Unspecified,
    None,
    Mean,
    Cls,
    Last,
    Rank,
    Other(i8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkloadInfo {
    pub kind: WorkloadKind,
    pub pooling: NativePooling,
    /// Set iff the session can embed.
    pub embedding_pooling: Option<Pooling>,
    /// Embedding only, else 0.
    pub output_dimensions: usize,
    /// Rerank only, else 0.
    pub classifier_outputs: usize,
    pub has_encoder: bool,
    pub has_decoder: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionConfig {
    /// Longest single input, in tokens. The whole input goes through in one
    /// physical batch, so a single-sequence session's context is this size.
    pub max_tokens: u32,
    /// How many inputs one batch call may decode together. A batch session's
    /// context holds `sequences * max_tokens` tokens and costs accordingly.
    pub sequences: u32,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self { max_tokens: 512, sequences: 1 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rerank {
    pub score: f32,
    /// Tokens the composed query/document prompt produced.
    pub token_count: usize,
}

pub struct Runtime {
    /// The serialiser the native declaration demands (see the crate doc);
    /// locked for one bridge call at a time via [`Runtime::lock`].
    bridge: Mutex<UniquePtr<sys::Bridge>>,
}

/// A model running as N chained stages. The caller drives it: one `run` call
/// pushes tokens through every stage and hands back the terminal frame, if the
/// terminal stage produces one.
pub struct StageChain<'rt> {
    runtime: &'rt Runtime,
    chain: UniquePtr<sys::StageChainHandle>,
    stages: usize,
}

impl Runtime {
    pub fn open_chain(&self, path: &str, ranges: &[(i32, i32)]) -> Result<StageChain<'_>, StageError> {
        let mut flat = Vec::with_capacity(ranges.len() * 2);
        for &(start, end) in ranges {
            flat.push(start);
            flat.push(end);
        }
        let result = self.lock().pin_mut().open_chain(path, &flat);
        if !result.ok {
            return Err(StageError::Rejected(result.error));
        }
        Ok(StageChain { runtime: self, chain: result.chain, stages: ranges.len() })
    }
}

impl StageChain<'_> {
    pub fn run(&mut self, tokens: &[Token]) -> Result<Option<Vec<u8>>, StageError> {
        let native: Vec<i32> = tokens.iter().map(|token| token.0).collect();
        let result = self.runtime.lock().pin_mut().run_chain(self.chain.pin_mut(), &native);
        if !result.ok {
            return Err(StageError::Rejected(result.error));
        }
        Ok(result.has_frame.then_some(result.frame))
    }

    /// How many stages this chain drives; the plan decides it, so it is fixed.
    pub fn stages(&self) -> usize {
        self.stages
    }
}

impl Runtime {
    pub fn new() -> Result<Self, RuntimeError> {
        let bridge = sys::new_bridge();
        if bridge.is_null() {
            return Err(RuntimeError::AllocationFailed);
        }
        Ok(Self { bridge: Mutex::new(bridge) })
    }

    /// Loads a GGUF model. The path is passed to llama as UTF-8.
    /// Plans `stages` contiguous stages over the model at `path`.
    ///
    /// A stage carries everything a split runtime needs: the layers it owns,
    /// its execution contract, and the values crossing each boundary. The
    /// planner's identities are opaque 32-byte tags, so they are hex-encoded.
    pub fn plan_stages(&self, path: &str, stages: u32) -> Result<Vec<StagePlan>, StagePlanError> {
        let result = self.lock().plan_stages(path, stages);
        if !result.ok {
            return Err(StagePlanError::Rejected(result.error));
        }
        let infos = result.stages;
        let mut plans = Vec::with_capacity(infos.len());
        for info in infos {
            plans.push(StagePlan {
                profile_id: info.profile_id_hex,
                graph_identity: info.graph_identity_hex,
                profile_identity: info.profile_identity_hex,
                slice_identity: info.slice_identity_hex,
                source_snapshot: info.source_snapshot_hex,
                graph_configuration: info.graph_configuration_hex,
                backend: info.backend_hex,
                layers: info.layer_start..info.layer_end,
                execution_contract_hex: info.execution_contract_hex,
                resident_identities: info.resident_identity_hex,
                imports: info
                    .import_identity_hex
                    .into_iter()
                    .zip(info.import_binding_hex)
                    .map(|(identity, binding)| FrontierValue { identity, binding })
                    .collect(),
                exports: info
                    .export_identity_hex
                    .into_iter()
                    .zip(info.export_binding_hex)
                    .map(|(identity, binding)| FrontierValue { identity, binding })
                    .collect(),
                request_inputs: info
                    .request_identity_hex
                    .into_iter()
                    .zip(info.request_binding_hex)
                    .map(|(identity, binding)| FrontierValue { identity, binding })
                    .collect(),
                semantics: StageSemantics {
                    frontier_values: info.frontier_values,
                    state_effects: info.state_effects,
                    written_state_effects: info.written_state_effects,
                    recurrent: info.recurrent_state,
                },
            });
        }
        plan_or_oom(plans)
    }

    /// Reads the model's family from its GGUF architecture.
    pub fn model_family(&self, path: &str) -> ModelFamily {
        self.lock().model_family(path)
    }

    /// Opens stage `index` of `stages` in `path` as a model of its own.
    pub fn load_stage(&self, path: &str, stages: u32, index: u32) -> Result<StagedModel<'_>, StageError> {
        let result = self.lock().pin_mut().load_staged_model(path, stages, index);
        if !result.ok || result.model.is_null() {
            return Err(StageError::Rejected(result.error));
        }
        Ok(StagedModel { runtime: self, handle: result.model })
    }

    /// Checks that the stages of `path` form one contiguous, handover-clean chain.
    pub fn validate_stage_chain(&self, path: &str, stages: u32) -> Result<(), ChainError> {
        let result = self.lock().validate_chain(path, stages);
        if result.ok { Ok(()) } else { Err(ChainError::Invalid(result.error)) }
    }

    pub fn load_model(&self, path: &str) -> Result<Model<'_>, RuntimeError> {
        let result = self.lock().pin_mut().load_model(path);
        model_result(&result)?;
        Ok(Model { runtime: self, handle: result.model })
    }

    fn lock(&self) -> MutexGuard<'_, UniquePtr<sys::Bridge>> {
        self.bridge.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// Model families the facade names explicitly.
///
/// Branch on this instead of parsing GGUF strings; unknown architectures are
/// `Unknown`, never an error.
pub use skippy_sys::ModelFamily;

/// A stage of a split model, opened as its own model.
///
/// A stage is a real model: it owns native resources and must outlive every
/// session created from it. The borrow in [`StagedSession`] enforces that.
pub struct StagedModel<'rt> {
    runtime: &'rt Runtime,
    handle: UniquePtr<sys::StagedModel>,
}

/// A session on a stage. Borrows the stage it belongs to.
pub struct StagedSession<'model> {
    model: &'model StagedModel<'model>,
    handle: UniquePtr<sys::StagedSession>,
}

impl<'rt> StagedModel<'rt> {
    /// Opens a session on this stage.
    pub fn create_session(&self) -> Result<StagedSession<'_>, StageError> {
        let result = self.runtime.lock().pin_mut().create_staged_session(&self.handle);
        if !result.ok || result.session.is_null() {
            return Err(StageError::Rejected(result.error));
        }
        Ok(StagedSession { model: self, handle: result.session })
    }
}

impl StagedSession<'_> {
    /// Embeds `tokens` into `output`, which must hold one embedding.
    ///
    /// Returns the number of values written, or the backend's reason: a
    /// non-terminal stage has no pooling head and refuses to embed.
    pub fn embed_into(&mut self, tokens: &[Token], output: &mut [f32]) -> Result<usize, StageError> {
        let native: Vec<i32> = tokens.iter().map(|token| token.0).collect();
        let result = self.model.runtime.lock().pin_mut().staged_embed(self.handle.pin_mut(), &native, output);
        if result.ok { Ok(result.dimensions) } else { Err(StageError::Rejected(result.error)) }
    }
}

/// Why a stage could not be opened or used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StageError {
    /// The backend refused; the message is its own explanation.
    Rejected(String),
}

impl std::fmt::Display for StageError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected(message) => write!(formatter, "stage rejected: {message}"),
        }
    }
}

impl std::error::Error for StageError {}

/// One stage of a model split across runtimes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StagePlan {
    /// Layers this stage owns, half-open.
    pub layers: std::ops::Range<i32>,
    /// Hex-encoded contract the planner produced for this stage.
    pub execution_contract_hex: String,
    /// Hex-encoded identities of the tensors the stage keeps resident.
    pub resident_identities: Vec<String>,
    /// Values consumed from the predecessor stage.
    pub imports: Vec<FrontierValue>,
    /// Values handed to the successor stage.
    pub exports: Vec<FrontierValue>,
    /// Values this stage is handed as input (attention masks, positions, ...).
    pub request_inputs: Vec<FrontierValue>,
    /// Hex-encoded identities of the slice this stage is.
    pub profile_id: String,
    pub graph_identity: String,
    pub profile_identity: String,
    pub slice_identity: String,
    pub source_snapshot: String,
    pub graph_configuration: String,
    pub backend: String,
    /// What the stage's state looks like, without materialising it.
    pub semantics: StageSemantics,
}

/// State shape of one stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StageSemantics {
    /// Values crossing the boundary in either direction.
    pub frontier_values: usize,
    /// State effects (KV or recurrent) this stage touches.
    pub state_effects: usize,
    /// How many of those are written.
    pub written_state_effects: usize,
    /// True when the stage holds recurrent state rather than attention KV.
    pub recurrent: bool,
}

/// Outcome of validating a plan chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainError {
    /// The chain is not a valid split; the message names the broken rule.
    Invalid(String),
}

impl std::fmt::Display for ChainError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(message) => write!(formatter, "invalid stage chain: {message}"),
        }
    }
}

impl std::error::Error for ChainError {}

/// One value crossing a stage boundary. Both halves are hex-encoded tags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrontierValue {
    pub identity: String,
    pub binding: String,
}

/// Why a model could not be split.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StagePlanError {
    /// The planner refused the request; the message says why.
    Rejected(String),
    /// The bridge could not even build the plan vector; an empty result is
    /// this surface's out-of-memory spelling, never a valid plan.
    OutOfMemory,
}

impl std::fmt::Display for StagePlanError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected(message) => write!(formatter, "stage plan rejected: {message}"),
            Self::OutOfMemory => write!(formatter, "stage plan out of memory"),
        }
    }
}

/// An empty plan vector is the bridge's out-of-memory spelling, never a
/// valid plan: a request for at least one stage always yields at least one.
fn plan_or_oom(plans: Vec<StagePlan>) -> Result<Vec<StagePlan>, StagePlanError> {
    if plans.is_empty() { Err(StagePlanError::OutOfMemory) } else { Ok(plans) }
}

impl std::error::Error for StagePlanError {}

/// Why the model's declared suppression policy could not be read.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SuppressTokensError {
    #[error("suppress tokens rejected: {0}")]
    Rejected(String),
    #[error("suppress tokens out of memory")]
    OutOfMemory,
}

fn suppression_result(result: sys::SuppressTokensResult) -> Result<Vec<Token>, SuppressTokensError> {
    if result.ok {
        Ok(tokens_from_native(result.tokens))
    } else if result.error.is_empty() {
        Err(SuppressTokensError::OutOfMemory)
    } else {
        Err(SuppressTokensError::Rejected(result.error))
    }
}

pub struct Model<'rt> {
    runtime: &'rt Runtime,
    handle: UniquePtr<sys::Model>,
}

impl Model<'_> {
    /// The vocab's declared suppress tokens (`tokenizer.ggml.suppress_tokens`),
    /// gathered once at load. The library never links a sampler: a caller
    /// holding logits applies this set before selecting a token. Returns an
    /// empty successful set when the GGUF declares none (patch 0012).
    ///
    /// # Errors
    /// Returns a typed error if the native model cannot be read or if copying
    /// the declared token set runs out of memory.
    pub fn suppress_tokens(&self) -> Result<Vec<Token>, SuppressTokensError> {
        suppression_result(self.runtime.lock().suppress_tokens(&self.handle))
    }

    pub fn create_session(&self, config: SessionConfig) -> Result<Session<'_>, RuntimeError> {
        let mut bridge = self.runtime.lock();
        let created = bridge.pin_mut().create_session(
            &self.handle,
            sys::SessionConfig { max_tokens: config.max_tokens, sequences: config.sequences },
        );
        runtime_result(created.status)?;
        let mut handle = created.session;

        let workload = bridge.workload(&handle);
        if let Err(error) = runtime_result(workload.status) {
            let _ = bridge.pin_mut().release_session(handle.pin_mut());
            return Err(error);
        }
        Ok(Session { runtime: self.runtime, _model: PhantomData, handle, workload: workload_info(workload.info) })
    }
}

impl Drop for Model<'_> {
    fn drop(&mut self) {
        // Sessions borrow the model, so none are live: this cannot fail with
        // ModelInUse, and a stale handle after shutdown is nothing to report.
        let _ = self.runtime.lock().pin_mut().unload_model(self.handle.pin_mut());
    }
}

pub struct Session<'m> {
    runtime: &'m Runtime,
    _model: PhantomData<&'m ()>,
    handle: UniquePtr<sys::Session>,
    workload: WorkloadInfo,
}

impl Session<'_> {
    #[must_use]
    pub fn workload(&self) -> &WorkloadInfo {
        &self.workload
    }

    pub fn tokenize(&mut self, text: &str) -> Result<Vec<Token>, WorkloadError> {
        let mut tokens = Vec::new();
        self.tokenize_into(text, &mut tokens)?;
        Ok(tokens)
    }

    /// Replaces the contents of `tokens`, reusing its allocation.
    pub fn tokenize_into(&mut self, text: &str, tokens: &mut Vec<Token>) -> Result<(), WorkloadError> {
        let mut native = tokens_into_native(std::mem::take(tokens));
        let status = self.runtime.lock().pin_mut().tokenize(&self.handle, text, &mut native);
        *tokens = tokens_from_native(native);
        workload_result(status, "")
    }

    /// One unit-normalised embedding of `tokens`.
    pub fn embed(&mut self, tokens: &[Token]) -> Result<Vec<f32>, WorkloadError> {
        let mut output = vec![0.0f32; self.workload.output_dimensions];
        let written = self.embed_into(tokens, &mut output)?;
        output.truncate(written);
        Ok(output)
    }

    /// One score per document for `query`, decoded in a single native pass.
    /// Needs a session created with `SessionConfig::sequences` at least
    /// `documents.len()`.
    pub fn rerank_batch(&mut self, query: &str, documents: &[&str]) -> Result<Vec<f32>, WorkloadError> {
        let mut scores = vec![0.0f32; documents.len()];
        let result = self.runtime.lock().pin_mut().rerank_batch(&self.handle, query, documents, &mut scores);
        workload_result(result.status, &result.error)?;
        Ok(scores)
    }

    /// One unit-normalised embedding per group, decoded in a single native
    /// pass. Needs a session created with `SessionConfig::sequences` at least
    /// `groups.len()`.
    pub fn embed_batch(&mut self, groups: &[&[Token]]) -> Result<Vec<Vec<f32>>, WorkloadError> {
        let mut flat: Vec<i32> = Vec::with_capacity(groups.iter().map(|group| group.len()).sum());
        let mut lengths: Vec<u32> = Vec::with_capacity(groups.len());
        for group in groups {
            flat.extend(group.iter().map(|token| token.0));
            lengths.push(u32::try_from(group.len()).map_err(|_| WorkloadError::TooManyTokens)?);
        }

        let dimensions = self.workload.output_dimensions;
        let mut output = vec![0.0f32; dimensions * groups.len()];
        let result = self.runtime.lock().pin_mut().embed_batch(&self.handle, &flat, &lengths, &mut output);
        workload_result(result.status, &result.error)?;
        Ok(output.chunks_exact(dimensions).map(<[f32]>::to_vec).collect())
    }

    /// Writes the embedding into the front of `output` and returns its width.
    pub fn embed_into(&mut self, tokens: &[Token], output: &mut [f32]) -> Result<usize, WorkloadError> {
        // The native check would report the zero width first; the kind is the
        // better answer and is already known here.
        if self.workload.kind != WorkloadKind::Embedding {
            return Err(WorkloadError::UnsupportedWorkload);
        }
        let result = self.runtime.lock().pin_mut().embed(&self.handle, tokens_as_native(tokens), output);
        if result.status == sys::WorkloadStatus::OutputTooSmall {
            return Err(WorkloadError::OutputTooSmall { required: result.dimensions });
        }
        workload_result(result.status, &result.error)?;
        Ok(result.dimensions)
    }

    pub fn rerank(&mut self, query: &str, document: &str) -> Result<Rerank, WorkloadError> {
        let result = self.runtime.lock().pin_mut().rerank(&self.handle, query, document);
        workload_result(result.status, &result.error)?;
        Ok(Rerank { score: result.score, token_count: result.token_count })
    }

    /// Runs the encoder over `tokens`; returns the token the decoder starts from.
    pub fn encode(&mut self, tokens: &[Token]) -> Result<Token, WorkloadError> {
        let result = self.runtime.lock().pin_mut().encode(&self.handle, tokens_as_native(tokens));
        workload_result(result.status, &result.error)?;
        Ok(Token(result.decoder_start))
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        let _ = self.runtime.lock().pin_mut().release_session(self.handle.pin_mut());
    }
}

/// The model path: the bridge carries the loader's reason in
/// `ModelResult.error`, and it stops here unless something reads it.
fn model_result(result: &sys::ModelResult) -> Result<(), RuntimeError> {
    use sys::RuntimeStatus as S;
    match result.status {
        S::LoadFailed => Err(RuntimeError::LoadFailed { message: result.error.to_string() }),
        status => runtime_result(status),
    }
}

fn runtime_result(status: sys::RuntimeStatus) -> Result<(), RuntimeError> {
    use sys::RuntimeStatus as S;
    Err(match status {
        S::Ok => return Ok(()),
        S::InvalidModel => RuntimeError::InvalidModel,
        S::InvalidSession => RuntimeError::InvalidSession,
        S::ModelInUse => RuntimeError::ModelInUse,
        S::AllocationFailed => RuntimeError::AllocationFailed,
        // Sessions never fail a load; a status-only caller has no message to
        // carry. The model path goes through model_result below.
        S::LoadFailed => RuntimeError::LoadFailed { message: String::new() },
        S::SessionCreationFailed => RuntimeError::SessionCreationFailed,
        S::UnsupportedConfiguration => RuntimeError::UnsupportedConfiguration,
        S::CapacityExceeded => RuntimeError::CapacityExceeded,
        S::InvalidBackendResult => RuntimeError::InvalidBackendResult,
        _ => RuntimeError::Internal,
    })
}

fn workload_result(status: sys::WorkloadStatus, error: &str) -> Result<(), WorkloadError> {
    use sys::WorkloadStatus as S;
    Err(match status {
        S::Ok => return Ok(()),
        S::InvalidSession => WorkloadError::InvalidSession,
        S::UnsupportedWorkload => WorkloadError::UnsupportedWorkload,
        S::UnsupportedClassifierCount => WorkloadError::UnsupportedClassifierCount,
        S::EmptyInput => WorkloadError::EmptyInput,
        S::TooManyTokens => WorkloadError::TooManyTokens,
        S::TooManySequences => WorkloadError::TooManySequences,
        S::TokenizationFailed => WorkloadError::TokenizationFailed,
        S::NoEmbeddingDimensions => WorkloadError::NoEmbeddingDimensions,
        // The width travels with the result, not the status; see embed_into.
        S::OutputTooSmall => WorkloadError::OutputTooSmall { required: 0 },
        S::AllocationFailed => WorkloadError::AllocationFailed,
        S::DecodeFailed => WorkloadError::DecodeFailed { message: error.to_owned() },
        S::EncodeFailed => WorkloadError::EncodeFailed { message: error.to_owned() },
        S::NoDecoderStartToken => WorkloadError::NoDecoderStartToken,
        S::NoOutput => WorkloadError::NoOutput,
        S::DimensionMismatch => WorkloadError::DimensionMismatch,
        _ => WorkloadError::Internal,
    })
}

fn workload_info(info: sys::WorkloadInfo) -> WorkloadInfo {
    use sys::WorkloadKind as K;
    let kind = match info.kind {
        K::CausalGeneration => WorkloadKind::CausalGeneration,
        K::Embedding => WorkloadKind::Embedding,
        K::Rerank => WorkloadKind::Rerank,
        K::EncoderDecoder => WorkloadKind::EncoderDecoder,
        other => unreachable!("facade produced WorkloadKind {}", other.repr),
    };
    let pooling = native_pooling(info.pooling);
    let embedding_pooling = match native_pooling(info.embedding_pooling) {
        NativePooling::None => Some(Pooling::None),
        NativePooling::Mean => Some(Pooling::Mean),
        NativePooling::Cls => Some(Pooling::Cls),
        NativePooling::Last => Some(Pooling::Last),
        NativePooling::Unspecified | NativePooling::Rank | NativePooling::Other(_) => None,
    };
    WorkloadInfo {
        kind,
        pooling,
        embedding_pooling,
        output_dimensions: info.output_dimensions,
        classifier_outputs: info.classifier_outputs,
        has_encoder: info.has_encoder,
        has_decoder: info.has_decoder,
    }
}

fn native_pooling(pooling: sys::NativePooling) -> NativePooling {
    use sys::NativePooling as P;
    match pooling {
        P::Unspecified => NativePooling::Unspecified,
        P::None => NativePooling::None,
        P::Mean => NativePooling::Mean,
        P::Cls => NativePooling::Cls,
        P::Last => NativePooling::Last,
        P::Rank => NativePooling::Rank,
        other => NativePooling::Other(other.repr),
    }
}

// Token is repr(transparent) over i32: same size, alignment and allocation
// layout, so a Vec or slice of one is a Vec or slice of the other.
fn tokens_as_native(tokens: &[Token]) -> &[i32] {
    // SAFETY: see above; the slice keeps its length and lifetime.
    unsafe { std::slice::from_raw_parts(tokens.as_ptr().cast::<i32>(), tokens.len()) }
}

fn tokens_into_native(mut tokens: Vec<Token>) -> Vec<i32> {
    let (ptr, len, capacity) = (tokens.as_mut_ptr(), tokens.len(), tokens.capacity());
    std::mem::forget(tokens);
    // SAFETY: see above; the allocation was made for `capacity` elements of
    // the same size and alignment, and `tokens` no longer owns it.
    unsafe { Vec::from_raw_parts(ptr.cast::<i32>(), len, capacity) }
}

fn tokens_from_native(mut tokens: Vec<i32>) -> Vec<Token> {
    let (ptr, len, capacity) = (tokens.as_mut_ptr(), tokens.len(), tokens.capacity());
    std::mem::forget(tokens);
    // SAFETY: as tokens_into_native, in the other direction.
    unsafe { Vec::from_raw_parts(ptr.cast::<Token>(), len, capacity) }
}

const _: () = {
    assert!(size_of::<Token>() == size_of::<i32>());
    assert!(align_of::<Token>() == align_of::<i32>());
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_vectors_round_trip_without_reallocating() {
        let mut tokens = Vec::with_capacity(16);
        tokens.extend([Token(1), Token(2)]);
        let before = tokens.as_ptr();

        let native = tokens_into_native(tokens);
        assert_eq!(native, [1, 2]);
        let tokens = tokens_from_native(native);

        assert_eq!(tokens, [Token(1), Token(2)]);
        assert_eq!(tokens.capacity(), 16);
        assert_eq!(tokens.as_ptr(), before);
    }

    #[test]
    fn status_mapping_is_total() {
        assert_eq!(runtime_result(sys::RuntimeStatus::Ok), Ok(()));
        assert_eq!(runtime_result(sys::RuntimeStatus::InternalError), Err(RuntimeError::Internal));
        assert_eq!(runtime_result(sys::RuntimeStatus { repr: 200 }), Err(RuntimeError::Internal));
        assert_eq!(workload_result(sys::WorkloadStatus::Ok, ""), Ok(()));
        assert_eq!(
            workload_result(sys::WorkloadStatus::DecodeFailed, ""),
            Err(WorkloadError::DecodeFailed { message: String::new() })
        );
        // A native failure with text carries it into the variant.
        assert_eq!(
            workload_result(sys::WorkloadStatus::DecodeFailed, "metal: command buffer failed"),
            Err(WorkloadError::DecodeFailed { message: "metal: command buffer failed".to_owned() })
        );
        assert_eq!(workload_result(sys::WorkloadStatus { repr: 200 }, ""), Err(WorkloadError::Internal));
    }

    #[test]
    fn unknown_pooling_is_preserved() {
        assert_eq!(native_pooling(sys::NativePooling { repr: 9 }), NativePooling::Other(9));
        assert_eq!(native_pooling(sys::NativePooling::Rank), NativePooling::Rank);
    }

    #[test]
    fn empty_plans_are_out_of_memory() {
        assert_eq!(plan_or_oom(Vec::new()), Err(StagePlanError::OutOfMemory));
    }

    #[test]
    fn suppression_allocation_failure_is_not_an_empty_success() {
        let failed = sys::SuppressTokensResult { ok: false, error: String::new(), tokens: Vec::new() };
        assert_eq!(suppression_result(failed), Err(SuppressTokensError::OutOfMemory));
        let empty = sys::SuppressTokensResult { ok: true, error: String::new(), tokens: Vec::new() };
        assert_eq!(suppression_result(empty), Ok(Vec::new()));
    }

    #[test]
    fn suppression_failure_preserves_the_native_message() {
        let failed = sys::SuppressTokensResult {
            ok: false,
            error: "suppress_tokens: invalid model".to_owned(),
            tokens: Vec::new(),
        };
        assert_eq!(
            suppression_result(failed),
            Err(SuppressTokensError::Rejected("suppress_tokens: invalid model".to_owned()))
        );
    }

    #[test]
    fn suppression_success_preserves_declared_token_order() {
        let declared = sys::SuppressTokensResult { ok: true, error: String::new(), tokens: vec![9, 2, 9] };
        assert_eq!(suppression_result(declared), Ok(vec![Token(9), Token(2), Token(9)]));
    }
}
