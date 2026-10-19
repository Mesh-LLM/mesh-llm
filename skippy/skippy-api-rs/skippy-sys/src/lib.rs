//! Raw cxx bindings to skippy_api's llama.cpp backend.
//!
//! Everything crosses the boundary by value or by borrowed slice; every
//! fallible call returns a status, never an exception or a panic. The safe
//! API is the `skippy` crate; use this crate directly only to build another.
//!
//! Ownership:
//!   `Bridge`           owns the native runtime and every model and session in it
//!   `Model`/`Session`  tokens minted by the runtime; they own nothing and must be
//!                      handed back to `unload_model`/`release_session` or they
//!                      stop resolving when the `Bridge` drops

#[cxx::bridge(namespace = "skippy::ffi")]
pub mod ffi {
    /// `skippy::RuntimeError` plus `Ok`, and `InternalError` for anything that
    /// tried to unwind across the boundary.
    #[repr(u8)]
    #[derive(Debug, Hash)]
    enum RuntimeStatus {
        Ok,
        InvalidModel,
        InvalidSession,
        ModelInUse,
        AllocationFailed,
        LoadFailed,
        SessionCreationFailed,
        UnsupportedConfiguration,
        CapacityExceeded,
        InvalidBackendResult,
        InternalError,
    }

    /// `skippy::WorkloadError` plus `Ok` and `InternalError`.
    #[repr(u8)]
    #[derive(Debug, Hash)]
    enum WorkloadStatus {
        Ok,
        InvalidSession,
        UnsupportedWorkload,
        UnsupportedClassifierCount,
        EmptyInput,
        TooManyTokens,
        TooManySequences,
        TokenizationFailed,
        NoEmbeddingDimensions,
        OutputTooSmall,
        AllocationFailed,
        DecodeFailed,
        EncodeFailed,
        NoDecoderStartToken,
        NoOutput,
        DimensionMismatch,
        InternalError,
    }

    #[repr(i8)]
    #[derive(Debug, Hash)]
    enum WorkloadKind {
        CausalGeneration,
        Embedding,
        Rerank,
        EncoderDecoder,
    }

    /// Mirrors `llama_pooling_type`. In `WorkloadInfo::embedding_pooling`,
    /// `Unspecified` means the context cannot embed.
    #[repr(i8)]
    #[derive(Debug, Hash)]
    enum NativePooling {
        Unspecified = -1,
        None = 0,
        Mean = 1,
        Cls = 2,
        Last = 3,
        Rank = 4,
    }

    /// Longest single input, in tokens, and how many inputs one batch call may
    /// decode together. A batch session's context holds sequences x max_tokens.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct SessionConfig {
        max_tokens: u32,
        sequences: u32,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct WorkloadInfo {
        kind: WorkloadKind,
        pooling: NativePooling,
        embedding_pooling: NativePooling,
        output_dimensions: usize,
        classifier_outputs: usize,
        has_encoder: bool,
        has_decoder: bool,
    }

    /// `model` is null unless `status` is `Ok`. `error` is the loader's own
    /// reason when `status` is `LoadFailed`, empty otherwise.
    struct ModelResult {
        status: RuntimeStatus,
        model: UniquePtr<Model>,
        error: String,
    }

    /// `session` is null unless `status` is `Ok`.
    struct SessionResult {
        status: RuntimeStatus,
        session: UniquePtr<Session>,
    }

    #[derive(Debug, Clone, Copy)]
    struct WorkloadResult {
        status: RuntimeStatus,
        info: WorkloadInfo,
    }

    /// `dimensions` is the model's width on `Ok` and on `OutputTooSmall`.
    #[derive(Debug, Clone)]
    struct EmbedResult {
        status: WorkloadStatus,
        dimensions: usize,
        /// The backend's failure text on a native DecodeFailed; empty otherwise.
        error: String,
    }

    /// plan_stages result: failure is the envelope's, never an entry's.
    #[derive(Debug)]
    struct PlanStagesResult {
        ok: bool,
        error: String,
        stages: Vec<StagePlanInfo>,
    }

    /// One stage of a split model.
    #[derive(Debug)]
    struct StagePlanInfo {
        /// Position in the native chain, also used for producer provenance.
        stage_index: i32,
        layer_start: i32,
        layer_end: i32,
        /// Hex-encoded: the contract is opaque bytes, not text.
        execution_contract_hex: String,
        /// Hex-encoded: the planner's resident entries are 32-byte identities.
        resident_identity_hex: Vec<String>,
        /// Hex-encoded: an activation identity is a 32-byte tag.
        import_identity_hex: Vec<String>,
        import_binding_hex: Vec<String>,
        export_identity_hex: Vec<String>,
        export_binding_hex: Vec<String>,
        /// Hex-encoded identities of the slice this stage is.
        profile_id_hex: String,
        graph_identity_hex: String,
        profile_identity_hex: String,
        slice_identity_hex: String,
        source_snapshot_hex: String,
        graph_configuration_hex: String,
        backend_hex: String,
        request_identity_hex: Vec<String>,
        request_binding_hex: Vec<String>,
        frontier_values: usize,
        state_effects: usize,
        written_state_effects: usize,
        recurrent_state: bool,
    }

    /// Model families the facade names explicitly.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(u8)]
    enum ModelFamily {
        Unknown = 0,
        Llama = 1,
        Qwen = 2,
        Mistral = 3,
        Gemma = 4,
        Phi = 5,
        Bert = 6,
        Mamba = 7,
        Rwkv = 8,
        T5 = 9,
    }

    /// A stage opened as its own model.
    struct ChainResult {
        ok: bool,
        error: String,
        chain: UniquePtr<StageChainHandle>,
    }

    struct ChainRunResult {
        ok: bool,
        error: String,
        has_frame: bool,
        frame: Vec<u8>,
    }

    struct StagedModelResult {
        ok: bool,
        error: String,
        /// Null unless `ok`.
        model: UniquePtr<StagedModel>,
    }

    /// A session on a stage.
    struct StagedSessionResult {
        ok: bool,
        error: String,
        /// Null unless `ok`.
        session: UniquePtr<StagedSession>,
    }

    /// One embedding from a staged session.
    #[derive(Debug)]
    struct StagedEmbedResult {
        ok: bool,
        error: String,
        dimensions: usize,
    }

    /// Outcome of validating a stage chain.
    #[derive(Debug)]
    struct ChainValidation {
        ok: bool,
        error: String,
    }

    #[derive(Debug, Clone)]
    struct RerankBatchResult {
        status: WorkloadStatus,
        /// The backend's failure text on a native DecodeFailed; empty otherwise.
        error: String,
    }

    #[derive(Debug, Clone)]
    struct RerankResult {
        status: WorkloadStatus,
        score: f32,
        token_count: usize,
        /// The backend's failure text on a native DecodeFailed; empty otherwise.
        error: String,
    }

    #[derive(Debug, Clone)]
    struct EncodeResult {
        status: WorkloadStatus,
        decoder_start: i32,
        /// The backend's failure text on a native EncodeFailed; empty otherwise.
        error: String,
    }

    unsafe extern "C++" {
        include!("skippy-sys/cpp/bridge.h");

        type Bridge;

        type StageChainHandle;
        #[namespace = "skippy"]
        type Model;
        #[namespace = "skippy"]
        type Session;

        /// Null when the runtime could not be allocated.
        fn new_bridge() -> UniquePtr<Bridge>;

        fn load_model(self: Pin<&mut Bridge>, path: &str) -> ModelResult;
        fn unload_model(self: Pin<&mut Bridge>, model: Pin<&mut Model>) -> RuntimeStatus;

        fn create_session(self: Pin<&mut Bridge>, model: &Model, config: SessionConfig) -> SessionResult;
        fn release_session(self: Pin<&mut Bridge>, session: Pin<&mut Session>) -> RuntimeStatus;

        fn workload(self: &Bridge, session: &Session) -> WorkloadResult;

        /// The model's declared vocab suppress tokens (patch 0012); empty when
        /// the GGUF declares none.
        fn suppress_tokens(self: &Bridge, model: &Model) -> Vec<i32>;

        /// Replaces the contents of `tokens`.
        fn tokenize(self: Pin<&mut Bridge>, session: &Session, text: &str, tokens: &mut Vec<i32>) -> WorkloadStatus;

        /// Writes the unit-normalised embedding into the first `dimensions`
        /// elements of `output`.
        /// Plans `stage_count` contiguous stages over the model at `path`.
        fn plan_stages(self: &Bridge, path: &str, stage_count: u32) -> PlanStagesResult;

        /// A stage opened as its own model.
        type StagedModel;
        /// A session on a stage.
        type StagedSession;

        /// Opens stage `index` of `stages` over the model at `path`.
        fn load_staged_model(self: Pin<&mut Bridge>, path: &str, stages: u32, index: u32) -> StagedModelResult;

        /// Opens a session on a stage.
        fn create_staged_session(self: Pin<&mut Bridge>, model: &StagedModel) -> StagedSessionResult;

        /// Embeds `tokens` into `output` on a stage.
        fn staged_embed(
            self: Pin<&mut Bridge>,
            session: Pin<&mut StagedSession>,
            tokens: &[i32],
            output: &mut [f32],
        ) -> StagedEmbedResult;

        /// Reads the model's family from its GGUF architecture.
        fn model_family(self: &Bridge, path: &str) -> ModelFamily;

        /// Checks that a plan chain is contiguous and hands over cleanly.
        fn validate_chain(self: &Bridge, path: &str, stage_count: u32) -> ChainValidation;

        fn open_chain(self: Pin<&mut Bridge>, path: &str, ranges: &[i32]) -> ChainResult;

        fn run_chain(self: Pin<&mut Bridge>, chain: Pin<&mut StageChainHandle>, tokens: &[i32]) -> ChainRunResult;

        fn embed(self: Pin<&mut Bridge>, session: &Session, tokens: &[i32], output: &mut [f32]) -> EmbedResult;

        /// One embedding per group, decoded in a single native pass. `tokens`
        /// is every group concatenated; `lengths[i]` is group i's token count.
        fn embed_batch(
            self: Pin<&mut Bridge>,
            session: &Session,
            tokens: &[i32],
            lengths: &[u32],
            output: &mut [f32],
        ) -> EmbedResult;

        fn rerank(self: Pin<&mut Bridge>, session: &Session, query: &str, document: &str) -> RerankResult;

        /// One score per document, decoded in a single native pass.
        fn rerank_batch(
            self: Pin<&mut Bridge>,
            session: &Session,
            query: &str,
            documents: &[&str],
            scores: &mut [f32],
        ) -> RerankBatchResult;

        fn encode(self: Pin<&mut Bridge>, session: &Session, tokens: &[i32]) -> EncodeResult;
    }
}

// The native runtime is single-threaded but not thread-affine: every object
// may be used from any thread as long as calls are serialised, which is what
// moving a Bridge (or a token) to another thread needs.
unsafe impl Send for ffi::Bridge {}
unsafe impl Send for ffi::Model {}
unsafe impl Send for ffi::Session {}

pub use ffi::*;

/// The cxx version these bindings were generated with; `UniquePtr` lives here.
pub use cxx;
