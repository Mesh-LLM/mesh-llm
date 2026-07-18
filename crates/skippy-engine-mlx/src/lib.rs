//! MLX (Metal) serving engine for mesh-llm.
//!
//! Serves HF safetensors tensor models over mesh-llm's real OpenAI-compatible
//! frontend (`openai_frontend::OpenAiBackend`), goose-style, on Apple Silicon.
//!
//! All MLX-touching code is gated behind BOTH the `mlx` cargo feature AND
//! `target_os = "macos"`. On any other target, or without the feature, this
//! crate compiles to an empty shell so it never burdens non-Apple builds.

#[cfg(all(feature = "mlx", target_os = "macos"))]
mod backend;
#[cfg(all(feature = "mlx", target_os = "macos"))]
mod boundary_bench;
#[cfg(all(feature = "mlx", target_os = "macos"))]
mod derived;
#[cfg(all(feature = "mlx", target_os = "macos"))]
mod distributed;
#[cfg(all(feature = "mlx", target_os = "macos"))]
mod engine;
#[cfg(all(feature = "mlx", target_os = "macos"))]
mod stage;

#[cfg(all(feature = "mlx", target_os = "macos"))]
pub use backend::MlxBackend;
#[cfg(all(feature = "mlx", target_os = "macos"))]
pub use boundary_bench::{
    MlxBoundaryBenchConfig, MlxBoundaryBenchReport, MlxTcpBoundaryBenchConfig,
    MlxTcpBoundaryBenchReport, MlxTcpBoundarySinkConfig, benchmark_mlx_boundary,
    benchmark_mlx_tcp_boundary, serve_mlx_tcp_boundary_sink,
};
#[cfg(all(feature = "mlx", target_os = "macos"))]
pub use derived::{
    MlxDerivationControl, MlxDerivedStageCacheConfig, MlxDerivedStageCacheResult,
    MlxDerivedStageConfig, MlxDerivedStageReport, MlxDerivedStageShard,
    MlxNemotronHValidationReport, derive_quantized_stage, derive_quantized_stage_cached,
    load_prepared_quantized_stage, mlx_derived_stage_cache_root, validate_nemotron_h_moe_stage,
};
#[cfg(all(feature = "mlx", target_os = "macos"))]
pub use distributed::{MlxDistributedEngine, MlxDistributedEngineConfig};
#[cfg(all(feature = "mlx", target_os = "macos"))]
pub use engine::{
    ChatTurn, GenerateRequest, MlxEngine, MlxEngineConfig, automatic_weight_quantization,
};
#[cfg(all(feature = "mlx", target_os = "macos"))]
pub use stage::{
    MlxComputeDtype, MlxNemotronHStageValidationReport, MlxNemotronHWireValidationReport,
    MlxStageEngine, MlxStageEngineConfig, MlxWeightQuantization, validate_nemotron_h_binary_wire,
    validate_nemotron_h_binary_wire_tokens, validate_nemotron_h_stage_engine,
};

/// True when this build actually contains the MLX engine.
pub const fn mlx_available() -> bool {
    cfg!(all(feature = "mlx", target_os = "macos"))
}
