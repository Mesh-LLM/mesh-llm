//! Mixture-of-Agents (MoA) gateway.
//!
//! Fan out to N heterogeneous LLM backends in parallel, arbitrate their
//! outputs with deterministic logic, and return one coherent OpenAI-
//! compatible response.  The client thinks it talks to one model.
//!
//! Transport is abstracted behind the [`ModelBackend`] trait (see
//! [`backend`]). The default [`HttpBackend`] talks to any
//! OpenAI-compatible HTTP endpoint and is suitable for standalone/test
//! use. The mesh host-runtime provides mesh-native backends that
//! dispatch local models via direct HTTP and remote models via QUIC
//! tunnel.
//!
//! ```text
//! Agent / Goose / pi
//!     │
//!     │  POST /v1/chat/completions { "model": "mesh" }
//!     ▼
//!  MoA Gateway  (handle_turn)
//!   ├─ session / context packing (role-shaped)        — context::*
//!   ├─ parallel fan-out via ModelBackend              — fanout::gather_workers_incremental
//!   ├─ incremental gathering with early-exit          — arbiter::try_early_decision
//!   ├─ deterministic arbiter (code, not models)       — arbiter::arbitrate
//!   └─ reducer escalation only on genuine conflict    — reducer::hedged_reducer_call
//! ```
//!
//! Modules:
//! - [`backend`] — `ModelBackend` trait, `HttpBackend`, `SamplingParams`,
//!   `ModelEntry`
//! - [`reducer`] — reducer candidate ordering, hedged ladder
//! - [`fanout`] — incremental worker gathering with early-exit
//! - [`arbiter`] — deterministic arbitration + early-exit consensus
//! - [`normalize`] — 3-tier dirty-output parsing
//! - [`session`] — canonical transcript + turn classification
//! - [`context`] — role-shaped context packing
//! - [`worker`] — role assignment, think-tag stripping

pub mod arbiter;
pub mod backend;
mod config;
pub mod context;
mod fanout;
mod gateway;
pub mod normalize;
mod reducer;
mod refinement;
mod resolve;
mod response;
pub mod session;
mod tool_guard;
mod tool_result;
mod tool_turn;
mod turn;
pub mod worker;

pub use backend::{HttpBackend, ModelBackend, ModelEntry, SamplingParams, apply_enable_thinking};
pub use config::{GatewayConfig, ReferencePolicy, RefinementPolicy};
pub use gateway::handle_turn;
pub(crate) use gateway::tool_names_for_turn;
pub(crate) use response::{
    chat_response, error_response, fallback_worker_response, tool_call_response,
    tool_proposal_response,
};
pub(crate) use tool_guard::enforce_tool_call_contract;
pub(crate) use turn::ForcedToolChoice;
pub use turn::{TurnKind, TurnResult, WorkerSummary};
pub use worker::{SMALL_TIER_MAX_B, entry_is_small_tier, strip_thinking, truncate_chars};

/// The virtual model name that triggers MoA routing.
pub const VIRTUAL_MODEL_NAME: &str = "mesh";

/// All fanned-out workers failed before the arbiter could pick a winner.
pub const MOA_ERR_ALL_WORKERS_FAILED: &str = "all_workers_failed";
/// Every reducer candidate failed (in both the tool-result and
/// arbiter-escalated paths).
pub const MOA_ERR_ALL_REDUCERS_FAILED: &str = "all_reducers_failed";
/// MoA only received silence directives or uncertainty after reduction.
pub const MOA_ERR_NO_USABLE_ANSWER: &str = "no_usable_answer";
