//! Iteration-level scheduling policy for Skippy staged serving.
//!
//! The crate owns policy, not a concrete server runtime. Every stage consumes
//! the same [`IterationPlan`], while `skippy-server` translates work items into
//! native ABI requests.

mod config;
mod engine;
mod sequence;
mod telemetry;

pub use config::{MemoryComponent, SchedulerConfig};
pub use engine::{AdmissionError, Scheduler, SchedulerSnapshot};
pub use sequence::{
    IterationPhase, IterationPlan, IterationWork, PrefixRestore, PrefixRestoreKind, Sequence,
    SequenceStatus,
};
pub use telemetry::{IterationTelemetry, SchedulerMetrics};

/// llama.cpp's hard upper bound for sequence identifiers in one context.
pub const LLAMA_MAX_SEQ: usize = 256;
