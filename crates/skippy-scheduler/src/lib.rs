//! Iteration-level scheduling policy for Skippy staged serving.
//!
//! The crate owns policy, not a concrete server runtime. Every stage consumes
//! the same [`IterationPlan`], while `skippy-server` translates work items into
//! native ABI requests.

mod cache_policy;
mod capacity;
mod config;
mod engine;
mod sequence;
mod telemetry;

pub use cache_policy::{
    CacheAffinity, CacheAwareCandidate, StageCacheAffinity, order_cache_aware_candidates,
    select_cache_aware_candidate,
};
pub use capacity::{
    CapacityDemand, CapacityPlan, ComponentCapacitySnapshot, EvictableCacheEntry,
    plan_component_capacity,
};
pub use config::{MemoryComponent, SchedulerConfig};
pub use engine::{AdmissionError, Scheduler, SchedulerSnapshot};
pub use sequence::{
    IterationPhase, IterationPlan, IterationPrediction, IterationWork, PrefixRestore,
    PrefixRestoreKind, Sequence, SequenceStatus,
};
pub use telemetry::{IterationTelemetry, SchedulerMetrics};

/// llama.cpp's hard upper bound for sequence identifiers in one context.
pub const LLAMA_MAX_SEQ: usize = 256;
