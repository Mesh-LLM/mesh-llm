//! Shared model preparation for standalone and embedded Skippy serving.
mod checkpoint;
pub mod family_policy;
pub mod stage;
pub use stage::{SingleStageOptions, StageSourceIdentity, single_stage_config};
