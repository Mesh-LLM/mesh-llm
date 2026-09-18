//! Shared model preparation for standalone and embedded Skippy serving.
mod checkpoint;
pub mod family_policy;
pub mod stage;
pub use stage::{SingleStageOptions, StageSourceIdentity, single_stage_config};

pub mod hash_cache;
pub mod package;

pub mod source;
pub mod source_registry;

pub mod materialization;
pub mod split_certification;
pub mod stage_admission;
pub mod stage_load;
