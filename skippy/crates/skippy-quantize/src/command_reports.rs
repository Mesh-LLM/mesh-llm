use std::path::PathBuf;

use serde::Serialize;

#[derive(Debug, Serialize)]
pub(crate) struct TensorTypeValidation {
    pub(crate) valid: bool,
    pub(crate) entry_count: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct QuantWindowPlan {
    pub(crate) first_split: u32,
    pub(crate) last_split: u32,
    pub(crate) staged_first_shard: PathBuf,
    pub(crate) output_prefix: PathBuf,
    pub(crate) command: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ConvertWindowPlan {
    pub(crate) first_split: u32,
    pub(crate) last_split: u32,
    pub(crate) output_prefix: PathBuf,
    pub(crate) command: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct SplitValidation {
    pub(crate) root: PathBuf,
    pub(crate) prefix: String,
    pub(crate) expected_splits: u32,
    pub(crate) completed_count: usize,
    pub(crate) first_missing: Option<u32>,
    pub(crate) last_present: Option<u32>,
    pub(crate) complete: bool,
}
