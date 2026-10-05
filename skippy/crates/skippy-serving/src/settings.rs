//! Operator overrides shared by standalone local and binary-stage serving.

use std::path::PathBuf;

use crate::frontend::{EmbeddedOpenAiRequestDefaults, InferenceGuardrailsConfig};

#[derive(Clone, Debug, Default)]
pub struct ServingTuning {
    pub n_threads: Option<usize>,
    pub n_threads_batch: Option<usize>,
    pub continuous_batching: Option<bool>,
    pub pipeline_decode_groups: Option<usize>,
    pub request_defaults: EmbeddedOpenAiRequestDefaults,
    pub guardrails: Option<InferenceGuardrailsConfig>,
    pub draft_model_path: Option<PathBuf>,
    pub speculative_window: Option<usize>,
    pub adaptive_speculative_window: Option<bool>,
    pub draft_n_gpu_layers: Option<i32>,
    pub native_mtp_draft_model_path: Option<PathBuf>,
}

impl ServingTuning {
    pub fn validate(&self) -> anyhow::Result<()> {
        for (name, value) in [
            ("threads", self.n_threads),
            ("threads-batch", self.n_threads_batch),
            ("pipeline-decode-groups", self.pipeline_decode_groups),
        ] {
            anyhow::ensure!(value != Some(0), "--{name} must be greater than zero");
        }
        self.request_defaults.validate()?;
        Ok(())
    }
}
