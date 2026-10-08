pub mod jobs;
pub mod permissions;
pub mod prepare;
pub mod script;

use anyhow::{Context, Result};

/// Build an `HFClient` from environment, suitable for API calls.
///
/// Resolves endpoint from `HF_ENDPOINT` and token from `HF_TOKEN` /
/// `HUGGING_FACE_HUB_TOKEN`.
pub fn build_hf_client() -> Result<hf_hub::HFClient> {
    let _ = skippy_model_hf::configure_hf_tls_provider();
    let mut builder =
        hf_hub::HFClientBuilder::new().cache_dir(skippy_model_hf::huggingface_hub_cache_dir());

    if let Some(endpoint) = std::env::var("HF_ENDPOINT")
        .ok()
        .filter(|s| !s.trim().is_empty())
    {
        builder = builder.endpoint(endpoint);
    }

    if let Some(token) = skippy_model_hf::hf_token_override() {
        builder = builder.token(token);
    }

    builder.build().context("build HuggingFace API client")
}

pub mod snapshot_promotion;

pub mod layer_job;

/// Immutable native competitive inputs and semantic tokenizer exports.
pub mod competitive_acquisition;

pub mod hf_checkpoint_stitch;
