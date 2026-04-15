//! Qwen2/Llama-style transformer model running on MLX via mlx-rs.
//!
//! Loads quantized safetensors and runs inference entirely on Metal GPU.
//! No Python, no subprocess — just Rust + MLX C library.

mod artifacts;
mod attention;
mod attention_kind;
mod cache;
mod config;
mod embedding;
mod families;
mod family;
mod kimi;
mod layer;
mod lfm2;
mod loader;
mod mlp;
mod primitives;

use anyhow::{bail, Context, Result};
#[cfg(test)]
use artifacts::patch_phi3_special_token_whitespace;
use artifacts::TensorPrefixes;
pub use artifacts::{is_mlx_model_dir, mlx_model_dir, TokenizerSpacingPatch};
use attention::{attention_mask, Attention, DeepseekV3Attention};
use attention_kind::AttentionKind;
pub use cache::KVCache;
use cache::{CachedKv, QuantizedCacheArrays};
#[cfg(test)]
use config::effective_text_config_json;
use config::experimental_quantized_kv_config;
pub(crate) use config::ModelConfig;
use embedding::{quant_params_for, QuantizedEmbedding};
#[cfg(test)]
use family::config_supports_mlx;
pub use family::ReasoningFamily;
use family::{reasoning_family, ModelArchitecture};
use kimi::{KimiDeltaAttention, KimiMlaAttention, KimiShortConv};
use layer::Layer;
use lfm2::Lfm2ShortConv;
use mlp::{Activation, DeepseekV3MoE, GptOssMoE, MlpKind, QuantizedSwitchLinear, MLP};
use mlx_rs::array;
use mlx_rs::ops::indexing::{IndexOp, TryIndexMutOp};
use mlx_rs::ops::{conv1d, pad};
use mlx_rs::Array;
use mlx_rs::Dtype;
use primitives::{
    cpu_dense_weight_t, layer_norm_kind, quantize_stacked_weights, rms_norm_kind, unit_rms_norm,
    NormKind, QuantizedLinear, QuantizedMultiLinear, RMSNorm,
};

#[derive(Debug, serde::Deserialize)]
pub struct QuantConfig {
    pub group_size: i32,
    pub bits: i32,
}

#[derive(Debug, serde::Deserialize)]
struct QuantOverride {
    #[serde(default)]
    group_size: Option<i32>,
    #[serde(default)]
    bits: Option<i32>,
}

// ── Full model ──

pub struct MlxModel {
    embed_tokens: QuantizedEmbedding,
    embed_scale: f32,
    embed_tokens_per_layer: Option<QuantizedEmbedding>,
    embed_tokens_per_layer_scale: Option<f32>,
    per_layer_projection_norm: Option<NormKind>,
    per_layer_model_projection: Option<QuantizedLinear>,
    per_layer_model_projection_scale: Option<f32>,
    per_layer_input_scale: Option<f32>,
    layers: Vec<Layer>,
    norm: NormKind,
    lm_head: Option<QuantizedLinear>,
    final_logit_softcapping: Option<f32>,
    pub config: ModelConfig,
    pub tokenizer: tokenizers::Tokenizer,
    pub tokenizer_spacing_patch: Option<TokenizerSpacingPatch>,
    pub prompt_template: crate::mlx::template::PromptTemplate,
    pub reasoning_family: ReasoningFamily,
    architecture: ModelArchitecture,
    tokenwise_prefill: bool,
    cacheless_generation: bool,
    prompt_cache_reuse: bool,
}

impl MlxModel {
    /// Run a forward pass. Input shape: [1, seq_len] of u32 token IDs.
    /// Returns logits [1, seq_len, vocab_size].
    pub fn forward(&self, tokens: &Array, caches: &mut [KVCache]) -> Result<Array> {
        let mut h = self.embed_tokens.forward(tokens)?;
        if self.embed_scale != 1.0 {
            h = h.multiply(&array!(self.embed_scale))?;
        }
        let per_layer_inputs = if let (
            Some(embed_tokens_per_layer),
            Some(embed_tokens_per_layer_scale),
            Some(per_layer_projection_norm),
            Some(per_layer_model_projection),
            Some(per_layer_model_projection_scale),
            Some(per_layer_input_scale),
            Some(hidden_size_per_layer_input),
        ) = (
            &self.embed_tokens_per_layer,
            self.embed_tokens_per_layer_scale,
            &self.per_layer_projection_norm,
            &self.per_layer_model_projection,
            self.per_layer_model_projection_scale,
            self.per_layer_input_scale,
            self.config.hidden_size_per_layer_input,
        ) {
            let per_layer_inputs = embed_tokens_per_layer
                .forward(tokens)?
                .multiply(&array!(embed_tokens_per_layer_scale))?
                .reshape(&[
                    tokens.shape()[0],
                    tokens.shape()[1],
                    self.config.num_hidden_layers,
                    hidden_size_per_layer_input,
                ])?;
            let per_layer_projection = per_layer_model_projection
                .forward(&h)?
                .multiply(&array!(per_layer_model_projection_scale))?
                .reshape(&[
                    h.shape()[0],
                    h.shape()[1],
                    self.config.num_hidden_layers,
                    hidden_size_per_layer_input,
                ])?;
            let per_layer_projection = per_layer_projection_norm.forward(&per_layer_projection)?;
            Some(
                (&per_layer_projection + &per_layer_inputs)
                    .multiply(&array!(per_layer_input_scale))?,
            )
        } else {
            None
        };
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_input = per_layer_inputs.as_ref().map(|inputs| {
                inputs.index((
                    std::ops::RangeFull,
                    std::ops::RangeFull,
                    i as i32,
                    std::ops::RangeFull,
                ))
            });
            let (before, current_and_after) = caches.split_at_mut(i);
            let current_cache = &mut current_and_after[0];
            let shared_cache = layer
                .attn
                .kv_shared_source()
                .and_then(|source| before.get(source));
            h = layer.forward(&h, layer_input.as_ref(), current_cache, shared_cache)?;
        }
        let h = self.norm.forward(&h)?;

        let h_for_logits = if matches!(self.norm, NormKind::Layer(_)) {
            h.as_dtype(Dtype::Float32)?
        } else {
            h.clone()
        };
        let logits = if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&h_for_logits)?
        } else {
            self.embed_tokens.as_linear().forward(&h_for_logits)?
        };
        if let Some(softcap) = self.final_logit_softcapping {
            let scaled = logits.divide(&array!(softcap))?;
            Ok(mlx_rs::ops::tanh(&scaled)?.multiply(&array!(softcap))?)
        } else {
            Ok(logits)
        }
    }

    pub fn forward_no_cache(&self, tokens: &Array) -> Result<Array> {
        let mut h = self.embed_tokens.forward(tokens)?;
        if self.embed_scale != 1.0 {
            h = h.multiply(&array!(self.embed_scale))?;
        }
        let per_layer_inputs = if let (
            Some(embed_tokens_per_layer),
            Some(embed_tokens_per_layer_scale),
            Some(per_layer_projection_norm),
            Some(per_layer_model_projection),
            Some(per_layer_model_projection_scale),
            Some(per_layer_input_scale),
            Some(hidden_size_per_layer_input),
        ) = (
            &self.embed_tokens_per_layer,
            self.embed_tokens_per_layer_scale,
            &self.per_layer_projection_norm,
            &self.per_layer_model_projection,
            self.per_layer_model_projection_scale,
            self.per_layer_input_scale,
            self.config.hidden_size_per_layer_input,
        ) {
            let per_layer_inputs = embed_tokens_per_layer
                .forward(tokens)?
                .multiply(&array!(embed_tokens_per_layer_scale))?
                .reshape(&[
                    tokens.shape()[0],
                    tokens.shape()[1],
                    self.config.num_hidden_layers,
                    hidden_size_per_layer_input,
                ])?;
            let per_layer_projection = per_layer_model_projection
                .forward(&h)?
                .multiply(&array!(per_layer_model_projection_scale))?
                .reshape(&[
                    h.shape()[0],
                    h.shape()[1],
                    self.config.num_hidden_layers,
                    hidden_size_per_layer_input,
                ])?;
            let per_layer_projection = per_layer_projection_norm.forward(&per_layer_projection)?;
            Some(
                (&per_layer_projection + &per_layer_inputs)
                    .multiply(&array!(per_layer_input_scale))?,
            )
        } else {
            None
        };
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_input = per_layer_inputs.as_ref().map(|inputs| {
                inputs.index((
                    std::ops::RangeFull,
                    std::ops::RangeFull,
                    i as i32,
                    std::ops::RangeFull,
                ))
            });
            h = layer.forward_no_cache(&h, layer_input.as_ref())?;
        }
        let h = self.norm.forward(&h)?;

        let h_for_logits = if matches!(self.norm, NormKind::Layer(_)) {
            h.as_dtype(Dtype::Float32)?
        } else {
            h.clone()
        };
        let logits = if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(&h_for_logits)?
        } else {
            self.embed_tokens.as_linear().forward(&h_for_logits)?
        };
        if let Some(softcap) = self.final_logit_softcapping {
            let scaled = logits.divide(&array!(softcap))?;
            Ok(mlx_rs::ops::tanh(&scaled)?.multiply(&array!(softcap))?)
        } else {
            Ok(logits)
        }
    }

    pub fn new_caches(&self) -> Vec<KVCache> {
        let quantized_kv = experimental_quantized_kv_config();
        self.layers
            .iter()
            .map(|layer| {
                if let Some(window_size) = layer.attn.sliding_window_size() {
                    KVCache::new_rotating(window_size, 0)
                } else if layer.attn.kv_shared_source().is_some() {
                    KVCache::new()
                } else if let Some((group_size, bits, min_dense_tokens)) = quantized_kv {
                    KVCache::new_quantized(group_size, bits, min_dense_tokens)
                } else {
                    KVCache::new()
                }
            })
            .collect()
    }

    pub fn tokenwise_prefill(&self) -> bool {
        self.tokenwise_prefill
    }

    pub fn can_replay_prompt_logits(&self) -> bool {
        !self.architecture.is_gemma3() && !self.architecture.is_gemma4()
    }

    pub fn cacheless_generation(&self) -> bool {
        self.cacheless_generation
    }

    pub fn prompt_cache_reuse(&self) -> bool {
        self.prompt_cache_reuse && experimental_quantized_kv_config().is_none()
    }
}

/// Argmax over the last position's logits. Returns the token ID.
pub fn argmax_last(logits: &Array) -> Result<u32> {
    let shape = logits.shape();
    let flat = if shape.len() == 3 {
        let last_idx = (shape[1] - 1) as i32;
        let idx = Array::from_int(last_idx);
        logits.take_axis(&idx, 1)?.reshape(&[-1])?
    } else {
        logits.reshape(&[-1])?
    };
    let token = mlx_rs::ops::indexing::argmax(&flat, false)?;
    mlx_rs::transforms::eval([&token])?;
    Ok(token.as_slice::<u32>()[0])
}

#[cfg(test)]
mod tests;
