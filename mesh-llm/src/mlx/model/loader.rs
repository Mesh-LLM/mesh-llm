use super::artifacts::{load_all_safetensors, load_tokenizer, tensor_prefixes};
use super::config::{
    attention_window_size_for_layer, kv_shared_source_for_layer, normalized_model_config_json,
};
use super::families::{
    apply_family_tensor_transforms, build_deepseek_v3_layer, build_gpt_oss_layer,
    build_kimi_linear_layer, build_lfm2_layer, build_standard_layer,
};
use super::family::{ensure_supported_mlx_model, model_architecture, uses_traditional_rope};
use super::*;
use serde_json::Value;
use std::path::Path;

impl MlxModel {
    /// Load an MLX model from a directory containing config.json,
    /// tokenizer.json, and model.safetensors.
    pub fn load(dir: &Path) -> Result<Self> {
        tracing::info!("MLX: loading model directory {}", dir.display());
        let config_text =
            std::fs::read_to_string(dir.join("config.json")).context("reading config.json")?;
        let config_json: Value =
            serde_json::from_str(&config_text).context("parsing config.json")?;
        ensure_supported_mlx_model(dir, &config_json)?;
        let effective_config_json = normalized_model_config_json(&config_json);
        let arch = model_architecture(&config_json);
        let mut config: ModelConfig =
            serde_json::from_value(effective_config_json).context("parsing config.json")?;
        if arch.is_gemma3() {
            config.eos_token_id.retain(|id| *id != 106);
        }
        let rope_traditional = uses_traditional_rope(&config_json);

        let quantized = config.quantization.as_ref();
        let default_group_size = quantized.map(|q| q.group_size).unwrap_or(0);
        let default_bits = quantized.map(|q| q.bits).unwrap_or(0);

        if let Some(qcfg) = quantized {
            tracing::info!(
                "MLX: loading {} layers, hidden={}, heads={}/{}, quant={}bit/g{}",
                config.num_hidden_layers,
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                qcfg.bits,
                qcfg.group_size,
            );
        } else {
            tracing::info!(
                "MLX: loading {} layers, hidden={}, heads={}/{}, dense_dtype={:?}",
                config.num_hidden_layers,
                config.hidden_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config_json
                    .get("torch_dtype")
                    .and_then(|value| value.as_str())
                    .unwrap_or("unknown"),
            );
        }

        let start = std::time::Instant::now();
        let mut tensors = load_all_safetensors(dir)?;
        tracing::info!(
            "MLX: loaded {} tensors in {:.2}s",
            tensors.len(),
            start.elapsed().as_secs_f64()
        );
        let prefixes = tensor_prefixes(&tensors)?;
        apply_family_tensor_transforms(
            arch,
            &mut tensors,
            &prefixes,
            &config,
            &config_json,
            default_group_size,
            default_bits,
        )?;

        let load_qlinear = |prefix: &str| -> Result<QuantizedLinear> {
            let weight = tensors
                .get(&format!("{prefix}.weight"))
                .cloned()
                .with_context(|| format!("missing {prefix}.weight"))?;
            let bias = tensors.get(&format!("{prefix}.bias")).cloned();
            let scales_key = format!("{prefix}.scales");
            let biases_key = format!("{prefix}.biases");
            let has_quantized_storage =
                tensors.contains_key(&scales_key) && tensors.contains_key(&biases_key);
            let dense_weight_t = if quantized.is_none() || !has_quantized_storage {
                Some(weight.transpose_axes(&[1, 0])?)
            } else {
                let (group_size, bits) =
                    quant_params_for(&config_json, prefix, default_group_size, default_bits);
                let scales = tensors
                    .get(&scales_key)
                    .cloned()
                    .with_context(|| format!("missing {prefix}.scales"))?;
                let biases = tensors
                    .get(&biases_key)
                    .cloned()
                    .with_context(|| format!("missing {prefix}.biases"))?;
                if bits == 5 {
                    Some(cpu_dense_weight_t(
                        &weight, &scales, &biases, group_size, bits,
                    )?)
                } else {
                    None
                }
            };
            let (group_size, bits) = if quantized.is_some() && has_quantized_storage {
                quant_params_for(&config_json, prefix, default_group_size, default_bits)
            } else {
                (0, 0)
            };
            let scales = tensors
                .get(&scales_key)
                .cloned()
                .unwrap_or_else(|| array!(0.0f32));
            let biases = tensors
                .get(&biases_key)
                .cloned()
                .unwrap_or_else(|| array!(0.0f32));
            Ok(QuantizedLinear {
                weight,
                scales,
                biases,
                bias,
                group_size,
                bits,
                dense_weight_t,
            })
        };

        let load_multi_linear = |prefix: &str| -> Result<QuantizedMultiLinear> {
            let (group_size, bits) =
                quant_params_for(&config_json, prefix, default_group_size, default_bits);
            Ok(QuantizedMultiLinear {
                weight: tensors
                    .get(&format!("{prefix}.weight"))
                    .cloned()
                    .with_context(|| format!("missing {prefix}.weight"))?,
                scales: tensors
                    .get(&format!("{prefix}.scales"))
                    .cloned()
                    .with_context(|| format!("missing {prefix}.scales"))?,
                biases: tensors
                    .get(&format!("{prefix}.biases"))
                    .cloned()
                    .with_context(|| format!("missing {prefix}.biases"))?,
                group_size,
                bits,
            })
        };

        let load_switch_linear = |prefix: &str| -> Result<QuantizedSwitchLinear> {
            let (group_size, bits) =
                quant_params_for(&config_json, prefix, default_group_size, default_bits);
            Ok(QuantizedSwitchLinear {
                weight: tensors
                    .get(&format!("{prefix}.weight"))
                    .cloned()
                    .with_context(|| format!("missing {prefix}.weight"))?,
                scales: tensors
                    .get(&format!("{prefix}.scales"))
                    .cloned()
                    .with_context(|| format!("missing {prefix}.scales"))?,
                biases: tensors
                    .get(&format!("{prefix}.biases"))
                    .cloned()
                    .with_context(|| format!("missing {prefix}.biases"))?,
                bias: tensors.get(&format!("{prefix}.bias")).cloned(),
                group_size,
                bits,
            })
        };

        let load_lfm2_conv_weight = |prefix: &str| -> Result<Array> {
            let weight = tensors
                .get(&format!("{prefix}.weight"))
                .cloned()
                .with_context(|| format!("missing {prefix}.weight"))?;
            if weight.ndim() == 3 && weight.shape()[2] > weight.shape()[1] {
                Ok(weight.transpose_axes(&[0, 2, 1])?)
            } else {
                Ok(weight)
            }
        };

        let (embed_group_size, embed_bits) = quant_params_for(
            &config_json,
            &format!("{}.embed_tokens", prefixes.model),
            default_group_size,
            default_bits,
        );
        let embed_weight = tensors
            .get(&format!("{}.embed_tokens.weight", prefixes.model))
            .cloned()
            .with_context(|| format!("missing {}.embed_tokens.weight", prefixes.model))?;
        let embed_scales = tensors
            .get(&format!("{}.embed_tokens.scales", prefixes.model))
            .cloned()
            .unwrap_or_else(|| array!(0.0f32));
        let embed_biases = tensors
            .get(&format!("{}.embed_tokens.biases", prefixes.model))
            .cloned()
            .unwrap_or_else(|| array!(0.0f32));
        let embed_dense_weight = quantized.is_none().then(|| embed_weight.clone());
        let embed_dense_weight_t = if quantized.is_none() {
            Some(embed_weight.transpose_axes(&[1, 0])?)
        } else {
            None
        };
        let embed_tokens = QuantizedEmbedding {
            weight: embed_weight,
            scales: embed_scales,
            biases: embed_biases,
            group_size: embed_group_size,
            bits: embed_bits,
            dense_weight: embed_dense_weight,
            dense_weight_t: embed_dense_weight_t,
        };
        let embed_scale = if arch.uses_gemma_scaled_embeddings() {
            (config.hidden_size as f32).sqrt()
        } else {
            1.0
        };
        let embed_tokens_per_layer = if arch.is_gemma4() {
            let (group_size, bits) = quant_params_for(
                &config_json,
                &format!("{}.embed_tokens_per_layer", prefixes.model),
                default_group_size,
                default_bits,
            );
            Some(QuantizedEmbedding {
                weight: tensors
                    .get(&format!("{}.embed_tokens_per_layer.weight", prefixes.model))
                    .cloned()
                    .with_context(|| {
                        format!("missing {}.embed_tokens_per_layer.weight", prefixes.model)
                    })?,
                scales: tensors
                    .get(&format!("{}.embed_tokens_per_layer.scales", prefixes.model))
                    .cloned()
                    .unwrap_or_else(|| array!(0.0f32)),
                biases: tensors
                    .get(&format!("{}.embed_tokens_per_layer.biases", prefixes.model))
                    .cloned()
                    .unwrap_or_else(|| array!(0.0f32)),
                group_size,
                bits,
                dense_weight: quantized.is_none().then(|| {
                    tensors[&format!("{}.embed_tokens_per_layer.weight", prefixes.model)].clone()
                }),
                dense_weight_t: if quantized.is_none() {
                    Some(
                        tensors[&format!("{}.embed_tokens_per_layer.weight", prefixes.model)]
                            .transpose_axes(&[1, 0])?,
                    )
                } else {
                    None
                },
            })
        } else {
            None
        };
        let per_layer_projection_norm = if arch.is_gemma4() {
            Some(rms_norm_kind(
                tensors
                    .get(&format!(
                        "{}.per_layer_projection_norm.weight",
                        prefixes.model
                    ))
                    .cloned()
                    .with_context(|| {
                        format!(
                            "missing {}.per_layer_projection_norm.weight",
                            prefixes.model
                        )
                    })?,
                config.rms_norm_eps,
                false,
            ))
        } else {
            None
        };
        let per_layer_model_projection = if arch.is_gemma4() {
            Some(load_qlinear(&format!(
                "{}.per_layer_model_projection",
                prefixes.model
            ))?)
        } else {
            None
        };

        let norm = if arch.is_olmo() {
            layer_norm_kind(1e-5)
        } else {
            rms_norm_kind(
                if arch.is_lfm2() {
                    tensors
                        .get(&format!("{}.embedding_norm.weight", prefixes.model))
                        .cloned()
                        .with_context(|| {
                            format!("missing {}.embedding_norm.weight", prefixes.model)
                        })?
                } else {
                    tensors
                        .get(&format!("{}.norm.weight", prefixes.model))
                        .cloned()
                        .with_context(|| format!("missing {}.norm.weight", prefixes.model))?
                },
                config.block_norm_eps.unwrap_or(config.rms_norm_eps),
                arch.uses_gemma_norm_offset(),
            )
        };

        let head_dim = config
            .head_dim
            .unwrap_or_else(|| config.hidden_size / config.num_attention_heads);
        let first_kv_shared_layer_idx = config
            .num_kv_shared_layers
            .map(|n| (config.num_hidden_layers - n).max(0) as usize)
            .unwrap_or(config.num_hidden_layers as usize);
        let non_shared_layer_types = config
            .layer_types
            .as_ref()
            .map(|types| types[..first_kv_shared_layer_idx.min(types.len())].to_vec());

        let mut layers = Vec::new();
        for i in 0..config.num_hidden_layers {
            let p = format!("{}.layers.{i}", prefixes.model);
            let layer_type = config
                .layer_types
                .as_ref()
                .and_then(|types| types.get(i as usize))
                .map(String::as_str);
            if arch.is_deepseek_v3() {
                layers.push(build_deepseek_v3_layer(
                    &tensors,
                    &p,
                    i,
                    &config,
                    &load_qlinear,
                    &load_multi_linear,
                    &load_switch_linear,
                )?);
                continue;
            }
            if arch.is_lfm2() {
                layers.push(build_lfm2_layer(
                    &tensors,
                    &p,
                    i,
                    &config,
                    head_dim,
                    &load_qlinear,
                    &load_lfm2_conv_weight,
                )?);
                continue;
            }
            if arch.is_kimi_linear() {
                layers.push(build_kimi_linear_layer(
                    &tensors,
                    &p,
                    i,
                    &config,
                    &load_qlinear,
                    &load_multi_linear,
                    &load_switch_linear,
                    &load_lfm2_conv_weight,
                )?);
                continue;
            }
            if arch.is_gpt_oss() {
                let window_size =
                    attention_window_size_for_layer(arch, &config, i as usize, layer_type)?;
                layers.push(build_gpt_oss_layer(
                    &tensors,
                    &p,
                    &config,
                    head_dim,
                    window_size,
                    &load_qlinear,
                    &load_switch_linear,
                )?);
                continue;
            }
            layers.push(build_standard_layer(
                &tensors,
                &p,
                i as usize,
                arch,
                &config,
                layer_type,
                head_dim,
                rope_traditional,
                non_shared_layer_types.as_deref(),
                &load_qlinear,
                &attention_window_size_for_layer,
                &kv_shared_source_for_layer,
            )?);
        }

        let lm_head = if config.tie_word_embeddings {
            None
        } else if let Some(prefix) = prefixes.lm_head.as_deref() {
            if tensors.contains_key(&format!("{prefix}.weight")) {
                Some(load_qlinear(prefix)?)
            } else {
                None
            }
        } else {
            None
        };

        let (tokenizer, tokenizer_spacing_patch) = load_tokenizer(dir, &config_json)?;
        let prompt_template = crate::mlx::template::PromptTemplate::detect(dir, &config_json);

        Ok(MlxModel {
            embed_tokens,
            embed_scale,
            embed_tokens_per_layer,
            embed_tokens_per_layer_scale: arch.is_gemma4().then(|| {
                config
                    .hidden_size_per_layer_input
                    .map(|dim| (dim as f32).sqrt())
                    .unwrap_or(1.0)
            }),
            per_layer_projection_norm,
            per_layer_model_projection,
            per_layer_model_projection_scale: arch
                .is_gemma4()
                .then_some((config.hidden_size as f32).powf(-0.5)),
            per_layer_input_scale: arch.is_gemma4().then_some(2.0f32.powf(-0.5)),
            layers,
            norm,
            lm_head,
            final_logit_softcapping: config.final_logit_softcapping,
            config,
            tokenizer,
            tokenizer_spacing_patch,
            prompt_template,
            reasoning_family: reasoning_family(&config_json),
            architecture: arch,
            tokenwise_prefill: arch.is_gemma2() || arch.is_gemma3() || arch.is_gemma4(),
            cacheless_generation: arch.is_gemma2()
                || arch.is_gpt_oss()
                || arch.is_kimi_linear()
                || arch.is_lfm2(),
            prompt_cache_reuse: !arch.is_gemma4(),
        })
    }
}
