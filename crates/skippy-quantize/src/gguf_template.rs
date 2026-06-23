use std::fs;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use crate::gguf_writer::GgufKv;
use crate::tokenizer_metadata::push_tokenizer_metadata;

#[derive(Debug, Clone, Copy)]
pub(crate) struct MetadataOptions {
    pub(crate) include_mtp: bool,
}

impl Default for MetadataOptions {
    fn default() -> Self {
        Self { include_mtp: true }
    }
}

pub(crate) fn metadata_from_hf_config(source: &Path, tensor_count: usize) -> Result<Vec<GgufKv>> {
    metadata_from_hf_config_with_options(source, tensor_count, MetadataOptions::default())
}

pub(crate) fn mtp_layer_start_from_hf_config(source: &Path) -> Result<Option<u32>> {
    let config = read_hf_config(source)?;
    let Some(nextn_layers) = optional_u32(&config, "num_nextn_predict_layers")
        .or_else(|| optional_u32(&config, "mtp_num_hidden_layers"))
    else {
        return Ok(None);
    };
    if nextn_layers == 0 {
        return Ok(None);
    }
    required_u32(&config, "num_hidden_layers").map(Some)
}

pub(crate) fn metadata_from_hf_config_with_options(
    source: &Path,
    tensor_count: usize,
    options: MetadataOptions,
) -> Result<Vec<GgufKv>> {
    let config = read_hf_config(source)?;
    let arch = architecture_name(&config)?;
    let mut metadata = vec![
        GgufKv::string("general.architecture", arch),
        GgufKv::string("general.name", model_name(source)),
        GgufKv::bool("skippy.convert.raw_safetensors", false),
        GgufKv::u64("skippy.convert.tensor_count", tensor_count as u64),
    ];
    push_common_llm_metadata(&mut metadata, arch, &config, options)?;
    push_attention_metadata(&mut metadata, arch, &config)?;
    push_glm_dsa_indexer_metadata(&mut metadata, arch, &config)?;
    push_moe_metadata(&mut metadata, arch, &config);
    push_tokenizer_metadata(&mut metadata, source, &config)?;
    if options.include_mtp {
        push_if_u32(
            &mut metadata,
            arch,
            "nextn_predict_layers",
            &config,
            "num_nextn_predict_layers",
        );
    }
    Ok(metadata)
}

fn read_hf_config(source: &Path) -> Result<Value> {
    let config_path = source.join("config.json");
    serde_json::from_slice(
        &fs::read(&config_path).with_context(|| format!("read {}", config_path.display()))?,
    )
    .with_context(|| format!("parse {}", config_path.display()))
}

fn architecture_name(config: &Value) -> Result<&'static str> {
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if matches!(model_type, "glm4_moe_lite" | "deepseek_v2") {
        return Ok("deepseek2");
    }
    if matches!(model_type, "glm4_moe" | "glm4v_moe") {
        return Ok("glm4moe");
    }
    if matches!(model_type, "glm_moe_dsa" | "glm-dsa") {
        return Ok("glm-dsa");
    }
    if is_unsupported_qwen3_variant(model_type) {
        anyhow::bail!(
            "native GGUF metadata for model_type={model_type:?} requires \
             Qwen3.5/Qwen3Next/Qwen3VL-specific metadata and tensor support; \
             use the external convert_hf_to_gguf.py backend"
        );
    }
    if model_type.starts_with("qwen3_moe") {
        return Ok("qwen3moe");
    }
    if model_type.starts_with("qwen2_moe") {
        return Ok("qwen2moe");
    }
    if model_type.starts_with("qwen3") {
        return Ok("qwen3");
    }
    if model_type.starts_with("qwen2") {
        return Ok("qwen2");
    }
    if matches!(model_type, "llama" | "mistral") {
        return Ok("llama");
    }
    anyhow::bail!("unsupported native GGUF metadata template for model_type={model_type:?}")
}

fn is_unsupported_qwen3_variant(model_type: &str) -> bool {
    let normalized = model_type
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    normalized.starts_with("qwen3next")
        || normalized.starts_with("qwen3vl")
        || normalized.starts_with("qwen35")
}

fn push_common_llm_metadata(
    metadata: &mut Vec<GgufKv>,
    arch: &str,
    config: &Value,
    options: MetadataOptions,
) -> Result<()> {
    push_required_u32(metadata, arch, "vocab_size", config, "vocab_size")?;
    push_required_u32(
        metadata,
        arch,
        "context_length",
        config,
        "max_position_embeddings",
    )?;
    push_required_u32(metadata, arch, "embedding_length", config, "hidden_size")?;
    let block_count = required_u32(config, "num_hidden_layers")?
        + if options.include_mtp {
            optional_u32(config, "num_nextn_predict_layers").unwrap_or(0)
        } else {
            0
        };
    metadata.push(GgufKv::u32(&format!("{arch}.block_count"), block_count));
    push_if_u32(
        metadata,
        arch,
        "feed_forward_length",
        config,
        "intermediate_size",
    );
    if let Some(eps) = optional_f32(config, "rms_norm_eps") {
        metadata.push(GgufKv::f32(
            &format!("{arch}.attention.layer_norm_rms_epsilon"),
            eps,
        ));
    }
    Ok(())
}

fn push_attention_metadata(metadata: &mut Vec<GgufKv>, arch: &str, config: &Value) -> Result<()> {
    let head_count = required_u32(config, "num_attention_heads")?;
    metadata.push(GgufKv::u32(
        &format!("{arch}.attention.head_count"),
        head_count,
    ));
    metadata.push(GgufKv::u32(
        &format!("{arch}.attention.head_count_kv"),
        optional_u32(config, "num_key_value_heads").unwrap_or(head_count),
    ));
    if let Some((nope, rope)) =
        optional_u32(config, "qk_nope_head_dim").zip(optional_u32(config, "qk_rope_head_dim"))
    {
        metadata.push(GgufKv::u32(
            &format!("{arch}.attention.key_length"),
            nope + rope,
        ));
        metadata.push(GgufKv::u32(&format!("{arch}.rope.dimension_count"), rope));
    } else {
        let head_dim = optional_u32(config, "head_dim").unwrap_or(
            required_u32(config, "hidden_size")? / required_u32(config, "num_attention_heads")?,
        );
        metadata.push(GgufKv::u32(
            &format!("{arch}.attention.key_length"),
            head_dim,
        ));
        metadata.push(GgufKv::u32(
            &format!("{arch}.rope.dimension_count"),
            head_dim,
        ));
    }
    let value_len = optional_u32(config, "v_head_dim")
        .or_else(|| optional_u32(config, "head_dim"))
        .unwrap_or(
            required_u32(config, "hidden_size")? / required_u32(config, "num_attention_heads")?,
        );
    metadata.push(GgufKv::u32(
        &format!("{arch}.attention.value_length"),
        value_len,
    ));
    if let Some(theta) = optional_f32(config, "rope_theta") {
        metadata.push(GgufKv::f32(&format!("{arch}.rope.freq_base"), theta));
    }
    push_if_u32(
        metadata,
        arch,
        "attention.q_lora_rank",
        config,
        "q_lora_rank",
    );
    push_if_u32(
        metadata,
        arch,
        "attention.kv_lora_rank",
        config,
        "kv_lora_rank",
    );
    Ok(())
}

fn push_glm_dsa_indexer_metadata(
    metadata: &mut Vec<GgufKv>,
    arch: &str,
    config: &Value,
) -> Result<()> {
    if arch != "glm-dsa" {
        return Ok(());
    }
    push_required_first_u32(
        metadata,
        arch,
        "attention.indexer.head_count",
        config,
        &["index_n_heads", "indexer_n_head"],
    )?;
    push_required_first_u32(
        metadata,
        arch,
        "attention.indexer.key_length",
        config,
        &["index_head_dim", "indexer_head_size"],
    )?;
    push_required_first_u32(
        metadata,
        arch,
        "attention.indexer.top_k",
        config,
        &["index_topk", "indexer_top_k"],
    )?;
    Ok(())
}

fn push_moe_metadata(metadata: &mut Vec<GgufKv>, arch: &str, config: &Value) {
    push_first_u32(
        metadata,
        arch,
        "expert_count",
        config,
        &["n_routed_experts", "num_experts"],
    );
    push_first_u32(
        metadata,
        arch,
        "expert_used_count",
        config,
        &["num_experts_per_tok"],
    );
    push_first_u32(
        metadata,
        arch,
        "expert_shared_count",
        config,
        &["n_shared_experts"],
    );
    push_first_u32(
        metadata,
        arch,
        "expert_feed_forward_length",
        config,
        &["moe_intermediate_size"],
    );
    push_first_u32(
        metadata,
        arch,
        "expert_shared_feed_forward_length",
        config,
        &["shared_expert_intermediate_size"],
    );
    push_first_u32(
        metadata,
        arch,
        "leading_dense_block_count",
        config,
        &["first_k_dense_replace"],
    );
    if let Some(scale) = optional_f32(config, "routed_scaling_factor") {
        metadata.push(GgufKv::f32(&format!("{arch}.expert_weights_scale"), scale));
    }
    if let Some(norm) = optional_bool(config, "norm_topk_prob") {
        metadata.push(GgufKv::bool(&format!("{arch}.expert_weights_norm"), norm));
    }
}

fn push_required_first_u32(
    metadata: &mut Vec<GgufKv>,
    arch: &str,
    gguf_suffix: &str,
    config: &Value,
    config_keys: &[&str],
) -> Result<()> {
    for config_key in config_keys {
        if let Some(value) = optional_u32(config, config_key) {
            ensure!(
                value > 0,
                "config value {config_key:?} must be greater than zero"
            );
            metadata.push(GgufKv::u32(&format!("{arch}.{gguf_suffix}"), value));
            return Ok(());
        }
    }
    anyhow::bail!("config missing one of {config_keys:?}")
}

fn push_first_u32(
    metadata: &mut Vec<GgufKv>,
    arch: &str,
    gguf_suffix: &str,
    config: &Value,
    config_keys: &[&str],
) {
    for config_key in config_keys {
        if let Some(value) = optional_u32(config, config_key) {
            metadata.push(GgufKv::u32(&format!("{arch}.{gguf_suffix}"), value));
            return;
        }
    }
}

fn push_required_u32(
    metadata: &mut Vec<GgufKv>,
    arch: &str,
    gguf_suffix: &str,
    config: &Value,
    config_key: &str,
) -> Result<()> {
    metadata.push(GgufKv::u32(
        &format!("{arch}.{gguf_suffix}"),
        required_u32(config, config_key)?,
    ));
    Ok(())
}

fn push_if_u32(
    metadata: &mut Vec<GgufKv>,
    arch: &str,
    gguf_suffix: &str,
    config: &Value,
    config_key: &str,
) {
    if let Some(value) = optional_u32(config, config_key) {
        metadata.push(GgufKv::u32(&format!("{arch}.{gguf_suffix}"), value));
    }
}

fn required_u32(config: &Value, key: &str) -> Result<u32> {
    let value = optional_u32(config, key).with_context(|| format!("config missing {key:?}"))?;
    ensure!(value > 0, "config value {key:?} must be greater than zero");
    Ok(value)
}

fn optional_u32(config: &Value, key: &str) -> Option<u32> {
    config
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
}

fn optional_f32(config: &Value, key: &str) -> Option<f32> {
    config
        .get(key)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
}

fn optional_bool(config: &Value, key: &str) -> Option<bool> {
    config.get(key).and_then(Value::as_bool)
}

fn model_name(source: &Path) -> &str {
    source
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("checkpoint")
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn builds_glm_moe_lite_metadata_from_config() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("config.json"),
            r#"{
              "model_type": "glm4_moe_lite",
              "vocab_size": 154880,
              "max_position_embeddings": 202752,
              "hidden_size": 2048,
              "intermediate_size": 10240,
              "num_hidden_layers": 47,
              "num_nextn_predict_layers": 1,
              "num_attention_heads": 20,
              "num_key_value_heads": 20,
              "qk_nope_head_dim": 192,
              "qk_rope_head_dim": 64,
              "v_head_dim": 256,
              "rope_theta": 1000000,
              "q_lora_rank": 768,
              "kv_lora_rank": 512,
              "n_routed_experts": 64,
              "num_experts_per_tok": 4,
              "n_shared_experts": 1,
              "moe_intermediate_size": 1536,
              "first_k_dense_replace": 1,
              "routed_scaling_factor": 1.8,
              "norm_topk_prob": true,
              "rms_norm_eps": 1e-5
            }"#,
        )
        .unwrap();

        let metadata = metadata_from_hf_config(&root, 3).unwrap();
        let text = format!("{metadata:?}");

        assert!(text.contains("general.architecture"));
        assert!(text.contains("deepseek2"));
        assert!(text.contains("deepseek2.block_count"));
        assert!(text.contains("48"));
        assert!(text.contains("deepseek2.attention.key_length"));
        assert!(text.contains("256"));
        assert!(text.contains("deepseek2.nextn_predict_layers"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn omits_mtp_metadata_when_requested() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("config.json"),
            r#"{
              "model_type": "glm4_moe_lite",
              "vocab_size": 154880,
              "max_position_embeddings": 202752,
              "hidden_size": 2048,
              "intermediate_size": 10240,
              "num_hidden_layers": 47,
              "num_nextn_predict_layers": 1,
              "num_attention_heads": 20,
              "num_key_value_heads": 20,
              "qk_nope_head_dim": 192,
              "qk_rope_head_dim": 64,
              "v_head_dim": 256
            }"#,
        )
        .unwrap();

        let metadata =
            metadata_from_hf_config_with_options(&root, 3, MetadataOptions { include_mtp: false })
                .unwrap();

        assert!(metadata.iter().any(|kv| {
            matches!(
                kv,
                GgufKv::U32 { key, value }
                    if key == "deepseek2.block_count" && *value == 47
            )
        }));
        assert!(!metadata.iter().any(|kv| {
            matches!(
                kv,
                GgufKv::U32 { key, .. } if key == "deepseek2.nextn_predict_layers"
            )
        }));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn builds_glm_dsa_indexer_metadata_from_config() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("config.json"),
            r#"{
              "model_type": "glm_moe_dsa",
              "vocab_size": 154880,
              "max_position_embeddings": 1048576,
              "hidden_size": 6144,
              "intermediate_size": 12288,
              "num_hidden_layers": 78,
              "num_nextn_predict_layers": 1,
              "num_attention_heads": 64,
              "num_key_value_heads": 64,
              "qk_nope_head_dim": 192,
              "qk_rope_head_dim": 64,
              "v_head_dim": 256,
              "q_lora_rank": 2048,
              "kv_lora_rank": 512,
              "index_n_heads": 32,
              "index_head_dim": 128,
              "index_topk": 2048,
              "n_routed_experts": 256,
              "num_experts_per_tok": 8,
              "n_shared_experts": 1,
              "moe_intermediate_size": 2048,
              "first_k_dense_replace": 3,
              "routed_scaling_factor": 2.5,
              "norm_topk_prob": true,
              "rms_norm_eps": 1e-5
            }"#,
        )
        .unwrap();

        let metadata = metadata_from_hf_config(&root, 3).unwrap();

        assert!(metadata.iter().any(|kv| {
            matches!(
                kv,
                GgufKv::U32 { key, value }
                    if key == "glm-dsa.attention.indexer.head_count" && *value == 32
            )
        }));
        assert!(metadata.iter().any(|kv| {
            matches!(
                kv,
                GgufKv::U32 { key, value }
                    if key == "glm-dsa.attention.indexer.key_length" && *value == 128
            )
        }));
        assert!(metadata.iter().any(|kv| {
            matches!(
                kv,
                GgufKv::U32 { key, value }
                    if key == "glm-dsa.attention.indexer.top_k" && *value == 2048
            )
        }));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn builds_llama_metadata_from_config() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("config.json"),
            r#"{
              "model_type": "llama",
              "vocab_size": 128256,
              "max_position_embeddings": 131072,
              "hidden_size": 4096,
              "intermediate_size": 14336,
              "num_hidden_layers": 32,
              "num_attention_heads": 32,
              "num_key_value_heads": 8,
              "head_dim": 128,
              "rope_theta": 500000,
              "rms_norm_eps": 1e-5
            }"#,
        )
        .unwrap();

        let metadata = metadata_from_hf_config(&root, 3).unwrap();
        let text = format!("{metadata:?}");

        assert!(text.contains("general.architecture"));
        assert!(text.contains("llama"));
        assert!(text.contains("llama.block_count"));
        assert!(text.contains("32"));
        assert!(text.contains("llama.attention.head_count_kv"));
        assert!(text.contains("8"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn defaults_kv_heads_to_attention_heads() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("config.json"),
            r#"{
              "model_type": "llama",
              "vocab_size": 32000,
              "max_position_embeddings": 4096,
              "hidden_size": 4096,
              "intermediate_size": 11008,
              "num_hidden_layers": 32,
              "num_attention_heads": 32
            }"#,
        )
        .unwrap();

        let metadata = metadata_from_hf_config(&root, 3).unwrap();
        let text = format!("{metadata:?}");

        assert!(text.contains("llama.attention.head_count_kv"));
        assert!(text.contains("32"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn builds_qwen2_moe_metadata_from_qwen_config_keys() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("config.json"),
            r#"{
              "model_type": "qwen2_moe",
              "vocab_size": 151936,
              "max_position_embeddings": 32768,
              "hidden_size": 2048,
              "intermediate_size": 5632,
              "num_hidden_layers": 24,
              "num_attention_heads": 16,
              "num_key_value_heads": 16,
              "head_dim": 128,
              "num_experts": 60,
              "num_experts_per_tok": 4,
              "moe_intermediate_size": 1408,
              "shared_expert_intermediate_size": 5632,
              "rope_theta": 1000000,
              "rms_norm_eps": 1e-6
            }"#,
        )
        .unwrap();

        let metadata = metadata_from_hf_config(&root, 3).unwrap();
        let text = format!("{metadata:?}");

        assert!(text.contains("qwen2moe"));
        assert!(text.contains("qwen2moe.expert_count"));
        assert!(text.contains("60"));
        assert!(text.contains("qwen2moe.expert_feed_forward_length"));
        assert!(text.contains("1408"));
        assert!(text.contains("qwen2moe.expert_shared_feed_forward_length"));
        assert!(text.contains("5632"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_newer_qwen_variants_that_need_specific_native_templates() {
        for model_type in [
            "qwen3_next",
            "qwen3next",
            "qwen3_vl",
            "qwen3vl_moe",
            "qwen3.5",
            "qwen3_5_moe",
            "qwen35",
            "qwen35moe",
        ] {
            let root = unique_temp_dir();
            fs::create_dir_all(&root).unwrap();
            fs::write(
                root.join("config.json"),
                format!(
                    r#"{{
                      "model_type": "{model_type}",
                      "vocab_size": 151936,
                      "max_position_embeddings": 32768,
                      "hidden_size": 2048,
                      "intermediate_size": 5632,
                      "num_hidden_layers": 24,
                      "num_attention_heads": 16
                    }}"#,
                ),
            )
            .unwrap();

            let err = metadata_from_hf_config(&root, 3).unwrap_err();
            let message = err.to_string();
            assert!(
                message.contains("Qwen3.5/Qwen3Next/Qwen3VL-specific"),
                "{model_type}: {message}"
            );
            assert!(
                message.contains("external convert_hf_to_gguf.py backend"),
                "{model_type}: {message}"
            );
            fs::remove_dir_all(root).unwrap();
        }
    }

    fn unique_temp_dir() -> PathBuf {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("skippy-gguf-template-{nanos}-{id}"))
    }
}
