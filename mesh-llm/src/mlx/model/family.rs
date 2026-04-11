use anyhow::{bail, Result};
use serde_json::Value;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningFamily {
    None,
    Qwen3,
    Glm,
    Kimi,
    GptOss,
    Lfm2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ModelArchitecture {
    LlamaLike,
    Olmo,
    Olmo2,
    DeepseekV3,
    GptOss,
    KimiLinear,
    Lfm2,
    Glm4,
    Gemma2,
    Gemma3,
    Gemma4,
}

impl ModelArchitecture {
    pub(super) fn is_olmo2(self) -> bool {
        matches!(self, Self::Olmo2)
    }

    pub(super) fn is_olmo(self) -> bool {
        matches!(self, Self::Olmo)
    }

    pub(super) fn is_deepseek_v3(self) -> bool {
        matches!(self, Self::DeepseekV3)
    }

    pub(super) fn is_glm4(self) -> bool {
        matches!(self, Self::Glm4)
    }

    pub(super) fn is_gpt_oss(self) -> bool {
        matches!(self, Self::GptOss)
    }

    pub(super) fn is_kimi_linear(self) -> bool {
        matches!(self, Self::KimiLinear)
    }

    pub(super) fn is_lfm2(self) -> bool {
        matches!(self, Self::Lfm2)
    }

    pub(super) fn is_gemma2(self) -> bool {
        matches!(self, Self::Gemma2)
    }

    pub(super) fn is_gemma3(self) -> bool {
        matches!(self, Self::Gemma3)
    }

    pub(super) fn is_gemma4(self) -> bool {
        matches!(self, Self::Gemma4)
    }

    pub(super) fn uses_gemma_norm_offset(self) -> bool {
        self.is_gemma2() || self.is_gemma3()
    }

    pub(super) fn uses_gemma_scaled_embeddings(self) -> bool {
        self.is_gemma2() || self.is_gemma3() || self.is_gemma4()
    }
}

pub(super) fn uses_traditional_rope(config: &Value) -> bool {
    config
        .get("rope_traditional")
        .and_then(|value| value.as_bool())
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|value| value.get("rope_traditional"))
                .and_then(|value| value.as_bool())
        })
        .unwrap_or(false)
}

pub(super) fn model_architecture(config: &Value) -> ModelArchitecture {
    let model_type = config
        .get("model_type")
        .and_then(|value| value.as_str())
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|value| value.get("model_type"))
                .and_then(|value| value.as_str())
        })
        .unwrap_or_default()
        .to_ascii_lowercase();

    if model_type.starts_with("glm4") {
        ModelArchitecture::Glm4
    } else if model_type.starts_with("gpt_oss") {
        ModelArchitecture::GptOss
    } else if model_type.starts_with("kimi_linear") {
        ModelArchitecture::KimiLinear
    } else if model_type.starts_with("deepseek_v3")
        || model_type.starts_with("kimi_k2")
        || model_type.starts_with("kimi_k25")
    {
        ModelArchitecture::DeepseekV3
    } else if model_type.starts_with("olmo2") {
        ModelArchitecture::Olmo2
    } else if model_type.starts_with("olmo") {
        ModelArchitecture::Olmo
    } else if model_type.starts_with("lfm2") {
        ModelArchitecture::Lfm2
    } else if model_type.starts_with("gemma4") {
        ModelArchitecture::Gemma4
    } else if model_type.starts_with("gemma2") {
        ModelArchitecture::Gemma2
    } else if model_type.starts_with("gemma3") {
        ModelArchitecture::Gemma3
    } else {
        ModelArchitecture::LlamaLike
    }
}

pub(super) fn reasoning_family(config: &Value) -> ReasoningFamily {
    let model_type = config
        .get("model_type")
        .and_then(|value| value.as_str())
        .or_else(|| {
            config
                .get("text_config")
                .and_then(|value| value.get("model_type"))
                .and_then(|value| value.as_str())
        })
        .unwrap_or_default()
        .to_ascii_lowercase();
    let architectures = config
        .get("architectures")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str())
        .map(|value| value.to_ascii_lowercase())
        .collect::<Vec<_>>();

    if model_type == "qwen3" || architectures.iter().any(|value| value.contains("qwen3")) {
        return ReasoningFamily::Qwen3;
    }
    if model_type.starts_with("glm") || architectures.iter().any(|value| value.contains("glm")) {
        return ReasoningFamily::Glm;
    }
    if model_type == "gpt_oss" || architectures.iter().any(|value| value.contains("gptoss")) {
        return ReasoningFamily::GptOss;
    }
    if model_type.starts_with("lfm2") || architectures.iter().any(|value| value.contains("lfm2")) {
        return ReasoningFamily::Lfm2;
    }
    if model_type.contains("kimi") || architectures.iter().any(|value| value.contains("kimi")) {
        return ReasoningFamily::Kimi;
    }
    ReasoningFamily::None
}

pub(super) fn config_supports_mlx(config: &Value) -> bool {
    let architectures = config
        .get("architectures")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter_map(|value| value.as_str());
    let model_type = config.get("model_type").and_then(|value| value.as_str());

    architectures.chain(model_type).any(|name| {
        let name = name.to_ascii_lowercase();
        matches!(
            name.as_str(),
            "llama"
                | "mistral"
                | "glm4"
                | "deepseek_v3"
                | "lfm2"
                | "phi3"
                | "qwen2"
                | "qwen3"
                | "gpt_oss"
                | "kimi_linear"
                | "olmo"
                | "olmo2"
                | "gemma2"
                | "gemma3"
                | "gemma3_text"
                | "gemma4"
                | "gemma4_text"
                | "glm4forcausallm"
                | "deepseekv3forcausallm"
                | "lfm2forcausallm"
                | "phi3forcausallm"
                | "llamaforcausallm"
                | "mistralforcausallm"
                | "qwen2forcausallm"
                | "qwen3forcausallm"
                | "gptossforcausallm"
                | "kimilinearforcausallm"
                | "olmoforcausallm"
                | "olmo2forcausallm"
                | "gemma2forcausallm"
                | "gemma3forcausallm"
                | "gemma3forconditionalgeneration"
                | "gemma4forcausallm"
                | "gemma4forconditionalgeneration"
        )
    })
}

pub(super) fn ensure_supported_mlx_model(dir: &Path, config: &Value) -> Result<()> {
    if config_supports_mlx(config) {
        return Ok(());
    }
    if let Some(architecture) = detect_architecture_from_safetensors_header(dir) {
        tracing::info!(
            "MLX loader: config.json did not identify a supported architecture, but safetensors headers matched {}",
            architecture
        );
        return Ok(());
    }

    let model_type = config
        .get("model_type")
        .and_then(|value| value.as_str())
        .unwrap_or("unknown");
    let architectures = config
        .get("architectures")
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "none".to_string());
    bail!(
        "unsupported MLX model architecture in {} (model_type={}, architectures={}); supported MLX models currently cover Llama/OLMo/DeepSeekV3/GPT-OSS/Kimi-Linear/LFM2/GLM4/Qwen/Gemma2/Gemma3/Gemma4-style safetensors checkpoints",
        dir.display(),
        model_type,
        architectures,
    )
}

pub(super) fn detect_architecture_from_safetensors_header(dir: &Path) -> Option<String> {
    let path = if dir.join("model.safetensors").exists() {
        dir.join("model.safetensors")
    } else {
        let text = std::fs::read_to_string(dir.join("model.safetensors.index.json")).ok()?;
        let index: Value = serde_json::from_str(&text).ok()?;
        let file = index
            .get("weight_map")
            .and_then(|value| value.as_object())?
            .values()
            .find_map(|value| value.as_str())?;
        dir.join(file)
    };

    let mut file = File::open(path).ok()?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).ok()?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    if header_len == 0 || header_len > 16 * 1024 * 1024 {
        return None;
    }
    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header).ok()?;
    let json: Value = serde_json::from_slice(&header).ok()?;
    let map = json.as_object()?;

    let keys: Vec<&str> = map
        .keys()
        .filter(|key| key.as_str() != "__metadata__")
        .map(|key| key.as_str())
        .collect();

    if keys.iter().any(|key| key.starts_with("model.layers."))
        && keys
            .iter()
            .any(|key| key.starts_with("model.embed_tokens."))
        && keys
            .iter()
            .any(|key| key.contains(".self_attn.q_proj.") || key.contains(".self_attn.q_proj"))
    {
        return Some("llama_like".to_string());
    }

    None
}
