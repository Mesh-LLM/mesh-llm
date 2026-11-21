//! Shape and decoder contract of one paired DFlash draft.

use serde::{Deserialize, Serialize};

#[cfg(feature = "conversion")]
use anyhow::{Context, Result, bail};
#[cfg(feature = "conversion")]
use runtimed_gguf::{GgufFile, MetaValue};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DFlashProfile {
    pub width: usize,
    pub vocabulary: usize,
    pub mask_token: usize,
    pub target_layers: Vec<u32>,
    pub query_heads: Vec<usize>,
    pub kv_heads: Vec<usize>,
    pub head_width: usize,
    pub rotary_width: usize,
    pub ffn_width: Vec<usize>,
    pub block_size: usize,
    pub decoder: DecoderContract,
    pub causal: bool,
    pub rope_base: f32,
    pub rms_epsilon: f32,
    pub embedding_scale: f32,
    pub value_scale: f32,
    pub attention_scale: f32,
    pub activation: HiddenActivation,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HiddenActivation {
    Silu,
    Gelu,
    GeluTanh,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecoderContract {
    Standard,
    Laguna,
}

impl DFlashProfile {
    pub fn layers(&self) -> usize {
        self.query_heads.len()
    }

    pub fn feature_width(&self) -> Option<usize> {
        self.width.checked_mul(self.target_layers.len())
    }

    #[cfg(feature = "conversion")]
    pub fn from_gguf(draft: &GgufFile, target: &GgufFile) -> Result<Self> {
        if draft.meta_str("general.architecture") != Some("dflash") {
            bail!("draft GGUF must have the dflash architecture");
        }
        validate_supported_metadata(draft)?;
        let layers = positive(draft, "dflash.block_count")?;
        let width = positive(draft, "dflash.embedding_length")?;
        let query_heads = per_layer(draft, "dflash.attention.head_count", layers)?;
        let head_width =
            if let Some(value) = optional_positive(draft, "dflash.attention.key_length")? {
                value
            } else {
                let heads = query_heads[0];
                if !width.is_multiple_of(heads) {
                    bail!("DFlash hidden width is not divisible by the query head count");
                }
                width / heads
            };
        if optional_positive(draft, "dflash.attention.value_length")?.unwrap_or(head_width)
            != head_width
            || head_width % 2 != 0
        {
            bail!("ANE DFlash requires equal, even key and value head widths");
        }
        let rotary_width =
            optional_positive(draft, "dflash.rope.dimension_count")?.unwrap_or(head_width);
        if rotary_width > head_width || rotary_width % 2 != 0 {
            bail!("DFlash rotary width must be even and within the key head width");
        }
        let block_size = optional_positive(draft, "dflash.block_size")?.unwrap_or(16);
        let kv_heads = if draft
            .metadata
            .contains_key("dflash.attention.head_count_kv")
        {
            per_layer(draft, "dflash.attention.head_count_kv", layers)?
        } else {
            query_heads.clone()
        };
        let ffn_width = per_layer(draft, "dflash.feed_forward_length", layers)?;
        for (&query, &kv) in query_heads.iter().zip(&kv_heads) {
            if query % kv != 0 {
                bail!("DFlash query head count must be divisible by KV head count");
            }
        }
        let target_layers = unsigned_array(draft, "dflash.target_layers")?;
        if target_layers.is_empty() {
            bail!("DFlash target feature layer list is empty");
        }
        let embedding = target
            .tensor("token_embd.weight")
            .context("target GGUF lacks token_embd.weight")?;
        if embedding.dims.len() != 2 || embedding.dims[0] != width as u64 {
            bail!("target token embeddings do not match the DFlash hidden width");
        }
        let vocabulary = usize::try_from(embedding.dims[1])?;
        let mask_token = positive_or_zero(draft, "tokenizer.ggml.mask_token_id")?;
        if mask_token >= vocabulary {
            bail!("DFlash mask token is outside the target vocabulary");
        }
        let decoder = match draft.meta_str("dflash.decoder_arch") {
            None | Some("") => DecoderContract::Standard,
            Some("laguna") => DecoderContract::Laguna,
            Some(other) => bail!("unsupported DFlash decoder contract {other}"),
        };
        for tensor in ["d2t", "markov_w1.weight", "selector_hidden.weight"] {
            if draft.tensor(tensor).is_some() {
                bail!("ANE DFlash graph does not implement {tensor}");
            }
        }
        let causal = match draft.metadata.get("dflash.attention.causal") {
            Some(MetaValue::Bool(value)) => *value,
            None => decoder == DecoderContract::Laguna,
            _ => bail!("dflash.attention.causal must be boolean"),
        };
        let rope_base = match draft.metadata.get("dflash.rope.freq_base") {
            Some(MetaValue::F32(value)) => *value,
            None => 10_000.0,
            _ => bail!("dflash.rope.freq_base must be a float"),
        };
        if !rope_base.is_finite() || rope_base <= 0.0 {
            bail!("DFlash RoPE base must be positive and finite");
        }
        let rms_epsilon =
            optional_float(draft, "dflash.attention.layer_norm_rms_epsilon")?.unwrap_or(1e-6);
        let embedding_scale = optional_float(draft, "dflash.embedding_scale")?.unwrap_or(1.0);
        let value_scale = optional_float(draft, "dflash.attention.value_scale")?.unwrap_or(1.0);
        let attention_scale = optional_float(draft, "dflash.attention.scale")?
            .unwrap_or_else(|| (head_width as f32).powf(-0.5));
        for (name, value) in [
            ("RMS epsilon", rms_epsilon),
            ("embedding scale", embedding_scale),
            ("attention scale", attention_scale),
            ("attention value scale", value_scale),
        ] {
            if !value.is_finite() || value <= 0.0 {
                bail!("DFlash {name} must be positive and finite");
            }
        }
        let activation = match draft.meta_str("dflash.hidden_activation") {
            None | Some("silu") => HiddenActivation::Silu,
            Some("gelu") => HiddenActivation::Gelu,
            Some("gelu_pytorch_tanh") => HiddenActivation::GeluTanh,
            Some(other) => bail!("unsupported DFlash hidden activation {other}"),
        };
        Ok(Self {
            width,
            vocabulary,
            mask_token,
            target_layers,
            query_heads,
            kv_heads,
            head_width,
            rotary_width,
            ffn_width,
            block_size,
            decoder,
            causal,
            rope_base,
            rms_epsilon,
            embedding_scale,
            value_scale,
            attention_scale,
            activation,
        })
    }
}

#[cfg(feature = "conversion")]
fn validate_supported_metadata(draft: &GgufFile) -> Result<()> {
    for key in draft.metadata.keys() {
        let unsupported_rope = key.starts_with("dflash.rope.")
            && key != "dflash.rope.freq_base"
            && key != "dflash.rope.dimension_count";
        let unsupported_attention = key.starts_with("dflash.attention.sliding_window");
        let unsupported_output = matches!(
            key.as_str(),
            "dflash.logit_scale" | "dflash.final_logit_softcapping"
        );
        if unsupported_rope || unsupported_attention || unsupported_output {
            bail!("ANE DFlash graph does not implement GGUF metadata {key}");
        }
    }
    Ok(())
}

#[cfg(feature = "conversion")]
fn unsigned(value: &MetaValue) -> Option<usize> {
    match value {
        MetaValue::U32(value) => Some(*value as usize),
        MetaValue::I32(value) => usize::try_from(*value).ok(),
        MetaValue::U64(value) => usize::try_from(*value).ok(),
        MetaValue::I64(value) => usize::try_from(*value).ok(),
        _ => None,
    }
}

#[cfg(feature = "conversion")]
fn positive_or_zero(gguf: &GgufFile, key: &str) -> Result<usize> {
    gguf.metadata
        .get(key)
        .and_then(unsigned)
        .with_context(|| format!("{key} must be a nonnegative integer"))
}

#[cfg(feature = "conversion")]
fn positive(gguf: &GgufFile, key: &str) -> Result<usize> {
    let value = positive_or_zero(gguf, key)?;
    if value == 0 {
        bail!("{key} must be positive");
    }
    Ok(value)
}

#[cfg(feature = "conversion")]
fn optional_positive(gguf: &GgufFile, key: &str) -> Result<Option<usize>> {
    if gguf.metadata.contains_key(key) {
        Ok(Some(positive(gguf, key)?))
    } else {
        Ok(None)
    }
}

#[cfg(feature = "conversion")]
fn optional_float(gguf: &GgufFile, key: &str) -> Result<Option<f32>> {
    match gguf.metadata.get(key) {
        Some(MetaValue::F32(value)) => Ok(Some(*value)),
        Some(_) => bail!("{key} must be a float"),
        None => Ok(None),
    }
}

#[cfg(feature = "conversion")]
fn unsigned_array(gguf: &GgufFile, key: &str) -> Result<Vec<u32>> {
    let Some(MetaValue::Arr(values)) = gguf.metadata.get(key) else {
        bail!("{key} must be an integer array");
    };
    values
        .iter()
        .map(|value| {
            unsigned(value)
                .and_then(|value| u32::try_from(value).ok())
                .with_context(|| format!("{key} contains an invalid layer"))
        })
        .collect()
}

#[cfg(feature = "conversion")]
fn per_layer(gguf: &GgufFile, key: &str, layers: usize) -> Result<Vec<usize>> {
    match gguf.metadata.get(key) {
        Some(MetaValue::Arr(values)) if values.len() == layers => values
            .iter()
            .map(|value| {
                unsigned(value)
                    .filter(|&value| value > 0)
                    .context(key.to_owned())
            })
            .collect(),
        Some(MetaValue::Arr(_)) => bail!("{key} length does not match DFlash layer count"),
        Some(value) => {
            let value = unsigned(value)
                .filter(|&value| value > 0)
                .context(key.to_owned())?;
            Ok(vec![value; layers])
        }
        None => bail!("DFlash GGUF lacks {key}"),
    }
}

#[cfg(all(test, feature = "conversion"))]
mod tests {
    use std::path::Path;

    use runtimed_gguf::{GgufFile, MetaValue};

    use super::{DFlashProfile, DecoderContract};

    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn value(bytes: &mut Vec<u8>, entry: &MetaValue) -> u32 {
        match entry {
            MetaValue::U32(number) => {
                bytes.extend_from_slice(&number.to_le_bytes());
                4
            }
            MetaValue::I32(number) => {
                bytes.extend_from_slice(&number.to_le_bytes());
                5
            }
            MetaValue::F32(number) => {
                bytes.extend_from_slice(&number.to_le_bytes());
                6
            }
            MetaValue::Bool(boolean) => {
                bytes.push(u8::from(*boolean));
                7
            }
            MetaValue::Str(text) => {
                string(bytes, text);
                8
            }
            MetaValue::Arr(entries) => {
                bytes.extend_from_slice(&4u32.to_le_bytes());
                bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
                for entry in entries {
                    assert_eq!(value(bytes, entry), 4);
                }
                9
            }
            _ => panic!("fixture metadata type is unsupported"),
        }
    }

    fn gguf(path: &Path, entries: &[(&str, MetaValue)], embedding: Option<(u64, u64)>) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&u64::from(embedding.is_some()).to_le_bytes());
        bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (key, entry) in entries {
            string(&mut bytes, key);
            let mut payload = Vec::new();
            let kind = value(&mut payload, entry);
            bytes.extend_from_slice(&kind.to_le_bytes());
            bytes.extend_from_slice(&payload);
        }
        if let Some((width, vocabulary)) = embedding {
            string(&mut bytes, "token_embd.weight");
            bytes.extend_from_slice(&2u32.to_le_bytes());
            bytes.extend_from_slice(&width.to_le_bytes());
            bytes.extend_from_slice(&vocabulary.to_le_bytes());
            bytes.extend_from_slice(&0u32.to_le_bytes());
            bytes.extend_from_slice(&0u64.to_le_bytes());
        }
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn parses_non_qwen_layer_shapes_and_laguna_contract() {
        let temp = tempfile::tempdir().unwrap();
        let draft = temp.path().join("draft.gguf");
        let target = temp.path().join("target.gguf");
        gguf(
            &draft,
            &[
                ("general.architecture", MetaValue::Str("dflash".into())),
                ("dflash.block_count", MetaValue::U32(2)),
                ("dflash.embedding_length", MetaValue::U32(96)),
                ("dflash.attention.key_length", MetaValue::U32(32)),
                ("dflash.attention.value_length", MetaValue::U32(32)),
                (
                    "dflash.attention.head_count",
                    MetaValue::Arr(vec![MetaValue::U32(3), MetaValue::U32(6)]),
                ),
                ("dflash.attention.head_count_kv", MetaValue::U32(1)),
                (
                    "dflash.feed_forward_length",
                    MetaValue::Arr(vec![MetaValue::U32(128), MetaValue::U32(192)]),
                ),
                (
                    "dflash.target_layers",
                    MetaValue::Arr(vec![MetaValue::U32(2), MetaValue::U32(4)]),
                ),
                ("tokenizer.ggml.mask_token_id", MetaValue::U32(7)),
                ("dflash.decoder_arch", MetaValue::Str("laguna".into())),
                ("dflash.rope.freq_base", MetaValue::F32(1_000_000.0)),
            ],
            None,
        );
        gguf(
            &target,
            &[("general.architecture", MetaValue::Str("laguna".into()))],
            Some((96, 100)),
        );
        let draft = GgufFile::open(&draft).unwrap();
        let target = GgufFile::open(&target).unwrap();
        let profile = DFlashProfile::from_gguf(&draft, &target).unwrap();
        assert_eq!(profile.decoder, DecoderContract::Laguna);
        assert_eq!(profile.query_heads, [3, 6]);
        assert_eq!(profile.ffn_width, [128, 192]);
        assert_eq!(profile.feature_width(), Some(192));
        assert_eq!(profile.vocabulary, 100);
        assert!(profile.causal);
    }

    #[test]
    fn rejects_rope_scaling_before_converting_a_mismatched_graph() {
        let temp = tempfile::tempdir().unwrap();
        let draft = temp.path().join("draft.gguf");
        let target = temp.path().join("target.gguf");
        gguf(
            &draft,
            &[
                ("general.architecture", MetaValue::Str("dflash".into())),
                ("dflash.rope.scaling.factor", MetaValue::F32(2.0)),
            ],
            None,
        );
        gguf(&target, &[], Some((8, 12)));
        let draft = GgufFile::open(&draft).unwrap();
        let target = GgufFile::open(&target).unwrap();
        let error = DFlashProfile::from_gguf(&draft, &target).unwrap_err();
        assert!(error.to_string().contains("dflash.rope.scaling.factor"));
    }
}
