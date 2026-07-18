use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};

const MAX_TENSOR_COUNT: usize = 2_000_000;
const MAX_TENSOR_RANK: usize = 16;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct TensorHeader {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub data_offsets: [u64; 2],
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct IndexMetadata {
    pub total_size: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct SafetensorsIndex {
    #[serde(default)]
    pub metadata: IndexMetadata,
    pub weight_map: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct LlamaConfig {
    pub model_type: String,
    pub hidden_size: u64,
    pub num_hidden_layers: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

pub fn parse_llama_config(bytes: &[u8]) -> Result<LlamaConfig> {
    let config: LlamaConfig =
        serde_json::from_slice(bytes).context("parse SafeTensors model config")?;
    ensure!(
        config.model_type == "llama",
        "MLX partial SafeTensors currently supports model_type=llama, got {:?}",
        config.model_type
    );
    ensure!(config.hidden_size > 0, "Llama hidden_size must be non-zero");
    ensure!(
        config.num_hidden_layers > 0,
        "Llama num_hidden_layers must be non-zero"
    );
    Ok(config)
}

pub fn parse_index(bytes: &[u8]) -> Result<SafetensorsIndex> {
    let index: SafetensorsIndex =
        serde_json::from_slice(bytes).context("parse SafeTensors index")?;
    ensure!(
        !index.weight_map.is_empty(),
        "SafeTensors index has no tensors"
    );
    ensure!(
        index.weight_map.len() <= MAX_TENSOR_COUNT,
        "SafeTensors index has too many tensors"
    );
    ensure!(
        index
            .weight_map
            .iter()
            .all(|(name, file)| !name.is_empty() && !file.is_empty()),
        "SafeTensors index contains an empty tensor or shard name"
    );
    Ok(index)
}

pub fn parse_header(bytes: &[u8], data_bytes: u64) -> Result<BTreeMap<String, TensorHeader>> {
    let raw: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(bytes).context("parse SafeTensors header")?;
    ensure!(
        raw.len() <= MAX_TENSOR_COUNT.saturating_add(1),
        "SafeTensors file has too many tensor entries"
    );
    let tensors = raw
        .into_iter()
        .filter(|(name, _)| name != "__metadata__")
        .map(|(name, value)| {
            ensure!(
                !name.is_empty(),
                "SafeTensors tensor name must not be empty"
            );
            serde_json::from_value(value)
                .map(|tensor| (name.clone(), tensor))
                .with_context(|| format!("parse SafeTensors tensor header {name}"))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    validate_headers(&tensors, data_bytes)?;
    Ok(tensors)
}

fn validate_headers(tensors: &BTreeMap<String, TensorHeader>, data_bytes: u64) -> Result<()> {
    ensure!(
        !tensors.is_empty(),
        "SafeTensors file has no tensor entries"
    );
    let mut extents = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        validate_tensor(name, tensor, data_bytes)?;
        extents.push((tensor.data_offsets[0], tensor.data_offsets[1], name));
    }
    extents.sort_by_key(|(start, _, _)| *start);
    let mut previous_end = 0;
    for (start, end, name) in extents {
        ensure!(
            start == previous_end,
            "SafeTensors tensor {name} leaves a gap or overlaps another tensor"
        );
        previous_end = end;
    }
    ensure!(
        previous_end == data_bytes,
        "SafeTensors tensor data does not cover the declared data section"
    );
    Ok(())
}

fn validate_tensor(name: &str, tensor: &TensorHeader, data_bytes: u64) -> Result<()> {
    ensure!(
        tensor.shape.len() <= MAX_TENSOR_RANK,
        "SafeTensors tensor {name} exceeds maximum rank {MAX_TENSOR_RANK}"
    );
    ensure!(
        tensor.data_offsets[0] <= tensor.data_offsets[1],
        "invalid data offsets for SafeTensors tensor {name}"
    );
    ensure!(
        tensor.data_offsets[1] <= data_bytes,
        "SafeTensors tensor {name} exceeds the data section"
    );
    let elements = tensor.shape.iter().try_fold(1_u64, |count, dimension| {
        count
            .checked_mul(*dimension)
            .context("SafeTensors tensor element count overflow")
    })?;
    let expected_bytes = elements
        .checked_mul(dtype_bytes(&tensor.dtype)?)
        .context("SafeTensors tensor byte count overflow")?;
    let actual_bytes = tensor.data_offsets[1] - tensor.data_offsets[0];
    ensure!(
        expected_bytes == actual_bytes,
        "SafeTensors tensor {name} has {actual_bytes} bytes but its dtype and shape require {expected_bytes}"
    );
    Ok(())
}

fn dtype_bytes(dtype: &str) -> Result<u64> {
    match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" | "F8_E8M0" => Ok(1),
        "U16" | "I16" | "F16" | "BF16" => Ok(2),
        "U32" | "I32" | "F32" => Ok(4),
        "U64" | "I64" | "F64" => Ok(8),
        _ => bail!("unsupported SafeTensors dtype {dtype:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_validates_a_complete_header() {
        let header = br#"{
            "a":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},
            "b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]},
            "__metadata__":{"format":"pt"}
        }"#;

        let tensors = parse_header(header, 8).unwrap();

        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors["a"].shape, vec![2]);
    }

    #[test]
    fn rejects_shape_byte_mismatch() {
        let header = br#"{
            "bad":{"dtype":"F16","shape":[3],"data_offsets":[0,4]}
        }"#;

        assert!(parse_header(header, 4).is_err());
    }

    #[test]
    fn rejects_gaps_and_overlaps() {
        let header = br#"{
            "a":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},
            "b":{"dtype":"F16","shape":[1],"data_offsets":[3,5]}
        }"#;

        assert!(parse_header(header, 5).is_err());
    }

    #[test]
    fn parses_nonempty_index() {
        let index = br#"{
            "metadata":{"total_size":4},
            "weight_map":{"model.layers.0.weight":"model-00001-of-00002.safetensors"}
        }"#;

        assert_eq!(parse_index(index).unwrap().metadata.total_size, Some(4));
    }

    #[test]
    fn accepts_only_supported_llama_config() {
        let llama = br#"{
            "model_type":"llama",
            "hidden_size":576,
            "num_hidden_layers":30,
            "tie_word_embeddings":true
        }"#;
        let qwen = br#"{
            "model_type":"qwen3",
            "hidden_size":1024,
            "num_hidden_layers":28
        }"#;

        assert_eq!(parse_llama_config(llama).unwrap().num_hidden_layers, 30);
        assert!(parse_llama_config(qwen).is_err());
    }
}
