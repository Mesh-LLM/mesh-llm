//! JSON shapes exchanged between the serving node and the auditing client.

use anyhow::{Context, Result, bail};
use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::commitment::MixingValue;
use crate::layer_tensors::TensorKind;

pub const RECEIPT_VERSION: u32 = 3;

/// What the node claims it ran, attached to a response.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Receipt {
    pub version: u32,
    /// Response id; the key for challenges.
    pub receipt_id: String,
    /// The GGUF the node claims, as configured by the operator.
    pub model: String,
    /// First and one-past-last sequence position the turn computed.
    pub position_start: u32,
    pub position_end: u32,
    pub stages: Vec<StageCommitment>,
}

/// One stage's commitment to the turn: a Merkle root over its leaves.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StageCommitment {
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    /// Hex BLAKE3 Merkle root over leaves ordered by position, then slot
    /// (see `commitment`).
    pub root: String,
    pub leaf_count: u64,
    /// Backend facts the verifier needs to pick rounding profiles.
    pub backend: String,
    /// ggml type names of the KV cache and state rows the stage stored,
    /// for example `f16` or `q8_0`. A client decides which it accepts.
    #[serde(default)]
    pub cache_types: Vec<String>,
}

impl Receipt {
    pub fn stage_for_layer(&self, layer: u32) -> Option<&StageCommitment> {
        self.stages
            .iter()
            .find(|stage| (stage.layer_start..stage.layer_end).contains(&layer))
    }

    pub fn layer_count(&self) -> u32 {
        self.stages.iter().map(|s| s.layer_end).max().unwrap_or(0)
    }
}

/// The client's challenge, sent after the receipt arrives.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Challenge {
    pub receipt_id: String,
    pub layer: u32,
    /// Absolute sequence positions inside the receipt's range.
    pub positions: Vec<u32>,
}

/// The node's answer: the challenged leaves and their Merkle paths.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Opening {
    pub receipt_id: String,
    pub stage_index: u32,
    pub layer: u32,
    pub leaves: Vec<LeafOpening>,
}

/// One challenged position: the layer's committed leaves with their Merkle
/// paths, and the layer's weight matmuls and selections recomputed by replay.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LeafOpening {
    pub position: u32,
    /// Residual stream entering the layer, base64 little-endian f32.
    pub residual_in: String,
    /// Residual stream leaving the layer. Absent where the layer kept no row
    /// for this position (the final layer of a final stage, non-output rows).
    pub residual_out: Option<String>,
    /// Hex sibling hashes, leaf to root, of the residual-in, mixing and
    /// residual-out leaves.
    pub residual_in_path: Vec<String>,
    pub mixing_path: Vec<String>,
    pub residual_out_path: Vec<String>,
    /// The mixing leaf's values, in leaf order.
    pub mixing: Vec<TensorOpening>,
    /// The layer's weight matmuls at this position, by index. Matmuls after
    /// the layer dropped this position's row are absent.
    pub matrices: Vec<MatrixOpening>,
    pub selections: Vec<SelectionOpening>,
}

impl LeafOpening {
    pub fn residual_in_values(&self) -> Result<Vec<f32>> {
        decode_f32(&self.residual_in).context("decode residual_in")
    }

    pub fn residual_out_values(&self) -> Result<Option<Vec<f32>>> {
        self.residual_out
            .as_deref()
            .map(|text| decode_f32(text).context("decode residual_out"))
            .transpose()
    }

    pub fn matrix(&self, index: u32) -> Option<&MatrixOpening> {
        self.matrices.iter().find(|m| m.index == index)
    }

    pub fn mixing_value(&self, kind: MixingKind, index: u32) -> Option<&TensorOpening> {
        self.mixing
            .iter()
            .find(|t| t.kind == kind && t.index == index)
    }

    pub fn selection(&self, index: u32) -> Option<&SelectionOpening> {
        self.selections.iter().find(|s| s.index == index)
    }
}

/// Which part of a token-mixing op a value is.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MixingKind {
    Input,
    /// A KV cache or state row as stored, in the cache's type.
    Stored,
    Output,
}

impl MixingKind {
    pub fn tensor_kind(self) -> TensorKind {
        match self {
            Self::Input => TensorKind::MixingInput,
            Self::Stored => TensorKind::MixingStored,
            Self::Output => TensorKind::MixingOutput,
        }
    }

    pub fn from_tensor_kind(kind: TensorKind) -> Option<Self> {
        match kind {
            TensorKind::MixingInput => Some(Self::Input),
            TensorKind::MixingStored => Some(Self::Stored),
            TensorKind::MixingOutput => Some(Self::Output),
            _ => None,
        }
    }
}

/// One committed token-mixing value.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TensorOpening {
    pub kind: MixingKind,
    pub index: u32,
    /// ggml type of the values.
    pub ggml_type: i32,
    /// Base64 little-endian values; absent where the layer kept no row.
    pub values: Option<String>,
}

impl TensorOpening {
    pub fn new(kind: MixingKind, index: u32, value: Option<(i32, &[u8])>) -> Self {
        Self {
            kind,
            index,
            ggml_type: value.map_or(crate::layer_tensors::GGML_F32, |(t, _)| t),
            values: value.map(|(_, bytes)| encode_bytes(bytes)),
        }
    }

    pub fn bytes(&self) -> Result<Option<Vec<u8>>> {
        self.values
            .as_deref()
            .map(|text| {
                base64::engine::general_purpose::STANDARD
                    .decode(text)
                    .context("decode a mixing value")
            })
            .transpose()
    }

    /// The f32 values; an error for another type.
    pub fn f32_values(&self) -> Result<Option<Vec<f32>>> {
        if self.ggml_type != crate::layer_tensors::GGML_F32 {
            bail!("mixing value {:?} {} is not f32", self.kind, self.index);
        }
        Ok(self
            .bytes()?
            .map(|bytes| crate::layer_tensors::decode_f32_bytes(&bytes)))
    }
}

/// A mixing value decoded for hashing: kind, index, ggml type and bytes.
pub type DecodedMixing = (MixingKind, u32, i32, Option<Vec<u8>>);

/// Decodes the mixing values of a leaf for hashing.
pub fn mixing_bytes(values: &[TensorOpening]) -> Result<Vec<DecodedMixing>> {
    values
        .iter()
        .map(|t| Ok((t.kind, t.index, t.ggml_type, t.bytes()?)))
        .collect()
}

/// Borrows decoded mixing values as leaf inputs.
pub fn mixing_values(decoded: &[DecodedMixing]) -> Vec<MixingValue<'_>> {
    decoded
        .iter()
        .map(|(kind, index, ggml_type, bytes)| MixingValue {
            kind: kind.tensor_kind(),
            index: *index,
            value: bytes.as_deref().map(|b| (*ggml_type, b)),
        })
        .collect()
}

/// One weight matmul for one token: `output = weight · input`. For a
/// `mul_mat_id`, one slice of input and output per expert used.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MatrixOpening {
    /// Ordinal among the layer's weight matmuls.
    pub index: u32,
    /// GGUF tensor name, e.g. `blk.12.attn_q.weight`.
    pub weight: String,
    /// The output feeds a selection (MoE routing); every row is checked.
    #[serde(default)]
    pub router: bool,
    /// The experts a `mul_mat_id` used, one per output slice.
    #[serde(default)]
    pub experts: Option<Vec<i32>>,
    /// Base64 little-endian f32.
    pub input: String,
    pub output: String,
}

impl MatrixOpening {
    pub fn input_values(&self) -> Result<Vec<f32>> {
        decode_f32(&self.input).with_context(|| format!("decode input of {}", self.weight))
    }

    pub fn output_values(&self) -> Result<Vec<f32>> {
        decode_f32(&self.output).with_context(|| format!("decode output of {}", self.weight))
    }
}

/// The ids a selection (`argsort` or top-k) picked for one token.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SelectionOpening {
    pub index: u32,
    pub ids: Vec<i32>,
}

pub fn encode_f32(values: &[f32]) -> String {
    encode_bytes(&crate::layer_tensors::f32_bytes(values))
}

pub fn encode_bytes(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

pub fn decode_f32(text: &str) -> Result<Vec<f32>> {
    let bytes = base64::engine::general_purpose::STANDARD.decode(text)?;
    if bytes.len() % 4 != 0 {
        bail!("f32 payload of {} bytes", bytes.len());
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_round_trip_is_bit_exact() {
        let values = [0.0f32, -1.5, f32::MIN_POSITIVE, 3.25e-7, f32::MAX];
        assert_eq!(decode_f32(&encode_f32(&values)).unwrap(), values);
    }

    #[test]
    fn stage_lookup_uses_half_open_ranges() {
        let stage = |index, start, end| StageCommitment {
            stage_index: index,
            layer_start: start,
            layer_end: end,
            root: String::new(),
            leaf_count: 0,
            backend: String::new(),
            cache_types: Vec::new(),
        };
        let receipt = Receipt {
            version: RECEIPT_VERSION,
            receipt_id: "r".into(),
            model: "m".into(),
            position_start: 0,
            position_end: 1,
            stages: vec![stage(0, 0, 18), stage(1, 18, 36)],
        };
        assert_eq!(receipt.stage_for_layer(17).unwrap().stage_index, 0);
        assert_eq!(receipt.stage_for_layer(18).unwrap().stage_index, 1);
        assert!(receipt.stage_for_layer(36).is_none());
        assert_eq!(receipt.layer_count(), 36);
    }
}
