//! JSON shapes exchanged between the serving node and the auditing client.

use anyhow::{Context, Result, bail};
use base64::Engine;
use serde::{Deserialize, Serialize};

pub const RECEIPT_VERSION: u32 = 2;

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
/// paths, and the layer's weight matmuls recomputed by replay.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LeafOpening {
    pub position: u32,
    /// Residual stream entering the layer, base64 little-endian f32.
    pub residual_in: String,
    /// Residual stream leaving the layer. Absent where the layer kept no row
    /// for this position (the final layer of a final stage, non-output rows).
    pub residual_out: Option<String>,
    /// Hex sibling hashes, leaf to root, of the residual-in, attention and
    /// residual-out leaves.
    pub residual_in_path: Vec<String>,
    pub attention_path: Vec<String>,
    pub residual_out_path: Vec<String>,
    /// The layer's weight matmuls at this position. The attention leaf is
    /// the attn_q, attn_k and attn_v outputs and the attn_output input. The
    /// FFN matmuls are absent with `residual_out`.
    pub matrices: Vec<MatrixOpening>,
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

    /// The opened matmul whose weight is `blk.<layer>.<role>.weight`.
    pub fn matrix(&self, layer: u32, role: &str) -> Option<&MatrixOpening> {
        let name = format!("blk.{layer}.{role}.weight");
        self.matrices.iter().find(|m| m.weight == name)
    }
}

/// One weight matmul for one token: `output = weight · input`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MatrixOpening {
    /// GGUF tensor name, e.g. `blk.12.attn_q.weight`.
    pub weight: String,
    /// Base64 little-endian f32.
    pub input: String,
    pub output: String,
}

impl MatrixOpening {
    pub fn new(weight: &str, input: &[f32], output: &[f32]) -> Self {
        Self {
            weight: weight.to_owned(),
            input: encode_f32(input),
            output: encode_f32(output),
        }
    }

    pub fn input_values(&self) -> Result<Vec<f32>> {
        decode_f32(&self.input).with_context(|| format!("decode input of {}", self.weight))
    }

    pub fn output_values(&self) -> Result<Vec<f32>> {
        decode_f32(&self.output).with_context(|| format!("decode output of {}", self.weight))
    }
}

pub fn encode_f32(values: &[f32]) -> String {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
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
