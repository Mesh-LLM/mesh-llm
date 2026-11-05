//! JSON shapes exchanged between the serving node and the auditing client.

use anyhow::{Context, Result, bail};
use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::attention::opening::{AttentionPartials, AttentionPick};
use crate::commitment::MixingValue;
use crate::layer_tensors::TensorKind;

/// The receipt format. A receipt carries the signed [`crate::turn::TurnBinding`]
/// and binds each committed value by its BLAKE3 digest
/// ([`crate::commitment::ValueDigest`]). A mixing leaf hashes its stored rows
/// apart from its other values, and a cell's digest covers its stored rows
/// alone, so an attention check can open the rows a turn attended to. On a
/// hybrid model the receipt binds the recurrent state the turn started from,
/// and the mixing leaf of each ubatch's first position commits the state the
/// ubatch read, so a client can recompute the ubatch.
pub const RECEIPT_VERSION: u32 = 5;

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
    /// The token at every position of the turn, from `position_start`. The
    /// stage holding layer 0 binds each one into the residual leaf of layer 0.
    pub tokens: Vec<i32>,
    /// The first position whose token this turn sampled. Earlier positions
    /// hold the prompt.
    pub generation_start: u32,
    /// The last token the turn sampled, which no position holds.
    pub final_token: Option<i32>,
    pub stages: Vec<StageCommitment>,
    /// The client key, parent turn and runtime, and the node's signature
    /// of the turn root. Absent when the request named no client key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binding: Option<crate::turn::TurnBinding>,
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
    /// The KV cells the turn attended to below its first replayed position,
    /// computed before the turn. `None` when the turn starts at position 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix: Option<PrefixCommitment>,
    /// The first position of every ubatch that computed positions of the
    /// turn, in order: the batch geometry a verifier's rounding depends on.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ubatches: Vec<u32>,
    /// On a stage with hybrid memory, the hex digest of the recurrent state
    /// the turn started from ([`crate::commitment::recurrent_state_digest`]).
    /// A replay of the turn starts from that state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recurrent_state: Option<String>,
}

/// A stage's commitment to the cached prefix a turn started from.
///
/// One leaf per position `0..positions`, in order: the position, the KV
/// cell that held it, and a digest of the leaves the stage committed when
/// it computed that position. A replay starts from those cells instead of
/// recomputing the prefix, and a client that holds the receipts of the
/// turns that computed the prefix can link them by those digests.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PrefixCommitment {
    /// Hex BLAKE3 Merkle root over the prefix leaves.
    pub root: String,
    pub positions: u32,
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

    /// The token at `position`, if the turn computed it.
    pub fn token_at(&self, position: u32) -> Option<i32> {
        let offset = position.checked_sub(self.position_start)?;
        self.tokens.get(usize::try_from(offset).ok()?).copied()
    }

    /// Positions whose output the turn sampled a token from, in order: the
    /// last prompt position, when the turn computed it, and every position
    /// after it.
    pub fn output_positions(&self) -> std::ops::Range<u32> {
        let first = self
            .generation_start
            .saturating_sub(1)
            .max(self.position_start);
        let end = if self.final_token.is_some() {
            self.position_end
        } else {
            self.position_end.saturating_sub(1)
        };
        first..end.max(first)
    }

    /// The token the turn sampled from the output at `position`.
    pub fn sampled_after(&self, position: u32) -> Option<i32> {
        if !self.output_positions().contains(&position) {
            return None;
        }
        if position + 1 == self.position_end {
            self.final_token
        } else {
            self.token_at(position + 1)
        }
    }

    /// The tokens the turn sampled, in order.
    pub fn sampled_tokens(&self) -> Vec<i32> {
        let start = self.generation_start.saturating_sub(self.position_start) as usize;
        let mut sampled = self.tokens.get(start..).unwrap_or_default().to_vec();
        sampled.extend(self.final_token);
        sampled
    }
}

/// One challenged layer and the positions opened in it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Pick {
    pub layer: u32,
    /// Absolute sequence positions inside the receipt's range.
    pub positions: Vec<u32>,
}

/// The client's challenge, sent after the receipt arrives. Besides the
/// picked layers, every challenge opens the layer-0 input and the stage
/// boundaries at every picked position.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Challenge {
    pub receipt_id: String,
    pub picks: Vec<Pick>,
    /// Hex VRF proof of the turn root under the receipt's client key, from
    /// which the picks are drawn. Required when the receipt names a key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proof: Option<String>,
    /// Positions whose sampled token the client checks against the output
    /// head, sorted. The final stage opens its output residual at each.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub head: Vec<u32>,
    /// Positions of picked layers whose attention the client checks, with
    /// a head each. A stage opens the head's partials at each one in a
    /// layer with attention.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attention: Vec<AttentionPick>,
}

impl Challenge {
    /// Every picked position, sorted and distinct.
    pub fn positions(&self) -> Vec<u32> {
        let set: std::collections::BTreeSet<u32> = self
            .picks
            .iter()
            .flat_map(|p| p.positions.iter().copied())
            .collect();
        set.into_iter().collect()
    }
}

/// The node's answer: one part per stage.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Opening {
    pub receipt_id: String,
    pub stages: Vec<StageOpening>,
}

impl Opening {
    pub fn stage(&self, index: u32) -> Option<&StageOpening> {
        self.stages.iter().find(|s| s.stage_index == index)
    }
}

/// What one stage opened: its picked layers, the layer-0 input if it holds
/// layer 0, and the leaves at its boundaries.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StageOpening {
    pub stage_index: u32,
    pub layers: Vec<LayerOpening>,
    /// The residual entering layer 0 with its token, at every picked
    /// position. Only the stage holding layer 0 opens these.
    #[serde(default)]
    pub anchors: Vec<AnchorOpening>,
    /// The leaf of the residual entering the stage, at every picked
    /// position. Empty for the stage holding layer 0.
    #[serde(default)]
    pub entry: Vec<BoundaryLeaf>,
    /// The leaf of the residual leaving the stage, at every picked position.
    #[serde(default)]
    pub exit: Vec<BoundaryLeaf>,
    /// The residual the output head reads, at every head position of the
    /// challenge. Only the final stage opens these.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub head: Vec<HeadOpening>,
    /// The partials of every attention pick of the challenge in a layer of
    /// the stage that has attention.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attention: Vec<AttentionPartials>,
    /// The recurrent token mixing of the first attention pick in a layer of
    /// the stage that keeps a recurrent state.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recurrent: Vec<RecurrentOpening>,
}

/// The ubatch that computed an attention pick in a layer that keeps a
/// recurrent state: the mixing values of each of its positions, and of the
/// position after it, whose leaf holds the state the ubatch left. A client
/// recomputes the token mixing from the inputs and the state the ubatch
/// read, and compares the outputs and the state it leaves.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RecurrentOpening {
    /// The attention pick.
    pub position: u32,
    pub layer: u32,
    /// The ubatch's positions of the turn's sequence: `start..end`.
    pub start: u32,
    pub end: u32,
    /// From `start` to `end`, or to `end - 1` when the turn ends there.
    pub leaves: Vec<MixingLeafOpening>,
}

/// One position's mixing leaf: its values and Merkle path.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MixingLeafOpening {
    pub position: u32,
    pub mixing: Vec<TensorOpening>,
    pub path: Vec<String>,
}

/// The residual leaving the model's final layer at one position, which the
/// output head turns into the logits the token was sampled from.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HeadOpening {
    pub position: u32,
    /// Base64 little-endian f32.
    pub residual: String,
    pub path: Vec<String>,
    /// The logits the node computed from the residual, base64 little-endian
    /// f32, one per vocabulary token. A client with a head key checks them
    /// instead of computing the whole output matrix. Absent from nodes that
    /// predate them.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logits: Option<String>,
}

impl HeadOpening {
    pub fn residual_values(&self) -> Result<Vec<f32>> {
        decode_f32(&self.residual).context("decode an output residual")
    }

    pub fn logit_values(&self) -> Result<Option<Vec<f32>>> {
        self.logits
            .as_deref()
            .map(|text| decode_f32(text).context("decode opened logits"))
            .transpose()
    }
}

/// The challenged leaves of one layer, and its weight matmuls and
/// selections recomputed by replay.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LayerOpening {
    pub layer: u32,
    pub leaves: Vec<LeafOpening>,
}

/// The residual entering layer 0 at one position, which the auditor
/// recomputes from the token.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AnchorOpening {
    pub position: u32,
    pub token: i32,
    /// Base64 little-endian f32.
    pub residual: String,
    pub path: Vec<String>,
}

impl AnchorOpening {
    pub fn residual_values(&self) -> Result<Vec<f32>> {
        decode_f32(&self.residual).context("decode an anchor residual")
    }
}

/// A residual leaf at a stage boundary, by hash: two stages commit the same
/// row, so the leaf must be the same in both trees.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BoundaryLeaf {
    pub position: u32,
    /// Hex leaf hash.
    pub leaf: String,
    pub path: Vec<String>,
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
    /// A KV cache row as stored, in the cache's type.
    Stored,
    Output,
    /// The recurrent state a sequence starts the ubatch from, in the leaf of
    /// its first position.
    State,
}

impl MixingKind {
    pub fn tensor_kind(self) -> TensorKind {
        match self {
            Self::Input => TensorKind::MixingInput,
            Self::Stored => TensorKind::MixingStored,
            Self::Output => TensorKind::MixingOutput,
            Self::State => TensorKind::MixingState,
        }
    }

    pub fn from_tensor_kind(kind: TensorKind) -> Option<Self> {
        match kind {
            TensorKind::MixingInput => Some(Self::Input),
            TensorKind::MixingStored => Some(Self::Stored),
            TensorKind::MixingOutput => Some(Self::Output),
            TensorKind::MixingState => Some(Self::State),
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
            .map(|text| decode_bytes(text).context("decode a mixing value"))
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

    /// How many values the input and output hold, without decoding them.
    /// Fails where [`Self::input_values`] or [`Self::output_values`] would.
    pub fn value_counts(&self) -> Result<(usize, usize)> {
        let input =
            f32_count(&self.input).with_context(|| format!("decode input of {}", self.weight))?;
        let output =
            f32_count(&self.output).with_context(|| format!("decode output of {}", self.weight))?;
        Ok((input, output))
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

pub fn decode_bytes(text: &str) -> Result<Vec<u8>> {
    Ok(base64::engine::general_purpose::STANDARD.decode(text)?)
}

/// How many values [`decode_f32`] decodes from `text`, without decoding it.
/// Fails on text it would refuse: not padded base64, or not whole f32s.
fn f32_count(text: &str) -> Result<usize> {
    let bytes = text.as_bytes();
    let padding = bytes.iter().rev().take_while(|&&b| b == b'=').count();
    let digits = &bytes[..bytes.len() - padding];
    let digit = |b: &u8| b.is_ascii_alphanumeric() || *b == b'+' || *b == b'/';
    if !bytes.len().is_multiple_of(4) || padding > 2 || !digits.iter().all(digit) {
        bail!("not base64: {} characters", bytes.len());
    }
    let len = bytes.len() / 4 * 3 - padding;
    if !len.is_multiple_of(4) {
        bail!("f32 payload of {len} bytes");
    }
    Ok(len / 4)
}

pub fn decode_f32(text: &str) -> Result<Vec<f32>> {
    let bytes = decode_bytes(text)?;
    if bytes.len() % 4 != 0 {
        bail!("f32 payload of {} bytes", bytes.len());
    }
    Ok(crate::layer_tensors::decode_f32_bytes(&bytes))
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
    fn f32_count_matches_the_decoded_values() {
        for n in 0..9 {
            let values = vec![1.0f32; n];
            assert_eq!(f32_count(&encode_f32(&values)).unwrap(), n);
        }
        for bad in ["AAA", "AA@A", "AAAAAA=", "A===", "AAAA"] {
            assert!(decode_f32(bad).is_err(), "{bad}");
            assert!(f32_count(bad).is_err(), "{bad}");
        }
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
            prefix: None,
            ubatches: Vec::new(),
            recurrent_state: None,
        };
        let receipt = Receipt {
            version: RECEIPT_VERSION,
            receipt_id: "r".into(),
            model: "m".into(),
            position_start: 0,
            position_end: 1,
            tokens: vec![5],
            generation_start: 1,
            final_token: Some(6),
            stages: vec![stage(0, 0, 18), stage(1, 18, 36)],
            binding: None,
        };
        assert_eq!(receipt.stage_for_layer(17).unwrap().stage_index, 0);
        assert_eq!(receipt.stage_for_layer(18).unwrap().stage_index, 1);
        assert!(receipt.stage_for_layer(36).is_none());
        assert_eq!(receipt.layer_count(), 36);
        assert_eq!(receipt.token_at(0), Some(5));
        assert_eq!(receipt.token_at(1), None);
        assert_eq!(receipt.sampled_tokens(), vec![6]);
        assert_eq!(receipt.output_positions(), 0..1);
        assert_eq!(receipt.sampled_after(0), Some(6));
    }

    #[test]
    fn output_positions_are_the_ones_tokens_were_sampled_from() {
        let receipt = |start, generation, end, final_token| Receipt {
            version: RECEIPT_VERSION,
            receipt_id: "r".into(),
            model: "m".into(),
            position_start: start,
            position_end: end,
            tokens: (start as i32..end as i32).collect(),
            generation_start: generation,
            final_token,
            stages: Vec::new(),
            binding: None,
        };
        // prompt 0..4, sampled tokens at 4..6 and a final one
        let r = receipt(0, 4, 6, Some(9));
        assert_eq!(r.output_positions(), 3..6);
        assert_eq!(r.sampled_after(3), Some(4));
        assert_eq!(r.sampled_after(5), Some(9));
        assert_eq!(r.sampled_after(2), None);
        // the turn continues a cached prefix it did not compute
        let r = receipt(10, 10, 12, Some(9));
        assert_eq!(r.output_positions(), 10..12);
        assert_eq!(r.sampled_after(10), Some(11));
        // no sampled token at all
        let r = receipt(0, 4, 4, None);
        assert!(r.output_positions().is_empty());
        // a turn that stopped without a final token
        let r = receipt(0, 2, 4, None);
        assert_eq!(r.output_positions(), 1..3);
        assert_eq!(r.sampled_after(2), Some(3));
    }
}
