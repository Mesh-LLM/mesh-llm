//! What the attention check sends: the heads a challenge picks, the
//! partials a stage opens for them, the blocks the client then picks, and
//! the K and V rows a stage opens for those blocks, with proofs that bind
//! them to the receipt.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::commitment::Hash;
use crate::wire::{TensorOpening, decode_f32, encode_f32};

/// A head of one challenged position and layer whose attention the client
/// checks. Part of the challenge, so the node opens the head's partials
/// before the client picks the blocks it recomputes.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct AttentionPick {
    pub position: u32,
    pub layer: u32,
    /// The head is this draw modulo the layer's query heads, which only
    /// the model knows.
    pub head_draw: u32,
}

/// One head's partials at a picked position, block after block over the
/// positions it attends to: per block, the largest logit, the sum of the
/// exponentials below it and the values they weight
/// ([`super::block_partial`]).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AttentionPartials {
    pub position: u32,
    pub layer: u32,
    pub head: u32,
    /// Base64 little-endian f32.
    pub partials: String,
}

impl AttentionPartials {
    pub fn new(pick: &AttentionPick, head: u32, partials: &[f32]) -> Self {
        Self {
            position: pick.position,
            layer: pick.layer,
            head,
            partials: encode_f32(partials),
        }
    }

    pub fn values(&self) -> Result<Vec<f32>> {
        decode_f32(&self.partials).context("decode attention partials")
    }
}

/// The blocks of one picked head the client recomputes, drawn after the
/// partials arrived.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct BlockPick {
    pub position: u32,
    pub layer: u32,
    pub blocks: Vec<u32>,
}

/// The client's second request: the K and V rows of the picked blocks.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct BlockChallenge {
    pub receipt_id: String,
    pub picks: Vec<BlockPick>,
    /// Hex VRF proof, under the receipt's client key, of the opening the
    /// blocks are drawn from ([`crate::turn::rows_input`]). Required when
    /// the receipt names a key: only its client reads the rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proof: Option<String>,
}

/// The node's answer: the rows each stage holds.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RowsOpening {
    pub receipt_id: String,
    pub stages: Vec<StageRows>,
}

/// The rows one stage opened, each once however many picks read it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StageRows {
    pub stage_index: u32,
    pub rows: Vec<RowOpening>,
}

/// What a stage stored at one attended position of one layer: every
/// stored row of the layer, by index, in the cache's types, and how they
/// are bound.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RowOpening {
    pub position: u32,
    pub layer: u32,
    pub stored: Vec<TensorOpening>,
    pub proof: RowProof,
}

/// How an opened row is bound to the receipt.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RowProof {
    /// The turn computed the position: the half of its mixing leaf that is
    /// not stored rows ([`crate::commitment::mixing_rest`]), and the leaf's
    /// path in the stage tree.
    Turn { rest: String, path: Vec<String> },
    /// An earlier computation stored it in a cell the turn read: the
    /// cell's index, the path of the layer's stored leaf in the cell's
    /// digest, and the cell's path in the prefix tree.
    Prefix {
        cell: u32,
        cell_path: Vec<String>,
        prefix_path: Vec<String>,
    },
}

pub(crate) fn parse_hash(text: &str) -> Result<Hash> {
    crate::commitment::hex_array(text, "a hash")
}
