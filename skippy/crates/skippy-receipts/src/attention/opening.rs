//! What the attention check sends: the heads a challenge picks, the
//! partials a stage opens for them, the blocks the client then picks, and
//! the K and V rows a stage opens for those blocks, with proofs that bind
//! them to the receipt.

use std::time::Duration;

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::commitment::{Hash, MixingDigest, ValueDigest};
use crate::layer_tensors::TensorKind;
use crate::wire::{MixingKind, TensorOpening, decode_f32, encode_f32};

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

/// How long after its opening a node keeps the rows of a challenged turn, at
/// most. A stage pins the cells of the turn for this long, and a client asks
/// for the rows ([`BlockChallenge`]) within it: the client checks the opening
/// before it asks for the rows.
pub const ROWS_PIN: Duration = Duration::from_secs(30);

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

impl BlockChallenge {
    /// Refuses a request for other rows than a draw asks: a pick that is
    /// not an attention pick of `challenge`, a pick twice, or more blocks
    /// than [`super::BLOCKS_PER_PICK`]. A node checks this before it reads
    /// any row.
    pub fn check_shape(&self, challenge: &crate::wire::Challenge) -> Result<()> {
        if self.receipt_id != challenge.receipt_id {
            anyhow::bail!("the rows name another receipt than the challenge");
        }
        let mut seen = std::collections::BTreeSet::new();
        for pick in &self.picks {
            let asked = challenge
                .attention
                .iter()
                .any(|a| a.position == pick.position && a.layer == pick.layer);
            if !asked || !seen.insert((pick.layer, pick.position)) {
                anyhow::bail!(
                    "position {} of layer {} is not one attention pick of the challenge, once",
                    pick.position,
                    pick.layer
                );
            }
            let increasing = pick.blocks.windows(2).all(|pair| pair[0] < pair[1]);
            if pick.blocks.len() > super::BLOCKS_PER_PICK || !increasing {
                anyhow::bail!(
                    "position {} of layer {} asks for {} blocks; a draw asks for at most {} in increasing order",
                    pick.position,
                    pick.layer,
                    pick.blocks.len(),
                    super::BLOCKS_PER_PICK
                );
            }
        }
        Ok(())
    }

    /// A digest of every field, which the node signs with the rows it
    /// opens ([`crate::turn::sign_rows`]): rows are evidence only of the
    /// blocks the node was asked for.
    pub fn digest(&self) -> Hash {
        let mut h = blake3::Hasher::new_derive_key("mesh-llm receipts block challenge v1");
        let text = |h: &mut blake3::Hasher, s: &str| {
            h.update(&(s.len() as u64).to_le_bytes());
            h.update(s.as_bytes());
        };
        text(&mut h, &self.receipt_id);
        h.update(&(self.picks.len() as u64).to_le_bytes());
        for pick in &self.picks {
            h.update(&pick.position.to_le_bytes());
            h.update(&pick.layer.to_le_bytes());
            h.update(&(pick.blocks.len() as u64).to_le_bytes());
            for block in &pick.blocks {
                h.update(&block.to_le_bytes());
            }
        }
        match &self.proof {
            Some(proof) => {
                h.update(&[1]);
                text(&mut h, proof);
            }
            None => {
                h.update(&[0]);
            }
        }
        *h.finalize().as_bytes()
    }
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
    /// digest, the hex digest of the leaves the stage committed at the
    /// position ([`crate::commitment::position_digest`]), which the
    /// prefix leaf binds with the cell, and the cell's path in the prefix
    /// tree.
    Prefix {
        cell: u32,
        cell_path: Vec<String>,
        leaves: String,
        prefix_path: Vec<String>,
    },
}

pub(crate) fn parse_hash(text: &str) -> Result<Hash> {
    crate::commitment::hex_array(text, "a hash")
}

/// The digests of a layer's stored rows, which must be opened in index
/// order.
pub(crate) fn stored_digests(stored: &[TensorOpening]) -> Result<Vec<MixingDigest>> {
    stored
        .iter()
        .enumerate()
        .map(|(i, row)| {
            ensure!(
                row.kind == MixingKind::Stored && row.index as usize == i,
                "stored rows must be opened in index order"
            );
            Ok(MixingDigest {
                kind: TensorKind::MixingStored,
                index: row.index,
                value: row
                    .bytes()?
                    .map(|bytes| (row.ggml_type, ValueDigest::of(&bytes))),
            })
        })
        .collect()
}
