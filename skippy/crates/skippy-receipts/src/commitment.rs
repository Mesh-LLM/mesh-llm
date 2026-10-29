//! Leaf hashing and the BLAKE3 Merkle tree over a stage's turn.
//!
//! A stage holding layers `start..end` commits, for every position, to
//! `2 · (end - start) + 1` leaves in this order:
//!
//! `residual(start), mixing(start), residual(start + 1), ..., mixing(end - 1), residual(end)`
//!
//! A residual leaf holds the residual stream entering a layer (`residual(end)`
//! is the stage output). A mixing leaf holds the inputs, stored KV or state
//! rows, and outputs of the layer's token-mixing ops (see
//! [`crate::layer_tensors`]). Leaves are ordered by position, then by that
//! slot order.

use anyhow::{Result, bail};

use crate::layer_tensors::TensorKind;

const RESIDUAL_DOMAIN: &[u8] = b"skippy-receipt-residual-v3";
const MIXING_DOMAIN: &[u8] = b"skippy-receipt-mixing-v3";
const NODE_DOMAIN: &[u8] = b"skippy-receipt-node-v1";

pub type Hash = [u8; 32];

/// Leaves per position for a stage holding `layers` layers.
pub fn slots_per_position(layer_start: u32, layer_end: u32) -> u64 {
    2 * u64::from(layer_end - layer_start) + 1
}

/// Slot of the residual stream entering `layer` (`layer == layer_end` is the output).
pub fn residual_slot(layer: u32, layer_start: u32) -> u64 {
    2 * u64::from(layer - layer_start)
}

/// Slot of the mixing leaf of `layer`.
pub fn mixing_slot(layer: u32, layer_start: u32) -> u64 {
    residual_slot(layer, layer_start) + 1
}

/// Index of a leaf in a stage tree.
pub fn leaf_index(position: u32, slot: u64, position_start: u32, slots: u64) -> u64 {
    u64::from(position - position_start) * slots + slot
}

/// Hashes the byte length, then the bytes. The bytes go to BLAKE3 in one
/// call, so it can hash many chunks in parallel lanes.
fn hash_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u32).to_le_bytes());
    hasher.update(bytes);
}

/// The residual leaf of `layer` at `position`, over little-endian f32
/// bytes. `None` marks a row the layer did not produce: the final layer of
/// a final stage keeps only output rows.
pub fn residual_leaf(position: u32, layer: u32, values: Option<&[u8]>) -> Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RESIDUAL_DOMAIN);
    hasher.update(&position.to_le_bytes());
    hasher.update(&layer.to_le_bytes());
    match values {
        Some(values) => {
            hasher.update(&[1]);
            hash_bytes(&mut hasher, values);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    *hasher.finalize().as_bytes()
}

/// One committed token-mixing value of a layer at a position.
#[derive(Clone, Copy, Debug)]
pub struct MixingValue<'a> {
    pub kind: TensorKind,
    pub index: u32,
    /// ggml type and little-endian bytes; `None` where the layer kept no row.
    pub value: Option<(i32, &'a [u8])>,
}

/// The mixing leaf of `layer` at `position`, over its values in leaf order
/// (inputs, stored rows, outputs, each by index).
pub fn mixing_leaf(position: u32, layer: u32, values: &[MixingValue<'_>]) -> Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(MIXING_DOMAIN);
    hasher.update(&position.to_le_bytes());
    hasher.update(&layer.to_le_bytes());
    hasher.update(&(values.len() as u32).to_le_bytes());
    for v in values {
        hasher.update(&v.kind.code().to_le_bytes());
        hasher.update(&v.index.to_le_bytes());
        match v.value {
            Some((ggml_type, bytes)) => {
                hasher.update(&[1]);
                hasher.update(&ggml_type.to_le_bytes());
                hash_bytes(&mut hasher, bytes);
            }
            None => {
                hasher.update(&[0]);
            }
        }
    }
    *hasher.finalize().as_bytes()
}

fn node(left: &Hash, right: &Hash) -> Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(NODE_DOMAIN);
    hasher.update(left);
    hasher.update(right);
    *hasher.finalize().as_bytes()
}

/// All levels of the tree, leaves first. An odd node is carried up unchanged.
pub struct MerkleTree {
    levels: Vec<Vec<Hash>>,
}

impl MerkleTree {
    pub fn new(leaves: Vec<Hash>) -> Result<Self> {
        if leaves.is_empty() {
            bail!("a receipt needs at least one leaf");
        }
        let mut levels = vec![leaves];
        while levels.last().is_some_and(|level| level.len() > 1) {
            let below = levels.last().expect("non-empty");
            let above = below
                .chunks(2)
                .map(|pair| {
                    if pair.len() == 2 {
                        node(&pair[0], &pair[1])
                    } else {
                        pair[0]
                    }
                })
                .collect();
            levels.push(above);
        }
        Ok(Self { levels })
    }

    pub fn root(&self) -> Hash {
        self.levels.last().expect("non-empty")[0]
    }

    pub fn leaf_count(&self) -> u64 {
        self.levels[0].len() as u64
    }

    pub fn leaf(&self, index: u64) -> Option<Hash> {
        self.levels[0].get(usize::try_from(index).ok()?).copied()
    }

    /// Sibling hashes from the leaf up; `None` marks a carried odd node.
    pub fn path(&self, index: u64) -> Result<Vec<Option<Hash>>> {
        let mut index = usize::try_from(index)?;
        if index >= self.levels[0].len() {
            bail!("leaf {index} outside a tree of {}", self.levels[0].len());
        }
        let mut path = Vec::new();
        for level in &self.levels[..self.levels.len() - 1] {
            path.push(level.get(index ^ 1).copied());
            index /= 2;
        }
        Ok(path)
    }

    /// Bytes the tree keeps in memory.
    pub fn bytes(&self) -> usize {
        self.levels.iter().map(|level| level.len() * 32).sum()
    }
}

/// Encodes a path for the wire: carried levels are the empty string.
pub fn encode_path(path: &[Option<Hash>]) -> Vec<String> {
    path.iter()
        .map(|h| h.map(hex::encode).unwrap_or_default())
        .collect()
}

/// Recomputes the root from a leaf and its encoded path.
pub fn root_from_path(leaf: Hash, index: u64, path: &[String]) -> Result<Hash> {
    let mut acc = leaf;
    let mut index = index;
    for sibling in path {
        if !sibling.is_empty() {
            let bytes: Hash = hex::decode(sibling)?
                .try_into()
                .map_err(|_| anyhow::anyhow!("sibling is not 32 bytes"))?;
            acc = if index.is_multiple_of(2) {
                node(&acc, &bytes)
            } else {
                node(&bytes, &acc)
            };
        }
        index /= 2;
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::layer_tensors::f32_bytes;

    fn leaf(i: u32) -> Hash {
        residual_leaf(i, 0, Some(&f32_bytes(&[i as f32, 1.0])))
    }

    #[test]
    fn every_path_reaches_the_root() {
        for n in [1u32, 2, 3, 5, 8, 13] {
            let leaves: Vec<Hash> = (0..n).map(leaf).collect();
            let tree = MerkleTree::new(leaves.clone()).unwrap();
            for (i, l) in leaves.iter().enumerate() {
                let path = encode_path(&tree.path(i as u64).unwrap());
                assert_eq!(
                    root_from_path(*l, i as u64, &path).unwrap(),
                    tree.root(),
                    "n={n} i={i}"
                );
            }
        }
    }

    #[test]
    fn a_changed_value_breaks_the_path() {
        let leaves: Vec<Hash> = (0..6).map(leaf).collect();
        let tree = MerkleTree::new(leaves).unwrap();
        let path = encode_path(&tree.path(2).unwrap());
        let forged = residual_leaf(2, 0, Some(&f32_bytes(&[2.0001, 1.0])));
        assert_ne!(root_from_path(forged, 2, &path).unwrap(), tree.root());
    }

    #[test]
    fn absent_and_empty_residuals_differ() {
        assert_ne!(residual_leaf(3, 1, None), residual_leaf(3, 1, Some(&[])));
    }

    #[test]
    fn mixing_values_bind_type_and_presence() {
        let bytes = [1u8, 2, 3, 4];
        let value = |ggml_type, present: bool| MixingValue {
            kind: TensorKind::MixingStored,
            index: 0,
            value: present.then_some((ggml_type, &bytes[..])),
        };
        let a = mixing_leaf(1, 2, &[value(1, true)]);
        assert_ne!(a, mixing_leaf(1, 2, &[value(8, true)]));
        assert_ne!(a, mixing_leaf(1, 2, &[value(1, false)]));
        assert_ne!(a, mixing_leaf(1, 3, &[value(1, true)]));
        let mut other = value(1, true);
        other.kind = TensorKind::MixingOutput;
        assert_ne!(a, mixing_leaf(1, 2, &[other]));
    }

    #[test]
    fn slots_order_residuals_and_mixing_by_layer() {
        assert_eq!(slots_per_position(18, 36), 37);
        assert_eq!(residual_slot(18, 18), 0);
        assert_eq!(mixing_slot(18, 18), 1);
        assert_eq!(residual_slot(36, 18), 36);
        assert_eq!(leaf_index(11, 36, 10, 37), 73);
    }
}
