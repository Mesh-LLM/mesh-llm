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
//! [`crate::layer_tensors`]): a hash of the stored rows ([`stored_leaf`]) and
//! a hash of the other values ([`mixing_rest`]), so a stored row can be
//! opened without the others. Leaves are ordered by position, then by that
//! slot order.

use anyhow::{Context, Result, anyhow, bail};

use crate::layer_tensors::TensorKind;

const RESIDUAL_DOMAIN: &[u8] = b"skippy-receipt-residual-v1";
const MIXING_DOMAIN: &[u8] = b"skippy-receipt-mixing-v2";
const MIXING_REST_DOMAIN: &[u8] = b"skippy-receipt-mixing-rest-v1";
const STORED_DOMAIN: &[u8] = b"skippy-receipt-stored-v1";
const NODE_DOMAIN: &[u8] = b"skippy-receipt-node-v0";
const RECURRENT_STATE_DOMAIN: &[u8] = b"skippy-receipt-recurrent-state-v1";
const PREFIX_DOMAIN: &[u8] = b"skippy-receipt-prefix-v2";
const POSITION_DOMAIN: &[u8] = b"skippy-receipt-position-v1";
const PREFIX_TOKENS_DOMAIN: &[u8] = b"skippy-receipt-prefix-tokens-v1";

pub type Hash = [u8; 32];

/// Decodes `N` bytes from hex; `what` names the value in errors.
pub fn hex_array<const N: usize>(text: &str, what: &str) -> Result<[u8; N]> {
    let bytes = hex::decode(text).with_context(|| format!("decode {what} from hex"))?;
    <[u8; N]>::try_from(bytes.as_slice())
        .map_err(|_| anyhow!("{what} has {} bytes, not {N}", bytes.len()))
}

/// Leaves per position for a stage holding `layers` layers.
pub fn slots_per_position(layer_start: u32, layer_end: u32) -> u64 {
    2 * u64::from(layer_end - layer_start) + 1
}

/// Slot of the residual stream entering `layer` (`layer == layer_end` is the output).
pub(crate) fn residual_slot(layer: u32, layer_start: u32) -> u64 {
    2 * u64::from(layer - layer_start)
}

/// Slot of the mixing leaf of `layer`.
pub(crate) fn mixing_slot(layer: u32, layer_start: u32) -> u64 {
    residual_slot(layer, layer_start) + 1
}

/// Index of a leaf in a stage tree.
pub(crate) fn leaf_index(position: u32, slot: u64, position_start: u32, slots: u64) -> u64 {
    u64::from(position - position_start) * slots + slot
}

/// A committed value as a leaf binds it: its byte length and the BLAKE3 hash
/// of its bytes. A recorder can take the digest where the value is computed,
/// on the device, and copy 32 bytes instead of the value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ValueDigest {
    pub len: u32,
    pub digest: Hash,
}

impl ValueDigest {
    pub fn of(bytes: &[u8]) -> Self {
        Self {
            len: bytes.len() as u32,
            digest: *blake3::hash(bytes).as_bytes(),
        }
    }

    fn hash_into(&self, hasher: &mut blake3::Hasher) {
        hasher.update(&self.len.to_le_bytes());
        hasher.update(&self.digest);
    }
}

/// The residual leaf of `layer` at `position`, over little-endian f32
/// bytes. `None` marks a row the layer did not produce: the final layer of
/// a final stage keeps only output rows. The residual entering layer 0 also
/// binds the position's token, so the auditor can recompute it.
pub(crate) fn residual_leaf(
    position: u32,
    layer: u32,
    token: Option<i32>,
    values: Option<&[u8]>,
) -> Hash {
    residual_leaf_of(position, layer, token, values.map(ValueDigest::of))
}

/// [`residual_leaf`] from the digest of the values.
pub(crate) fn residual_leaf_of(
    position: u32,
    layer: u32,
    token: Option<i32>,
    value: Option<ValueDigest>,
) -> Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RESIDUAL_DOMAIN);
    hasher.update(&position.to_le_bytes());
    hasher.update(&layer.to_le_bytes());
    match token {
        Some(token) => {
            hasher.update(&[1]);
            hasher.update(&token.to_le_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    match value {
        Some(value) => {
            hasher.update(&[1]);
            value.hash_into(&mut hasher);
        }
        None => {
            hasher.update(&[0]);
        }
    }
    *hasher.finalize().as_bytes()
}

/// One committed token-mixing value of a layer at a position.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MixingValue<'a> {
    pub kind: TensorKind,
    pub index: u32,
    /// ggml type and little-endian bytes; `None` where the layer kept no row.
    pub value: Option<(i32, &'a [u8])>,
}

/// A [`MixingValue`] by its digest.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MixingDigest {
    pub kind: TensorKind,
    pub index: u32,
    /// ggml type and the value's digest; `None` where the layer kept no row.
    pub value: Option<(i32, ValueDigest)>,
}

/// The mixing leaf of `layer` at `position`, over its values in leaf order
/// (inputs, stored rows, outputs, each by index).
pub(crate) fn mixing_leaf(position: u32, layer: u32, values: &[MixingValue<'_>]) -> Hash {
    let digests: Vec<MixingDigest> = values
        .iter()
        .map(|v| MixingDigest {
            kind: v.kind,
            index: v.index,
            value: v.value.map(|(t, bytes)| (t, ValueDigest::of(bytes))),
        })
        .collect();
    mixing_leaf_of(position, layer, &digests)
}

/// [`mixing_leaf`] from the digests of the values.
pub(crate) fn mixing_leaf_of(position: u32, layer: u32, values: &[MixingDigest]) -> Hash {
    let (stored, rest): (Vec<MixingDigest>, Vec<MixingDigest>) = values
        .iter()
        .partition(|v| v.kind == TensorKind::MixingStored);
    mixing_leaf_from(
        position,
        layer,
        &mixing_rest(&rest),
        &stored_leaf(position, layer, &stored),
    )
}

/// The mixing leaf from its two halves: the digest of the values that are
/// not stored rows ([`mixing_rest`]) and the leaf of the stored rows
/// ([`stored_leaf`]). An attention check opens a stored row with the other
/// half alone.
pub(crate) fn mixing_leaf_from(position: u32, layer: u32, rest: &Hash, stored: &Hash) -> Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(MIXING_DOMAIN);
    hasher.update(&position.to_le_bytes());
    hasher.update(&layer.to_le_bytes());
    hasher.update(rest);
    hasher.update(stored);
    *hasher.finalize().as_bytes()
}

/// The digest of a mixing leaf's values that are not stored rows, in leaf
/// order.
pub(crate) fn mixing_rest(values: &[MixingDigest]) -> Hash {
    digest_list(MIXING_REST_DOMAIN, None, values)
}

/// The digest of the recurrent state a turn of a hybrid model started from,
/// as the stage exported it: every recurrent layer's state for the turn's
/// sequence.
pub(crate) fn recurrent_state_digest(state: &[u8]) -> Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RECURRENT_STATE_DOMAIN);
    hasher.update(&(state.len() as u64).to_le_bytes());
    hasher.update(state);
    *hasher.finalize().as_bytes()
}

/// The leaf of the rows a layer stored in the KV cache at `position`, by
/// index. It is half of the layer's mixing leaf, and a leaf of the cell's
/// digest ([`cell_digest`]).
pub(crate) fn stored_leaf(position: u32, layer: u32, stored: &[MixingDigest]) -> Hash {
    digest_list(STORED_DOMAIN, Some((position, layer)), stored)
}

/// The digest of a KV cell a stage wrote at one position: the Merkle root
/// over its layers' [`stored_leaf`]s, in layer order. A cell's rows of one
/// layer are opened with that leaf's path.
pub(crate) fn cell_digest(stored_leaves: Vec<Hash>) -> Result<Hash> {
    Ok(MerkleTree::new(stored_leaves)?.root())
}

/// The digest of every leaf a stage committed at one position when it
/// computed it, in slot order. A prefix leaf binds it ([`prefix_leaf`]), so
/// a client can open a leaf of a position the turn read from cache: the
/// node sends the position's leaves, and they must hash to this digest.
pub(crate) fn position_digest(leaves: &[Hash]) -> Hash {
    let mut h = blake3::Hasher::new();
    h.update(POSITION_DOMAIN);
    h.update(&(leaves.len() as u64).to_le_bytes());
    for leaf in leaves {
        h.update(leaf);
    }
    *h.finalize().as_bytes()
}

/// The leaf of one KV cell that a turn read below its first position, in the
/// tree a stage commits as the prefix of the turn: the position the cell
/// holds, the index of the cell in its stream of the cache, the digest of
/// what the cell stores ([`cell_digest`]), and the digest of the leaves the
/// stage committed when it computed the position ([`position_digest`]).
pub(crate) fn prefix_leaf(
    position: i32,
    cell: u32,
    cell_digest: &Hash,
    position_digest: &Hash,
) -> Hash {
    let mut h = blake3::Hasher::new();
    h.update(PREFIX_DOMAIN);
    h.update(&position.to_le_bytes());
    h.update(&cell.to_le_bytes());
    h.update(cell_digest);
    h.update(position_digest);
    *h.finalize().as_bytes()
}

/// The digest of the tokens of a turn's cached prefix, from position 0. The
/// stage that holds layer 0 commits it with the prefix, so the prompt a
/// client checks covers the tokens the turn did not compute.
pub(crate) fn prefix_tokens_digest(tokens: &[i32]) -> Hash {
    let mut h = blake3::Hasher::new();
    h.update(PREFIX_TOKENS_DOMAIN);
    h.update(&(tokens.len() as u64).to_le_bytes());
    for token in tokens {
        h.update(&token.to_le_bytes());
    }
    *h.finalize().as_bytes()
}

fn digest_list(domain: &[u8], at: Option<(u32, u32)>, values: &[MixingDigest]) -> Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    if let Some((position, layer)) = at {
        hasher.update(&position.to_le_bytes());
        hasher.update(&layer.to_le_bytes());
    }
    hasher.update(&(values.len() as u32).to_le_bytes());
    for v in values {
        hasher.update(&v.kind.code().to_le_bytes());
        hasher.update(&v.index.to_le_bytes());
        match v.value {
            Some((ggml_type, value)) => {
                hasher.update(&[1]);
                hasher.update(&ggml_type.to_le_bytes());
                value.hash_into(&mut hasher);
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
pub(crate) struct MerkleTree {
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
pub(crate) fn encode_path(path: &[Option<Hash>]) -> Vec<String> {
    path.iter()
        .map(|h| h.map(hex::encode).unwrap_or_default())
        .collect()
}

/// Which levels of the path to `index` in a tree of `leaf_count` leaves
/// have a sibling (`true`) and which carry an odd node up (`false`).
pub(crate) fn path_shape(index: u64, leaf_count: u64) -> Result<Vec<bool>> {
    if index >= leaf_count {
        bail!("leaf {index} outside a tree of {leaf_count}");
    }
    let (mut index, mut width) = (index, leaf_count);
    let mut shape = Vec::new();
    while width > 1 {
        shape.push((index ^ 1) < width);
        index /= 2;
        width = width.div_ceil(2);
    }
    Ok(shape)
}

/// Recomputes the root from a leaf and its encoded path. The path must have
/// the shape of the tree the receipt names: a free choice of where siblings
/// sit would let one index open two different leaves.
pub(crate) fn root_from_path(
    leaf: Hash,
    index: u64,
    leaf_count: u64,
    path: &[String],
) -> Result<Hash> {
    let shape = path_shape(index, leaf_count)?;
    let fits = shape.len() == path.len()
        && shape
            .iter()
            .zip(path)
            .all(|(&sibling, hash)| sibling != hash.is_empty());
    if !fits {
        bail!("the path to leaf {index} does not fit a tree of {leaf_count} leaves");
    }
    let mut acc = leaf;
    let mut index = index;
    for sibling in path {
        if !sibling.is_empty() {
            let bytes: Hash = hex_array(sibling, "a path sibling")?;
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
        residual_leaf(i, 0, None, Some(&f32_bytes(&[i as f32, 1.0])))
    }

    fn digest(kind: TensorKind, index: u32, byte: u8) -> MixingDigest {
        MixingDigest {
            kind,
            index,
            value: Some((1, ValueDigest::of(&[byte; 8]))),
        }
    }

    #[test]
    fn stored_rows_open_with_the_other_half_alone() {
        let values = [
            digest(TensorKind::MixingInput, 0, 1),
            digest(TensorKind::MixingStored, 0, 2),
            digest(TensorKind::MixingStored, 1, 3),
            digest(TensorKind::MixingOutput, 0, 4),
        ];
        let rest = mixing_rest(&[values[0], values[3]]);
        let stored = stored_leaf(7, 2, &values[1..3]);
        assert_eq!(
            mixing_leaf_of(7, 2, &values),
            mixing_leaf_from(7, 2, &rest, &stored)
        );
        let changed = digest(TensorKind::MixingStored, 1, 9);
        assert_ne!(stored_leaf(7, 2, &[values[1], changed]), stored);
        assert_ne!(
            stored_leaf(8, 2, &values[1..3]),
            stored,
            "binds the position"
        );
        assert_ne!(stored_leaf(7, 3, &values[1..3]), stored, "binds the layer");
    }

    #[test]
    fn every_path_reaches_the_root() {
        for n in [1u32, 2, 3, 5, 8, 13] {
            let leaves: Vec<Hash> = (0..n).map(leaf).collect();
            let tree = MerkleTree::new(leaves.clone()).unwrap();
            for (i, l) in leaves.iter().enumerate() {
                let path = encode_path(&tree.path(i as u64).unwrap());
                assert_eq!(
                    root_from_path(*l, i as u64, u64::from(n), &path).unwrap(),
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
        let forged = residual_leaf(2, 0, None, Some(&f32_bytes(&[2.0001, 1.0])));
        assert_ne!(root_from_path(forged, 2, 6, &path).unwrap(), tree.root());
    }

    #[test]
    fn one_index_cannot_open_two_leaves() {
        // A root over [b, a] would verify both a (path "", b) and b (path a)
        // at index 2 if the auditor let the prover choose where siblings sit.
        let (a, b) = (leaf(1), leaf(2));
        let root = node(&b, &a);
        let carried = vec![String::new(), hex::encode(b)];
        let direct = vec![hex::encode(a)];
        for leaf_count in [3, 4, 5, 6] {
            let opened = [
                root_from_path(a, 2, leaf_count, &carried),
                root_from_path(b, 2, leaf_count, &direct),
            ];
            assert!(
                opened
                    .iter()
                    .filter(|r| r.as_ref().ok() == Some(&root))
                    .count()
                    < 2,
                "leaf_count {leaf_count}"
            );
        }
    }

    #[test]
    fn path_shapes_follow_the_tree() {
        assert_eq!(path_shape(0, 1).unwrap(), Vec::<bool>::new());
        assert_eq!(path_shape(4, 5).unwrap(), vec![false, false, true]);
        assert_eq!(path_shape(2, 3).unwrap(), vec![false, true]);
        assert!(path_shape(5, 5).is_err());
        for n in [1u64, 2, 3, 5, 8, 13] {
            let tree = MerkleTree::new((0..n as u32).map(leaf).collect()).unwrap();
            for i in 0..n {
                let path = tree.path(i).unwrap();
                let shape: Vec<bool> = path.iter().map(Option::is_some).collect();
                assert_eq!(shape, path_shape(i, n).unwrap(), "n={n} i={i}");
            }
        }
    }

    #[test]
    fn absent_and_empty_residuals_differ() {
        assert_ne!(
            residual_leaf(3, 1, None, None),
            residual_leaf(3, 1, None, Some(&[]))
        );
    }

    #[test]
    fn the_token_is_part_of_the_leaf() {
        let values = f32_bytes(&[1.0]);
        let leaf = |token| residual_leaf(0, 0, token, Some(&values));
        assert_ne!(leaf(Some(7)), leaf(Some(8)));
        assert_ne!(leaf(Some(7)), leaf(None));
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
    fn digests_from_the_device_make_the_same_leaves() {
        // more than one BLAKE3 chunk, as a device hashes it
        let values = f32_bytes(&(0..2560).map(|i| i as f32 * 0.5).collect::<Vec<_>>());
        let digest = ValueDigest::of(&values);
        assert_eq!(digest.len, 10240);
        assert_eq!(
            residual_leaf(3, 4, None, Some(&values)),
            residual_leaf_of(3, 4, None, Some(digest))
        );
        let value = MixingValue {
            kind: TensorKind::MixingStored,
            index: 1,
            value: Some((1, &values[..])),
        };
        let by_digest = MixingDigest {
            kind: TensorKind::MixingStored,
            index: 1,
            value: Some((1, digest)),
        };
        assert_eq!(
            mixing_leaf(3, 4, &[value]),
            mixing_leaf_of(3, 4, &[by_digest])
        );
        // the length is bound as well as the hash
        let short = ValueDigest {
            len: 10236,
            ..digest
        };
        assert_ne!(
            residual_leaf_of(3, 4, None, Some(digest)),
            residual_leaf_of(3, 4, None, Some(short))
        );
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
