//! The interop profile's Checkpointed Local Log proof items, as a peer sends
//! them (profile §6.2.1 steps 12 and 13): a `cll-inclusion-proof` is
//! `[MMR node index, inclusion path]` and a `cll-consistency-proof` is
//! `[from size, to size, one path per earlier peak, the peaks to the
//! right]`, both in CBOR. Reading one never trusts its sizes or indices
//! beyond the profile's integers.

use cll::mmr::{
    height_at, interior_hash, pos_to_leaf_index, root_from_peaks, verify_inclusion, Hash,
    InclusionProof,
};

/// The longest inclusion or consistency path an MMR of profile size can
/// have.
const MAX_PATH: usize = 64;

/// The largest MMR node index read from a peer: the profile's integers.
const MAX_NODE_INDEX: u64 = (1 << 53) - 1;

/// A `cll-inclusion-proof` item: `[MMR node index, inclusion path]`.
pub(super) fn inclusion_parts(octets: &[u8]) -> Option<(u64, Vec<Hash>)> {
    let value: ciborium::Value = ciborium::from_reader(octets).ok()?;
    let ciborium::Value::Array(items) = value else {
        return None;
    };
    let [index, path] = items.as_slice() else {
        return None;
    };
    let index = u64::try_from(index.as_integer()?).ok()?;
    if index > MAX_NODE_INDEX {
        return None;
    }
    Some((index, hash_list(path)?))
}

/// The leaf an inclusion proof is for.
pub(crate) fn inclusion_leaf(octets: &[u8]) -> Option<u64> {
    let (index, _) = inclusion_parts(octets)?;
    pos_to_leaf_index(index).ok()
}

/// Whether `proof` proves the leaf `leaf` with entry `entry` in the MMR of
/// `size` nodes whose peaks are `peaks` (profile §6.2.1 step 12). The item's
/// node index must be the leaf's own, and `cll::mmr::verify_inclusion`
/// derives the path's shape from the leaf and `size`, so the path must climb
/// from that node to the one peak that holds it. A leaf cannot be passed off
/// at another position or under another size: a peak that is itself a leaf
/// would otherwise match any position given an empty path.
pub(crate) fn proves(proof: &[u8], leaf: u64, entry: &Hash, size: u64, peaks: &[Hash]) -> bool {
    let Some((index, path)) = inclusion_parts(proof) else {
        return false;
    };
    if cll::mmr::leaf_index_to_pos(leaf).ok() != Some(index) {
        return false;
    }
    let Ok(positions) = cll::mmr::peaks(size) else {
        return false;
    };
    // The peak that holds the leaf: peaks ascend and each holds the nodes
    // after the one before it.
    let Some(k) = positions.iter().position(|&p| p >= index) else {
        return false;
    };
    if positions.len() != peaks.len() {
        return false;
    }
    let hex = |hashes: &[Hash]| hashes.iter().map(hex::encode).collect();
    let item = InclusionProof {
        v: 1,
        kind: "inclusion".into(),
        size,
        leaf_index: leaf,
        witness: hex(&path),
        peaks_left: hex(&peaks[..k]),
        peaks_right: hex(&peaks[k + 1..]),
    };
    verify_inclusion(&root_from_peaks(peaks), size, leaf, entry, &item)
}

/// MMR's `included_root`: the node at `index` with hash `node`, carried up
/// its path, as the position it reaches and its hash. `None` when the path
/// climbs past any MMR of profile size.
fn included_root(mut index: u64, mut node: Hash, path: &[Hash]) -> Option<(u64, Hash)> {
    for (height, sibling) in (height_at(index)..).zip(path) {
        if index.checked_add(1)? > MAX_NODE_INDEX {
            return None;
        }
        if height_at(index + 1) > height {
            index += 1;
            node = interior_hash(sibling, &node, index);
        } else {
            index = index.checked_add(2u64.checked_shl(height)?)?;
            if index > MAX_NODE_INDEX {
                return None;
            }
            node = interior_hash(&node, sibling, index);
        }
    }
    Some((index, node))
}

/// Whether `proof` (a `cll-consistency-proof` item) proves that the MMR of
/// `to` nodes with peaks `to_peaks` extends the one of `from` nodes with
/// peaks `from_peaks` (profile §6.2.1 step 13).
pub(super) fn consistent(
    from_size: u64,
    from_peaks: &[Hash],
    to_size: u64,
    to_peaks: &[Hash],
    proof: &[u8],
) -> bool {
    (|| -> Option<bool> {
        let value: ciborium::Value = ciborium::from_reader(proof).ok()?;
        let ciborium::Value::Array(items) = value else {
            return None;
        };
        let [from, to, paths, right] = items.as_slice() else {
            return None;
        };
        let from = u64::try_from(from.as_integer()?).ok()?;
        let to = u64::try_from(to.as_integer()?).ok()?;
        if from != from_size || to != to_size || from == 0 || from > to {
            return Some(false);
        }
        let ciborium::Value::Array(paths) = paths else {
            return None;
        };
        let peak_indices = cll::mmr::peaks(from).ok()?;
        if paths.len() != peak_indices.len() || paths.len() != from_peaks.len() {
            return Some(false);
        }
        let mut roots: Vec<Hash> = Vec::new();
        for ((index, path), peak) in peak_indices.iter().zip(paths).zip(from_peaks) {
            let path = hash_list(path)?;
            if path.len() > MAX_PATH {
                return Some(false);
            }
            let (_, root) = included_root(*index, *peak, &path)?;
            if roots.last() != Some(&root) {
                roots.push(root);
            }
        }
        roots.extend(hash_list(right)?);
        Some(roots == to_peaks)
    })()
    .unwrap_or(false)
}

fn hash_list(value: &ciborium::Value) -> Option<Vec<Hash>> {
    let ciborium::Value::Array(items) = value else {
        return None;
    };
    if items.len() > MAX_PATH {
        return None;
    }
    items
        .iter()
        .map(|item| match item {
            ciborium::Value::Bytes(b) => b.as_slice().try_into().ok(),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_request_parity::{corpus_ledger, NODE_KEY_SEED};
    use cll::checkpoint::verify_checkpoint_cose_offline;
    use cll::mmr::leaf_hash;
    use evidence_request::cll_material::encode_inclusion_proof;

    /// A leaf that is itself a peak cannot be passed off at another
    /// position: in a three-leaf MMR, leaf 2 is the right peak, and an empty
    /// path from leaf 0's position must not reach it.
    #[test]
    fn an_inclusion_proof_proves_its_leafs_position() {
        use cll::mmr::{add_leaf, peaks, MemoryNodeStore, NodeReader};
        let entries = [[1u8; 32], [2u8; 32], [3u8; 32]];
        let mut nodes = MemoryNodeStore::new();
        for e in &entries {
            add_leaf(&mut nodes, leaf_hash(e)).unwrap();
        }
        let size = nodes.size();
        let peak_hashes: Vec<Hash> = peaks(size)
            .unwrap()
            .iter()
            .map(|&p| nodes.node(p))
            .collect();
        let proof = |leaf: u64| {
            let path = cll::mmr::inclusion_proof(&nodes, leaf, size)
                .unwrap()
                .witness;
            let path: Vec<Hash> = path
                .iter()
                .map(|h| hex::decode(h).unwrap().try_into().unwrap())
                .collect();
            encode_inclusion_proof(leaf, &path).unwrap()
        };
        for (leaf, entry) in entries.iter().enumerate() {
            assert!(
                proves(&proof(leaf as u64), leaf as u64, entry, size, &peak_hashes),
                "{leaf}"
            );
        }
        let forged = encode_inclusion_proof(0, &[]).unwrap();
        assert!(!proves(&forged, 0, &entries[2], size, &peak_hashes));
        assert!(!proves(&proof(2), 2, &entries[2], size, &peak_hashes[..1]));
    }

    /// `included_root` stays inside the profile's integers on a hostile
    /// path, and an index that is not a leaf proves nothing.
    #[test]
    fn a_hostile_proof_is_refused_without_panicking() {
        assert!(included_root(MAX_NODE_INDEX, [0; 32], &[[1; 32]; 64]).is_none());
        let entry = [3u8; 32];
        let mut proof = Vec::new();
        ciborium::into_writer(
            &ciborium::Value::Array(vec![
                ciborium::Value::Integer(2.into()),
                ciborium::Value::Array(vec![]),
            ]),
            &mut proof,
        )
        .unwrap();
        assert!(!proves(&proof, 1, &entry, 3, &[leaf_hash(&entry)]));
        assert!(inclusion_parts(b"not cbor").is_none());
        let ledger = corpus_ledger();
        let cps = crate::evidence_request_parity::corpus_checkpoints(
            &ledger,
            &ed25519_dalek::SigningKey::from_bytes(&NODE_KEY_SEED),
        );
        let decoded = |i: usize| {
            verify_checkpoint_cose_offline(&cps[i].statement)
                .decoded
                .unwrap()
        };
        let (a, b) = (decoded(0), decoded(1));
        assert!(!consistent(
            a.mmr_size,
            &a.new_peak_hashes,
            b.mmr_size,
            &b.new_peak_hashes,
            b"\x84\x00"
        ));
    }
}
