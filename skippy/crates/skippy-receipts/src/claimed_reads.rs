//! The bytes of the claimed GGUF the glue check of an opening reads, so a
//! client that holds a sparse copy fetches them before it loads a layer.
//!
//! A layer loads whole, so the copy must hold every tensor the evaluation
//! reads before the load: the loader copies the tensors and does not see
//! bytes fetched later. The glue evaluation does not read the weights of
//! the matmuls the opening holds; their opened outputs replace them. The
//! input of layer 0 reads one row of the token embedding per anchor.
//!
//! The row check and the head check read their rows through the
//! [`WeightFile`] and fetch them themselves.

use std::collections::BTreeSet;

use anyhow::Result;

use crate::weights::WeightFile;
use crate::wire::Opening;

/// The token embedding, which the input of layer 0 reads a row of per token.
pub const TOKEN_EMBEDDING: &str = "token_embd.weight";

/// The ranges, `(offset, len)`, that loading `layers` and checking the
/// anchors of `opening` read.
pub fn glue_ranges(
    weights: &WeightFile,
    opening: &Opening,
    layers: &[u32],
) -> Result<Vec<(u64, u64)>> {
    let opened: BTreeSet<&str> = opening
        .stages
        .iter()
        .flat_map(|s| &s.layers)
        .flat_map(|l| &l.leaves)
        .flat_map(|leaf| &leaf.matrices)
        .map(|m| m.weight.as_str())
        .collect();
    let mut layers: BTreeSet<u32> = layers.iter().copied().collect();
    let anchors: Vec<i32> = opening
        .stages
        .first()
        .map(|s| s.anchors.iter().map(|a| a.token).collect())
        .unwrap_or_default();
    if !anchors.is_empty() {
        layers.insert(0);
    }
    let mut ranges = Vec::new();
    for t in weights.tensors.values() {
        let Some(layer) = t.layer() else {
            continue;
        };
        // a layer the opening has no matmuls of reads none of its matrices
        let skipped =
            opened.contains(t.name.as_str()) || (t.is_matrix() && !opened_layer(&opened, layer));
        if layers.contains(&layer) && !skipped {
            ranges.push((t.offset, t.total_bytes()));
        }
    }
    if weights.tensors.contains_key(TOKEN_EMBEDDING) {
        for token in anchors {
            ranges.push(weights.row_range(TOKEN_EMBEDDING, u64::try_from(token)?)?);
        }
    }
    Ok(ranges)
}

/// Whether the opening holds any matmul of `layer`.
fn opened_layer(opened: &BTreeSet<&str>, layer: u32) -> bool {
    let prefix = format!("blk.{layer}.");
    opened.iter().any(|name| name.starts_with(&prefix))
}

#[cfg(test)]
mod tests;
