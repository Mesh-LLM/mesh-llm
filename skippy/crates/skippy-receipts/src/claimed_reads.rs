//! The bytes of the claimed GGUF the glue check of an opening reads, so a
//! client that holds a sparse copy fetches them before it loads a layer.
//!
//! A layer loads whole, so the copy must hold every tensor the evaluation
//! reads before the load: the loader copies the tensors and does not see
//! bytes fetched later. The glue evaluation does not read the weights of
//! the matmuls the opening holds; their opened outputs replace them. The
//! input of layer 0 reads one row of the token embedding per anchor.
//!
//! A layer also reads tensors that are not its own: frequency factors of
//! its position encoding (`rope_freqs.weight`), shared norms and
//! projections, whatever its graph names. Which ones is the native
//! planner's to say, and this module has only the header, so it takes every
//! tensor outside the layers but the two large matrices, the token
//! embedding and the output head, which are read by row. A hole where a
//! layer reads one of them is zeros to the loader, and would fail an honest
//! node. One that is too large to fetch whole
//! ([`WHOLE_TENSOR_LIMIT`]) cannot be audited from a sparse copy.
//!
//! The row check and the head check read their rows through the
//! [`WeightFile`] and fetch them themselves. Everything here and there
//! counts against the file's fetch budget.

use std::collections::BTreeSet;

use anyhow::Result;

use crate::weights::limits::{WHOLE_TENSOR_LIMIT, refuse};
use crate::weights::{TensorInfo, WeightFile};
use crate::wire::Opening;

/// The token embedding, which the input of layer 0 reads a row of per token.
pub(crate) const TOKEN_EMBEDDING: &str = "token_embd.weight";
/// The output head, which the head check reads by row.
pub(crate) const OUTPUT_HEAD: &str = "output.weight";

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
            ranges.extend(shared_range(weights, t)?);
            continue;
        };
        // a layer the opening has no matmuls of reads none of its matrices
        let skipped =
            opened.contains(t.name.as_str()) || (t.is_matrix() && !opened_layer(&opened, layer));
        if layers.contains(&layer) && !skipped {
            ranges.push((t.offset, t.total_bytes()?));
        }
    }
    if weights.tensors.contains_key(TOKEN_EMBEDDING) {
        for token in anchors {
            ranges.push(weights.row_range(TOKEN_EMBEDDING, u64::try_from(token)?)?);
        }
    }
    Ok(ranges)
}

/// The range of a tensor outside the layers, which any layer may read: all
/// of it, but nothing of the two matrices read by row. A tensor too large
/// to fetch whole is refused when the copy does not hold it; a local file
/// holds every tensor.
fn shared_range(weights: &WeightFile, t: &TensorInfo) -> Result<Option<(u64, u64)>> {
    if t.name == TOKEN_EMBEDDING || t.name == OUTPUT_HEAD {
        return Ok(None);
    }
    let range = (t.offset, t.total_bytes()?);
    if range.1 > WHOLE_TENSOR_LIMIT && weights.missing_bytes(&[range])? > 0 {
        return refuse(format!(
            "a layer may read {}, which is outside the layers and, at {} bytes, more than the {WHOLE_TENSOR_LIMIT} a sparse copy fetches whole; audit against a local copy of the model",
            t.name, range.1
        ));
    }
    Ok(Some(range))
}

/// Whether the opening holds any matmul of `layer`.
fn opened_layer(opened: &BTreeSet<&str>, layer: u32) -> bool {
    let prefix = format!("blk.{layer}.");
    opened.iter().any(|name| name.starts_with(&prefix))
}

#[cfg(test)]
mod tests;
