//! The types of stored KV rows: the ones a row can be decoded in, and the
//! ones an audit covers.
//!
//! A stored row arrives from the node with a ggml type id. A row decodes in
//! the KV cache types a stage can serve with, and in f32 ([`DECODED`]). An
//! id outside that list is never decoded: the runtime may not know the type
//! at all, or may read more bytes for it than the node sent.
//!
//! An audit covers fewer types ([`AUDITED`]): f32 and f16. A quantized
//! cache (q8_0, q4_0) is not covered. Its rows differ from what the
//! attention of a prompt ubatch read by more than the loose attention check
//! allows: on CUDA, honest q8_0 turns were up to 5.5 times past the limit
//! (`mesh/docs/design/RECEIPTS_MEASUREMENTS.md`). So a stage with a
//! quantized cache issues no receipt, a client does not audit a receipt
//! that names one, and the auditor builds its own layer with an audited
//! type only.
//!
//! A node decodes its own cache rows here too, when it computes attention
//! partials ([`crate::attention`]), so both sides read a row the same way.

use anyhow::Result;

use crate::dequant::dequantize_row;
use crate::ggml_types::{GGML_F16, GGML_F32, GGML_Q4_0, GGML_Q8_0, type_layout, type_name};
use crate::node_fault::{BlameNode, ensure_node};

/// The ggml types a stored row can be decoded in.
pub(crate) const DECODED: [u32; 4] = [GGML_F32, GGML_F16, GGML_Q4_0, GGML_Q8_0];

/// The ggml types of the stored rows an audit covers: a cache that is not
/// quantized.
pub(crate) const AUDITED: [u32; 2] = [GGML_F32, GGML_F16];

/// The type `ggml_type` names, if a stored row can be decoded in it.
pub fn decoded(ggml_type: i32) -> Option<u32> {
    u32::try_from(ggml_type)
        .ok()
        .filter(|t| DECODED.contains(t))
}

/// The type `ggml_type` names, if an audit covers stored rows of it.
pub fn audited(ggml_type: i32) -> Option<u32> {
    u32::try_from(ggml_type)
        .ok()
        .filter(|t| AUDITED.contains(t))
}

/// Whether a receipt's cache type name is a type an audit covers.
pub(crate) fn audited_name(name: &str) -> bool {
    AUDITED.iter().any(|&t| type_name(t) == name)
}

/// How a receipt names the type `ggml_type`. An id that is no type has a
/// name no receipt uses.
pub fn name(ggml_type: i32) -> String {
    u32::try_from(ggml_type).map_or_else(|_| format!("type{ggml_type}"), type_name)
}

/// The bytes of a row of `values` values in the type `ggml_type`; `None` when the type holds no whole row of that length.
pub fn row_bytes(ggml_type: u32, values: usize) -> Option<usize> {
    let (block, bytes) = type_layout(ggml_type)?;
    let (block, bytes) = (usize::try_from(block).ok()?, usize::try_from(bytes).ok()?);
    values
        .is_multiple_of(block)
        .then(|| (values / block).checked_mul(bytes))
        .flatten()
}

/// The values a stored row of `ggml_type` holds, decoded. Fails, as the
/// node's fault, for a type outside [`DECODED`], a row that is not whole
/// blocks of the type, or a value that is not a number.
pub(crate) fn decode_row(ggml_type: i32, bytes: &[u8]) -> Result<Vec<f64>> {
    let Some(ggml_type) = decoded(ggml_type) else {
        crate::node_fault::node_fault!(
            "a stored row is of ggml type {ggml_type}, which the auditor does not decode"
        );
    };
    let name = type_name(ggml_type);
    let (block, block_bytes) = type_layout(ggml_type).unwrap_or((1, 4));
    let (block, block_bytes) = (block as usize, block_bytes as usize);
    ensure_node!(
        !bytes.is_empty() && bytes.len().is_multiple_of(block_bytes),
        "a stored {name} row of {} bytes is not whole blocks of {block_bytes}",
        bytes.len()
    );
    let values = bytes.len() / block_bytes * block;
    let row = dequantize_row(ggml_type, bytes, values).or_node_fault()?;
    ensure_node!(
        row.iter().all(|v| v.is_finite()),
        "a stored {name} row holds a value that is not a number"
    );
    Ok(row)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node_fault::NodeFault;

    #[test]
    fn a_row_decodes_only_in_the_cache_types_a_stage_serves_with() {
        assert_eq!(decoded(1), Some(GGML_F16));
        assert_eq!(decoded(8), Some(GGML_Q8_0));
        // q4_1 is a ggml type no stage stores rows in; 4 and 39 are no types
        for unknown in [3, 4, 39, -1, i32::MAX] {
            assert_eq!(decoded(unknown), None, "{unknown}");
        }
        assert_eq!(name(1), "f16");
        assert_eq!(name(-3), "type-3");
    }

    #[test]
    fn an_audit_covers_no_quantized_cache() {
        assert_eq!(audited(0), Some(GGML_F32));
        assert_eq!(audited(1), Some(GGML_F16));
        // a row decodes in q4_0 and q8_0, and no audit covers them
        for quantized in [GGML_Q4_0, GGML_Q8_0] {
            assert_eq!(decoded(quantized as i32), Some(quantized));
            assert_eq!(audited(quantized as i32), None, "{quantized}");
        }
        for unknown in [3, 39, -1, i32::MAX] {
            assert_eq!(audited(unknown), None, "{unknown}");
        }
        assert!(audited_name("f16") && audited_name("f32"));
        for name in ["q8_0", "q4_0", "q4_1", "type39", ""] {
            assert!(!audited_name(name), "{name}");
        }
    }

    #[test]
    fn a_row_has_the_bytes_its_type_packs_its_values_into() {
        assert_eq!(row_bytes(GGML_F32, 6), Some(24));
        assert_eq!(row_bytes(GGML_F16, 6), Some(12));
        assert_eq!(row_bytes(GGML_Q8_0, 64), Some(68));
        // q8_0 packs 32 values to a block
        assert_eq!(row_bytes(GGML_Q8_0, 48), None);
        assert_eq!(row_bytes(77, 4), None);
    }

    #[test]
    fn a_stored_row_decodes_only_when_whole_and_finite() {
        let half = |v: f32| half::f16::from_f32(v).to_le_bytes();
        let row: Vec<u8> = [1.5f32, -2.0].into_iter().flat_map(half).collect();
        assert_eq!(decode_row(1, &row).unwrap(), vec![1.5, -2.0]);
        let faults = [
            decode_row(1, &row[..3]),
            decode_row(1, &[]),
            decode_row(39, &row),
            decode_row(1, &half(f32::NAN)),
            decode_row(1, &half(f32::INFINITY)),
            decode_row(0, &f32::NAN.to_le_bytes()),
            // a q8_0 block needs 34 bytes
            decode_row(8, &[0; 33]),
        ];
        for fault in faults {
            let error = fault.unwrap_err();
            assert!(NodeFault::find(&error).is_some(), "{error:#}");
        }
    }
}
