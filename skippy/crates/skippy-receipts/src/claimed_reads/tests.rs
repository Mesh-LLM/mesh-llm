use super::*;
use crate::ggml_types::GGML_F32;
use crate::weights::limits::CannotAudit;
use crate::weights::test_gguf;

fn opening(json: serde_json::Value) -> Opening {
    serde_json::from_value(json).unwrap()
}

fn matrix(weight: &str) -> serde_json::Value {
    serde_json::json!({"index": 0, "weight": weight, "input": "", "output": ""})
}

fn leaf(matrices: Vec<serde_json::Value>) -> serde_json::Value {
    serde_json::json!({
        "position": 3, "residual_in": "", "residual_out": null,
        "residual_in_path": [], "mixing_path": [], "residual_out_path": [],
        "mixing": [], "matrices": matrices, "selections": [],
    })
}

/// An opening of layer 1 that holds its `attn_q` matmul and one anchor, of
/// token 2.
fn layer_one() -> Opening {
    opening(serde_json::json!({
        "receipt_id": "r",
        "stages": [{
            "stage_index": 0,
            "layers": [{"layer": 1, "leaves": [leaf(vec![matrix("blk.1.attn_q.weight")])]}],
            "anchors": [{"position": 3, "token": 2, "residual": "", "path": []}],
        }],
    }))
}

#[test]
fn a_layer_reads_its_small_tensors_the_shared_ones_and_the_anchor_rows() {
    let weights = test_gguf::open(
        &[],
        &[
            ("token_embd.weight", vec![2, 4], vec![0.0; 8]),
            ("blk.0.attn_norm.weight", vec![2], vec![1.0; 2]),
            ("blk.0.attn_q.weight", vec![2, 2], vec![0.0; 4]),
            ("blk.1.attn_norm.weight", vec![2], vec![1.0; 2]),
            ("blk.1.attn_q.weight", vec![2, 2], vec![0.0; 4]),
            ("blk.1.attn_k.weight", vec![2, 2], vec![0.0; 4]),
            ("blk.2.attn_norm.weight", vec![2], vec![1.0; 2]),
            ("output_norm.weight", vec![2], vec![1.0; 2]),
            ("rope_freqs.weight", vec![1], vec![1.0]),
            ("per_layer_proj.weight", vec![2, 2], vec![0.0; 4]),
            ("output.weight", vec![2, 4], vec![0.0; 8]),
        ],
    );
    let mut ranges = glue_ranges(&weights, &layer_one(), &[1]).unwrap();
    ranges.sort_unstable();
    let range = |name: &str| weights.tensor_range(name).unwrap();
    let mut expected = vec![
        range("blk.0.attn_norm.weight"),
        range("blk.1.attn_norm.weight"),
        // not opened, so the glue may read it
        range("blk.1.attn_k.weight"),
        // outside the layers: a layer's graph may name any of them, and a
        // hole where it reads one would be zeros to the loader
        range("output_norm.weight"),
        range("rope_freqs.weight"),
        range("per_layer_proj.weight"),
        weights.row_range("token_embd.weight", 2).unwrap(),
    ];
    expected.sort_unstable();
    assert_eq!(ranges, expected);
}

/// A remote model, its header held, with the tensor `shared` outside the
/// layers.
fn remote_with(shared: (&str, Vec<u64>)) -> WeightFile {
    let (header, data) = test_gguf::header(
        &[],
        &[
            ("token_embd.weight", vec![2, 4], GGML_F32),
            ("blk.1.attn_norm.weight", vec![2], GGML_F32),
            ("blk.1.attn_q.weight", vec![2, 2], GGML_F32),
            (shared.0, shared.1, GGML_F32),
        ],
    );
    let size = header.len() as u64 + data;
    WeightFile::open(Box::new(test_gguf::Remote::new(header, size))).unwrap()
}

#[test]
fn a_shared_tensor_too_large_to_fetch_whole_is_not_audited_from_a_copy() {
    // a float past the limit
    let rows = WHOLE_TENSOR_LIMIT / 16 + 1;
    let weights = remote_with(("per_layer_token_embd.weight", vec![4, rows]));
    let error = glue_ranges(&weights, &layer_one(), &[1]).unwrap_err();
    assert!(CannotAudit::caused(&error), "{error:#}");
    assert!(
        error.to_string().contains("per_layer_token_embd.weight"),
        "{error}"
    );
    // at the limit it is fetched
    let weights = remote_with(("per_layer_proj.weight", vec![4, WHOLE_TENSOR_LIMIT / 16]));
    let ranges = glue_ranges(&weights, &layer_one(), &[1]).unwrap();
    let shared = weights.tensor_range("per_layer_proj.weight").unwrap();
    assert_eq!(shared.1, WHOLE_TENSOR_LIMIT);
    assert!(ranges.contains(&shared));
    // the two matrices read by row are never fetched whole, however large
    let weights = remote_with(("output.weight", vec![4, rows]));
    let ranges = glue_ranges(&weights, &layer_one(), &[1]).unwrap();
    assert!(!ranges.contains(&weights.tensor_range("output.weight").unwrap()));
}

#[test]
fn the_tensors_a_layer_loads_count_against_the_budget() {
    let weights = remote_with(("rope_freqs.weight", vec![64]));
    let ranges = glue_ranges(&weights, &layer_one(), &[1]).unwrap();
    // the norm, the frequency factors and the anchor's row
    let bytes: u64 = ranges.iter().map(|(_, len)| len).sum();
    assert_eq!(bytes, 8 + 256 + 8);
    weights.set_fetch_budget(bytes - 1);
    let error = weights.prefetch(&ranges).unwrap_err();
    assert!(CannotAudit::caused(&error), "{error:#}");
    weights.set_fetch_budget(bytes);
    weights.prefetch(&ranges).unwrap();
    assert_eq!(weights.fetched_bytes(), bytes);
}

#[test]
fn a_tensor_of_a_type_without_a_size_is_an_error_not_a_hole() {
    let mut weights = test_gguf::open(
        &[],
        &[
            ("blk.1.attn_norm.weight", vec![2], vec![1.0; 2]),
            ("rope_freqs.weight", vec![1], vec![1.0]),
        ],
    );
    // a type id ggml removed: once sized zero, so never fetched
    weights
        .tensors
        .get_mut("blk.1.attn_norm.weight")
        .unwrap()
        .ggml_type = 4;
    assert!(glue_ranges(&weights, &layer_one(), &[1]).is_err());
}
