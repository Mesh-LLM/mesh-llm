use super::*;
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

#[test]
fn a_layer_reads_its_small_tensors_and_the_anchor_rows() {
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
        ],
    );
    let opening = opening(serde_json::json!({
        "receipt_id": "r",
        "stages": [{
            "stage_index": 0,
            "layers": [{"layer": 1, "leaves": [leaf(vec![matrix("blk.1.attn_q.weight")])]}],
            "anchors": [{"position": 3, "token": 2, "residual": "", "path": []}],
        }],
    }));
    let mut ranges = glue_ranges(&weights, &opening, &[1]).unwrap();
    ranges.sort_unstable();
    let range = |name: &str| weights.tensor_range(name).unwrap();
    let mut expected = vec![
        range("blk.0.attn_norm.weight"),
        range("blk.1.attn_norm.weight"),
        // not opened, so the glue may read it
        range("blk.1.attn_k.weight"),
        weights.row_range("token_embd.weight", 2).unwrap(),
    ];
    expected.sort_unstable();
    assert_eq!(ranges, expected);
}
