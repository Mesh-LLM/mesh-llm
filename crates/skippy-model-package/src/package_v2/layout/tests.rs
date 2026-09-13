use super::*;
use skippy_runtime::TensorInfo;

fn info(name: &str, role: TensorRole, layer: i32) -> TensorInfo {
    TensorInfo {
        name: name.to_string(),
        layer_index: u32::try_from(layer).ok(),
        role,
        ggml_type: 0,
        byte_size: 16,
        element_count: 4,
    }
}

fn plan_artifacts(tensors: &[TensorInfo]) -> Result<Vec<PlannedArtifact>> {
    plan_artifacts_with_budget(tensors, DEFAULT_MAX_ARTIFACT_BYTES)
}

#[test]
fn assigns_every_role_to_its_canonical_artifact() {
    let planned = plan_artifacts(&[
        info("tokenizer.ggml.tokens", TensorRole::Tokenizer, -1),
        info("unknown.thing", TensorRole::Unknown, -1),
        info("caption.notes", TensorRole::Metadata, -1),
        info("token_embd.weight", TensorRole::Embedding, -1),
        info("output_norm.weight", TensorRole::FinalNorm, -1),
        info("output.weight", TensorRole::Output, -1),
        info("blk.0.attn_q.weight", TensorRole::Layer, 0),
        info("blk.1.attn_q.weight", TensorRole::Layer, 1),
        info("blk.11.ffn_down.weight", TensorRole::Layer, 11),
    ])
    .unwrap();
    assert_eq!(
        planned
            .iter()
            .map(|artifact| artifact.path.as_str())
            .collect::<Vec<_>>(),
        [
            "shared/common.gguf",
            "shared/embeddings.gguf",
            "shared/output.gguf",
            "layers/layer-00000.gguf",
            "layers/layer-00001.gguf",
            "layers/layer-00011.gguf",
        ]
    );
    let by_id = planned
        .iter()
        .map(|artifact| (artifact.id.as_str(), artifact))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        by_id["common"].tensor_names,
        ["caption.notes", "tokenizer.ggml.tokens", "unknown.thing"]
    );
    assert_eq!(by_id["embeddings"].tensor_names, ["token_embd.weight"]);
    assert_eq!(
        by_id["output"].tensor_names,
        ["output.weight", "output_norm.weight"]
    );
    assert_eq!(by_id["layer-00000"].tensor_names, ["blk.0.attn_q.weight"]);
    assert_eq!(
        by_id["layer-00011"].tensor_names,
        ["blk.11.ffn_down.weight"]
    );
}

#[test]
fn omits_empty_shared_artifacts_but_never_a_layer() {
    let planned = plan_artifacts(&[info("blk.0.attn_q.weight", TensorRole::Layer, 0)]).unwrap();
    assert_eq!(planned.len(), 1);
    assert_eq!(planned[0].path, "layers/layer-00000.gguf");
    assert!(matches!(
        planned[0].kind,
        PlannedArtifactKind::Layer { ordinal: 0 }
    ));
}

#[test]
fn layer_tensor_without_index_fails_closed() {
    assert!(plan_artifacts(&[info("blk.0.attn_q.weight", TensorRole::Layer, -1)]).is_err());
}

#[test]
fn every_tensor_is_bound_exactly_once() {
    let tensors = vec![
        info("tokenizer.ggml.tokens", TensorRole::Tokenizer, -1),
        info("token_embd.weight", TensorRole::Embedding, -1),
        info("output_norm.weight", TensorRole::FinalNorm, -1),
        info("blk.0.attn_q.weight", TensorRole::Layer, 0),
        info("blk.1.attn_q.weight", TensorRole::Layer, 1),
    ];
    let planned = plan_artifacts(&tensors).unwrap();
    let mut bound = planned
        .iter()
        .flat_map(|artifact| &artifact.tensor_names)
        .cloned()
        .collect::<Vec<_>>();
    bound.sort();
    let mut expected = tensors
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();
    expected.sort();
    assert_eq!(bound, expected);
}

#[test]
fn oversized_layer_splits_into_byte_balanced_parts() {
    let tensors = [
        info("blk.0.attn_q.weight", TensorRole::Layer, 0),
        info("blk.0.attn_k.weight", TensorRole::Layer, 0),
        info("blk.0.attn_v.weight", TensorRole::Layer, 0),
        info("blk.0.ffn_down.weight", TensorRole::Layer, 0),
    ]
    .map(|mut tensor| {
        tensor.byte_size = 10;
        tensor
    });
    let planned = plan_artifacts_with_budget(&tensors, 21).unwrap();
    let paths: Vec<&str> = planned.iter().map(|a| a.path.as_str()).collect();
    assert_eq!(
        paths,
        [
            "layers/layer-00000-part00.gguf",
            "layers/layer-00000-part01.gguf",
        ]
    );
    assert!(planned.iter().all(PlannedArtifact::is_part));
    // Two byte-balanced parts of a 40-byte layer hold 20/20 within budget.
    for part in &planned {
        let bytes: u64 = part
            .tensor_names
            .iter()
            .map(|name| tensors.iter().find(|t| t.name == *name).unwrap().byte_size)
            .sum();
        assert!(bytes <= 21);
    }
    let mut bound: Vec<&str> = planned
        .iter()
        .flat_map(|artifact| artifact.tensor_names.iter().map(String::as_str))
        .collect();
    bound.sort_unstable();
    assert_eq!(
        bound,
        [
            "blk.0.attn_k.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_v.weight",
            "blk.0.ffn_down.weight",
        ]
    );
}

#[test]
fn dominating_tensor_stays_whole_rather_than_exceeding_budget_in_parts() {
    let tensors = [
        info("blk.0.attn_q.weight", TensorRole::Layer, 0),
        info("blk.0.ffn_down.weight", TensorRole::Layer, 0),
    ]
    .map(|mut tensor| {
        tensor.byte_size = 100;
        tensor
    });
    let planned = plan_artifacts_with_budget(&tensors, 21).unwrap();
    // The oversized tensor is indivisible: keep it whole instead of emitting
    // parts that still exceed the budget.
    assert_eq!(planned.len(), 1);
    assert_eq!(planned[0].path, "layers/layer-00000.gguf");
    assert!(!planned[0].is_part());
    assert_eq!(planned[0].tensor_names.len(), 2);
}

#[test]
fn zero_budget_fails_closed() {
    assert!(
        plan_artifacts_with_budget(&[info("blk.0.attn_q.weight", TensorRole::Layer, 0)], 0)
            .is_err()
    );
}
