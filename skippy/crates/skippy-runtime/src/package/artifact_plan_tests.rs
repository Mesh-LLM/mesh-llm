use super::*;
use serde_json::{Value, json};
fn artifact(path: &str) -> Value {
    json!({"path":path,"tensor_count":1,"tensor_bytes":1,"artifact_bytes":4,"sha256":"a".repeat(64)})
}
fn manifest() -> Value {
    let layers: Vec<_> = (0..3)
        .map(|index| {
            let mut value = artifact(&format!("layers/{index}.gguf"));
            value["layer_index"] = json!(index);
            value
        })
        .collect();
    let mut projector = artifact("projectors/vision.gguf");
    projector["kind"] = json!("mmproj");
    json!({"schema_version":1,"model_id":"test/model","format":"layer-package","layer_count":3,
        "source_model":{"path":"source.gguf","sha256":"a".repeat(64)},
        "shared":{"metadata":artifact("metadata.gguf"),"embeddings":artifact("embeddings.gguf"),"output":artifact("output.gguf")},
        "layers":layers,"projectors":[projector],
        "skippy_abi_version":format!("{}.{}.{}",skippy_ffi::ABI_VERSION_MAJOR,skippy_ffi::ABI_VERSION_MINOR,skippy_ffi::ABI_VERSION_PATCH)})
}
#[test]
fn balanced_ranges_cover_every_layer_once_with_early_remainders() {
    assert_eq!(even_stage_range(0, 3, 8).unwrap(), (0, 3));
    assert_eq!(even_stage_range(1, 3, 8).unwrap(), (3, 6));
    assert_eq!(even_stage_range(2, 3, 8).unwrap(), (6, 8));
    for layers in 1..32 {
        for count in 1..=layers {
            let mut previous = 0;
            for index in 0..count {
                let (start, end) = even_stage_range(index, count, layers).unwrap();
                assert_eq!(start, previous);
                assert!(end > start);
                previous = end;
            }
            assert_eq!(previous, layers);
        }
    }
    assert_eq!(
        even_stage_range(u32::MAX - 1, u32::MAX, u32::MAX).unwrap(),
        (u32::MAX - 1, u32::MAX)
    );
}
#[test]
fn impossible_counts_indices_and_empty_ranges_refuse() {
    for (index, count, layers) in [(0, 0, 3), (0, 1, 0), (1, 1, 3), (0, 4, 3)] {
        assert!(even_stage_range(index, count, layers).is_err());
    }
    let bytes = serde_json::to_vec(&manifest()).unwrap();
    for (index, count, start, end) in [(0, 3, 1, 1), (0, 3, 0, 4), (3, 3, 0, 1), (0, 4, 0, 1)] {
        assert!(declared_stage_parts(&bytes, index, count, start, end).is_err());
    }
}
#[test]
fn declaration_only_plan_preserves_head_tail_middle_and_all_projectors() {
    let bytes = serde_json::to_vec(&manifest()).unwrap();
    for (index, expected) in [
        (0, vec!["metadata", "embeddings", "layer", "projector"]),
        (1, vec!["metadata", "layer", "projector"]),
        (2, vec!["metadata", "output", "layer", "projector"]),
    ] {
        let parts = declared_stage_parts(&bytes, index, 3, index, index + 1).unwrap();
        assert_eq!(
            parts
                .iter()
                .map(|part| part.role.as_str())
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(parts.iter().find_map(|part| part.layer_index), Some(index));
        assert!(parts.iter().all(|part| part.artifact_bytes == 4));
    }
    let parts = declared_stage_parts(&bytes, 0, 1, 0, 3).unwrap();
    assert_eq!(
        parts
            .iter()
            .map(|part| part.role.as_str())
            .collect::<Vec<_>>(),
        [
            "metadata",
            "embeddings",
            "output",
            "layer",
            "layer",
            "layer",
            "projector"
        ]
    );
}
#[test]
fn malformed_closure_refuses_before_any_declarations() {
    let mut mutations = Vec::new();
    for path in [
        "../escape.gguf",
        "/absolute.gguf",
        "layers/x\ty.gguf",
        "layers\\x.gguf",
        "layers//x.gguf",
        "layers/./x.gguf",
    ] {
        let mut value = manifest();
        value["layers"][1]["path"] = json!(path);
        mutations.push(value);
    }
    let mut value = manifest();
    value["layers"][1]["layer_index"] = json!(0);
    mutations.push(value);
    let mut value = manifest();
    value["projectors"][0]["path"] = json!("metadata.gguf");
    mutations.push(value);
    let mut value = manifest();
    value["layers"][1]["sha256"] = json!("bad");
    mutations.push(value);
    let mut value = manifest();
    value["shared"]["metadata"]["artifact_bytes"] = json!(0);
    mutations.push(value);
    let mut value = manifest();
    value["layers"].as_array_mut().unwrap().remove(1);
    mutations.push(value);
    for value in mutations {
        assert!(declared_stage_parts(&serde_json::to_vec(&value).unwrap(), 1, 3, 1, 2).is_err());
    }
    assert!(declared_stage_parts(&vec![b' '; MAX_MANIFEST_BYTES + 1], 0, 1, 0, 1).is_err());
}

#[test]
fn acquisition_geometry_uses_admitted_schema_and_requires_positive_width() {
    let mut value = manifest();
    assert!(declared_geometry(&serde_json::to_vec(&value).unwrap()).is_err());
    value["activation_width"] = json!(4096);
    assert_eq!(
        declared_geometry(&serde_json::to_vec(&value).unwrap()).unwrap(),
        ("test/model".into(), 3, 4096)
    );
    value["activation_width"] = json!(0);
    assert!(declared_geometry(&serde_json::to_vec(&value).unwrap()).is_err());
    value["activation_width"] = json!(4096);
    value["layers"][0]["path"] = json!("../escape");
    assert!(declared_geometry(&serde_json::to_vec(&value).unwrap()).is_err());
}
