use super::placement::project;
use serde_json::{Value, json};
const GIB: u64 = 1024 * 1024 * 1024;

fn artifact(size: u64) -> Value {
    json!({"files":["weights.gguf"],"file_integrity":{"weights.gguf":{"size_bytes":size}}})
}
fn model(family: &str, workload: &str, size: u64) -> Value {
    json!({"family":family,"class":workload,"certification_lanes":[],"artifact":artifact(size),
        "draft_artifact":null,"mmproj_artifact":null,"resources":{"estimated_model_bytes":size}})
}
fn plan(models: Vec<Value>) -> Value {
    let rows = models
        .iter()
        .enumerate()
        .map(|(index, model)| {
            json!({
        "id":format!("shard-{index}"),"shard_index":index,"families":model["family"],
        "estimated_work_bytes":model["resources"]["estimated_model_bytes"]})
        })
        .collect::<Vec<_>>();
    let shards = models
        .iter()
        .enumerate()
        .map(|(index, model)| {
            json!({
        "shard_index":index,"families":[model["family"].clone()]})
        })
        .collect::<Vec<_>>();
    json!({"selected_models":models,"github_matrix":{"include":rows},"shards":shards,
        "required_certification_lanes":["single-step","chain","state-handoff"]})
}
fn project_json(plan: &Value) -> Value {
    serde_json::to_value(project(&serde_json::to_vec(plan).unwrap()).unwrap()).unwrap()
}

#[test]
fn causal_and_workload_residency_include_auxiliary_weights_and_runtime_allowances() {
    let mut causal = model("causal", "causal_generation", 8 * GIB);
    causal["draft_artifact"] = artifact(GIB);
    causal["mmproj_artifact"] = artifact(GIB);
    causal["resources"]["estimated_model_bytes"] = (12 * GIB).into();
    let projected = project_json(&plan(vec![causal, model("embed", "embedding", 8 * GIB)]));
    let rows = projected["include"].as_array().unwrap();
    assert_eq!(rows[0]["families"], "embed");
    assert_eq!(rows[0]["resident_model_bytes"], 16 * GIB);
    assert_eq!(rows[0]["runtime_allowance_bytes"], 8 * GIB);
    assert_eq!(rows[1]["resident_model_bytes"], 14 * GIB);
    assert_eq!(rows[1]["runtime_allowance_bytes"], 14 * GIB / 4 + 6 * GIB);
}

#[test]
fn source_minimum_can_promote_but_never_demote_estimated_placement() {
    let mut small = model("small", "causal_generation", GIB);
    small["resources"]["minimum_runner_memory_gib"] = 256.into();
    let mut large = model("large", "causal_generation", 100 * GIB);
    large["resources"]["minimum_runner_memory_gib"] = 128.into();
    let projected = project_json(&plan(vec![large, small]));
    for row in projected["include"].as_array().unwrap() {
        assert_eq!(row["memory_tier"], "accelerator-memory-256plus");
    }
}

#[test]
fn largest_tier_boundary_reserves_ten_percent_and_fails_closed() {
    // Peak is weights + ceil(weights / 4) + 6 GiB, so this is admitted.
    let admitted = 179 * GIB;
    assert!(
        project(
            &serde_json::to_vec(&plan(vec![model("fits", "causal_generation", admitted)])).unwrap()
        )
        .is_ok()
    );
    assert!(
        project(
            &serde_json::to_vec(&plan(vec![model(
                "oversize",
                "causal_generation",
                180 * GIB
            )]))
            .unwrap()
        )
        .is_err()
    );
}

#[test]
fn malformed_sizes_classes_minima_and_overflow_reject_without_projection() {
    let baseline = model("invalid", "causal_generation", GIB);
    for (pointer, invalid) in [
        ("/resources/estimated_model_bytes", json!(0)),
        ("/resources/minimum_runner_memory_gib", json!(192)),
        ("/class", json!("unknown")),
        ("/artifact/files", json!(["weights.gguf", "weights.gguf"])),
        ("/artifact/file_integrity/weights.gguf/size_bytes", json!(0)),
        ("/resources/estimated_model_bytes", json!(u64::MAX)),
    ] {
        let mut changed = baseline.clone();
        if pointer.ends_with("minimum_runner_memory_gib") {
            changed["resources"]["minimum_runner_memory_gib"] = invalid;
        } else {
            *changed.pointer_mut(pointer).unwrap() = invalid;
        }
        assert!(
            project(&serde_json::to_vec(&plan(vec![changed])).unwrap()).is_err(),
            "{pointer}"
        );
    }
}

#[test]
fn stable_projection_orders_equal_size_families_without_modifying_source_plan() {
    let value = plan(vec![
        model("zeta", "rerank", GIB),
        model("alpha", "rerank", GIB),
    ]);
    let bytes = serde_json::to_vec(&value).unwrap();
    let result = serde_json::to_value(project(&bytes).unwrap()).unwrap();
    assert_eq!(result["include"][0]["families"], "alpha");
    assert_eq!(result["include"][0]["shard_index"], 1);
    assert_eq!(result["include"][1]["families"], "zeta");
    assert_eq!(serde_json::from_slice::<Value>(&bytes).unwrap(), value);
}
