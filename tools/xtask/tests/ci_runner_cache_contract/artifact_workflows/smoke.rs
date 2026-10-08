use super::{document, input, job, named, steps, support, text};
use serde_json::{Value, json};
use std::fs;

#[test]
fn artifact_smoke_callers_keep_dense_recurrent_models_and_bounded_state() {
    let document = document("smoke.yml");
    for platform in ["smoke_tests", "smoke_tests_macos"] {
        let smoke = job(&document, platform);
        for (id, artifact) in [
            ("dense_model", "smollm2-q8-inference"),
            ("recurrent_model", "family-granite-hybrid"),
        ] {
            let restore = steps(smoke)
                .iter()
                .filter(|step| text(step, "id") == Some(id))
                .collect::<Vec<_>>();
            assert_eq!(restore.len(), 1);
            assert_eq!(input(restore[0], "model_artifact_id"), Some(artifact));
        }
        for (phase, script, prefix) in [
            ("standalone inference", "ci-smoke-test.sh", "MESH_CI"),
            (
                "OpenAI client compatibility",
                "ci-compat-smoke.sh",
                "MESH_COMPAT",
            ),
            ("constrained-stack", "ci-smoke-test.sh", "MESH_CI"),
        ] {
            for family in ["Dense", "Recurrent"] {
                let step = named(smoke, &format!("{family} {phase} smoke"));
                let run = text(step, "run").unwrap();
                assert!(run.contains(&format!("scripts/{script}")));
                let owner = if family == "Dense" {
                    "dense_model"
                } else {
                    "recurrent_model"
                };
                assert!(run.contains(&format!("${{{{ steps.{owner}.outputs.model_path }}}}")));
                if family == "Recurrent" {
                    let env = step.get("env").unwrap();
                    for field in ["CTX_SIZE", "BATCH_SIZE", "UBATCH_SIZE"] {
                        assert_eq!(text(env, &format!("{prefix}_{field}")), Some("128"));
                    }
                }
                if phase == "constrained-stack" {
                    assert_eq!(
                        text(step.get("env").unwrap(), "MESH_TOKIO_STACK_SIZE"),
                        Some("2097152")
                    );
                }
            }
        }
    }
}

#[test]
fn artifact_split_smoke_forwards_recurrent_identity_and_durable_restarts() {
    let lane = document("ci-linux-product-smoke-slice.yml");
    let split = job(&lane, "two_node_split");
    assert_eq!(
        text(split, "uses"),
        Some("./.github/workflows/scripted-binary-smoke.yml")
    );
    for (key, value) in [
        ("model_artifact_id", "smollm2-q8-inference"),
        ("kv_recurrent_model_artifact_id", "family-granite-hybrid"),
        ("kv_recurrent_expected_exact_payload_kind", "kv-recurrent"),
        (
            "split_evidence_artifact_name",
            "two-node-split-dense-recurrent-evidence",
        ),
    ] {
        assert_eq!(input(split, key), Some(value));
    }
    assert!(
        input(split, "smoke_script")
            .unwrap()
            .split_whitespace()
            .any(|argument| argument == "MESH_TWO_NODE_SPLIT_DURABLE_L3=1")
    );
    let scripted = document("scripted-binary-smoke.yml");
    let restore = named(
        job(&scripted, "scripted_binary_smoke"),
        "Restore recurrent smoke model",
    );
    assert_eq!(
        text(restore, "uses"),
        Some("./.github/actions/restore-test-model")
    );
    assert_eq!(
        input(restore, "model_artifact_id"),
        Some("${{ inputs.kv_recurrent_model_artifact_id }}")
    );
}

#[test]
fn artifact_main_budget_and_cuda_catalog_preserve_registered_product_smokes() {
    let catalog: Value =
        serde_json::from_slice(&fs::read(support::root().join("ci/slices.yml")).unwrap()).unwrap();
    assert_eq!(catalog["profiles"]["main"]["all_rows"], true);
    assert_eq!(
        catalog["profiles"]["main"]["budgets"],
        json!({"total_max_workers":18,"linux_max_parallel":12,"macos_max_parallel":4,"windows_max_parallel":2})
    );
    let cuda = catalog["runtime_rows"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["id"] == "linux-cuda")
        .unwrap();
    assert_eq!(cuda["cuda_architectures"], "86;120");
    let build_dir = cuda["build_dir"].as_str().unwrap();
    assert!(build_dir.contains("sm86") && build_dir.contains("sm120"));
    let ids = catalog["smoke_rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_str().unwrap())
        .collect::<Vec<_>>();
    for id in [
        "core",
        "two-node-client",
        "two-node-split",
        "core-cuda",
        "metal-model-load",
    ] {
        assert!(ids.contains(&id));
    }
    assert!(
        !ids.iter()
            .any(|id| id.contains("product-integration") || *id == "qwen-recurrent-gate")
    );
    let linux = document("ci-linux-product-smoke-slice.yml");
    for (key, id) in [
        ("core", "core"),
        ("two_node_client", "two-node-client"),
        ("two_node_split", "two-node-split"),
    ] {
        assert_eq!(
            text(job(&linux, key), "if"),
            Some(
                format!("${{{{ contains(fromJson(inputs.smoke_matrix).*.id, '{id}') }}}}").as_str()
            )
        );
    }
    assert_eq!(
        input(job(&linux, "core_cuda"), "runner"),
        Some("gpu-nvidia")
    );
}
