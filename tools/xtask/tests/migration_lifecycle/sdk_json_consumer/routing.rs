use super::{
    fixture::{capture, repository},
    workflow_yaml::{self, Node},
};
use crate::process::{ProcessSpec, Value};
use serde_json::json;
use std::{collections::BTreeMap, fs};
fn document(path: &str) -> Node {
    workflow_yaml::parse(&fs::read_to_string(repository().join(path)).unwrap()).unwrap()
}
fn steps(node: &Node) -> &[Node] {
    let Node::Seq(items) = node.get("steps").unwrap() else {
        panic!("steps must be a sequence")
    };
    items
}
fn wiring(lane: &Node, workflow: &Node, action: &Node) -> bool {
    let product = lane
        .get("jobs")
        .and_then(|jobs| jobs.get("runtime_product"))
        .unwrap();
    let mut needs = product.get("needs").unwrap().list();
    needs.sort_unstable();
    let job = workflow.get("jobs").unwrap().get("linux_product").unwrap();
    needs == ["authority_linux_x64", "hosts", "native_runtimes"]
        && product
            .get("with")
            .and_then(|inputs| inputs.get("authority_linux_x64"))
            .and_then(Node::text)
            == Some("${{ needs.authority_linux_x64.outputs.identity_json }}")
        && product.get("uses").and_then(Node::text)
            == Some("./.github/workflows/ci-linux-product-slice.yml")
        && steps(job).iter().any(|step| {
            step.get("with")
                .and_then(|with| with.get("ref"))
                .and_then(Node::text)
                == Some("${{ inputs.source_sha || github.sha }}")
        })
        && steps(job).iter().any(|step| {
            step.get("uses").and_then(Node::text) == Some("./.github/actions/compose-product-input")
                && step
                    .get("with")
                    .and_then(|with| with.get("readiness_smoke"))
                    .is_none()
        })
        && action
            .get("inputs")
            .unwrap()
            .get("readiness_smoke")
            .unwrap()
            .get("default")
            .and_then(Node::text)
            == Some("true")
        && steps(action.get("runs").unwrap()).iter().any(|step| {
            step.get("run").and_then(Node::text).is_some_and(|run| {
                run.lines()
                    .any(|line| line.trim() == "scripts/ci-compose-product-input.sh")
            })
        })
}
#[test]
fn sdk_json_consumer_cli_changes_route_producers_and_default_required_consumer() {
    let scratch = tempfile::tempdir().unwrap();
    let root = scratch.path().canonicalize().unwrap();
    let mut payload: serde_json::Value = serde_json::from_slice(
        &fs::read(repository().join("scripts/tests/fixtures/ci-plan/runtime-catalog-pr-1675.json"))
            .unwrap(),
    )
    .unwrap();
    for changed in [
        payload["changed_files"].clone(),
        json!(["crates/mesh-llm-commands/src/runtime_native/formatters.rs"]),
    ] {
        payload["changed_files"] = changed;
        fs::write(
            root.join("input.json"),
            serde_json::to_vec(&payload).unwrap(),
        )
        .unwrap();
        let result = capture(ProcessSpec {
            executable: "/bin/bash".into(),
            cwd: repository(),
            environment: BTreeMap::new(),
            arguments: [
                "-c".into(),
                "exec \"$1\" ci plan < \"$2\"".into(),
                "fixture".into(),
                env!("CARGO_BIN_EXE_xtask").into(),
                root.join("input.json").into_os_string(),
            ]
            .into_iter()
            .map(Value::Public)
            .collect(),
        });
        assert!(result.process.success(), "{:?}", result.process);
        let plan: serde_json::Value =
            serde_json::from_slice(result.stdout.unwrap().as_bytes()).unwrap();
        assert!(
            !plan["domains"]
                .as_array()
                .unwrap()
                .contains(&json!("ci-control"))
        );
        assert_eq!(plan["matrices"]["sdk"], json!([]));
        for slice in ["ui-artifact", "runtime-product"] {
            assert!(
                plan["required_slices"]
                    .as_array()
                    .unwrap()
                    .contains(&json!(slice))
            );
        }
        for (matrix, id) in [
            ("runtime_products", "linux-cpu"),
            ("hosts", "linux-amd64-host"),
        ] {
            let ids: Vec<_> = plan["matrices"][matrix]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| row["id"].clone())
                .collect();
            assert_eq!(ids, vec![json!(id)]);
        }
    }
    let lane = document(".github/workflows/ci-linux-lane.yml");
    let workflow = document(".github/workflows/ci-linux-product-slice.yml");
    let action = document(".github/actions/compose-product-input/action.yml");
    assert!(wiring(&lane, &workflow, &action));
    let lane_source =
        fs::read_to_string(repository().join(".github/workflows/ci-linux-lane.yml")).unwrap();
    let edge = "needs: [hosts, native_runtimes, authority_linux_x64]";
    assert!(lane_source.contains(edge));
    for omitted in ["hosts", "native_runtimes", "authority_linux_x64"] {
        let retained = ["hosts", "native_runtimes", "authority_linux_x64"]
            .into_iter()
            .filter(|job| *job != omitted)
            .collect::<Vec<_>>()
            .join(", ");
        let changed =
            workflow_yaml::parse(&lane_source.replacen(edge, &format!("needs: [{retained}]"), 1))
                .unwrap();
        assert!(!wiring(&changed, &workflow, &action), "dropped {omitted}");
    }
    let substituted = workflow_yaml::parse(&lane_source.replace(
        "authority_linux_x64: ${{ needs.authority_linux_x64.outputs.identity_json }}",
        "authority_linux_x64: ${{ inputs.source_sha }}",
    ))
    .unwrap();
    assert!(!wiring(&substituted, &workflow, &action));
    let source =
        fs::read_to_string(repository().join(".github/actions/compose-product-input/action.yml"))
            .unwrap();
    let disabled =
        workflow_yaml::parse(&source.replace("default: \"true\"", "default: \"false\"")).unwrap();
    assert!(!wiring(&lane, &workflow, &disabled));
}
