use super::fixture::{Fixture, repository, stderr};
use crate::workflow_yaml::{self, Node};
use std::fs;

// These finite observers stop at the actual readiness call. They establish
// caller ordering and fail-closed wiring, not model or durable-cache behavior.
const OBSERVERS: &str = r#"
SEED_API_PORT=3131; SEED_CONSOLE_PORT=3231; SEED_BIND_PORT=3331
WORKER_API_PORT=3132; WORKER_CONSOLE_PORT=3232; WORKER_BIND_PORT=3332
SEED_LOG="$WORK_DIR/seed.log"; WORKER_LOG="$WORK_DIR/worker.log"
CLIENT_PID=100; WORKER_PID=101; SEED_PID=102
TOKEN=; MODEL_LABEL=dense; DURABLE_L3=1
RECURRENT_MODEL=fixture-recurrent; RECURRENT_CTX_SIZE=4096
RECURRENT_EXPECTED_EXACT_PAYLOAD_KIND=recurrent
mkdir -p "$WORK_DIR"
printf '{}\n' > "$WORK_DIR/request.json"
printf '{}\n' > "$WORK_DIR/response.json"
start_node() {
  [[ "$#" == 6 ]]
  if [[ "$1" == seed ]]; then [[ -z "$2" ]];
  else [[ "$1" == worker && "$2" == fixture-token ]]; fi
  printf 'start:%s\n' "$1" >> "$EVENTS"
  printf '200\n'
}
wait_for_seed_token() {
  printf 'token:%s\n' "$1" >> "$EVENTS"
  TOKEN=fixture-token
}
kill_tree() { printf 'stop:%s\n' "$1" >> "$EVENTS"; }
wait_for_durable_population() { printf 'population\n' >> "$EVENTS"; }
wait_for_split_topology() {
  [[ -z "$DRIVER_LABEL" && -z "$DRIVER_API_PORT" ]]
  printf 'readiness:%s\n' "$1" >> "$EVENTS"
  exit 73
}
"#;

#[test]
fn actual_dense_durable_and_recurrent_callers_require_readiness_after_both_nodes_start() {
    for leg in ["dense", "durable", "recurrent"] {
        let fixture = Fixture::new();
        let body = match leg {
            "dense" => fixture.section("\nSEED_PID=\"$(start_node seed", "\nPREFIX_PAYLOAD_ROOT="),
            "durable" => {
                fixture.functions(&[("run_durable_restart_probe", "write_durable_l3_evidence")])
                    + "run_durable_restart_probe \"$WORK_DIR/request.json\" \"$WORK_DIR/response.json\"\n"
            }
            _ => {
                fixture.section(
                    "run_recurrent_leg() {",
                    "\nif [[ -n \"$RECURRENT_MODEL\" ]]; then",
                ) + "\nrun_recurrent_leg\n"
            }
        };
        let script = format!(
            "set -euo pipefail\n{OBSERVERS}\n{body}\nprintf 'unexpected continuation\\n' >> \"$EVENTS\"\n"
        );
        let events = fixture.root.join("events");
        let result = fixture.run(script, &[("EVENTS", events.display().to_string())]);
        assert_eq!(
            result.process.status.unwrap().code(),
            Some(73),
            "{leg}: {}",
            stderr(&result)
        );
        let expected = match leg {
            "dense" => "start:seed\ntoken:\nstart:worker\nreadiness:\n",
            "durable" => {
                "population\nstop:100\nstop:101\nstop:102\nstart:seed\ntoken:dense durable restart: \nstart:worker\nreadiness:dense durable restart: \n"
            }
            _ => {
                "stop:100\nstop:101\nstop:102\nstart:seed\ntoken:recurrent leg: \nstart:worker\nreadiness:recurrent leg: \n"
            }
        };
        assert_eq!(fs::read_to_string(events).unwrap(), expected, "{leg}");
    }
}

#[test]
fn structured_split_smoke_retains_the_same_evidence_directory_after_success_or_failure() {
    let source =
        fs::read_to_string(repository().join(".github/workflows/scripted-binary-smoke.yml"))
            .unwrap();
    let document = workflow_yaml::parse(&source).unwrap();
    let job = document
        .get("jobs")
        .unwrap()
        .get("scripted_binary_smoke")
        .unwrap();
    let Node::Seq(steps) = job.get("steps").unwrap() else {
        panic!("steps must be a sequence");
    };
    let named = |name| {
        steps
            .iter()
            .enumerate()
            .filter(|(_, step)| step.get("name").and_then(Node::text) == Some(name))
            .collect::<Vec<_>>()
    };
    let smoke = named("Run scripted smoke");
    let upload = named("Upload split-smoke evidence");
    assert_eq!(smoke.len(), 1);
    assert_eq!(upload.len(), 1);
    let (smoke_index, smoke) = smoke[0];
    let (upload_index, upload) = upload[0];
    assert!(upload_index > smoke_index);
    assert_eq!(
        smoke.get("run").and_then(Node::text),
        Some(
            "${{ inputs.smoke_script }} \"${{ inputs.staged_binary_path }}\" \"${{ inputs.artifact_path }}\" \"${{ steps.model_inputs.outputs.model_path }}\""
        )
    );
    assert_eq!(
        smoke
            .get("env")
            .unwrap()
            .get("MESH_TWO_NODE_SPLIT_WORK_DIR")
            .and_then(Node::text),
        Some(
            "${{ inputs.split_evidence_artifact_name != '' && format('{0}/{1}', runner.temp, inputs.split_evidence_artifact_name) || '' }}"
        )
    );
    assert_eq!(
        upload.get("if").and_then(Node::text),
        Some("${{ (success() || failure()) && inputs.split_evidence_artifact_name != '' }}")
    );
    assert!(
        upload
            .get("uses")
            .and_then(Node::text)
            .unwrap()
            .starts_with("actions/upload-artifact@")
    );
    let inputs = upload.get("with").unwrap();
    assert_eq!(
        inputs.get("name").and_then(Node::text),
        Some("${{ inputs.split_evidence_artifact_name }}")
    );
    assert_eq!(
        inputs.get("path").and_then(Node::text),
        Some("${{ runner.temp }}/${{ inputs.split_evidence_artifact_name }}")
    );
    assert_eq!(
        inputs.get("if-no-files-found").and_then(Node::text),
        Some("error")
    );
}
