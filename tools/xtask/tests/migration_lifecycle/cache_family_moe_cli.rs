//! Actual MoE frontend/inspection/correctness composition, inert supplied artifacts.
use super::cache_family_full_matrix_cli::Fixture;
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Arg,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::Path,
    thread,
    time::{Duration, Instant},
};
fn quoted(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}
fn borrow_forwarder(f: &Fixture) -> String {
    fs::write(
        f.root.join("moe-borrow-required"),
        b"observed switch required",
    )
    .unwrap();
    let binary = f.root.join("correctness");
    let body = fs::read_to_string(&binary).unwrap();
    let forwarder = r#"/bin/cp "$CACHE_MATRIX_ARGS" "$CACHE_FIXTURE_ROOT/incoming-correctness-args"
if [ -f "$CACHE_FIXTURE_ROOT/drop-borrow-flag" ]; then
  : > "$CACHE_MATRIX_ARGS"
  for arg; do
    if [ "$arg" != '--borrow-resident-hits' ]; then
      printf '%s\n' "$arg" >> "$CACHE_MATRIX_ARGS"
    fi
  done
fi
"#;
    assert_eq!(body.matches("exec ").count(), 1);
    let body = body.replacen("exec ", &format!("{forwarder}exec "), 1);
    fs::write(binary, &body).unwrap();
    hex::encode(Sha256::digest(body.as_bytes()))
}
fn input(f: &Fixture) -> std::path::PathBuf {
    let correctness_pin = borrow_forwarder(f);
    let matrix: Value = serde_json::from_slice(&fs::read(&f.input).unwrap()).unwrap();
    let inspector = f.root.join("inspector");
    let toolkit = f.root.join("moe-toolkit");
    fs::create_dir(&toolkit).unwrap();
    fs::write(toolkit.join("version"), b"inert").unwrap();
    let mut tree = Sha256::new();
    tree.update(7_u64.to_be_bytes());
    tree.update(b"version");
    tree.update(Sha256::digest(b"inert"));
    let toolkit_pin = hex::encode(tree.finalize());
    let model = matrix["profiles"]["qwen3_dense"]["correctness"]["model"]
        .as_str()
        .unwrap();
    let body = format!(
        "#!/bin/sh\n[ \"$#\" = 2 ] && [ \"$1\" = inspect ] && [ \"$2\" = {} ] || exit 64\nexport MOE_FIXTURE_ROOT={}\n{} --ignored --exact cache_family_moe_cli::inert_moe_inspector --nocapture >/dev/null 2>&1 || exit $?\n/bin/cat {}/inspection.json\n",
        quoted(model),
        quoted(f.root.to_str().unwrap()),
        quoted(std::env::current_exe().unwrap().to_str().unwrap()),
        quoted(f.root.to_str().unwrap())
    );
    let body = body.replacen(
        "export MOE_FIXTURE_ROOT",
        &format!(
            "[ \"$CUDA_PATH\" = {} ] || exit 65\nexport MOE_FIXTURE_ROOT",
            quoted(toolkit.to_str().unwrap())
        ),
        1,
    );
    fs::write(&inspector, &body).unwrap();
    fs::set_permissions(&inspector, fs::Permissions::from_mode(0o700)).unwrap();
    let mut correctness = matrix["profiles"]["qwen3_dense"]["correctness"].clone();
    correctness["toolkit_directories"] = json!({"CUDA_PATH":{"path":toolkit,"sha256":toolkit_pin}});
    correctness["correctness_sha256"] = json!(correctness_pin);
    correctness["topologies"] = json!(["one-stage", "split-middle", "split-final"]);
    let path = f.root.join("moe-input.json");
    fs::write(&path,serde_json::to_vec(&json!({"schema_version":1,"inspector":inspector,"inspector_sha256":hex::encode(Sha256::digest(body.as_bytes())),"inspector_source_commit":"a".repeat(40),"cases":[{"layer_end":28,"correctness":correctness}],"execution_seconds":120})).unwrap()).unwrap();
    path
}
fn spec(f: &Fixture, input: &Path) -> ProcessSpec {
    ProcessSpec {
        executable: env!("CARGO_BIN_EXE_xtask").into(),
        arguments: [
            "automation".into(),
            "cache-family-moe".into(),
            "--input".into(),
            input.as_os_str().into(),
            "--output".into(),
            f.root.join("moe").into_os_string(),
        ]
        .into_iter()
        .map(Arg::Public)
        .collect(),
        cwd: f.root.clone(),
        environment: BTreeMap::new(),
    }
}
fn invoke(spec: &ProcessSpec, cancel: &Cancellation) -> process::ProcessReport {
    let r = process::supervise(
        spec,
        &Limits {
            execution: Duration::from_secs(130),
            graceful_shutdown: Duration::from_secs(15),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancel,
        OutputFiles::default(),
    )
    .unwrap();
    assert!(
        r.failure.is_none()
            && r.cleanup.complete
            && !r.cleanup.forced
            && !r.cleanup.graceful_signal_failed
            && r.cleanup.failure.is_none()
    );
    assert!(
        [&r.stdout, &r.stderr]
            .iter()
            .all(|s| s.line_capture_complete && !s.truncated && s.oversized_lines == 0)
    );
    r
}
#[test]
fn moe_actual_cli_observes_three_topologies_expert_presence_and_zero_serialized_resident_payload() {
    for missing in [false, true] {
        let f = Fixture::new();
        let input = input(&f);
        if missing {
            fs::write(f.root.join("moe-mode"), b"missing").unwrap();
        }
        let r = invoke(&spec(&f, &input), &Cancellation::default());
        assert_eq!(r.outcome, process::Outcome::Exited);
        assert_eq!(r.status.and_then(|s| s.code()), Some(i32::from(missing)));
        let receipt: Value =
            serde_json::from_slice(&fs::read(f.root.join("moe/moe-expert-smoke.json")).unwrap())
                .unwrap();
        assert_eq!(
            receipt["status"],
            if missing { "incomplete" } else { "completed" }
        );
        let rows = receipt["cases"][0]["observation"]["rows"]
            .as_array()
            .unwrap();
        assert_eq!(rows.len(), 3);
        for row in rows {
            assert_eq!(row["status"], if missing { "fail" } else { "pass" });
            assert!(row["native_seq_remapped"].is_null());
            assert!(row["source_native_seq_id"].is_null());
            assert_eq!(row["serialized_payload_bytes"], 0);
            assert_eq!(row["resident_state_bytes"], 512);
            assert_eq!(row["suffix_prefill_matches"], true);
            assert_eq!(row["borrowed_resident_hits"], true);
        }
        let table: Value = serde_json::from_slice(
            &fs::read(f.root.join("moe/moe-expert-smoke-table.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(table["status"], receipt["status"]);
        assert_eq!(table["rows"].as_array().unwrap().len(), 3);
        assert_eq!(
            fs::read(f.root.join("moe/moe-expert-smoke-table.md")).unwrap(),
            fs::read(f.root.join("moe/moe-expert-smoke.md")).unwrap()
        );
        let observed: Value =
            serde_json::from_slice(&fs::read(f.root.join("borrow-observation.json")).unwrap())
                .unwrap();
        assert_eq!(observed["borrow_resident_hits"], true);
        assert!(
            observed["argv"]
                .as_array()
                .unwrap()
                .iter()
                .any(|a| a == "--borrow-resident-hits")
        );
        assert!(
            fs::read_to_string(f.root.join("moe/moe-expert-smoke.md"))
                .unwrap()
                .contains("sequence IDs unmeasured")
        );
        f.directory.close().unwrap();
    }
}
#[test]
fn moe_actual_cli_inspector_pin_refusal_prevents_inspector_and_correctness() {
    let f = Fixture::new();
    let path = input(&f);
    let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    value["inspector_sha256"] = json!("f".repeat(64));
    fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    let r = invoke(&spec(&f, &path), &Cancellation::default());
    assert_eq!(r.outcome, process::Outcome::Exited);
    assert_eq!(r.status.and_then(|s| s.code()), Some(1));
    assert!(!f.root.join("inspector-admitted").exists());
    assert!(!f.root.join("moe/case-00/correctness").exists());
    f.directory.close().unwrap();
}
#[test]
fn moe_actual_cli_causal_held_inspector_cancellation_publishes_partial_and_reaps() {
    let f = Fixture::new();
    let path = input(&f);
    fs::write(f.root.join("moe-mode"), b"hold").unwrap();
    let cancel = Cancellation::default();
    let child_cancel = cancel.clone();
    let spec = spec(&f, &path);
    let task = thread::spawn(move || invoke(&spec, &child_cancel));
    let until = Instant::now() + Duration::from_secs(15);
    while !f.root.join("inspector-admitted").exists() && Instant::now() < until {
        thread::sleep(Duration::from_millis(5));
    }
    let admitted = f.root.join("inspector-admitted").exists();
    cancel.cancel();
    let r = task.join().unwrap();
    assert!(admitted, "actual inspect call not admitted");
    assert_eq!(r.outcome, process::Outcome::Cancelled);
    let receipt: Value =
        serde_json::from_slice(&fs::read(f.root.join("moe/moe-expert-smoke.json")).unwrap())
            .unwrap();
    assert_ne!(receipt["status"], "completed");
    assert!(!f.root.join("moe/case-00/correctness").exists());
    f.directory.close().unwrap();
}
#[test]
fn moe_actual_cli_dropped_borrow_switch_refuses_report_after_observed_projection() {
    let f = Fixture::new();
    let path = input(&f);
    fs::write(f.root.join("drop-borrow-flag"), b"owned forwarder fault").unwrap();
    let r = invoke(&spec(&f, &path), &Cancellation::default());
    assert_eq!(r.outcome, process::Outcome::Exited);
    assert_eq!(r.status.and_then(|s| s.code()), Some(1));
    let incoming = fs::read_to_string(f.root.join("incoming-correctness-args")).unwrap();
    assert!(incoming.lines().any(|a| a == "--borrow-resident-hits"));
    let observed: Value =
        serde_json::from_slice(&fs::read(f.root.join("borrow-observation.json")).unwrap()).unwrap();
    assert_eq!(observed["borrow_resident_hits"], false);
    assert!(
        !observed["argv"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a == "--borrow-resident-hits")
    );
    let receipt: Value =
        serde_json::from_slice(&fs::read(f.root.join("moe/moe-expert-smoke.json")).unwrap())
            .unwrap();
    assert_ne!(receipt["status"], "completed");
    let rows = receipt["cases"][0]["observation"]["rows"]
        .as_array()
        .unwrap();
    assert_eq!(rows.len(), 3);
    assert!(rows.iter().all(|row| row["status"] != "pass"));
    for index in 0..3 {
        let root = f.root.join(format!(
            "moe/case-00/correctness/output/topology-{index:02}"
        ));
        assert!(!root.join("state-handoff.json").exists());
        let lifecycle: Value =
            serde_json::from_slice(&fs::read(root.join("correctness-process.json")).unwrap())
                .unwrap();
        assert_eq!(lifecycle["outcome"], "Exited");
        assert_ne!(lifecycle["exit_code"], 0);
        assert_eq!(lifecycle["cleanup_complete"], true);
        assert_eq!(lifecycle["cleanup_forced"], false);
    }
    f.directory.close().unwrap();
}
#[test]
#[ignore = "subprocess-only pinned inert MoE inspector"]
fn inert_moe_inspector() {
    let root = std::path::PathBuf::from(std::env::var_os("MOE_FIXTURE_ROOT").unwrap());
    fs::write(
        root.join("inspector-admitted"),
        b"inspect fixed canonical model",
    )
    .unwrap();
    let mode = fs::read_to_string(root.join("moe-mode")).unwrap_or_default();
    if mode == "hold" {
        loop {
            thread::sleep(Duration::from_millis(10));
        }
    }
    let tensors = if mode == "missing" {
        vec![]
    } else {
        [0,10,20].into_iter().map(|layer|json!({"name":format!("blk.{layer}.ffn_up_exps.weight"),"layer_index":layer,"role":"layer","ggml_type":0,"byte_size":100})).collect()
    };
    fs::write(
        root.join("inspection.json"),
        serde_json::to_vec(&json!({"tensor_count":tensors.len(),"tensors":tensors})).unwrap(),
    )
    .unwrap();
}

#[test]
fn moe_actual_cli_explicit_stage0_range_and_toolkit_pin_refusal() {
    for bad_toolkit in [false, true] {
        let f = Fixture::new();
        let path = input(&f);
        let mut request: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        let correctness = &mut request["cases"][0]["correctness"];
        correctness["topologies"] =
            json!(["one-stage", "split-stage0", "split-middle", "split-final"]);
        if bad_toolkit {
            correctness["toolkit_directories"]["CUDA_PATH"]["sha256"] = json!("f".repeat(64));
        }
        fs::write(&path, serde_json::to_vec(&request).unwrap()).unwrap();
        let result = invoke(&spec(&f, &path), &Cancellation::default());
        assert_eq!(
            result.status.and_then(|s| s.code()),
            Some(i32::from(bad_toolkit))
        );
        let receipt: Value =
            serde_json::from_slice(&fs::read(f.root.join("moe/moe-expert-smoke.json")).unwrap())
                .unwrap();
        if bad_toolkit {
            assert_ne!(receipt["status"], "completed");
            assert!(!f.root.join("inspector-admitted").exists());
            assert!(!f.root.join("moe/case-00/correctness").exists());
        } else {
            assert_eq!(receipt["status"], "completed");
            let rows = receipt["cases"][0]["observation"]["rows"]
                .as_array()
                .unwrap();
            assert_eq!(rows.len(), 4);
            assert_eq!(rows[1]["topology"], "split-stage0");
            assert_eq!(rows[1]["layer_start"], 0);
            assert_eq!(rows[1]["layer_end"], 9);
            assert_eq!(rows[1]["expert"]["expert_layers"], json!([0]));
        }
        f.directory.close().unwrap();
    }
}
