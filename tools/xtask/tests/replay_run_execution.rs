#[test]
fn run_execution_qualifies_two_arms_and_reverses_second_pass() {
    execute_case(70.0, true, "captured");
}

#[test]
fn configured_cache_gate_fails_after_retaining_complete_run() {
    execute_case(80.0, false, "captured");
}

#[test]
fn pinned_reference_is_retained_while_local_model_is_launched() {
    execute_case(70.0, true, "reference");
}

#[cfg(unix)]
#[test]
fn retained_reader_feeds_integrated_run_without_generic_python_execution() {
    execute_case(70.0, true, "dataset");
}

#[cfg(unix)]
#[test]
fn build_jobs_feed_integrated_run_through_bounded_executable_fixtures() {
    execute_case(70.0, true, "build");
}

fn execute_case(minimum_cache: f64, passed: bool, mode: &str) {
    use sha2::{Digest, Sha256};
    let state = tempfile::tempdir().unwrap();
    let model = state.path().join("model.gguf");
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.extend(2_u64.to_le_bytes());
    for (key, value) in [
        ("general.architecture", Some("fixture")),
        ("fixture.context_length", None),
    ] {
        bytes.extend(u64::try_from(key.len()).unwrap().to_le_bytes());
        bytes.extend(key.as_bytes());
        if let Some(value) = value {
            bytes.extend(8_u32.to_le_bytes());
            bytes.extend(u64::try_from(value.len()).unwrap().to_le_bytes());
            bytes.extend(value.as_bytes());
        } else {
            bytes.extend(4_u32.to_le_bytes());
            bytes.extend(131072_u32.to_le_bytes());
        }
    }
    let model_digest = hex::encode(Sha256::digest(&bytes));
    std::fs::write(&model, bytes).unwrap();
    let runtime_root = state.path().join("native-runtimes");
    let runtime = runtime_root.join("fixture");
    std::fs::create_dir_all(&runtime).unwrap();
    let runtime_bytes = b"fixture runtime";
    std::fs::write(runtime.join("runtime.so"), runtime_bytes).unwrap();
    let mut tree = Sha256::new();
    tree.update(10_u64.to_be_bytes());
    tree.update(b"runtime.so");
    tree.update(Sha256::digest(runtime_bytes));
    let runtime_digest = hex::encode(tree.finalize());
    let binary = std::path::Path::new(env!("CARGO_BIN_EXE_laya-product-fixture"));
    let binary_digest = hex::encode(Sha256::digest(std::fs::read(binary).unwrap()));
    let builds:Vec<_> = ["baseline","candidate"].into_iter().map(|label|serde_json::json!({
        "label":label,"ref":label,"commit":label,"binary":binary,"binary_sha256":binary_digest,
        "runtime_root":runtime_root,"runtime":runtime,"runtime_sha256":runtime_digest
    })).collect();
    let trajectory = |session: &str| {
        serde_json::json!({"session_id":session,"source_dataset":"fixture","agent_framework":"goose","recorded_model":null,
        "messages":[{"role":"user","content":"task"},{"role":"assistant","content":"first"},
            {"role":"user","content":"next"},{"role":"assistant","content":"final"}]})
    };
    let manifest = state.path().join("manifest.json");
    std::fs::write(
        &manifest,
        serde_json::to_vec(&serde_json::json!({"metadata":{"name":"real harness capture","revision":"r1"},"cohorts":{
            "warmup":[trajectory("warmup")],"1":[trajectory("first"),trajectory("second")]
        }}))
        .unwrap(),
    )
    .unwrap();
    let input = state.path().join("input.json");
    let output = state.path().join("run");
    std::fs::write(&input,serde_json::to_vec(&serde_json::json!({"manifest":manifest,
        "requirements":{"concurrency":[1],"minimum_worker_waves":2,"warmup_turns":1,"required_frameworks":["goose"]},
        "builds":builds,"model":model,"model_sha256":model_digest,"minimum_context_tokens":131072,
        "minimum_session_prompt_tokens":1,"require_recurrent_restores":true,"passes":2,"max_output_tokens":2048,
        "request_timeout_seconds":2,"startup_timeout_seconds":3,"timeout_seconds":10,"output":output,
        "require_output_match":true,"min_cache_pct":minimum_cache
    })).unwrap()).unwrap();
    if mode == "reference" {
        let mut request: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&input).unwrap()).unwrap();
        request["model_reference"] = "fixture/model@0123456789abcdef/model.gguf".into();
        std::fs::write(&input, serde_json::to_vec(&request).unwrap()).unwrap();
    }
    #[cfg(unix)]
    if mode == "dataset" {
        use std::os::unix::fs::PermissionsExt;
        let generated = state.path().join("generated.json");
        let reader = state.path().join("reader-fixture");
        std::fs::write(&reader, format!("#!/bin/sh\ncase \"$1\" in */mesh/evals/agentic-trajectory-manifest.py) ;; *) exit 8;; esac\nshift\noutput=''\nwhile [ \"$#\" -gt 0 ]; do\nif [ \"$1\" = '--output' ]; then output=\"$2\"; fi\nshift 2\ndone\ncp '{}' \"$output\"\n",manifest.display())).unwrap();
        std::fs::set_permissions(&reader, std::fs::Permissions::from_mode(0o700)).unwrap();
        let dataset = state.path().join("input.parquet");
        std::fs::write(&dataset, b"fixture dataset").unwrap();
        let mut request: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&input).unwrap()).unwrap();
        request["manifest"] = serde_json::to_value(generated).unwrap();
        request["dataset"] = serde_json::json!({"file":dataset,"sha256":hex::encode(Sha256::digest(b"fixture dataset")),
            "revision":"fixture","python":reader,"timeout_seconds":2,"sessions_per_cohort":2,
            "min_isl":1,"max_isl":100,"min_turns":1,"frameworks":["goose"],"source_datasets":["fixture"]});
        std::fs::write(&input, serde_json::to_vec(&request).unwrap()).unwrap();
    }
    #[cfg(unix)]
    if mode == "build" {
        use std::os::unix::fs::PermissionsExt;
        let worktrees = state.path().join("worktrees");
        let git = state.path().join("git-fixture");
        let just = state.path().join("just-fixture");
        let commit = "0123456789abcdef0123456789abcdef01234567";
        std::fs::write(&git,format!("#!/bin/sh\ncase \"$1\" in rev-parse) printf '%s\\n' '{commit}';; status) exit 0;; *) exit 9;; esac\n")).unwrap();
        std::fs::write(&just, "#!/bin/sh\nexit 0\n").unwrap();
        for path in [&git, &just] {
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        let mut jobs = Vec::new();
        for label in ["baseline", "candidate"] {
            let worktree = worktrees.join(format!("{label}-0123456789"));
            std::fs::create_dir_all(worktree.join("target/release")).unwrap();
            std::fs::copy(binary, worktree.join("target/release/mesh-llm")).unwrap();
            let runtime = worktree.join("dist/native-runtimes/fixture");
            std::fs::create_dir_all(&runtime).unwrap();
            std::fs::write(runtime.join("runtime.so"), runtime_bytes).unwrap();
            std::fs::write(
                runtime.join("manifest.json"),
                br#"{"runtime":{"backend":{"kind":"cpu"}}}"#,
            )
            .unwrap();
            jobs.push(serde_json::json!({"repo":state.path(),"worktree_root":worktrees,"label":label,"ref":label,
                "backend":"cpu","git":git,"just":just,"timeout_seconds":5,"logs":state.path().join(format!("build-logs/{label}"))}));
        }
        let mut request: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&input).unwrap()).unwrap();
        request["builds"] = serde_json::json!([]);
        request["build_jobs"] = jobs.into();
        std::fs::write(&input, serde_json::to_vec(&request).unwrap()).unwrap();
    }
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "execute-run", "--input"])
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        result.status.success() == passed,
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let run: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("run.json")).unwrap()).unwrap();
    let labels: Vec<_> = run["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|result| result["label"].as_str().unwrap())
        .collect();
    assert_eq!(labels, ["baseline", "candidate", "candidate", "baseline"]);
    assert_eq!(
        run["order"],
        serde_json::json!([
            {"pass":1,"label":"baseline"},{"pass":1,"label":"candidate"},
            {"pass":2,"label":"candidate"},{"pass":2,"label":"baseline"}
        ])
    );
    assert!(run["completed_at"].as_str().unwrap().contains('T'));
    assert_eq!(
        run["config"]["model_file"],
        serde_json::to_value(&model).unwrap()
    );
    assert_eq!(run["config"]["model_sha256"], model_digest);
    if mode == "reference" {
        assert_eq!(
            run["config"]["model"],
            "fixture/model@0123456789abcdef/model.gguf"
        );
    }
    assert_eq!(run["gates"]["passed"], passed);
    if !passed {
        report::assert_failed_report(&output, &run);
    }
    assert_eq!(run["context_preflight"]["baseline"]["passed"], true);
    assert_eq!(
        run["inputs"]["kind"],
        if mode == "dataset" {
            "thoughtworks"
        } else {
            "captured"
        }
    );
    assert_imported_capture(&manifest, &output, &run, mode);
    let comparison: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("summary/comparison.json")).unwrap())
            .unwrap();
    assert_eq!(comparison[0]["passes"], 2);
    assert_eq!(comparison[1]["delta_comparable"], true);
    resume::assert_resume_integrity(&input, &run, passed);
}
#[path = "replay_run_report/mod.rs"]
mod report;
#[path = "replay_run_report/resume.rs"]
mod resume;

fn assert_imported_capture(
    manifest: &std::path::Path,
    output: &std::path::Path,
    run: &serde_json::Value,
    mode: &str,
) {
    use sha2::{Digest, Sha256};
    let copied = output.join("inputs/captured-trajectories.json");
    let source_bytes = std::fs::read(manifest).unwrap();
    let copied_bytes = std::fs::read(copied).unwrap();
    assert_eq!(copied_bytes, source_bytes);
    let source_digest = hex::encode(Sha256::digest(&source_bytes));
    assert_eq!(run["inputs"]["manifest_sha256"], source_digest);
    assert_eq!(run["inputs"]["cohorts"]["1"]["assistant_turns"], 4);
    let copied_document: serde_json::Value = serde_json::from_slice(&copied_bytes).unwrap();
    assert_eq!(copied_document["metadata"]["revision"], "r1");
    if mode != "dataset" {
        assert_eq!(run["inputs"]["dataset"]["revision"], "r1");
        assert_eq!(run["inputs"]["metadata"], copied_document["metadata"]);
        assert_eq!(run["inputs"]["source_manifest_sha256"], source_digest);
    }
}
