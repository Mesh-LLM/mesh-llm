//! Linux-owned inert tool fixture; no source checkout, compiler or public network.
use crate::process::{
    self, Cancellation, Completion, Limits, Outcome, ProcessSpec, Readiness, Value,
};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
pub(super) fn fixture(mode: &str) -> (tempfile::TempDir, PathBuf, PathBuf) {
    assert!(["ok", "wrong-head", "build-fail", "hold"].contains(&mode));
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().canonicalize().unwrap();
    let tools = root.join("tools");
    std::fs::create_dir(&tools).unwrap();
    let script = format!(
        r#"#!/bin/sh
set -eu
mode='{mode}'
if [ "$1" = '--version' ]; then printf 'inert pinned tool 1\n'; exit 0; fi
case "${{0##*/}}" in
 git)
  case "$1" in
   clone) test "$2" = '--no-checkout'; test "$3" = '--filter=blob:none'; test "$4" = '--'; test "$5" = 'https://github.com/Mesh-LLM/mesh-llm.git'; /bin/mkdir -p "$6/third_party/llama.cpp"; printf '%s\n' 'cccccccccccccccccccccccccccccccccccccccc' > "$6/third_party/llama.cpp/upstream.txt" ;;
   checkout) test "$2" = '--detach'; test "$3" = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' ;;
   rev-parse) if [ "$2" = HEAD ]; then if [ "${{PWD##*/}}" = llama.cpp ]; then printf '%s\n' 'dddddddddddddddddddddddddddddddddddddddd'; elif [ "$mode" = wrong-head ]; then printf '%s\n' 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'; else printf '%s\n' 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'; fi; else test "$2" = 'HEAD^{{tree}}'; printf '%s\n' 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'; fi ;;
   merge-base) test "$2" = --is-ancestor; test "$3" = cccccccccccccccccccccccccccccccccccccccc; test "$4" = HEAD ;;
   diff-index) test "$2" = '--quiet'; test "$3" = HEAD; test "$4" = '--' ;;
   *) exit 64 ;;
  esac ;;
 just)
  case "$1" in
   llama-prepare) test "$#" = 1; printf 'prepared\n' > prepared; /bin/mkdir -p .deps/llama.cpp; printf '%s\n' cccccccccccccccccccccccccccccccccccccccc > .deps/llama.cpp/.mesh-llm-upstream-sha; printf '%s\n' dddddddddddddddddddddddddddddddddddddddd > .deps/llama.cpp/.mesh-llm-patched-sha ;;
   skippy-quantize-standalone-release-build) test "$2" = cpu; test "$#" = 2; test -f prepared; printf 'started\n' > build-started; if [ "$mode" = build-fail ]; then exit 7; fi; if [ "$mode" = hold ]; then trap 'exit 1' TERM; while :; do /bin/sleep 0.05; done; fi; /bin/mkdir -p target/release; printf 'inert built artifact\n' > target/release/skippy-quantize ;;
   *) exit 64 ;;
  esac ;;
 *) exit 64 ;;
esac
"#
    );
    use std::os::unix::fs::PermissionsExt as _;
    let names = [
        "git", "just", "cargo", "rustc", "cmake", "c++", "ld.lld", "curl",
    ];
    let declared=names.iter().map(|name|{let path=tools.join(name);std::fs::write(&path,&script).unwrap();std::fs::set_permissions(&path,std::fs::Permissions::from_mode(0o700)).unwrap();serde_json::json!({"name":name,"path":path,"sha256":hex::encode(Sha256::digest(script.as_bytes()))})}).collect::<Vec<_>>();
    let request = serde_json::json!({"schema_version":1,"mesh_commit":"a".repeat(40),"git_tree":"b".repeat(40),"llama_commit":"c".repeat(40),"upstream_file_sha256":hex::encode(Sha256::digest(format!("{}\n","c".repeat(40)).as_bytes())),"image":format!("ghcr.io/declared/prepared@sha256:{}","e".repeat(64)),"native_profile":"standalone-static-skippy-quantize-cpu","tools":declared,"path_directories":[tools],"timeout_seconds":30,"cpu_plan_receipt_sha256":"1".repeat(64),"declared_estimate_usd":1.0,"max_cost_usd":2.0});
    let input = root.join("input.json");
    std::fs::write(&input, serde_json::to_vec(&request).unwrap()).unwrap();
    (temporary, input, root.join("evidence"))
}
fn invoke(input: &Path, output: &Path, cancel: &Cancellation) -> process::ProcessReport {
    let spec = ProcessSpec {
        executable: Path::new(env!("CARGO_BIN_EXE_xtask")).into(),
        cwd: input.parent().unwrap().into(),
        environment: Default::default(),
        arguments: [
            "automation",
            "hf-certify",
            "bootstrap",
            "--input",
            input.to_str().unwrap(),
            "--output-directory",
            output.to_str().unwrap(),
        ]
        .map(|s| Value::Public(s.into()))
        .into(),
    };
    let report = process::supervise(
        &spec,
        &Limits {
            execution: Duration::from_secs(40),
            graceful_shutdown: Duration::from_secs(4),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 16384,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        cancel,
        Default::default(),
    )
    .unwrap();
    assert!(report.failure.is_none() && report.cleanup.failure.is_none());
    assert!(
        report.cleanup.complete && !report.cleanup.forced && !report.cleanup.graceful_signal_failed
    );
    for stream in [&report.stdout, &report.stderr] {
        assert!(stream.line_capture_complete && !stream.truncated);
        assert_eq!(stream.oversized_lines, 0);
    }
    report
}
#[test]
fn actual_bootstrap_cli_observes_ordered_checkout_prepare_build_and_custody() {
    let (root, input, output) = fixture("ok");
    let report = invoke(&input, &output, &Cancellation::default());
    assert_eq!(report.outcome, Outcome::Exited);
    assert!(report.status.unwrap().success());
    let receipt: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("bootstrap.json")).unwrap()).unwrap();
    assert_eq!(receipt["status"], "BOOTSTRAP_COMPLETED");
    let phases = receipt["phases"].as_array().unwrap();
    assert!(
        phases
            .iter()
            .position(|p| p["phase"] == "checkout")
            .unwrap()
            < phases.iter().position(|p| p["phase"] == "prepare").unwrap()
    );
    assert!(
        phases.iter().position(|p| p["phase"] == "prepare").unwrap()
            < phases.iter().position(|p| p["phase"] == "build").unwrap()
    );
    assert_eq!(
        phases.last().unwrap()["sha256"],
        hex::encode(Sha256::digest(b"inert built artifact\n"))
    );
    for field in [
        "image_observed",
        "cpu_plan_and_rate_observed",
        "acquisition_completed",
        "publication_completed",
    ] {
        assert_eq!(receipt[field], false);
    }
    root.close().unwrap();
}
#[test]
fn actual_bootstrap_cli_refuses_wrong_source_and_retains_failed_build_phase() {
    for mode in ["wrong-head", "build-fail"] {
        let (root, input, output) = fixture(mode);
        let report = invoke(&input, &output, &Cancellation::default());
        assert_eq!(report.outcome, Outcome::Exited);
        assert_eq!(report.status.unwrap().code(), Some(1));
        let receipt: serde_json::Value =
            serde_json::from_slice(&std::fs::read(output.join("bootstrap.json")).unwrap()).unwrap();
        assert_eq!(receipt["status"], "FAILED");
        assert!(
            !output
                .join("mesh-source/target/release/skippy-quantize")
                .exists()
        );
        let phases = receipt["phases"].as_array().unwrap();
        if mode == "wrong-head" {
            assert!(!phases.iter().any(|p| p["phase"] == "prepare"));
        } else {
            assert_eq!(phases.last().unwrap()["phase"], "build");
            assert_eq!(phases.last().unwrap()["status"], 7);
        }
        root.close().unwrap();
    }
}
#[test]
fn actual_bootstrap_cli_cancel_after_observed_build_start_retains_partial_phases() {
    let (root, input, output) = fixture("hold");
    let cancel = Cancellation::default();
    let child_cancel = cancel.clone();
    let worker = std::thread::spawn(move || invoke(&input, &output, &child_cancel));
    let marker = root.path().join("evidence/mesh-source/build-started");
    let deadline = Instant::now() + Duration::from_secs(15);
    while !marker.exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(5));
    }
    let observed = marker.exists();
    cancel.cancel();
    let report = worker.join().unwrap();
    assert!(observed);
    assert_eq!(report.outcome, Outcome::Cancelled);
    let receipt: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.path().join("evidence/bootstrap.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(receipt["status"], "FAILED");
    assert!(
        !root
            .path()
            .join("evidence/mesh-source/target/release/skippy-quantize")
            .exists()
    );
    assert!(
        receipt["phases"]
            .as_array()
            .unwrap()
            .iter()
            .any(|p| p["phase"] == "prepare")
    );
    root.close().unwrap();
}
