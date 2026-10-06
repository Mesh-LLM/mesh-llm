use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness,
    Value as Argument,
};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use std::{collections::BTreeMap, path::Path, time::Duration};
fn digest(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend((value.len() as u64).to_le_bytes());
    bytes.extend(value.as_bytes());
}
fn gguf() -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.extend(3u64.to_le_bytes());
    string(&mut bytes, "general.architecture");
    bytes.extend(8u32.to_le_bytes());
    string(&mut bytes, "llama");
    for (key, value) in [
        ("llama.block_count", 12u32),
        ("llama.embedding_length", 64u32),
    ] {
        string(&mut bytes, key);
        bytes.extend(4u32.to_le_bytes());
        bytes.extend(value.to_le_bytes());
    }
    bytes
}
fn input(root: &Path, bytes: &[u8]) -> Value {
    json!({"cache_root":root.join("cache"),"source_root":root,"candidates":[{"family":"llama","llama_model":"llama","status":"candidate_multimodal","priority":"p0","model_pin":{"repo":"fixture/model","revision":"a".repeat(40),"file":"model-00001-of-00002.gguf","blob_sha256":digest(bytes),"size_bytes":bytes.len()},"layer_end":null,"split_layer":null,"splits":null,"recurrent_all":true,"recurrent_ranges":[]}],"statuses":[],"families":[],"llama_models":[],"priorities":[],"limit":null,"policy":{"ctx_size":128,"n_gpu_layers":999,"prompt":"Hello from fixture","run_id":"local-fixture","skip_build":true,"skip_state":false,"state_payload_kind":null,"prefix_token_count":256,"cache_hit_repeats":2,"borrow_resident_hits":true,"cache_decoded_result_hits":true,"startup_timeout_secs":15}})
}
fn run(root: &Path, input: &Value) -> (bool, String, String) {
    run_verb(root, input, "parity-local-plan")
}
fn run_verb(root: &Path, input: &Value, verb: &str) -> (bool, String, String) {
    run_verb_with_environment(root, input, verb, BTreeMap::new())
}
fn run_verb_with_environment(
    root: &Path,
    input: &Value,
    verb: &str,
    environment: BTreeMap<std::ffi::OsString, Argument>,
) -> (bool, String, String) {
    let path = root.join("input.json");
    std::fs::write(&path, serde_json::to_vec(input).unwrap()).unwrap();
    let raw = process::supervise_raw(
        &ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            arguments: [
                "automation",
                "replay-matrix",
                verb,
                "--input",
                path.to_str().unwrap(),
            ]
            .into_iter()
            .map(|s| Argument::Public(s.into()))
            .collect(),
            cwd: root.into(),
            environment,
        },
        &Limits {
            execution: Duration::from_secs(
                if verb == "parity-local-run" || verb == "parity-local" {
                    10
                } else {
                    5
                },
            ),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(65536),
            stderr: std::num::NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    assert_eq!(raw.process.outcome, process::Outcome::Exited);
    assert!(
        raw.process.cleanup.complete
            && raw.process.cleanup.failure.is_none()
            && !raw.process.cleanup.graceful_signal_failed
            && !raw.process.cleanup.forced
            && raw.process.failure.is_none(),
        "{:?}",
        raw.process
    );
    for (bytes, stream) in [
        (raw.stdout.as_ref().unwrap(), &raw.process.stdout),
        (raw.stderr.as_ref().unwrap(), &raw.process.stderr),
    ] {
        assert_eq!(bytes.as_bytes().len() as u64, stream.bytes_seen);
        // Diagnostic redaction can suppress token-related JSON while raw capture stays complete.
        assert!(!stream.truncated && stream.line_capture_complete);
    }
    (
        raw.process.success(),
        String::from_utf8(raw.stdout.unwrap().as_bytes().to_vec()).unwrap(),
        String::from_utf8(raw.stderr.unwrap().as_bytes().to_vec()).unwrap(),
    )
}
fn setup(root: &Path, bytes: &[u8]) -> std::path::PathBuf {
    let snapshot = root
        .join("cache/models--fixture--model/snapshots")
        .join("a".repeat(40));
    std::fs::create_dir_all(&snapshot).unwrap();
    let file = snapshot.join("model-00001-of-00002.gguf");
    std::fs::write(&file, bytes).unwrap();
    file
}
#[test]
fn parity_local_actual_cli_derives_shape_and_recurrent_cache_context_with_broad_candidate_selection()
 {
    let root = tempfile::tempdir().unwrap();
    let bytes = gguf();
    let file = setup(root.path(), &bytes);
    let mut request = input(root.path(), &bytes);
    // Normal HF snapshot link points into the same cache's immutable blobs.
    let blob = root.path().join("cache/blobs/model");
    std::fs::create_dir_all(blob.parent().unwrap()).unwrap();
    std::fs::rename(&file, &blob).unwrap();
    std::os::unix::fs::symlink(&blob, &file).unwrap();
    let (ok, stdout, stderr) = run(root.path(), &request);
    assert!(ok, "{stderr}");
    let result: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(result["rows"][0]["status"], "local_verified");
    assert_eq!(result["rows"][0]["layer_end"], 12);
    assert_eq!(result["rows"][0]["activation_width"], 64);
    assert_eq!(
        result["rows"][0]["local_path"],
        json!(blob.canonicalize().unwrap())
    );
    let argv: Vec<_> = result["invocations"][0]["arguments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    for pair in [
        ["--ctx-size", "544"],
        ["--state-payload-kind", "kv-recurrent"],
        ["--splits", "4,8"],
        ["--prefix-token-count", "256"],
        ["--cache-hit-repeats", "2"],
    ] {
        assert!(argv.windows(2).any(|p| p == pair), "{argv:?}");
    }
    for flag in [
        "--borrow-resident-hits",
        "--cache-decoded-result-hits",
        "--recurrent-all",
    ] {
        assert!(argv.contains(&flag));
    }
    request["families"] = json!(["another"]);
    let (ok, stdout, stderr) = run(root.path(), &request);
    assert!(ok, "{stderr}");
    assert_eq!(
        serde_json::from_str::<Value>(&stdout).unwrap()["invocations"],
        json!([])
    );
    root.close().unwrap();
}
#[test]
fn parity_local_actual_cli_keeps_corrupt_missing_and_unwritten_fifo_rows_without_invocations() {
    use std::os::unix::ffi::OsStrExt as _;
    let root = tempfile::tempdir().unwrap();
    let bytes = gguf();
    let file = setup(root.path(), &bytes);
    let request = input(root.path(), &bytes);
    std::fs::write(&file, b"changed").unwrap();
    let (ok, stdout, stderr) = run(root.path(), &request);
    assert!(ok, "{stderr}");
    let result: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(result["rows"][0]["status"], "inspect_error");
    assert!(
        result["rows"][0]["inspect_error"]
            .as_str()
            .unwrap()
            .contains("size differs")
    );
    assert_eq!(result["invocations"], json!([]));
    std::fs::remove_file(&file).unwrap();
    let (ok, stdout, stderr) = run(root.path(), &request);
    assert!(ok, "{stderr}");
    assert_eq!(
        serde_json::from_str::<Value>(&stdout).unwrap()["rows"][0]["status"],
        "missing"
    );
    let name = std::ffi::CString::new(file.as_os_str().as_bytes()).unwrap();
    // SAFETY: creates only this owned private fixture name; CString remains valid.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let (ok, stdout, stderr) = run(root.path(), &request);
    assert!(ok, "{stderr}");
    let result: Value = serde_json::from_str(&stdout).unwrap();
    assert!(
        result["rows"][0]["inspect_error"]
            .as_str()
            .unwrap()
            .contains("regular file")
    );
    assert_eq!(result["invocations"], json!([]));
    root.close().unwrap();
}
#[test]
fn parity_local_actual_cli_refuses_later_shard_traversal_and_invalid_boundary_before_planning() {
    let root = tempfile::tempdir().unwrap();
    let bytes = gguf();
    setup(root.path(), &bytes);
    let base = input(root.path(), &bytes);
    for name in [
        "../outside.gguf",
        "/outside.gguf",
        "model-00002-of-00002.gguf",
        "mmproj.gguf",
    ] {
        let mut request = base.clone();
        request["candidates"][0]["model_pin"]["file"] = name.into();
        let (ok, stdout, stderr) = run(root.path(), &request);
        assert!(!ok);
        assert!(stdout.is_empty());
        assert!(stderr.contains("safe immutable serving-file pin"));
    }
    let mut request = base;
    request["candidates"][0]["splits"] = json!([8, 4]);
    let (ok, stdout, stderr) = run(root.path(), &request);
    assert!(ok, "{stderr}");
    let result: Value = serde_json::from_str(&stdout).unwrap();
    assert!(
        result["rows"][0]["inspect_error"]
            .as_str()
            .unwrap()
            .contains("increasing interior")
    );
    assert_eq!(result["invocations"], json!([]));
    root.close().unwrap();
}
fn execution_input(root: &Path, plan: Value, prompt: &str) -> Value {
    let mut plan = plan;
    plan["policy"]["prompt"] = prompt.into();
    let harness = root.join("scripts/family-certify.sh");
    std::fs::create_dir_all(harness.parent().unwrap()).unwrap();
    let example = Path::new(env!("CARGO_BIN_EXE_xtask"))
        .parent()
        .unwrap()
        .join("examples/l7_daemon_fixture");
    std::fs::copy(example, &harness).unwrap();
    json!({"plan":plan,"evidence":root.join("run"),"harness_sha256":digest(&std::fs::read(harness).unwrap()),"max_seconds":12,"candidate_seconds":2,"stop_on_failure":false,"stage_build_dir":null,"path":"/usr/bin:/bin"})
}
fn run_request(root: &Path, input: &Value) -> (bool, String, String) {
    run_verb(root, input, "parity-local-run")
}
#[test]
fn parity_run_actual_cli_binds_harness_manifest_and_preserves_failed_partial_receipts() {
    for (prompt, stop) in [
        ("successful", false),
        ("sanitized-path", false),
        ("fail", false),
        ("wrong-identity", false),
        ("missing-manifest", false),
        ("fail", true),
    ] {
        let root = tempfile::tempdir().unwrap();
        let bytes = gguf();
        setup(root.path(), &bytes);
        let mut plan = input(root.path(), &bytes);
        let mut second = plan["candidates"][0].clone();
        second["family"] = "another".into();
        second["status"] = "candidate_stateful".into();
        plan["candidates"].as_array_mut().unwrap().push(second);
        let mut request = execution_input(root.path(), plan, prompt);
        request["stop_on_failure"] = stop.into();
        let (ok, stdout, stderr) = run_request(root.path(), &request);
        assert_eq!(
            ok,
            matches!(prompt, "successful" | "sanitized-path"),
            "{stderr}"
        );
        let receipt: Value = serde_json::from_slice(
            &std::fs::read(root.path().join("run/run-receipt.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(receipt["planned_candidates"], 2);
        assert_eq!(receipt["completed_receipts"], if stop { 1 } else { 2 });
        if !stop {
            assert_eq!(
                receipt["receipts"][1]["invocation"]["arguments"][1],
                "another"
            );
        }
        assert_eq!(receipt["receipts"][0]["cleanup_complete"], true);
        assert_eq!(receipt["receipts"][0]["cleanup_forced"], false);
        if ok {
            assert_eq!(
                receipt["receipts"][0]["stdout_suppressed_lines"]
                    .as_u64()
                    .unwrap()
                    > 0,
                prompt == "sanitized-path"
            );
            assert!(
                receipt["receipts"][0]["stdout_bytes_seen"]
                    .as_u64()
                    .unwrap()
                    > 0
            );
            assert_eq!(
                serde_json::from_str::<Value>(&stdout).unwrap()["status"],
                "completed"
            );
            assert_eq!(
                receipt["receipts"][0]["status"],
                "process_completed_manifest_bound"
            );
        } else {
            assert_eq!(receipt["status"], "failed");
            assert_eq!(receipt["receipts"][0]["status"], "failed");
        }
        root.close().unwrap();
    }
}
#[test]
fn parity_run_actual_cli_rejects_harness_drift_and_retains_deadline_receipt() {
    let root = tempfile::tempdir().unwrap();
    let bytes = gguf();
    setup(root.path(), &bytes);
    let mut request = execution_input(root.path(), input(root.path(), &bytes), "hold");
    request["harness_sha256"] = "0".repeat(64).into();
    let (ok, _, stderr) = run_request(root.path(), &request);
    assert!(!ok);
    assert!(stderr.contains("source digest"));
    assert!(!root.path().join("run").exists());
    request["harness_sha256"] =
        digest(&std::fs::read(root.path().join("scripts/family-certify.sh")).unwrap()).into();
    let (ok, _, stderr) = run_request(root.path(), &request);
    assert!(!ok, "{stderr}");
    let receipt: Value =
        serde_json::from_slice(&std::fs::read(root.path().join("run/run-receipt.json")).unwrap())
            .unwrap();
    assert_eq!(receipt["receipts"][0]["outcome"], "Deadline");
    assert_eq!(receipt["receipts"][0]["cleanup_complete"], true);
    root.close().unwrap();
}
#[test]
fn parity_run_actual_cli_interrupts_held_harness_and_retains_cancelled_receipt() {
    let root = tempfile::tempdir().unwrap();
    let bytes = gguf();
    setup(root.path(), &bytes);
    let mut request = execution_input(root.path(), input(root.path(), &bytes), "hold");
    request["candidate_seconds"] = 30.into();
    request["max_seconds"] = 40.into();
    let fixture_root = root.path().to_path_buf();
    let pidpath = fixture_root.join("run/candidate-0000/harness/parent.pid");
    let signal = std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + Duration::from_secs(6);
        loop {
            if let Ok(pid) = std::fs::read_to_string(&pidpath) {
                let pid = pid.parse::<i32>().unwrap();
                assert!(pid > 0); /* SAFETY: owned xtask parent PID published by its held fixture child. */
                assert_eq!(unsafe { libc::kill(pid, libc::SIGINT) }, 0);
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "held fixture did not publish parent"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    });
    let (ok, _, stderr) = run_request(root.path(), &request);
    signal.join().unwrap();
    assert!(!ok, "{stderr}");
    let receipt: Value =
        serde_json::from_slice(&std::fs::read(root.path().join("run/run-receipt.json")).unwrap())
            .unwrap();
    assert_eq!(receipt["status"], "cancelled");
    assert_eq!(receipt["receipts"][0]["outcome"], "Cancelled");
    assert_eq!(receipt["receipts"][0]["cleanup_complete"], true);
    assert_eq!(receipt["receipts"][0]["cleanup_forced"], false);
    root.close().unwrap();
}
fn local_git(root: &Path, args: &[&str]) -> String {
    let executable = std::env::var_os("MIGRATION_TEST_GIT").expect("normal requires actual Git");
    let mut argv = vec![
        "-c",
        "user.name=Parity Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "-c",
        "core.hooksPath=/dev/null",
    ];
    argv.extend_from_slice(args);
    let report = process::supervise_raw(
        &ProcessSpec {
            executable: executable.into(),
            arguments: argv.iter().map(|s| Argument::Public((*s).into())).collect(),
            cwd: root.into(),
            environment: BTreeMap::from([
                ("PATH".into(), Argument::Public("/usr/bin:/bin".into())),
                ("GIT_MASTER".into(), Argument::Public("1".into())),
                ("GIT_OPTIONAL_LOCKS".into(), Argument::Public("0".into())),
                ("GIT_CONFIG_NOSYSTEM".into(), Argument::Public("1".into())),
                (
                    "GIT_CONFIG_GLOBAL".into(),
                    Argument::Public("/dev/null".into()),
                ),
            ]),
        },
        &Limits {
            execution: Duration::from_secs(5),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(65536),
            stderr: std::num::NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    assert!(report.process.success(), "{:?}", report.process);
    assert!(
        report.process.cleanup.failure.is_none()
            && !report.process.cleanup.graceful_signal_failed
            && !report.process.cleanup.forced
    );
    for (bytes, stream) in [
        (report.stdout.as_ref().unwrap(), &report.process.stdout),
        (report.stderr.as_ref().unwrap(), &report.process.stderr),
    ] {
        assert_eq!(bytes.as_bytes().len() as u64, stream.bytes_seen);
        assert!(stream.line_capture_complete && !stream.truncated && stream.suppressed_lines == 0);
    }
    String::from_utf8(report.stdout.unwrap().as_bytes().to_vec())
        .unwrap()
        .trim()
        .into()
}
fn source_authority(root: &Path, bytes: &[u8]) -> Value {
    let patches = root.join("third_party/llama.cpp/patches");
    std::fs::create_dir_all(&patches).unwrap();
    std::fs::write(patches.join("0001-fixture.patch"), b"inert fixture").unwrap();
    std::fs::write(
        root.join("third_party/llama.cpp/upstream.txt"),
        "a".repeat(40),
    )
    .unwrap();
    let patch_digest =
        digest(format!("0001-fixture.patch\n{}\n", digest(b"inert fixture")).as_bytes());
    for directory in ["crates/skippy-ffi/src", "docs/skippy", "ci/llama-canary"] {
        std::fs::create_dir_all(root.join(directory)).unwrap();
    }
    std::fs::write(root.join("crates/skippy-ffi/src/lib.rs"),"pub const ABI_VERSION_MAJOR: u32 = 1;\npub const ABI_VERSION_MINOR: u32 = 2;\npub const ABI_VERSION_PATCH: u32 = 3;\n").unwrap();
    let pin = json!({"repo":"fixture/pinned","revision":"a".repeat(40),"file":"model.gguf","blob_sha256":digest(bytes),"size_bytes":bytes.len(),"selector":"fixture"});
    let candidates = json!({"defaults":{"ctx_size":77,"n_gpu_layers":9,"prompt":"successful"},"candidates":[{"llama_model":"fixture","family":"pinned","status":"certified","repo":"wrong/legacy","model_pin":pin},{"llama_model":"fixture","family":"discovered","status":"candidate_multimodal","repo":"fixture/discovered","include":"*.gguf","recurrent":"all"},{"llama_model":"fixture","family":"corrupt","status":"candidate","repo":"fixture/corrupt"},{"llama_model":"fixture","family":"missing","status":"candidate_stateful","repo":"fixture/missing"}],"support_priority":{"p0":{"families":["pinned","discovered","corrupt","missing"]}}});
    std::fs::write(
        root.join("docs/skippy/llama-parity-candidates.json"),
        serde_json::to_vec(&candidates).unwrap(),
    )
    .unwrap();
    let family = json!({"models":[{"family":"pinned","artifact":{"repo":"fixture/pinned","revision":"a".repeat(40),"selector":"fixture","file_integrity":{"model.gguf":{"blob_id":digest(bytes),"size_bytes":bytes.len()}}}}]});
    std::fs::write(
        root.join("ci/llama-canary/family-certified.json"),
        serde_json::to_vec(&family).unwrap(),
    )
    .unwrap();
    std::fs::write(
        root.join(".gitignore"),
        ".deps/\ncache/\nscripts/family-certify.sh\ninput.json\nrun/\n",
    )
    .unwrap();
    local_git(root, &["init", "--quiet"]);
    local_git(root, &["add", "-A"]);
    local_git(
        root,
        &["commit", "--quiet", "-m", "owned controller fixture"],
    );
    let revision = local_git(root, &["rev-parse", "HEAD"]);
    let native = root.join(".deps/llama.cpp");
    std::fs::create_dir_all(native.join("src/models")).unwrap();
    std::fs::create_dir_all(native.join("src/skippy")).unwrap();
    std::fs::write(
        native.join("src/models/fixture.cpp"),
        "begin_block(layer); end_block(layer);\n",
    )
    .unwrap();
    std::fs::write(native.join("src/skippy/model_loading.cpp"),include_bytes!("../../src/automation/canary_receipts/package_closure/runtime_slice/prepared_admission.cpp")).unwrap();
    local_git(&native, &["init", "--quiet"]);
    local_git(&native, &["add", "-A"]);
    local_git(
        &native,
        &["commit", "--quiet", "-m", "owned native fixture"],
    );
    let head = local_git(&native, &["rev-parse", "HEAD"]);
    std::fs::write(native.join(".git/info/exclude"), ".mesh-llm-*\n").unwrap();
    for (name, value) in [
        (".mesh-llm-upstream-sha", "a".repeat(40)),
        (".mesh-llm-patched-sha", head),
        (".mesh-llm-patch-digest", patch_digest),
        (".mesh-llm-prepare-schema", "5".into()),
    ] {
        std::fs::write(native.join(name), format!("{value}\n")).unwrap();
    }
    json!({"controller":{"root":root,"revision":revision,"executable_sha256":digest(&std::fs::read(env!("CARGO_BIN_EXE_xtask")).unwrap())},"root":root,"base":revision})
}
#[test]
fn parity_frontdoor_actual_cli_composes_full_source_roster_offline_defaults_and_retained_run() {
    let root = tempfile::tempdir().unwrap();
    let bytes = gguf();
    let authority = source_authority(root.path(), &bytes);
    for (repo, name, data) in [
        ("pinned", "model.gguf", bytes.clone()),
        ("discovered", "model-00002-of-00002.gguf", b"later".to_vec()),
        ("discovered", "model-00001-of-00002.gguf", bytes.clone()),
        ("corrupt", "broken.gguf", b"bad GGUF".to_vec()),
    ] {
        let dir = root.path().join(format!(
            "cache/models--fixture--{repo}/snapshots/{}",
            "a".repeat(40)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(name), data).unwrap();
    }
    let mut request = json!({"authority":authority,"cache_root":root.path().join("cache"),"mode":"inventory","statuses":[],"families":[],"llama_models":[],"priorities":[],"limit":null,"missing_only":false,"local_only":false,"policy":{"skip_build":true,"prefix_token_count":8,"cache_hit_repeats":2,"borrow_resident_hits":true},"run":null,"admission_seconds":20});
    let (ok, stdout, stderr) = run_verb(root.path(), &request, "parity-local");
    assert!(ok, "{stderr}");
    let result: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(result["admission"]["model_sources"], 1);
    assert_eq!(result["inventory"].as_array().unwrap().len(), 4);
    assert_eq!(result["inventory"][0]["status"], "local_verified");
    assert_eq!(result["inventory"][0]["classification"], "certified");
    assert!(
        result["inventory"][0]["local_path"]
            .as_str()
            .unwrap()
            .contains("models--fixture--pinned")
    );
    assert_eq!(result["inventory"][1]["status"], "local_verified");
    assert_eq!(
        result["inventory"][1]["identity_scope"],
        "observed_offline_cache_not_provider_pin"
    );
    assert_eq!(result["inventory"][2]["status"], "inspect_error");
    assert_eq!(result["inventory"][3]["status"], "missing");
    assert_eq!(result["prepared_plan"]["policy"]["ctx_size"], 77);
    assert_eq!(result["prepared_plan"]["policy"]["n_gpu_layers"], 9);
    request["missing_only"] = true.into();
    let (ok, stdout, stderr) = run_verb(root.path(), &request, "parity-local");
    assert!(ok, "{stderr}");
    assert_eq!(
        serde_json::from_str::<Value>(&stdout).unwrap()["inventory"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    request["missing_only"] = false.into();
    let setup = execution_input(root.path(), result["prepared_plan"].clone(), "successful");
    let mut settings = setup.as_object().unwrap().clone();
    settings.remove("plan");
    request["policy"]["ctx_size"] = 20.into();
    request["policy"]["n_gpu_layers"] = (-1).into();
    request["mode"] = "run".into();
    request["run"] = json!(settings);
    let (ok, _, stderr) = run_verb(root.path(), &request, "parity-local");
    assert!(ok, "{stderr}");
    let receipt: Value =
        serde_json::from_slice(&std::fs::read(root.path().join("run/run-receipt.json")).unwrap())
            .unwrap();
    assert_eq!(receipt["planned_candidates"], 2);
    assert_eq!(receipt["completed_receipts"], 2);
    for (index, kind) in [(0, "resident-kv"), (1, "kv-recurrent")] {
        let args: Value = serde_json::from_slice(
            &std::fs::read(
                root.path()
                    .join(format!("run/candidate-{index:04}/harness/arguments.json")),
            )
            .unwrap(),
        )
        .unwrap();
        let args: Vec<_> = args
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap())
            .collect();
        assert!(args.windows(2).any(|pair| pair == ["--ctx-size", "48"]));
        assert!(args.windows(2).any(|pair| pair == ["--n-gpu-layers", "-1"]));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--state-payload-kind", kind])
        );
    }
    assert_final_source_refusal(root.path(), &mut request);
    root.close().unwrap();
}
#[test]
fn parity_frontdoor_actual_cli_refuses_wrong_controller_and_unpinned_outside_cache() {
    let root = tempfile::tempdir().unwrap();
    let bytes = gguf();
    let mut authority = source_authority(root.path(), &bytes);
    std::fs::create_dir(root.path().join("cache")).unwrap();
    authority["controller"]["executable_sha256"] = "0".repeat(64).into();
    let mut request = json!({"authority":authority,"cache_root":root.path().join("cache"),"mode":"inventory","statuses":[],"families":[],"llama_models":[],"priorities":[],"limit":null,"missing_only":false,"local_only":false,"policy":{},"run":null,"admission_seconds":20});
    let (ok, stdout, stderr) = run_verb(root.path(), &request, "parity-local");
    assert!(!ok);
    assert!(stdout.is_empty());
    assert!(stderr.contains("frozen executable"));
    request["authority"]["controller"]["executable_sha256"] =
        digest(&std::fs::read(env!("CARGO_BIN_EXE_xtask")).unwrap()).into();
    let outside = tempfile::tempdir().unwrap();
    let snapshots = root.path().join("cache/models--fixture--discovered");
    std::fs::create_dir(&snapshots).unwrap();
    std::os::unix::fs::symlink(outside.path(), snapshots.join("snapshots")).unwrap();
    let (ok, stdout, stderr) = run_verb(root.path(), &request, "parity-local");
    assert!(ok, "{stderr}");
    let result: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(result["inventory"][1]["status"], "inspect_error");
    assert!(
        result["inventory"][1]["inspect_error"]
            .as_str()
            .unwrap()
            .contains("snapshot directory")
    );
    assert!(!root.path().join("run").exists());
    root.close().unwrap();
    outside.close().unwrap();
}
#[test]
fn parity_run_actual_cli_preserves_reviewed_device_profile_and_typed_build_root() {
    let root = tempfile::tempdir().unwrap();
    let bytes = gguf();
    setup(root.path(), &bytes);
    let mut request = execution_input(root.path(), input(root.path(), &bytes), "successful");
    let build = root.path().join("typed-build");
    std::fs::create_dir(&build).unwrap();
    request["stage_build_dir"] = json!(build);
    let settings = [
        ("LLAMA_STAGE_BACKEND", "cpu"),
        ("SKIPPY_LLAMA_BACKEND", "cuda"),
        ("LLAMA_STAGE_LINK_MODE", "static"),
        ("SKIPPY_LLAMA_LINK_MODE", "dynamic"),
        ("LLAMA_STAGE_CUDA_ARCHITECTURES", "80;90"),
        ("SKIPPY_CUDA_ARCHITECTURES", "86"),
        ("LLAMA_STAGE_AMDGPU_TARGETS", "gfx1100"),
        ("SKIPPY_AMDGPU_TARGETS", "gfx1030"),
        ("LLAMA_STAGE_GGML_NATIVE", "OFF"),
        ("SKIPPY_GGML_NATIVE", "ON"),
        ("GGML_CUDA_NO_VMM", "1"),
        ("CUDA_VISIBLE_DEVICES", "2,0"),
        ("HIP_VISIBLE_DEVICES", "1"),
        ("ROCR_VISIBLE_DEVICES", ""),
    ];
    let mut env = BTreeMap::new();
    for (name, value) in settings {
        env.insert(name.into(), Argument::Public(value.into()));
    }
    for (name, value) in [
        ("HF_TOKEN", "credential-must-not-cross"),
        ("LLAMA_STAGE_BUILD_DIR", "/unowned/build"),
        ("SKIPPY_LLAMA_BUILD_DIR", "/unowned/alias"),
    ] {
        env.insert(name.into(), Argument::Public(value.into()));
    }
    let (ok, stdout, stderr) =
        run_verb_with_environment(root.path(), &request, "parity-local-run", env);
    assert!(ok, "{stderr}");
    let output: Value = serde_json::from_str(&stdout).unwrap();
    let captured: Value = serde_json::from_slice(
        &std::fs::read(
            root.path()
                .join("run/candidate-0000/harness/environment.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for (name, value) in settings {
        assert_eq!(captured[name], value);
        assert_eq!(
            output["effective_profile"]["inherited_settings"][name],
            value
        );
    }
    for name in [
        "HF_TOKEN",
        "SKIPPY_LLAMA_BUILD_DIR",
        "CUDA_PATH",
        "HIP_PATH",
        "ROCM_PATH",
        "VULKAN_SDK",
    ] {
        assert!(captured.get(name).is_none(), "{name}");
        assert!(
            output["effective_profile"]["inherited_settings"]
                .get(name)
                .is_none()
        );
    }
    assert_eq!(captured["LLAMA_STAGE_BUILD_DIR"], json!(build));
    assert_eq!(captured["GIT_MASTER"], "1");
    assert_eq!(captured["GIT_OPTIONAL_LOCKS"], "0");
    assert_eq!(captured["PATH"], request["path"]);
    let recorded: Value = serde_json::from_slice(
        &std::fs::read(root.path().join("run/effective-profile.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(recorded, output["effective_profile"]);
    assert_eq!(recorded["stage_build_dir"], json!(build));
    root.close().unwrap();
}
fn toolkit_request(root: &Path) -> Value {
    let bytes = gguf();
    setup(root, &bytes);
    let mut request = execution_input(root, input(root, &bytes), "successful");
    let mut directories = serde_json::Map::new();
    for name in [
        "CUDA_PATH",
        "HIP_PATH",
        "ROCM_PATH",
        "LLVMInstallDir",
        "VULKAN_SDK",
    ] {
        let path = root.join(format!("toolkit-{name}"));
        std::fs::create_dir(&path).unwrap();
        for suffix in ["lib/x64", "hip/lib", "llvm/lib", "Lib"] {
            std::fs::create_dir_all(path.join(suffix)).unwrap();
        }
        directories.insert(name.into(), json!(path));
    }
    request["toolkit_dirs"] = directories.into();
    request
}
#[test]
fn parity_run_actual_cli_delivers_typed_toolkit_directories_and_refuses_missing_conflicts() {
    use std::os::unix::ffi::OsStrExt as _;
    let root = tempfile::tempdir().unwrap();
    let mut request = toolkit_request(root.path());
    let requested = root.path().join("cuda-link");
    std::os::unix::fs::symlink(
        request["toolkit_dirs"]["CUDA_PATH"].as_str().unwrap(),
        &requested,
    )
    .unwrap();
    request["toolkit_dirs"]["CUDA_PATH"] = json!(requested);
    let mut env = BTreeMap::new();
    env.insert(
        "CUDA_PATH".into(),
        Argument::Public(requested.into_os_string()),
    );
    let (ok, stdout, stderr) =
        run_verb_with_environment(root.path(), &request, "parity-local-run", env);
    assert!(ok, "{stderr}");
    let output: Value = serde_json::from_str(&stdout).unwrap();
    let profile = &output["effective_profile"]["toolkit_observations"];
    let captured: Value = serde_json::from_slice(
        &std::fs::read(
            root.path()
                .join("run/candidate-0000/harness/environment.json"),
        )
        .unwrap(),
    )
    .unwrap();
    for name in [
        "CUDA_PATH",
        "HIP_PATH",
        "ROCM_PATH",
        "LLVMInstallDir",
        "VULKAN_SDK",
    ] {
        let path = Path::new(request["toolkit_dirs"][name].as_str().unwrap())
            .canonicalize()
            .unwrap();
        assert_eq!(captured[name], json!(path));
        assert_eq!(profile["directories"][name]["canonical"], json!(path));
        assert!(profile["directories"][name]["directory"]["inode"].is_u64());
    }
    root.close().unwrap();
    for mode in [
        "relative",
        "missing",
        "file",
        "fifo",
        "unknown",
        "ambient-conflict",
        "ambient-untyped",
    ] {
        let root = tempfile::tempdir().unwrap();
        let mut request = toolkit_request(root.path());
        let mut env = BTreeMap::new();
        match mode {
            "relative" => request["toolkit_dirs"]["CUDA_PATH"] = "relative/cuda".into(),
            "missing" => request["toolkit_dirs"]["CUDA_PATH"] = json!(root.path().join("absent")),
            "file" => {
                let path = root.path().join("ordinary-file");
                std::fs::write(&path, b"not a directory").unwrap();
                request["toolkit_dirs"]["CUDA_PATH"] = json!(path);
            }
            "fifo" => {
                let path = root.path().join("toolkit-fifo");
                let name = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
                // SAFETY: creates only the owned fixture name; no reader/writer is started.
                assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
                request["toolkit_dirs"]["CUDA_PATH"] = json!(path);
            }
            "unknown" => request["toolkit_dirs"]["UNOWNED_ROOT"] = "/unowned".into(),
            "ambient-conflict" => {
                env.insert(
                    "CUDA_PATH".into(),
                    Argument::Public(root.path().as_os_str().into()),
                );
            }
            "ambient-untyped" => {
                request["toolkit_dirs"]
                    .as_object_mut()
                    .unwrap()
                    .remove("CUDA_PATH");
                env.insert(
                    "CUDA_PATH".into(),
                    Argument::Public(root.path().as_os_str().into()),
                );
            }
            _ => unreachable!(),
        }
        let (ok, stdout, stderr) =
            run_verb_with_environment(root.path(), &request, "parity-local-run", env);
        assert!(!ok, "{mode}");
        assert!(stdout.is_empty());
        assert!(
            stderr.contains(if mode == "unknown" {
                "unknown field"
            } else {
                "toolkit_dirs"
            }),
            "{mode}: {stderr}"
        );
        assert!(
            !root.path().join("run").exists(),
            "refused before any harness/evidence creation"
        );
        root.close().unwrap();
    }
}
#[test]
fn parity_run_actual_cli_refuses_toolkit_directory_replacement_after_harness() {
    let root = tempfile::tempdir().unwrap();
    let mut request = toolkit_request(root.path());
    request["plan"]["policy"]["prompt"] = "replace-toolkit".into();
    let (ok, stdout, stderr) = run_request(root.path(), &request);
    assert!(!ok);
    assert!(stdout.is_empty());
    assert!(stderr.contains("partial receipts retained"));
    let receipt: Value =
        serde_json::from_slice(&std::fs::read(root.path().join("run/run-receipt.json")).unwrap())
            .unwrap();
    assert_eq!(receipt["status"], "failed");
    assert_eq!(receipt["receipts"][0]["status"], "refused");
    assert!(
        receipt["receipts"][0]["error"]
            .as_str()
            .unwrap()
            .contains("toolkit directory observation changed across harness execution")
    );
    assert!(
        root.path()
            .join("run/candidate-0000/harness/manifest.json")
            .is_file()
    );
    assert!(root.path().join("run/candidate-0000/stdout.log").is_file());
    root.close().unwrap();
}

fn assert_final_source_refusal(root: &Path, request: &mut Value) {
    request["policy"]["prompt"] = "source-guard-drift".into();
    request["run"]["evidence"] = json!(root.join("run-final-source-refusal"));
    let (ok, stdout, stderr) = run_verb(root, request, "parity-local");
    assert!(!ok && stdout.is_empty(), "{stderr}");
    assert!(stderr.contains("parity manifest changed during manual operation"));
    assert_eq!(
        std::fs::read(root.join("source-guard-mutated")).unwrap(),
        b"observed"
    );
    let evidence = root.join("run-final-source-refusal");
    let receipt: Value =
        serde_json::from_slice(&std::fs::read(evidence.join("run-receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt["status"], "refused");
    assert_eq!(receipt["completed_receipts"], 2);
    assert_eq!(receipt["receipts"].as_array().unwrap().len(), 2);
    assert!(
        receipt["terminal_errors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|error| error
                .as_str()
                .unwrap()
                .contains("parity manifest changed during manual operation"))
    );
    let observed: Value =
        serde_json::from_slice(&std::fs::read(evidence.join("observed-receipt.json")).unwrap())
            .unwrap();
    assert_eq!(observed["status"], "observed_pending_final_admission");
    assert_eq!(observed["observed_status"], "completed");
    assert_eq!(observed["receipts"], receipt["receipts"]);
}
