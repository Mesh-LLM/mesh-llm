//! Complete current battery dry-run with actual typed plan admission and no interpreter on PATH.
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::symlink,
    path::{Path, PathBuf},
    time::Duration,
};

fn native_tool(name: &str) -> PathBuf {
    let configured = (name == "jq")
        .then(|| std::env::var_os("MIGRATION_TEST_JQ"))
        .flatten();
    let path = configured.map(PathBuf::from).unwrap_or_else(|| {
        ["/usr/bin", "/bin", "/opt/homebrew/bin"]
            .iter()
            .map(|directory| Path::new(directory).join(name))
            .find(|path| path.is_file())
            .unwrap_or_else(|| panic!("native fixture tool missing: {name}"))
    });
    assert!(path.is_absolute() && path.is_file());
    path.canonicalize().unwrap()
}
fn finite_path(directory: &Path) {
    fs::create_dir(directory).unwrap();
    for name in [
        "awk", "bash", "basename", "cat", "cp", "cut", "date", "dirname", "mkdir", "paste", "sed",
        "tail", "tr", "wc", "jq",
    ] {
        symlink(native_tool(name), directory.join(name)).unwrap();
    }
    assert!(!directory.join("python3").exists());
    assert!(!directory.join("python").exists());
}
fn run(
    root: &Path,
    tools: &Path,
    artifact: &Path,
    id: &str,
    supplied: Option<&Path>,
) -> process::ProcessReport {
    let script = root.join("scripts/skippy-family-battery.sh");
    let mut arguments: Vec<std::ffi::OsString> = vec!["-c".into(), "if command -v python3 >/dev/null || command -v python >/dev/null; then exit 97; fi; exec /bin/bash \"$@\"".into(), "finite-no-python".into(), script.into(), "--skip-build".into(), "--dry-run".into()];
    if let Some(plan) = supplied {
        arguments.extend(["--plan".into(), plan.into()]);
    }
    let environment = BTreeMap::from([
        ("PATH".into(), Value::Public(tools.into())),
        (
            "MESH_LLM_AUTOMATION_BIN".into(),
            Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
        ),
        ("FAMILY_BATTERY_RUN_ID".into(), Value::Public(id.into())),
        (
            "FAMILY_BATTERY_ARTIFACT_ROOT".into(),
            Value::Public(artifact.into()),
        ),
        (
            "FAMILY_BATTERY_BIN_DIR".into(),
            Value::Public(root.join("absent-native-bin").into()),
        ),
    ]);
    let report = process::supervise(
        &ProcessSpec {
            executable: "/bin/bash".into(),
            cwd: root.into(),
            arguments: arguments.into_iter().map(Value::Public).collect(),
            environment,
        },
        &Limits {
            execution: Duration::from_secs(8),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert!(
        report.cleanup.complete && !report.stdout.truncated && !report.stderr.truncated,
        "{report:?}"
    );
    report
}
#[test]
fn actual_battery_causal_dry_run_needs_no_python_and_rejects_tampered_plan_before_lanes() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    fs::create_dir(root.join("scripts")).unwrap();
    for relative in [
        "scripts/skippy-family-battery.sh",
        "skippy/scripts/skippy-family-battery.sh",
    ] {
        let destination = root.join(relative);
        fs::create_dir_all(destination.parent().unwrap()).unwrap();
        fs::copy(super::support::repository().join(relative), destination).unwrap();
    }
    fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
    fs::create_dir_all(root.join("tools/xtask")).unwrap();
    fs::write(
        root.join("tools/xtask/Cargo.toml"),
        "[package]\nname='fixture'\n",
    )
    .unwrap();
    let manifest = root.join("ci/llama-canary/family-certified.json");
    fs::create_dir_all(manifest.parent().unwrap()).unwrap();
    let mut policy: serde_json::Value = serde_json::from_slice(
        &fs::read(
            super::support::repository()
                .join("tools/xtask/tests/fixtures/family_evidence/synthetic-manifest.json"),
        )
        .unwrap(),
    )
    .unwrap();
    policy["models"].as_array_mut().unwrap().truncate(1);
    policy["models"][0]["execution"]["trunk_layers"] = json!(8);
    policy["models"][0]["execution"]["activation_width"] = json!(64);
    let original = serde_json::to_vec(&policy).unwrap();
    fs::write(&manifest, &original).unwrap();
    let tools = root.join("finite-tools");
    finite_path(&tools);
    let artifacts = root.join("artifacts");
    let accepted = run(&root, &tools, &artifacts, "accepted", None);
    assert!(accepted.success(), "{accepted:?}");
    assert!(
        String::from_utf8_lossy(&accepted.stdout.bytes_retained)
            .contains("1 certifications planned; no lanes executed")
    );
    assert!(!root.join("absent-native-bin").exists());
    assert_eq!(fs::read(&manifest).unwrap(), original);
    let admitted = artifacts.join("accepted/policy-plan.json");
    let mut plan: serde_json::Value =
        serde_json::from_slice(&fs::read(&admitted).unwrap()).unwrap();
    assert_eq!(plan["selected_models"][0]["class"], "causal_generation");
    plan["selected_models"][0]["execution"]["activation_width"] = json!(65);
    let forged = root.join("forged-plan.json");
    fs::write(&forged, serde_json::to_vec(&plan).unwrap()).unwrap();
    let rejected = run(&root, &tools, &artifacts, "rejected", Some(&forged));
    assert!(!rejected.success(), "{rejected:?}");
    assert!(
        !String::from_utf8_lossy(&rejected.stdout.bytes_retained).contains("==> family-certify:")
    );
    assert!(
        fs::read_dir(artifacts.join("rejected/certifications"))
            .unwrap()
            .next()
            .is_none()
    );
    assert_eq!(fs::read(&manifest).unwrap(), original);
}

// Current canary dry-run ownership: the complete retained battery still owns all planning.
fn intent_policy() -> serde_json::Value {
    let mut policy: serde_json::Value = serde_json::from_slice(
        &fs::read(
            super::support::repository()
                .join("tools/xtask/tests/fixtures/family_evidence/synthetic-manifest.json"),
        )
        .unwrap(),
    )
    .unwrap();
    policy["models"].as_array_mut().unwrap().truncate(1);
    policy["models"][0]["execution"]["trunk_layers"] = json!(8);
    policy["models"][0]["execution"]["activation_width"] = json!(1024);
    policy
}
fn intent_root(policy: &serde_json::Value) -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    fs::create_dir(root.join("scripts")).unwrap();
    for relative in [
        "scripts/skippy-family-battery.sh",
        "skippy/scripts/skippy-family-battery.sh",
    ] {
        let destination = root.join(relative);
        fs::create_dir_all(destination.parent().unwrap()).unwrap();
        fs::copy(super::support::repository().join(relative), destination).unwrap();
    }
    fs::write(root.join("Cargo.toml"), "[workspace]\n").unwrap();
    fs::create_dir_all(root.join("tools/xtask")).unwrap();
    fs::write(
        root.join("tools/xtask/Cargo.toml"),
        "[package]\nname='fixture'\n",
    )
    .unwrap();
    let manifest = root.join("ci/llama-canary/family-certified.json");
    fs::create_dir_all(manifest.parent().unwrap()).unwrap();
    fs::write(manifest, serde_json::to_vec(policy).unwrap()).unwrap();
    finite_path(&root.join("finite-tools"));
    directory
}
fn intent_run(root: &Path, id: &str, extra: &[&str]) -> (bool, String) {
    let mut argv = vec![
        Value::Public(root.join("scripts/skippy-family-battery.sh").into()),
        Value::Public("--dry-run".into()),
    ];
    argv.extend(extra.iter().map(|v| Value::Public((*v).into())));
    intent_process(root, id, argv)
}
fn intent_process(root: &Path, id: &str, argv: Vec<Value>) -> (bool, String) {
    intent_process_with_environment(root, id, argv, BTreeMap::new())
}
fn intent_process_with_environment(
    root: &Path,
    id: &str,
    argv: Vec<Value>,
    extra: BTreeMap<std::ffi::OsString, Value>,
) -> (bool, String) {
    let mut spec = ProcessSpec {
        executable: "/bin/bash".into(),
        cwd: root.into(),
        arguments: argv,
        environment: BTreeMap::from([
            (
                "PATH".into(),
                Value::Public(root.join("finite-tools").into()),
            ),
            (
                "MESH_LLM_AUTOMATION_BIN".into(),
                Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
            ),
            ("FAMILY_BATTERY_RUN_ID".into(), Value::Public(id.into())),
            (
                "FAMILY_BATTERY_ARTIFACT_ROOT".into(),
                Value::Public(root.join("artifacts").into()),
            ),
            (
                "FAMILY_BATTERY_BIN_DIR".into(),
                Value::Public(root.join("absent-native-bin").into()),
            ),
        ]),
    };
    spec.environment.extend(extra);
    let result = process::supervise_raw(
        &spec,
        &Limits {
            execution: Duration::from_secs(20),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 262144,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        process::RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(262144),
            stderr: std::num::NonZeroUsize::new(262144),
        },
    )
    .unwrap();
    let p = &result.process;
    assert_eq!(p.outcome, process::Outcome::Exited, "{p:?}");
    assert!(
        p.failure.is_none()
            && p.cleanup.complete
            && !p.cleanup.forced
            && !p.cleanup.graceful_signal_failed
            && p.cleanup.failure.is_none(),
        "{p:?}"
    );
    assert!(
        p.stdout.line_capture_complete
            && p.stderr.line_capture_complete
            && !p.stdout.truncated
            && !p.stderr.truncated,
        "{p:?}"
    );
    let stdout = result.stdout.as_ref().unwrap().as_bytes();
    let stderr = result.stderr.as_ref().unwrap().as_bytes();
    assert_eq!(u64::try_from(stdout.len()).unwrap(), p.stdout.bytes_seen);
    assert_eq!(u64::try_from(stderr.len()).unwrap(), p.stderr.bytes_seen);
    // Sanitized diagnostic suppression is independent of complete bounded raw command evidence.
    (
        p.success(),
        format!(
            "{}{}",
            String::from_utf8_lossy(stdout),
            String::from_utf8_lossy(stderr)
        ),
    )
}
#[test]
fn complete_battery_dry_run_preserves_build_skip_roster_filter_and_workload_startup() {
    let mut policy = intent_policy();
    let mut workload = policy["models"][0].clone();
    workload["family"] = json!("embedding-family");
    workload["class"] = json!("embedding");
    workload["profile"] = json!("workload-oracle");
    workload["evidence"] = json!({"fixture":"fixture","comparison":"fixture"});
    workload["resources"]["startup_timeout_secs"] = json!(600);
    policy["models"].as_array_mut().unwrap().push(workload);
    let directory = intent_root(&policy);
    let root = directory.path().canonicalize().unwrap();
    let (success, built) = intent_run(&root, "build", &[]);
    assert!(success, "{built}");
    assert_eq!(
        built.matches("cargo build -p skippy-correctness").count(),
        1
    );
    assert_eq!(built.matches("/scripts/family-certify.sh ").count(), 1);
    assert_eq!(
        built
            .matches("/scripts/skippy-workload-certify.sh ")
            .count(),
        1
    );
    assert!(built.contains("--startup-timeout-secs 600") && built.contains("--require-oracle"));
    assert!(built.contains("2 certifications planned; no lanes executed"));
    assert!(!built.contains("--wire-dtype") && !built.contains("--strict-dtype"));
    assert!(!built.contains("SKIPPY_MM_PROJECTOR="));
    let (success, skipped) = intent_run(&root, "skip", &["--skip-build"]);
    assert!(
        success && !skipped.contains("cargo build -p skippy-correctness"),
        "{skipped}"
    );
    let (success, filtered) = intent_run(&root, "filter", &["--skip-build", "--families", "zeta"]);
    assert!(
        success && filtered.contains("1 certifications planned; no lanes executed"),
        "{filtered}"
    );
    assert!(!filtered.contains("/scripts/skippy-workload-certify.sh "));
    let (success, unknown) = intent_run(
        &root,
        "unknown",
        &["--skip-build", "--families", "unknown-family"],
    );
    assert!(
        !success && unknown.contains("unknown selected families"),
        "{unknown}"
    );
    assert!(!unknown.contains("==> family-certify:"));
    assert!(!root.join("absent-native-bin").exists());
    directory.close().unwrap();
    let mut causal = intent_policy();
    let mut second = causal["models"][0].clone();
    second["family"] = json!("second-family");
    causal["models"].as_array_mut().unwrap().push(second);
    let second_directory = intent_root(&causal);
    let second_root = second_directory.path().canonicalize().unwrap();
    let (success, complete) = intent_run(&second_root, "two-causal", &["--skip-build"]);
    assert!(success, "{complete}");
    assert_eq!(complete.matches("/scripts/family-certify.sh ").count(), 2);
    assert!(complete.contains("--family zeta") && complete.contains("--family second-family"));
    assert!(complete.contains("2 certifications planned; no lanes executed"));
    second_directory.close().unwrap();
}
#[test]
fn complete_battery_dry_run_preserves_all_head_mtp_budget_and_projector_lane() {
    for (startup, mtp, deadline) in [(300, 3, 2700), (1800, 3, 7200), (1800, 0, 6600)] {
        let mut policy = intent_policy();
        policy["models"][0]["execution"]["mtp_layers"] = json!(mtp);
        policy["models"][0]["execution"]["speculative_policy"] = json!("mtp-if-present");
        policy["models"][0]["resources"]["startup_timeout_secs"] = json!(startup);
        policy["models"][0]["mmproj_artifact"] = json!({"repo":"fixture/projector","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","files":["projector.gguf"],"file_integrity":{"projector.gguf":{"size_bytes":1,"blob_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}},"selector":"f16"});
        let directory = intent_root(&policy);
        let root = directory.path().canonicalize().unwrap();
        let (success, output) = intent_run(&root, "mtp", &["--skip-build"]);
        assert!(success, "{output}");
        assert!(
            output.contains(&format!(
                "startup_timeout={startup}s cert_timeout={deadline}s"
            )),
            "{output}"
        );
        assert_eq!(output.contains("--require-native-mtp-draft"), mtp > 0);
        assert!(
            output.contains("--require-lanes")
                && output.contains("--skip-build")
                && output.contains("--skip-speculative")
        );
        assert!(!output.contains("--draft-model") && !output.contains("llama-spec-bench"));
        assert!(
            output.contains("SKIPPY_MM_PROJECTOR=")
                && output.contains("frontend::tests::multimodal")
                && output.contains("--test-threads=1")
        );
        assert!(output.contains("1 certifications planned; no lanes executed"));
        assert_eq!(
            multimodal_fixture_digest(),
            "308ff69210df5efdcc7c79abd65f68f7ed8545f469222e0a3c7f774d074a5034"
        );
        directory.close().unwrap();
    }
}
fn multimodal_fixture_digest() -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(
        fs::read(
            super::support::repository().join("ci/llama-canary/fixtures/multimodal-smoke.png"),
        )
        .unwrap(),
    ))
}
#[test]
fn actual_projector_smoke_failure_preserves_separate_count_and_terminal_receipt() {
    let directory = intent_root(&intent_policy());
    let root = directory.path().canonicalize().unwrap();
    let source = fs::read_to_string(
        super::support::repository().join("skippy/scripts/skippy-family-battery.sh"),
    )
    .unwrap();
    let body = source
        .split_once("\nrun_mmproj_smoke() {\n")
        .unwrap()
        .1
        .split_once("\n}\n\n")
        .unwrap()
        .0;
    let script = format!(
        "set -euo pipefail\nrun_mmproj_smoke() {{\n{body}\n}}\n{}",
        r#"
ROOT="$PWD"
CERT_DIR="$PWD/certs"
RESULTS_JSONL="$PWD/results.jsonl"
TOTAL=1
DRY_RUN=0
MM_SMOKE_TOTAL=0
MM_SMOKE_FAILURE_COUNT=0
CERT_FAILURE_COUNT=0
FAILURES=()
FAMILY_BATTERY_MM_TEST_BIN=/bin/echo
cert_timeout_for_startup() { printf '10\n'; }
slugify() { printf '%s\n' "$1"; }
run_battery_timeout() { return "$observer_status"; }
observer_status=23
run_mmproj_smoke family model.gguf projector.gguf model-id 300 64 4
observer_status=0
TOTAL=2
run_mmproj_smoke family model.gguf projector.gguf model-id 300 64 4
printf 'separate-counts mm_total=%s mm_failed=%s core_failed=%s failures=%s\n' "$MM_SMOKE_TOTAL" "$MM_SMOKE_FAILURE_COUNT" "$CERT_FAILURE_COUNT" "${FAILURES[*]}"
"#
    );
    let (success, text) = intent_process(
        &root,
        "projector",
        vec![Value::Public("-c".into()), Value::Public(script.into())],
    );
    assert!(
        success
            && text.contains(
                "separate-counts mm_total=2 mm_failed=1 core_failed=0 failures=family@mmproj"
            ),
        "{text}"
    );
    let rows: Vec<serde_json::Value> = fs::read_to_string(root.join("results.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 2);
    for (row, status, code) in [(&rows[0], "fail", 23), (&rows[1], "pass", 0)] {
        assert_eq!(row["mmproj_smoke"], true);
        assert_eq!(row["exit_code"], code);
        assert_eq!(row["outcomes"][0]["name"], "mmproj-smoke");
        assert_eq!(row["outcomes"][0]["status"], status);
        assert_eq!(row["outcomes"][0]["exit_code"], code);
    }
    directory.close().unwrap();
}
fn mtp_gguf_fixture() -> Vec<u8> {
    fn text(bytes: &mut Vec<u8>, s: &str) {
        bytes.extend(u64::try_from(s.len()).unwrap().to_le_bytes());
        bytes.extend(s.as_bytes());
    }
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3u32.to_le_bytes());
    bytes.extend(0u64.to_le_bytes());
    bytes.extend(3u64.to_le_bytes());
    text(&mut bytes, "general.architecture");
    bytes.extend(8u32.to_le_bytes());
    text(&mut bytes, "qwen3");
    for (key, value) in [
        ("qwen3.block_count", 6u32),
        ("qwen3.embedding_length", 1024),
    ] {
        text(&mut bytes, key);
        bytes.extend(4u32.to_le_bytes());
        bytes.extend(value.to_le_bytes());
    }
    bytes
}
fn scan_fixture(complete: bool) -> serde_json::Value {
    let mut tensors=(0..5).map(|layer|json!({"name":format!("blk.{layer}.weight"),"layer_index":layer,"role":"layer","ggml_type":1,"byte_size":0})).collect::<Vec<_>>();
    for (name, size) in [("eh_proj", 1024), ("enorm", 0), ("hnorm", 0)] {
        if complete || name == "eh_proj" {
            tensors.push(json!({"name":format!("blk.5.nextn.{name}.weight"),"layer_index":5,"role":"layer","ggml_type":1,"byte_size":size}));
        }
    }
    json!({"tensor_count":tensors.len(),"tensors":tensors})
}
fn battery_executable(path: &Path, body: &str) {
    use std::os::unix::fs::PermissionsExt;
    fs::write(path, body).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
#[test]
fn complete_battery_preflight_pins_snapshot_and_refuses_incomplete_nextn_head_before_certification()
{
    use sha2::{Digest, Sha256};
    let bytes = mtp_gguf_fixture();
    let revision = "a".repeat(40);
    let mut policy = intent_policy();
    let row = &mut policy["models"][0];
    row["execution"]["trunk_layers"] = json!(5);
    row["execution"]["mtp_layers"] = json!(1);
    row["execution"]["speculative_policy"] = json!("mtp-if-present");
    row["resources"]["estimated_model_bytes"] = json!(1024);
    row["artifact"] = json!({"repo":"fixture/zeta","revision":revision,"files":["model.gguf"],"selector":"fixture","file_integrity":{"model.gguf":{"size_bytes":bytes.len(),"blob_id":hex::encode(Sha256::digest(&bytes))}}});
    let temp = intent_root(&policy);
    let root = temp.path().canonicalize().unwrap();
    let hf = root.join("hf");
    let repository = hf.join("hub/models--fixture--zeta");
    let snapshot = repository
        .join("snapshots")
        .join(&revision)
        .join("model.gguf");
    let blob = repository
        .join("blobs")
        .join(hex::encode(Sha256::digest(&bytes)));
    fs::create_dir_all(blob.parent().unwrap()).unwrap();
    fs::create_dir_all(snapshot.parent().unwrap()).unwrap();
    fs::write(&blob, &bytes).unwrap();
    std::os::unix::fs::symlink(&blob, &snapshot).unwrap();
    let native = root.join("native");
    fs::create_dir(&native).unwrap();
    battery_executable(
        &native.join("skippy-package-builder"),
        "#!/bin/bash\n[[ $# == 2 && $1 == inspect && $2 == \"$FAKE_MODEL_PATH\" ]] || exit 91\ncat \"$FAKE_SCAN_PATH\"\n",
    );
    battery_executable(
        &native.join("skippy-topology-plan"),
        "#!/bin/bash\n[[ $# == 3 && $1 == fixture/zeta:fixture && $2 == 6 && $3 == 1024 ]] || exit 92\ncat \"$FAKE_TOPOLOGY_PATH\"\n",
    );
    for name in ["skippy-correctness", "skippy"] {
        battery_executable(&native.join(name), "#!/bin/sh\nexit 93\n");
    }
    battery_executable(
        &root.join("finite-tools/hf"),
        "#!/bin/bash\n[[ $# == 5 && $1 == download && $2 == fixture/zeta && $3 == model.gguf && $4 == --revision && $5 == aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa ]] || exit 94\nprintf 'path=%s\\n' \"$FAKE_MODEL_PATH\"\n",
    );
    let topology = json!({"boundaries":(1..6).map(|layer|json!({"layer":layer,"decision":"accepted"})).collect::<Vec<_>>(),"two_stage_splits":[3],"three_stage_splits":[2,4]});
    fs::write(
        root.join("topology.json"),
        serde_json::to_vec(&topology).unwrap(),
    )
    .unwrap();
    let extra = BTreeMap::from([
        ("HF_HOME".into(), Value::Public(hf.into())),
        ("HF_HUB_OFFLINE".into(), Value::Public("1".into())),
        (
            "FAMILY_BATTERY_BIN_DIR".into(),
            Value::Public(native.into()),
        ),
        (
            "FAMILY_BATTERY_MIN_FREE_GIB".into(),
            Value::Public("0".into()),
        ),
        (
            "FAKE_MODEL_PATH".into(),
            Value::Public(snapshot.clone().into()),
        ),
        (
            "FAKE_SCAN_PATH".into(),
            Value::Public(root.join("scan.json").into()),
        ),
        (
            "FAKE_TOPOLOGY_PATH".into(),
            Value::Public(root.join("topology.json").into()),
        ),
    ]);
    for complete in [true, false] {
        fs::write(
            root.join("scan.json"),
            serde_json::to_vec(&scan_fixture(complete)).unwrap(),
        )
        .unwrap();
        let id = if complete { "complete" } else { "incomplete" };
        let argv = vec![
            Value::Public(root.join("scripts/skippy-family-battery.sh").into()),
            Value::Public("--preflight-only".into()),
            Value::Public("--skip-build".into()),
        ];
        let (success, output) = intent_process_with_environment(&root, id, argv, extra.clone());
        assert_eq!(success, complete, "{output}");
        let artifact = root.join("artifacts").join(id);
        let corpus = fs::read_to_string(artifact.join("native-mtp-models.tsv")).unwrap();
        assert_eq!(corpus.lines().count(), if complete { 2 } else { 1 });
        assert!(!artifact.join("preflight/speculative-smoke.json").exists());
        assert!(
            fs::read_dir(artifact.join("certifications"))
                .unwrap()
                .next()
                .is_none()
        );
        if complete {
            assert!(
                corpus.contains(&revision)
                    && corpus.contains("zeta")
                    && corpus.lines().last().unwrap().ends_with("\t5")
            );
            let resolved = fs::read_to_string(artifact.join("resolved-models.tsv")).unwrap();
            assert!(resolved.contains(&revision) && resolved.contains("|1|1024|5|"));
            let environment: serde_json::Value = serde_json::from_slice(
                &fs::read(artifact.join("preflight/environment.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(environment["ports"]["allocation"], "os-assigned-at-launch");
        } else {
            let rows = fs::read_to_string(artifact.join("results.jsonl")).unwrap();
            assert!(
                rows.contains("model-invalid")
                    && rows.contains("planned MTP layer count 1")
                    && rows.contains("scanned complete-head count 0")
            );
        }
    }
    temp.close().unwrap();
}
