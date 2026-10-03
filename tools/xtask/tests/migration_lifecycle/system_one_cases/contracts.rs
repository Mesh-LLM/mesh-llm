use super::serialization::{run, run_spec};
use crate::{
    process::{ProcessSpec, Value},
    workflow_yaml::{self, Node},
};
use serde_json::{Value as Json, json};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
};

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}
fn source(path: &str) -> String {
    fs::read_to_string(repository().join(path)).unwrap()
}
fn environment(root: &Path) -> BTreeMap<std::ffi::OsString, Value> {
    BTreeMap::from([
        (
            "PATH".into(),
            Value::Public(std::env::var_os("PATH").unwrap()),
        ),
        (
            "MESH_LLM_AUTOMATION_BIN".into(),
            Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
        ),
        ("WORK_DIR".into(), Value::Public(root.into())),
    ])
}
fn shell(
    root: &Path,
    body: String,
    env: BTreeMap<std::ffi::OsString, Value>,
) -> crate::process::RawProcessReport {
    fs::write(root.join("caller.sh"), body).unwrap();
    run_spec(ProcessSpec {
        executable: "/bin/bash".into(),
        arguments: vec![Value::Public("caller.sh".into())],
        cwd: root.into(),
        environment: env,
    })
}
fn function(script: &str, name: &str, next: &str) -> String {
    let suffix = script.split(&format!("{name}() {{\n")).nth(1).unwrap();
    format!(
        "{name}() {{\n{}",
        suffix.split(&format!("\n{next}() {{")).next().unwrap()
    )
}
fn resolve(id: &str, cadence: &str, cwd: &Path) -> Json {
    let manifest = repository().join("ci/model-artifacts/manifests/skippy-system-one-smoke.json");
    let result = run(
        vec![
            "models".into(),
            "resolve".into(),
            manifest.to_str().unwrap().into(),
            "--artifact-id".into(),
            id.into(),
            "--cadence".into(),
            cadence.into(),
            "--require-single-file".into(),
        ],
        cwd,
    );
    assert!(result.process.success());
    serde_json::from_slice(result.stdout.unwrap().as_bytes()).unwrap()
}
#[test]
fn actual_pinned_resolver_and_direct_prewarm_preserve_published_identity_and_suite_boundaries() {
    let scratch = tempfile::tempdir().unwrap();
    let pinned = resolve("family-diffusion-gemma", "manual", scratch.path());
    for (key, value) in [
        ("repo", "unsloth/diffusiongemma-26B-A4B-it-GGUF"),
        ("revision", "f4183a2c7a354128d02545752303c4354d165bf0"),
        ("file", "diffusiongemma-26B-A4B-it-Q4_K_M.gguf"),
        (
            "sha256",
            "24523b6c833c9ce9f5f34f9b333ab1517d73d6f1e76a103645353114c8028bc5",
        ),
    ] {
        assert_eq!(pinned[key], value);
    }
    assert_eq!(
        pinned["size_bytes"]
            .as_str()
            .unwrap()
            .parse::<u64>()
            .unwrap(),
        16806810208_u64
    );
    for cadence in ["llama-bump", "manual-full", "nightly"] {
        assert_eq!(
            resolve("family-diffusion-gemma", cadence, scratch.path()),
            pinned
        );
    }
    let manifest: Json = serde_json::from_str(&source(
        "ci/model-artifacts/manifests/skippy-system-one-smoke.json",
    ))
    .unwrap();
    let ids: std::collections::BTreeSet<_> = manifest["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap())
        .collect();
    assert_eq!(
        ids,
        [
            "family-qwen3-dense",
            "family-laya-multilingual",
            "family-diffusion-gemma"
        ]
        .into_iter()
        .collect()
    );
    let family: Json =
        serde_json::from_str(&source("ci/llama-canary/family-certified.json")).unwrap();
    assert!(
        family["models"]
            .as_array()
            .unwrap()
            .iter()
            .all(|row| row["family"] != "diffusion-gemma")
    );
    let map: Json =
        serde_json::from_str(&source("ci/llama-canary/generated-family-map.json")).unwrap();
    assert!(
        map["families"]
            .as_object()
            .unwrap()
            .get("diffusion-gemma")
            .is_none()
    );
    let wrapper = repository().join("scripts/skippy-system-one-smoke.sh");
    assert_ne!(
        fs::metadata(&wrapper).unwrap().permissions().mode() & 0o111,
        0
    );
    let report = run_spec(ProcessSpec {
        executable: wrapper,
        arguments: vec![Value::Public("--prewarm".into())],
        cwd: scratch.path().into(),
        environment: environment(scratch.path()),
    });
    assert!(report.process.success());
    let output = String::from_utf8_lossy(report.stdout.as_ref().unwrap().as_bytes());
    for key in ["repo", "revision", "file", "sha256", "url"] {
        assert!(output.contains(pinned[key].as_str().unwrap()), "{key}");
    }
    assert!(output.contains(pinned["size_bytes"].as_str().unwrap()));
    assert!(output.contains("hf download"));
}

#[test]
fn actual_wrapper_resolver_authorization_and_digest_refusals_stop_before_native_launch() {
    let scratch = tempfile::tempdir().unwrap();
    let root = scratch.path();
    let script = source("scripts/skippy-system-one-smoke.sh");
    let mut manifest: Json = serde_json::from_str(&source(
        "ci/model-artifacts/manifests/skippy-system-one-smoke.json",
    ))
    .unwrap();
    let row = &mut manifest["artifacts"][0];
    let file = row["file"].as_str().unwrap().to_owned();
    row["size_bytes"] = json!(7);
    row["sha256"] = json!("0".repeat(64));
    row["file_integrity"][&file]["size_bytes"] = json!(7);
    row["file_integrity"][&file]["blob_id"] = json!("0".repeat(64));
    fs::write(root.join(&file), b"corrupt").unwrap();
    fs::write(
        root.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    for cadence in ["unauthorized-finite-cadence", "manual"] {
        let caller = format!(
            r#"set -euo pipefail
automation=("$MESH_LLM_AUTOMATION_BIN")
SMOKE_MANIFEST="$PWD/manifest.json"
SMOKE_CADENCE='{cadence}'
REPORT_DIR="$PWD"
require_smoke_binaries() {{ :; }}
start_stage_server() {{ : > native-launch; return 97; }}
{}
{}
{}
if run_cases_against_stage contract family-qwen3-dense "$PWD/{file}" contract 128 2; then exit 98; else exit "$?"; fi
"#,
            function(&script, "artifact_summary", "cached_artifact_path"),
            function(&script, "verify_artifact_digest", "write_stage_config"),
            function(&script, "run_cases_against_stage", "prewarm_plan")
        );
        let report = shell(root, caller, environment(root));
        assert_eq!(report.process.status.unwrap().code(), Some(2));
        assert!(!root.join("native-launch").exists());
        assert!(!root.join("contract-stage.json").exists());
        assert!(!root.join("system-one-contract.json").exists());
    }
}

fn document(path: &str) -> Node {
    workflow_yaml::parse(&source(path)).unwrap()
}
fn text<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap()
}
#[test]
fn parsed_both_canary_passes_preserve_report_upload_and_execute_actual_cache_state_summary() {
    let top = document(".github/workflows/llama-upstream-canary.yml");
    let jobs = top.get("jobs").unwrap();
    for name in ["candidate", "verification"] {
        let job = jobs.get(name).unwrap();
        assert_eq!(
            text(job, "uses"),
            "./.github/workflows/llama-canary-family-pass.yml"
        );
        assert!(text(job, "if").contains("!cancelled()"));
        assert!(job.get("with").unwrap().get("pass_id").is_some());
    }
    assert_eq!(
        text(
            jobs.get("verification").unwrap().get("with").unwrap(),
            "mode"
        ),
        "verify-build"
    );
    assert!(
        jobs.get("verification")
            .unwrap()
            .get("needs")
            .unwrap()
            .list()
            .contains(&"candidate")
    );
    let family = document(".github/workflows/llama-canary-family-pass.yml");
    let build = family.get("jobs").unwrap().get("build").unwrap();
    let Node::Seq(steps) = build.get("steps").unwrap() else {
        panic!("build steps")
    };
    let index = |name| {
        steps
            .iter()
            .position(|step| step.get("name").and_then(Node::text) == Some(name))
            .unwrap()
    };
    let report = &steps[index("Report System One smoke result")];
    let upload = &steps[index("Upload System One smoke evidence")];
    assert!(index("Report System One smoke result") < index("Upload System One smoke evidence"));
    for step in [report, upload] {
        assert!(text(step, "if").contains("!cancelled()"));
    }
    let evidence = upload.get("with").unwrap();
    assert!(text(upload, "uses").starts_with("actions/upload-artifact@"));
    assert!(text(evidence, "name").contains("inputs.pass_id"));
    assert!(
        text(evidence, "path")
            .contains("${{ env.CANARY_SOURCE_ROOT }}/target/skippy-system-one-smoke/")
    );
    assert_eq!(text(evidence, "if-no-files-found"), "warn");
    assert!(text(report.get("env").unwrap(), "WORK_DIR").contains("env.CANARY_SOURCE_ROOT"));
    let laya = &steps[index("Report Laya smoke result")];
    assert!(text(laya, "run").contains("reports/laya.json"));
    let scratch = tempfile::tempdir().unwrap();
    let root = scratch.path();
    fs::create_dir(root.join("reports")).unwrap();
    for (checked, path, expected) in [
        (false, "", "cache state not checked"),
        (true, "", "not present in the offline cache"),
        (true, "/offline/pinned.gguf", "present in the offline cache"),
    ] {
        fs::write(root.join("reports/system-one.json"), serde_json::to_vec(&json!({
            "status":"unqualified", "contract":{"status":"pass"},
            "full_model_read":{"status":"unqualified","backend":"metal","certified_backends":["cuda"],"artifact_cache_checked":checked,"artifact_path":path},
            "reasons":["finite qualification reason"]
        })).unwrap()).unwrap();
        let summary = root.join("summary.md");
        fs::write(&summary, "").unwrap();
        let mut env = environment(root);
        env.insert(
            "GITHUB_STEP_SUMMARY".into(),
            Value::Public(summary.clone().into_os_string()),
        );
        env.insert("BUILD_OUTCOME".into(), Value::Public("success".into()));
        let result = shell(root, text(report, "run").to_owned(), env);
        assert!(result.process.success());
        let rendered = fs::read_to_string(summary).unwrap();
        assert!(rendered.contains(expected));
        assert!(rendered.contains("NOT CERTIFIED"));
        assert!(rendered.contains("finite qualification reason"));
    }
}

#[test]
fn actual_canary_build_qualification_and_verification_relocation_preserve_required_smoke() {
    let script = source("scripts/llama-canary-agent-repair.sh");
    let scratch = tempfile::tempdir().unwrap();
    let root = scratch.path();
    let verification = root.join("verification");
    fs::create_dir(&verification).unwrap();
    let verification = verification.canonicalize().unwrap();
    let materialize = function(&script, "materialize_verification_tree", "run_prepare");
    let full_build = function(&script, "run_full_build", "run_certification");
    let caller = format!(
        r#"set -euo pipefail
ROOT="$PWD"
TRUSTED_ROOT="$PWD"
VERIFY_ROOT="$PWD/verification"
CERTIFIED_SHA=fixture
RUN_KEY=finite
LLAMA_STAGE_BUILD_DIR="$PWD/native"
SYSTEMONE_SMOKE_DIR="$PWD/old-smoke"
cleanup_verification_worktree() {{ :; }}
git() {{ :; }}
rm() {{ :; }}
verify_repair_pin() {{ :; }}
{materialize}
materialize_verification_tree
[[ "$SYSTEMONE_SMOKE_DIR" == "$VERIFY_ROOT/target/skippy-system-one-smoke" ]] || exit 96
BUILD_LOG="$ROOT/build.log"
STATE_DIR="$ROOT"
HARNESS_MODE=verify-build
lipo() {{ printf 'arm64\n'; }}
run_verification_logged() {{
 local label="$1"
 shift 2
 if [[ "$label" == 'System One smoke' ]]; then
   printf '%s\n' "$@" > "$ROOT/smoke-arguments"
 fi
}}
{full_build}
run_full_build
"#
    );
    let result = shell(root, caller, environment(root));
    assert!(result.process.success());
    let args = fs::read_to_string(verification.join("smoke-arguments")).unwrap();
    let arguments: std::collections::BTreeSet<_> = args.lines().collect();
    for required in [
        "env",
        "SYSTEMONE_SMOKE_CADENCE=llama-bump",
        "SYSTEMONE_SMOKE_BUILD_BACKEND=metal",
        "SYSTEMONE_SMOKE_CERTIFIED_BACKENDS=metal",
        "SYSTEMONE_SMOKE_REQUIRE_QUALIFIED=1",
        "scripts/skippy-system-one-smoke.sh",
    ] {
        assert!(arguments.contains(required));
    }
    assert!(
        arguments.contains(
            format!(
                "WORK_DIR={}/target/skippy-system-one-smoke",
                verification.display()
            )
            .as_str()
        )
    );
}
