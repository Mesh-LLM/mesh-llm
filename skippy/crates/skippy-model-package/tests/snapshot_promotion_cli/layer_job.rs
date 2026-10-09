#![cfg(unix)]
use super::fixtures;
use std::{
    fs,
    path::Path,
    process::{Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};
fn finish(mut child: std::process::Child) -> Output {
    let until = Instant::now() + Duration::from_secs(10);
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= until {
            child.kill().unwrap();
            let result = child.wait_with_output().unwrap();
            panic!("owned layer CLI deadline: {}", result.status);
        }
        thread::sleep(Duration::from_millis(5));
    }
    child.wait_with_output().unwrap()
}
fn inputs(root: &Path) -> (std::path::PathBuf, std::path::PathBuf) {
    let manifest = root.join("model-package.json");
    fs::write(
        &manifest,
        fixtures::manifest("metadata.gguf", 4, "a".repeat(64)),
    )
    .unwrap();
    let source = root.join("source.json");
    fs::write(&source,serde_json::to_vec(&serde_json::json!({"repo":"fixture/model","revision":"b".repeat(40),"metadata_sha256":{},"missing":[],"license":"apache-2.0"})).unwrap()).unwrap();
    (manifest, source)
}
fn invoke(manifest: &Path, source: &Path, output: &Path) -> Output {
    finish(
        Command::new(env!("CARGO_BIN_EXE_model-package-layer-job"))
            .env_clear()
            .args(["project", "--manifest"])
            .arg(manifest)
            .args([
                "--source-repo",
                "fixture/model",
                "--source-revision",
                &"b".repeat(40),
                "--source-admission-file",
            ])
            .arg(source)
            .args([
                "--target-repo",
                "fixture/package",
                "--experimental",
                "--output-directory",
            ])
            .arg(output)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    )
}
#[test]
fn layer_job_actual_cli_projects_full_root_card_and_refuses_nonregular_or_identity_mismatch() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let (manifest, source) = inputs(&root);
    let output = root.join("projection");
    let report = invoke(&manifest, &source, &output);
    assert!(report.status.success());
    assert_eq!(
        String::from_utf8(report.stdout).unwrap(),
        "model.gguf\n1\n4\n"
    );
    let projection: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("projection.json")).unwrap()).unwrap();
    assert_eq!(projection["total_bytes"], 4);
    assert_eq!(projection["layer_count"], 1);
    assert_eq!(projection["artifacts"][0]["path"], "metadata.gguf");
    assert_eq!(projection["source_revision"], "b".repeat(40));
    let card = fs::read_to_string(output.join("README.preview.md")).unwrap();
    assert!(card.contains("does not promote"));
    assert!(card.contains("license: \"apache-2.0\""));
    assert!(!invoke(&manifest, &source, &output).status.success());
    let mut value: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    value["layer_count"] = serde_json::json!(2);
    fs::write(&manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    let refused = root.join("refused");
    assert!(!invoke(&manifest, &source, &refused).status.success());
    assert!(!refused.exists());
    let fifo = root.join("fifo");
    let name = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
    // SAFETY: CString owns a valid NUL-terminated fixture path; no writer is opened.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    assert!(!invoke(&fifo, &source, &refused).status.success());
    assert!(!refused.exists());
    temp.close().unwrap();
}
#[test]
fn layer_job_actual_embedded_projection_block_forwards_source_and_experimental_policy() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let (_, source) = inputs(&root);
    let package = root.join("package");
    fs::create_dir(&package).unwrap();
    fs::copy(
        root.join("model-package.json"),
        package.join("model-package.json"),
    )
    .unwrap();
    let source_dir = root.join("source-admission");
    fs::create_dir(&source_dir).unwrap();
    fs::copy(source, source_dir.join("source.json")).unwrap();
    let script = include_str!("../../src/scripts/split-model-job.sh");
    let start = script.find("# The root/card projection").unwrap();
    let end = script[start..].find("SOURCE_IDENTITY=").unwrap() + start;
    let block = &script[start..end];
    let bash = if std::path::Path::new("/opt/homebrew/bin/bash").exists() {
        "/opt/homebrew/bin/bash"
    } else {
        "bash"
    };
    let report = finish(
        Command::new(bash)
            .env_clear()
            .env("LAYER_JOB", env!("CARGO_BIN_EXE_model-package-layer-job"))
            .env("LOCAL_WORK_DIR", &root)
            .env("PACKAGE_DIR", package)
            .env("SOURCE_ADMISSION_DIR", source_dir)
            .env("SOURCE_REPO", "fixture/model")
            .env("SOURCE_REVISION", "b".repeat(40))
            .env("TARGET_REPO", "fixture/package")
            .env("SOURCE_PIPELINE_TAG", "text-generation")
            .env("PACKAGE_EXPERIMENTAL", "true")
            .args(["-euc", block])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    assert!(
        report.status.success(),
        "{}",
        String::from_utf8_lossy(&report.stderr)
    );
    let value: serde_json::Value =
        serde_json::from_slice(&fs::read(root.join("layer-projection/projection.json")).unwrap())
            .unwrap();
    assert_eq!(value["experimental"], true);
    assert_eq!(value["source_repo"], "fixture/model");
    assert!(
        fs::read_to_string(root.join("layer-projection/README.preview.md"))
            .unwrap()
            .contains("mesh-llm serve --model \"fixture/package\" --split")
    );
    temp.close().unwrap();
}

fn upload_admit(root: &Path, artifact: &Path, credential: &Path, output: &Path) -> Output {
    finish(
        Command::new(env!("CARGO_BIN_EXE_model-package-layer-job"))
            .env_clear()
            .args([
                "upload",
                "--admit-only",
                "--repo",
                "fixture/package",
                "--revision",
                "automation/staging",
                "--relative-path",
                "layers/layer.gguf",
                "--artifact",
            ])
            .arg(artifact)
            .arg("--credential-file")
            .arg(credential)
            .arg("--output-directory")
            .arg(output)
            .current_dir(root)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    )
}
#[test]
fn layer_upload_actual_cli_admits_observed_fd_identity_and_refuses_fifo_or_private_credentials() {
    use sha2::{Digest as _, Sha256};
    use std::os::unix::fs::PermissionsExt as _;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let artifact = root.join("layer.gguf");
    fs::write(&artifact, b"finite artifact").unwrap();
    let credential = root.join("credential");
    fs::write(&credential, b"private-fixture-token").unwrap();
    fs::set_permissions(&credential, fs::Permissions::from_mode(0o600)).unwrap();
    let output = root.join("admission");
    let report = upload_admit(&root, &artifact, &credential, &output);
    assert!(report.status.success());
    let value: serde_json::Value =
        serde_json::from_slice(&fs::read(output.join("upload.json")).unwrap()).unwrap();
    assert_eq!(value["status"], "ADMITTED");
    assert_eq!(value["publication_performed"], false);
    assert_eq!(value["source_identity_origin"], "observed_local_fd");
    assert_eq!(value["identity"]["byte_size"], 15);
    let digest: String = Sha256::digest(b"finite artifact")
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(value["identity"]["sha256"], digest);
    assert_eq!(value["request_sha256"].as_str().unwrap().len(), 64);
    assert!(artifact.is_file());
    assert!(!String::from_utf8_lossy(&report.stdout).contains("private-fixture-token"));
    assert!(!String::from_utf8_lossy(&report.stderr).contains("private-fixture-token"));
    assert!(
        !fs::read_to_string(output.join("upload.json"))
            .unwrap()
            .contains("private-fixture-token")
    );
    let refused = root.join("refused");
    fs::set_permissions(&credential, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(
        !upload_admit(&root, &artifact, &credential, &refused)
            .status
            .success()
    );
    assert!(!refused.exists());
    fs::set_permissions(&credential, fs::Permissions::from_mode(0o600)).unwrap();
    let fifo = root.join("fifo");
    let name = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
    // SAFETY: valid owned fixture path, with no writer to conceal a blocking open.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    assert!(
        !upload_admit(&root, &fifo, &credential, &refused)
            .status
            .success()
    );
    assert!(!refused.exists());
    assert!(
        !upload_admit(&root, &artifact, &credential, &output)
            .status
            .success()
    );
    temp.close().unwrap();
}
#[test]
fn layer_upload_actual_embedded_hook_preserves_scope_attempts_and_private_file_delivery() {
    use std::os::unix::fs::PermissionsExt as _;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let helper = root.join("helper");
    fs::write(
        &helper,
        br##"#!/bin/bash
set -euo pipefail
[[ "$1" == upload && "$2" == --confirm ]]
shift 2
while (($#)); do
 case "$1" in
 --repo) [[ "$2" == fixture/package ]]; shift 2;;
 --revision) [[ "$2" == automation/staging ]]; shift 2;;
 --artifact) artifact="$2"; shift 2;;
 --relative-path) [[ "$2" == layers/layer.gguf ]]; shift 2;;
 --credential-file) credential="$2"; shift 2;;
 --output-directory) output="$2"; shift 2;;
 --maximum-attempts) [[ "$2" == 8 ]]; shift 2;;
 --timeout-seconds) [[ "$2" == 3600 ]]; shift 2;;
 --unlink-after-success) shift;;
 *) exit 64;;
 esac
done
[[ "$(<"$credential")" == private-fixture-token ]]
[[ -f "$artifact" ]]
mkdir "$output"
printf '%s' '{"status":"inert-projection-only"}' > "$output/upload.json"
printf '%s' "$credential" > "$JOB_TMP_DIR/credential-path"
"##,
    )
    .unwrap();
    fs::set_permissions(&helper, fs::Permissions::from_mode(0o700)).unwrap();
    let script = include_str!("../../src/scripts/split-model-job.sh");
    let begin = script.find("# Native one-artifact publication.").unwrap();
    let end = script[begin..]
        .find("chmod +x \"$ARTIFACT_UPLOAD_HOOK\"")
        .unwrap()
        + begin;
    let hook = root.join("upload-hook");
    let report = finish(
        Command::new("/bin/bash")
            .env_clear()
            .env("LAYER_JOB", &helper)
            .env("ARTIFACT_UPLOAD_HOOK", &hook)
            .args(["-euc", &script[begin..end]])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    assert!(report.status.success());
    let artifact = root.join("layer.gguf");
    fs::write(&artifact, b"finite artifact").unwrap();
    let report = finish(
        Command::new("/bin/bash")
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("LAYER_JOB", &helper)
            .env("JOB_TMP_DIR", &root)
            .env("HF_TOKEN", "private-fixture-token")
            .env("TARGET_REPO", "fixture/package")
            .env("TARGET_UPLOAD_REVISION", "automation/staging")
            .env("SKIPPY_PACKAGE_ARTIFACT_PATH", &artifact)
            .env("SKIPPY_PACKAGE_ARTIFACT_RELATIVE_PATH", "layers/layer.gguf")
            .arg(&hook)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    assert!(report.status.success());
    let credential = fs::read_to_string(root.join("credential-path")).unwrap();
    assert!(!Path::new(&credential).exists());
    assert!(artifact.is_file()); // Inert adapter does not impersonate actual publication/unlink.
    assert!(!String::from_utf8_lossy(&report.stdout).contains("private-fixture-token"));
    assert!(!String::from_utf8_lossy(&report.stderr).contains("private-fixture-token"));
    temp.close().unwrap();
}

#[test]
fn layer_repository_actual_cli_refuses_unconfirmed_or_invalid_scope_before_output() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let output = root.join("refused");
    for (confirmed, repo) in [
        (false, "fixture/package"),
        (true, "../foreign"),
        (true, "single"),
    ] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_model-package-layer-job"));
        command
            .env_clear()
            .args(["ensure-repo", "--repo", repo, "--credential-file"])
            .arg(root.join("missing-secret"))
            .arg("--output-directory")
            .arg(&output);
        if confirmed {
            command.arg("--confirm");
        }
        let report = finish(
            command
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        assert!(!report.status.success());
        assert!(!output.exists());
        let error = String::from_utf8_lossy(&report.stderr);
        assert!(error.contains("layer job operation incomplete"));
        assert!(report.stdout.is_empty());
        assert!(!error.contains("missing-secret"));
    }
    temp.close().unwrap();
}

#[test]
fn layer_catalog_actual_cli_requires_confirmation_before_credentials_output_or_transport() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let output = root.join("refused");
    let report = finish(
        Command::new(env!("CARGO_BIN_EXE_model-package-layer-job"))
            .env_clear()
            .args(["update-catalog", "--manifest"])
            .arg(root.join("missing.json"))
            .args([
                "--source-repo",
                "fixture/model",
                "--source-revision",
                &"a".repeat(40),
                "--source-file",
                "model.gguf",
                "--target-repo",
                "fixture/package",
                "--credential-file",
            ])
            .arg(root.join("missing-secret"))
            .arg("--output-directory")
            .arg(&output)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    assert!(!report.status.success());
    let error = String::from_utf8_lossy(&report.stderr);
    assert!(error.contains("layer job operation incomplete"));
    assert!(report.stdout.is_empty());
    assert!(!error.contains("missing-secret"));
    assert!(!output.exists());
    temp.close().unwrap();
}
#[test]
fn layer_publication_actual_embedded_tail_preserves_manifest_staging_catalog_pr_and_card_main_scope()
 {
    use std::os::unix::fs::PermissionsExt as _;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let helper = root.join("helper");
    fs::write(&helper,br##"#!/bin/bash
set -euo pipefail
command="$1"; shift
printf '%s\n' "$command" >> "$LOCAL_WORK_DIR/calls"
while (($#)); do
 case "$1" in
 --confirm|--create-pr|--experimental) printf '%s\n' "$1" >> "$LOCAL_WORK_DIR/calls"; shift;;
 --output-directory) output="$2"; shift 2;;
 --artifact) artifact="$2"; shift 2;;
 --relative-path) relative="$2"; shift 2;;
 --revision) revision="$2"; shift 2;;
 --credential-file) credential="$2"; shift 2;;
 --repo|--source-repo|--source-revision|--source-file|--target-repo|--maximum-attempts|--timeout-seconds|--manifest|--pipeline-tag|--mesh-llm-ref) shift 2;;
 *) exit 64;;
 esac
done
[[ "$(<"$credential")" == private-fixture-token ]]
mkdir "$output"
case "$command" in
 upload) [[ -f "$artifact" ]]; printf '%s:%s\n' "$relative" "$revision" >> "$LOCAL_WORK_DIR/calls";;
 update-catalog) printf '%s' '{"status":"inert-projection"}' > "$output/catalog.json";;
 prepare-card) printf '%s' 'finite inert card' > "$output/README.md";;
 *) exit 64;;
esac
"##).unwrap();
    fs::set_permissions(&helper, fs::Permissions::from_mode(0o700)).unwrap();
    let package = root.join("package");
    fs::create_dir(&package).unwrap();
    fs::write(package.join("model-package.json"), b"{}").unwrap();
    let script = include_str!("../../src/scripts/split-model-job.sh");
    let start = script.find("# Shared native publication entry").unwrap();
    let end = script[start..].find("# ─── Summary").unwrap() + start;
    let report = finish(
        Command::new("/bin/bash")
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("LAYER_JOB", &helper)
            .env("LOCAL_WORK_DIR", &root)
            .env("JOB_TMP_DIR", &root)
            .env("PACKAGE_DIR", package)
            .env("HF_TOKEN", "private-fixture-token")
            .env("TARGET_REPO", "fixture/package")
            .env("TARGET_UPLOAD_REVISION", "automation/staging")
            .env("SOURCE_REPO", "fixture/model")
            .env("SOURCE_REVISION", "a".repeat(40))
            .env("SOURCE_FILE", "model.gguf")
            .env("REPUBLISH", "false")
            .env("CATALOG_CREATE_PR", "true")
            .env("PACKAGE_EXPERIMENTAL", "true")
            .args(["-euc", &script[start..end]])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    assert!(report.status.success());
    let calls = fs::read_to_string(root.join("calls")).unwrap();
    assert_eq!(calls.lines().filter(|l| *l == "upload").count(), 2);
    assert!(calls.contains("model-package.json:automation/staging"));
    assert!(calls.contains("README.md:main"));
    assert!(calls.contains("update-catalog\n--confirm\n--create-pr"));
    assert!(calls.contains("prepare-card\n--experimental"));
    for name in ["layer-catalog-credential", "layer-card-credential"] {
        assert!(!root.join(name).exists());
    }
    assert!(!String::from_utf8_lossy(&report.stdout).contains("private-fixture-token"));
    assert!(!String::from_utf8_lossy(&report.stderr).contains("private-fixture-token"));
    temp.close().unwrap();
}

#[test]
fn layer_workspace_actual_cli_formats_supplied_observations_and_refuses_overflow() {
    for (verb, bytes, expected, success) in [
        ("format-bytes", "1024", "1.0 KiB\n", true),
        ("workspace-estimate", "7", "34359738375\n", true),
        ("workspace-estimate", "18446744073709551615", "", false),
    ] {
        let report = finish(
            Command::new(env!("CARGO_BIN_EXE_model-package-layer-job"))
                .env_clear()
                .args([verb, "--bytes", bytes])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        assert_eq!(report.status.success(), success);
        assert_eq!(String::from_utf8(report.stdout).unwrap(), expected);
    }
}

#[test]
fn layer_projector_actual_cli_refuses_invalid_selection_before_credential_or_output() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    for (ordinal, path) in ["../mm.gguf", "nested//mm.gguf", "mm.bin"]
        .iter()
        .enumerate()
    {
        let output = root.join(format!("refused-{ordinal}"));
        let report = finish(
            Command::new(env!("CARGO_BIN_EXE_model-package-layer-job"))
                .env_clear()
                .args([
                    "projector",
                    "--repo",
                    "fixture/model",
                    "--revision",
                    &"b".repeat(40),
                    "--file",
                    path,
                    "--credential-file",
                ])
                .arg(root.join("unread-missing-credential"))
                .arg("--output-directory")
                .arg(&output)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        assert!(!report.status.success());
        assert!(report.stdout.is_empty());
        assert!(
            String::from_utf8_lossy(&report.stderr)
                .contains("operation incomplete; inspect receipts")
        );
        assert!(!output.exists());
    }
    temp.close().unwrap();
}
#[test]
fn layer_projector_actual_embedded_loop_preserves_literal_files_and_private_credential_cleanup() {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let helper = root.join("inert-helper");
    fs::write(
        &helper,
        r#"#!/bin/bash
set -eu
printf '%s\n' "$@" >> "$CALLS"
test "$1" = projector
shift
while [ "$#" -gt 0 ]; do
 case "$1" in
 --file) selected="$2";;
 --output-directory) output="$2";;
 --credential-file) credential="$2";;
 esac
 shift 2
done
test "$(cat "$credential")" = private-fixture-token
test "$(stat -c '%a' "$credential" 2>/dev/null || stat -f '%Lp' "$credential")" = 600
if [ "${FAIL:-0}" = 1 ]; then exit 41; fi
mkdir "$output"
printf GGUFfixture > "$output/projector.gguf"
printf '%s\n' "$output/projector.gguf"
"#,
    )
    .unwrap();
    fs::set_permissions(&helper, fs::Permissions::from_mode(0o700)).unwrap();
    let script = include_str!("../../src/scripts/split-model-job.sh");
    let start = script.find("WRITE_PACKAGE_PROJECTOR_ARGS=()").unwrap();
    let end = script[start..].find("PUBLISHER_METADATA_DIR=").unwrap() + start;
    let block = format!(
        "{}\nprintf '%s\\n' \"${{WRITE_PACKAGE_PROJECTOR_ARGS[@]}}\" > \"$ARGS\"\n",
        &script[start..end]
    );
    for failed in [false, true] {
        let workspace = root.join(if failed { "failed" } else { "success" });
        fs::create_dir(&workspace).unwrap();
        let calls = workspace.join("calls");
        let args = workspace.join("args");
        let report = finish(
            Command::new("/bin/bash")
                .env_clear()
                .env("PATH", "/usr/bin:/bin")
                .env("LAYER_JOB", &helper)
                .env("JOB_TMP_DIR", &workspace)
                .env("SOURCE_REPO", "fixture/model")
                .env("SOURCE_REVISION", "b".repeat(40))
                .env(
                    "SOURCE_PROJECTOR_FILES",
                    "__layer_projector_inert__/mm proj.gguf\n__layer_projector_inert__/second.gguf",
                )
                .env("HF_TOKEN", "private-fixture-token")
                .env("CALLS", &calls)
                .env("ARGS", &args)
                .env("FAIL", if failed { "1" } else { "0" })
                .args(["-euc", &block])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        assert_eq!(report.status.code(), Some(if failed { 41 } else { 0 }));
        let calls = fs::read_to_string(calls).unwrap();
        assert!(calls.contains("--file\n__layer_projector_inert__/mm proj.gguf\n"));
        assert!(calls.contains(&format!("--revision\n{}\n", "b".repeat(40))));
        assert!(!calls.contains("private-fixture-token"));
        for entry in fs::read_dir(&workspace)
            .unwrap()
            .flatten()
            .filter(|e| e.file_type().unwrap().is_dir())
        {
            assert!(!entry.path().join("credential").exists());
        }
        assert!(!String::from_utf8_lossy(&report.stdout).contains("private-fixture-token"));
        assert!(!String::from_utf8_lossy(&report.stderr).contains("private-fixture-token"));
        if failed {
            assert!(!args.exists());
        } else {
            let args = fs::read_to_string(args).unwrap();
            assert_eq!(args.lines().filter(|s| *s == "--projector").count(), 2);
            assert!(calls.contains("--file\n__layer_projector_inert__/second.gguf\n"));
        }
    }
    temp.close().unwrap();
}
