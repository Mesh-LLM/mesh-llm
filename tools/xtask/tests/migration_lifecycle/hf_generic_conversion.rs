#![cfg(unix)]
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, time::Duration};
fn sha(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn succeeded(report: &crate::process::RawProcessReport) -> bool {
    report.process.status.and_then(|s| s.code()) == Some(0)
}
fn diagnostics(report: &crate::process::RawProcessReport) -> String {
    report.stderr.as_ref().map_or(String::new(), |b| {
        String::from_utf8_lossy(b.as_bytes()).into_owned()
    })
}
fn invoke(root: &Path, input: &Value, label: &str) -> (crate::process::RawProcessReport, Value) {
    use crate::process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, RawCaptureOptions,
        Readiness, Value as Argument,
    };
    let path = root.join(format!("{label}.json"));
    fs::write(&path, serde_json::to_vec(input).unwrap()).unwrap();
    let out = root.join(label);
    let seconds = input["timeout_seconds"].as_u64().unwrap() + 10;
    let result = process::supervise_raw_with_files(
        &ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            cwd: root.into(),
            environment: std::collections::BTreeMap::from([(
                "PATH".into(),
                Argument::Public("/usr/bin:/bin".into()),
            )]),
            arguments: [
                "automation",
                "hf-certify",
                "generic-conversion",
                "--input",
                path.to_str().unwrap(),
                "--output-directory",
                out.to_str().unwrap(),
            ]
            .into_iter()
            .map(|s| Argument::Public(s.into()))
            .collect(),
        },
        &Limits {
            execution: Duration::from_secs(seconds),
            graceful_shutdown: Duration::from_secs(4),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 1048576,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
        RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(1048576),
            stderr: std::num::NonZeroUsize::new(1048576),
        },
    )
    .unwrap();
    assert!(result.process.failure.is_none());
    assert!(
        result.process.cleanup.complete
            && !result.process.cleanup.forced
            && !result.process.cleanup.graceful_signal_failed
            && result.process.cleanup.failure.is_none()
    );
    for stream in [&result.process.stdout, &result.process.stderr] {
        assert!(stream.line_capture_complete && !stream.truncated);
        assert_eq!(stream.oversized_lines, 0);
    }
    assert_eq!(
        result.stdout.as_ref().unwrap().as_bytes().len() as u64,
        result.process.stdout.bytes_seen
    );
    assert_eq!(
        result.stderr.as_ref().unwrap().as_bytes().len() as u64,
        result.process.stderr.bytes_seen
    );
    let evidence = fs::read(out.join("generic-conversion.json"))
        .map(|b| serde_json::from_slice(&b).unwrap())
        .unwrap_or(Value::Null);
    (result, evidence)
}
pub(super) fn input(root: &Path, failure: bool) -> Value {
    use std::os::unix::fs::PermissionsExt;
    let source = root.join("source");
    fs::create_dir(&source).unwrap();
    fs::write(source.join("config.json"), b"{}").unwrap();
    let work = root.join("work");
    fs::create_dir(&work).unwrap();
    let binary = root.join("native-inert");
    let script = format!(
        r#"#!/bin/bash
set -eu
verb="$1"
shift
if [ "$verb" = convert-job ]; then
  dry=0
  while [ "$#" -gt 0 ]; do
    printf '%s\n' "$1" >> '{work}/argv'
    case "$1" in
      --mtp) shift;;
      --dry-run) dry=1; shift;;
      *) printf '%s\n' "$2" >> '{work}/argv'; shift 2;;
    esac
  done
  if [ "$dry" = 1 ]; then exit 0; fi
  mkdir -p '{work}/target/BF16'
  printf GGUFfixture1 > '{work}/target/BF16/model-00001-of-00002.gguf'
  printf GGUFfixture2 > '{work}/target/BF16/model-00002-of-00002.gguf'
  printf '%s' '{{"schema_version":1,"kind":"convert","source":"{source}","target":"{work}/target","target_prefix":"BF16","output_basename":"model","expected_splits":2,"window_size":1,"output_type":"bf16","quant":null,"tensor_type_file":null}}' > '{work}/convert-manifest.json'
  printf '%s' '{{"phase":"complete","events":[]}}' > '{work}/status.json'
  exit 0
fi
test "$verb" = verify-job
if [ '{failure}' = true ]; then exit 23; fi
printf '%s\n' '{{"root":"{work}/target","prefix":"BF16","basename":"model","expected_splits":2,"completed_count":2,"first_missing":null,"last_present":2,"first_shard":"model-00001-of-00002.gguf","last_shard":"model-00002-of-00002.gguf","complete":true}}'
"#,
        work = work.display(),
        source = source.display()
    );
    fs::write(&binary, script.as_bytes()).unwrap();
    fs::set_permissions(&binary, fs::Permissions::from_mode(0o700)).unwrap();
    json!({"schema_version":1,"source_repo":"fixture/source","target_repo":"fixture/result","mesh_revision":"b".repeat(40),"output_basename":"model","source":source,"source_files":[{"path":source.join("config.json"),"sha256":sha(b"{}")}],"work_directory":work,"binary":{"path":binary,"sha256":sha(script.as_bytes())},"expected_splits":2,"timeout_seconds":15})
}
#[test]
fn generic_conversion_actual_cli_preserves_split_status_card_and_upload_only_complete_roster() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let mut request = input(&root, false);
    let (report, evidence) = invoke(&root, &request, "converted");
    assert!(succeeded(&report), "{}", diagnostics(&report));
    assert_eq!(evidence["status"], "LOCAL_ARTIFACT_READY");
    assert_eq!(evidence["publication_completed"], false);
    assert_eq!(evidence["artifact_roster"].as_object().unwrap().len(), 6);
    let artifact = root.join("work/target/BF16");
    assert!(
        fs::read_to_string(artifact.join("README.md"))
            .unwrap()
            .contains("not a standalone chat model")
    );
    assert_eq!(
        fs::read(artifact.join("skippy-convert-status.json")).unwrap(),
        fs::read(root.join("work/status.json")).unwrap()
    );
    let calls = fs::read(root.join("work/argv")).unwrap();
    let calls_text = String::from_utf8(calls.clone()).unwrap();
    for needle in [
        "--window-size\n1\n",
        "--split-max-size\n50G\n",
        "--max-memory\n24G\n",
        "--watchdog-seconds\n300\n",
        "--stream-buffer-bytes\n8388608\n",
    ] {
        assert!(calls_text.contains(needle));
    }
    request["upload_only"] = json!(true);
    request["binary"] = Value::Null;
    request["source_files"] = json!([]);
    fs::remove_file(root.join("native-inert")).unwrap();
    let (report, evidence) = invoke(&root, &request, "upload-only");
    assert!(succeeded(&report));
    assert_eq!(evidence["status"], "LOCAL_ARTIFACT_READY");
    assert_eq!(evidence["artifact_roster"].as_object().unwrap().len(), 5);
    assert_eq!(fs::read(root.join("work/argv")).unwrap(), calls);
    assert!(evidence["convert_process"].is_null());
    fs::remove_file(artifact.join("model-00002-of-00002.gguf")).unwrap();
    let (report, evidence) = invoke(&root, &request, "missing");
    assert!(!succeeded(&report));
    assert_eq!(evidence["status"], "FAILED");
    assert_eq!(evidence["publication_completed"], false);
    temp.close().unwrap();
}
#[test]
fn generic_conversion_actual_cli_dry_run_and_failed_native_verify_never_publish() {
    for failed in [false, true] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let mut request = input(&root, failed);
        request["dry_run"] = json!(!failed);
        let (report, evidence) = invoke(&root, &request, "result");
        assert_eq!(succeeded(&report), !failed);
        assert_eq!(
            evidence["status"],
            if failed {
                "FAILED"
            } else {
                "DRY_RUN_COMPLETED"
            }
        );
        assert_eq!(evidence["publication_completed"], false);
        if failed {
            assert_eq!(evidence["verify_process"]["exit_code"], 23);
            assert!(
                root.join("work/target/BF16/model-00001-of-00002.gguf")
                    .exists()
            );
            assert!(!root.join("work/target/BF16/README.md").exists());
        } else {
            assert!(evidence["verification"].is_null());
            assert!(!root.join("work/target").exists());
        }
        temp.close().unwrap();
    }
}

#[test]
#[ignore = "isolated inert generic publisher helper; invoked only by owning CLI fixture"]
fn generic_conversion_inert_publication_helper() {
    let mode = std::env::var("GENERIC_HELPER_MODE").unwrap();
    let output = std::path::PathBuf::from(std::env::var("GENERIC_HELPER_OUTPUT").unwrap());
    fs::create_dir(&output).unwrap();
    let value = if mode == "repo" {
        #[derive(serde::Serialize)]
        struct Options {
            repo: String,
            credential_file: std::path::PathBuf,
            output_directory: std::path::PathBuf,
            timeout_seconds: u64,
            confirm: bool,
        }
        let opts = Options {
            repo: std::env::var("GENERIC_HELPER_REPO").unwrap(),
            credential_file: std::env::var("GENERIC_HELPER_CREDENTIAL").unwrap().into(),
            output_directory: output.clone(),
            timeout_seconds: std::env::var("GENERIC_HELPER_TIMEOUT")
                .unwrap()
                .parse()
                .unwrap(),
            confirm: true,
        };
        json!({"schema_version":1,"request_sha256":sha(&serde_json::to_vec(&opts).unwrap()),"status":"REPOSITORY_READY","repository":{"repo":opts.repo,"completed":true,"error":null,"observed_parent":"c".repeat(40),"mutation_attempted":true}})
    } else {
        let input = fs::read(std::env::var("GENERIC_HELPER_INPUT").unwrap()).unwrap();
        // The native publisher hashes its typed compact input, rather than the pretty transport.
        #[derive(serde::Deserialize, serde::Serialize)]
        struct Artifact {
            path: std::path::PathBuf,
            path_in_repo: String,
            sha256: String,
            byte_size: u64,
        }
        #[derive(serde::Deserialize, serde::Serialize)]
        struct PublisherInput {
            schema_version: u32,
            repo: String,
            parent_commit: String,
            shards: Vec<Artifact>,
            sidecars: Vec<Artifact>,
            credential_file: Option<std::path::PathBuf>,
            execution_timeout_ms: u64,
        }
        let typed: PublisherInput = serde_json::from_slice(&input).unwrap();
        let request_sha256 = sha(&serde_json::to_vec(&typed).unwrap());
        let request: Value = serde_json::from_slice(&input).unwrap();
        assert_eq!(request["parent_commit"], "c".repeat(40));
        assert_eq!(request["shards"].as_array().unwrap().len(), 2);
        assert_eq!(request["sidecars"].as_array().unwrap().len(), 3);
        let paths = request["shards"]
            .as_array()
            .unwrap()
            .iter()
            .chain(request["sidecars"].as_array().unwrap())
            .map(|a| a["path_in_repo"].clone())
            .collect::<Vec<_>>();
        let objects = request["shards"].as_array().unwrap().iter().map(|a|json!({"oid":a["sha256"],"size":a["byte_size"],"mutation_attempted":true,"uploaded_parts":1,"object_present":true,"source_custody_verified":true,"completed":true,"error":null})).collect::<Vec<_>>();
        json!({"schema_version":1,"request_sha256":request_sha256,"status":"PUBLISHED","source_custody_verified":true,"error":null,"publication":{"schema_version":1,"repo":request["repo"],"parent_commit":request["parent_commit"],"ordered_paths":paths,"objects":objects,"object_attempted_paths":request["shards"].as_array().unwrap().iter().map(|a|a["path_in_repo"].clone()).collect::<Vec<_>>(),"commit_attempted":true,"commit_oid":"d".repeat(40),"remote_verified_paths":paths,"final_source_custody_verified":true,"completed":true,"error":null}})
    };
    let name = if mode == "repo" {
        "repository.json"
    } else {
        "publication.json"
    };
    fs::write(output.join(name), serde_json::to_vec(&value).unwrap()).unwrap();
}
fn publisher(root: &Path, mode: &str) -> Value {
    use std::os::unix::fs::PermissionsExt;
    let path = root.join(format!("{mode}-helper"));
    let test = std::env::current_exe().unwrap();
    let script = format!(
        r#"#!/bin/bash
set -eu
export GENERIC_HELPER_MODE='{mode}'
if [ '{mode}' = repo ]; then
  test "$1" = ensure-repo; shift
  test "$1" = --confirm; shift
  test "$1" = --repo; export GENERIC_HELPER_REPO="$2"; shift 2
  test "$1" = --credential-file; export GENERIC_HELPER_CREDENTIAL="$2"; shift 2
  test "$1" = --output-directory; export GENERIC_HELPER_OUTPUT="$2"; shift 2
  test "$1" = --timeout-seconds; export GENERIC_HELPER_TIMEOUT="$2"; shift 2
else
  test "$1" = publish; shift
  test "$1" = --input; export GENERIC_HELPER_INPUT="$2"; shift 2
  test "$1" = --output-directory; export GENERIC_HELPER_OUTPUT="$2"; shift 2
fi
test "$#" = 0
exec '{test}' --ignored --exact hf_generic_conversion::generic_conversion_inert_publication_helper --nocapture
"#,
        test = test.display()
    );
    fs::write(&path, script.as_bytes()).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
    json!({"path":path,"sha256":sha(script.as_bytes())})
}
#[test]
fn generic_conversion_actual_cli_provisions_then_admits_complete_folder_at_one_commit() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let mut request = input(&root, false);
    request["expected_splits"] = json!(1);
    let repo = publisher(&root, "repo");
    let model = publisher(&root, "model");
    let credential = root.join("credential");
    fs::write(&credential, b"inert-not-an-account-token").unwrap();
    request["helper"] = repo.clone();
    request["helper_source"] = repo;
    request["model_publisher"] = model.clone();
    request["model_publisher_source"] = model;
    request["credential_file"] = json!(credential);
    request["publish_confirmed"] = json!(true);
    request["timeout_seconds"] = json!(30);
    let (process, receipt) = invoke(&root, &request, "published");
    assert!(
        succeeded(&process),
        "{}; receipt: {receipt}",
        diagnostics(&process)
    );
    assert_eq!(receipt["status"], "PUBLISHED");
    assert_eq!(receipt["publication_completed"], true);
    let publication = &receipt["model_publication"]["final_receipt"]["publication"];
    assert_eq!(publication["commit_oid"], "d".repeat(40));
    assert_eq!(publication["ordered_paths"].as_array().unwrap().len(), 5);
    assert_eq!(
        publication["remote_verified_paths"],
        publication["ordered_paths"]
    );
    assert_eq!(publication["final_source_custody_verified"], true);
    request["upload_only"] = json!(true);
    request["binary"] = Value::Null;
    request["source_files"] = json!([]);
    fs::write(
        root.join("work/target/BF16/unpublished.bin"),
        b"not a supported sidecar",
    )
    .unwrap();
    let (refused, partial) = invoke(&root, &request, "refused-roster");
    assert!(!succeeded(&refused));
    assert_eq!(partial["status"], "FAILED");
    assert_eq!(partial["publication_completed"], false);
    assert!(!root.join("refused-roster/repository").exists());
    temp.close().unwrap();
}
#[test]
fn generic_conversion_actual_upload_only_refuses_oversized_and_unbounded_split_manifest() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let mut request = input(&root, false);
    let (started, _) = invoke(&root, &request, "ready");
    assert!(succeeded(&started));
    request["upload_only"] = json!(true);
    request["binary"] = Value::Null;
    request["source_files"] = json!([]);
    let manifest = root.join("work/target/BF16/skippy-convert-manifest.json");
    for (label, bytes) in [
        ("oversized", vec![b' '; 8 * 1048576 + 1]),
        ("huge-count", serde_json::to_vec(&json!({"expected_splits":u64::MAX,"output_basename":"model","target_prefix":"BF16"})).unwrap()),
    ] {
        fs::write(&manifest, &bytes).unwrap();
        let (refused, receipt) = invoke(&root, &request, label);
        assert!(!succeeded(&refused));
        assert_eq!(receipt["status"], "FAILED");
        assert_eq!(receipt["publication_completed"], false);
        assert!(receipt["repository"].is_null());
        assert_eq!(fs::read(&manifest).unwrap(), bytes);
    }
    temp.close().unwrap();
}
#[test]
fn generic_conversion_actual_cli_admits_native_effective_split_increase_and_upload_only_roster() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let mut request = input(&root, false);
    request["expected_splits"] = json!(1);
    request["split_max_size"] = json!("1G");
    let (report, receipt) = invoke(&root, &request, "effective");
    assert!(succeeded(&report), "{}", diagnostics(&report));
    assert_eq!(receipt["effective_splits"], 2);
    assert_eq!(receipt["verification"]["expected_splits"], 2);
    assert_eq!(receipt["artifact_roster"].as_object().unwrap().len(), 6);
    assert!(
        root.join("work/target/BF16/model-00002-of-00002.gguf")
            .is_file()
    );
    request["upload_only"] = json!(true);
    request["binary"] = Value::Null;
    request["source_files"] = json!([]);
    let (report, receipt) = invoke(&root, &request, "effective-upload");
    assert!(succeeded(&report));
    assert_eq!(receipt["effective_splits"], 2);
    assert_eq!(receipt["artifact_roster"].as_object().unwrap().len(), 5);
    assert!(receipt["convert_process"].is_null());
    temp.close().unwrap();
}
