//! Actual full native frontend with a privately injected native helper transport.
//! Requires separately built xtask; synthetic four-family assets are not real model qualification.
use super::{
    contract::digest,
    full_chain::{Peer, fixture},
};
use serde_json::{Value, json};
use std::os::unix::{fs::PermissionsExt as _, process::CommandExt as _};
use std::{
    fs,
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant},
};
fn quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}
fn executable(path: &Path, text: &str) {
    fs::write(path, text).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).unwrap();
}
pub(super) fn call(
    xtask: &Path,
    root: &Path,
    args: &[String],
    label: &str,
) -> (bool, String, String) {
    let out = root.join(format!("{label}-outer-stdout"));
    let err = root.join(format!("{label}-outer-stderr"));
    let mut child = Command::new(xtask)
        .args(args)
        .current_dir(root)
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .stdout(Stdio::from(fs::File::create(&out).unwrap()))
        .stderr(Stdio::from(fs::File::create(&err).unwrap()))
        .process_group(0)
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(45);
    let (status, timed_out) = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break (status, false);
        }
        if Instant::now() >= deadline {
            // Still-owned unreaped leader prevents PID reuse during group cleanup.
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            break (child.wait().unwrap(), true);
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    let stdout = fs::read(&out).unwrap();
    let stderr = fs::read(&err).unwrap();
    assert!(
        !timed_out,
        "owned finite frontend exceeded its budget after forced cleanup"
    );
    assert!(stdout.len() < 1024 * 1024 && stderr.len() < 1024 * 1024);
    (
        status.success(),
        String::from_utf8(stdout).unwrap(),
        String::from_utf8(stderr).unwrap(),
    )
}
fn reader(root: &Path) -> std::path::PathBuf {
    let response = root.join("inert-reader-response.json");
    fs::write(&response,serde_json::to_vec(&json!({"schema_version":1,"rows":[{"session_id":"fixture-session","source_dataset":"fixture","n_turns":1,"max_isl":3,"total_tokens":4,"messages_json":"[{\"role\":\"user\",\"content\":\"finite fixture\"}]"}]})).unwrap()).unwrap();
    let reader = root.join("reader");
    executable(
        &reader,
        &format!(
            "#!/bin/sh\nif [ \"$1\" = '--help' ]; then printf 'trajectory-reader --request FILE --response FILE\\n'; exit 0; fi\n[ \"$1\" = '--request' ] && [ \"$3\" = '--response' ] || exit 91\n/bin/cat {} > \"$4\"\n",
            quote(response.to_str().unwrap())
        ),
    );
    reader
}
fn manifest(root: &Path, config: &Value) -> std::path::PathBuf {
    let artifacts:Vec<_>=config["models"].as_array().unwrap().iter().map(|row|{let file=row["filename"].as_str().unwrap();let bytes=format!("finite-model-{}",row["key"].as_str().unwrap());json!({"id":row["artifact_id"],"repo":row["repo"],"revision":row["revision"],"selector":"fixture","model_ref":format!("{}:fixture",row["repo"].as_str().unwrap()),"cadences":["manual"],"files":[file],"urls":[format!("https://huggingface.co/{}/resolve/{}/{}",row["repo"].as_str().unwrap(),row["revision"].as_str().unwrap(),file)],"file_integrity":{(file):{"size_bytes":bytes.len(),"blob_id":digest(bytes.as_bytes())}}})}).collect();
    let path = root.join("model-manifest.json");
    fs::write(&path,serde_json::to_vec(&json!({"schema_version":1,"manifest_kind":"test-model-artifacts","artifacts":artifacts})).unwrap()).unwrap();
    path
}
#[test]
#[ignore = "requires separately built xtask plus tokenizers locked graph; no acquisition outside owned loopback"]
fn actual_full_native_prefetch_acquires_four_families_exports_and_generates_manifest_and_readerless_skips()
 {
    let xtask = std::path::PathBuf::from(
        std::env::var("COMPETITIVE_FIXTURE_XTASK_BIN")
            .expect("explicit separately built xtask required"),
    )
    .canonicalize()
    .unwrap();
    assert!(xtask.is_file());
    for skip in [false, true] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let f = fixture(&root);
        let peer = Peer::new(f.files, f.revision);
        let reader = reader(&root);
        let manifest = manifest(&root, &f.config);
        let mut input: Value = serde_json::from_slice(&fs::read(&f.request).unwrap()).unwrap();
        let mut config = f.config;
        if !skip {
            let seed = root.join("seed.parquet");
            fs::write(&seed, b"inert protocol fixture; not actual parquet").unwrap();
            let generated = root.join("expected-manifest.json");
            let launcher = root.join("manifest-launcher");
            executable(
                &launcher,
                &format!(
                    "#!/bin/sh\nexport MESH_LLM_TRAJECTORY_READER_BIN={}\nexec {} \"$@\"\n",
                    quote(reader.to_str().unwrap()),
                    quote(xtask.to_str().unwrap())
                ),
            );
            let (ok, _, error) = call(
                &launcher,
                &root,
                &[
                    "automation".into(),
                    "agentic-prompt-manifest".into(),
                    "--dataset-file".into(),
                    seed.to_str().unwrap().into(),
                    "--dataset-revision".into(),
                    "a".repeat(40),
                    "--output".into(),
                    generated.to_str().unwrap().into(),
                    "--families".into(),
                    "1".into(),
                    "--requests-per-family".into(),
                    "2".into(),
                    "--min-isl".into(),
                    "1".into(),
                    "--max-isl".into(),
                    "20".into(),
                    "--min-turns".into(),
                    "1".into(),
                    "--source-dataset".into(),
                    "fixture".into(),
                ],
                "expected",
            );
            assert!(ok, "native manifest baseline failed: {error}");
            let bytes = fs::read(generated).unwrap();
            let produced: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(produced["prompts"].as_array().unwrap().len(), 2);
            assert_eq!(
                produced["metadata"]["selection"]["sources"],
                json!(["fixture"])
            );
            config["thoughtworks"]["selection"]["manifest_sha256"] = json!(digest(&bytes));
        } else {
            input["model_keys"] = json!(["granite-h1-hybrid"]);
            for group in ["skip_dataset", "skip_tokenizers", "skip_vllm_configs"] {
                input[group] = json!(true);
            }
            fs::remove_file(&reader).unwrap();
        }
        let config_bytes = serde_json::to_vec(&config).unwrap();
        fs::write(input["config"].as_str().unwrap(), &config_bytes).unwrap();
        input["config_sha256"] = json!(digest(&config_bytes));
        input["model_manifest"] = json!(&manifest);
        input["model_manifest_sha256"] = json!(digest(&fs::read(&manifest).unwrap()));
        fs::write(&f.request, serde_json::to_vec(&input).unwrap()).unwrap();
        let helper = root.join("helper");
        executable(
            &helper,
            &format!(
                "#!/bin/sh\nexport COMPETITIVE_FIXTURE_ENDPOINT={}\nif [ \"$1\" = 'verify-acquired' ]; then export COMPETITIVE_FIXTURE_INPUT=\"$3\" COMPETITIVE_FIXTURE_PHASE=\"$5\"; else export COMPETITIVE_FIXTURE_INPUT=\"$2\"; unset COMPETITIVE_FIXTURE_PHASE; fi\nexec {} --exact competitive_acquisition::full_chain::fixture_native_helper_process --ignored --nocapture\n",
                quote(&peer.endpoint),
                quote(std::env::current_exe().unwrap().to_str().unwrap())
            ),
        );
        let evidence = root.join("evidence");
        let mut args = vec![
            "automation".into(),
            "replay-matrix".into(),
            "competitive-inputs-prefetch".into(),
            "--request".into(),
            f.request.to_str().unwrap().into(),
            "--helper".into(),
            helper.to_str().unwrap().into(),
            "--helper-sha256".into(),
            digest(&fs::read(&helper).unwrap()),
            "--evidence-directory".into(),
            evidence.to_str().unwrap().into(),
            "--timeout-seconds".into(),
            "35".into(),
        ];
        if !skip {
            args.extend([
                "--reader".into(),
                reader.to_str().unwrap().into(),
                "--reader-sha256".into(),
                digest(&fs::read(reader).unwrap()),
            ]);
        }
        let (ok, _, error) = call(&xtask, &root, &args, "full");
        assert!(ok, "actual finite prefetch refused: {error}");
        let receipt: Value =
            serde_json::from_slice(&fs::read(evidence.join("prefetch.json")).unwrap()).unwrap();
        assert_eq!(
            receipt["status"],
            if skip {
                "MATERIALIZED_SELECTED"
            } else {
                "MATERIALIZED"
            }
        );
        assert!(receipt["error"].is_null());
        assert_eq!(
            receipt["helper_final"]["families"]
                .as_array()
                .unwrap()
                .len(),
            if skip { 1 } else { 4 }
        );
        assert_eq!(receipt["before-manifest-process-clean"], true);
        assert_eq!(receipt["after-manifest-process-clean"], true);
        if !skip {
            assert_eq!(
                receipt["manifest_sha256"],
                config["thoughtworks"]["selection"]["manifest_sha256"]
            );
            assert!(root.join("owned/thoughtworks/manifest.json").is_file());
            assert!(
                root.join("owned/tokenizers/granite-h1-hybrid/model.safetensors")
                    .is_file()
            );
            assert!(
                !root
                    .join("owned/tokenizers/granite-h1-hybrid/README.md")
                    .exists()
            );
        } else {
            assert!(!root.join("owned/thoughtworks").exists());
            assert!(receipt["reader_sha256"].is_null());
        }
        let requests = peer.close();
        assert!(
            requests
                .iter()
                .all(|r| !r.to_ascii_lowercase().contains("authorization:")),
            "anonymous native Rust HF requests must omit Authorization entirely"
        );
        assert!(
            requests
                .iter()
                .all(|r| r.contains("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"))
        );
        temp.close().unwrap();
    }
}
