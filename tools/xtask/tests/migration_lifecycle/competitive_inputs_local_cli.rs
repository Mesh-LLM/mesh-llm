//! Actual native local component CLI; no acquisition or tokenizer inference.
use crate::process;
use sha2::{Digest as _, Sha256};
use std::{collections::BTreeMap, fs, num::NonZeroUsize, path::Path, time::Duration};
fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}
fn tree(root: &Path) -> String {
    let mut names: Vec<_> = fs::read_dir(root)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    let mut h = Sha256::new();
    for name in names {
        h.update((name.len() as u64).to_be_bytes());
        h.update(name.as_bytes());
        h.update(Sha256::digest(fs::read(root.join(name)).unwrap()));
    }
    hex::encode(h.finalize())
}
#[test]
fn actual_local_tokenizer_materializer_cli_success_pin_refusal_and_fresh_output_custody() {
    for mode in [
        "success",
        "pin-refusal",
        "existing-output",
        "source-output-overlap",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let source = root.join("source");
        fs::create_dir(&source).unwrap();
        fs::write(source.join("tokenizer.json"), b"{\"fixture\":true}").unwrap();
        let source_pin = tree(&source);
        let mut config: serde_json::Value = serde_json::from_slice(
            &fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../skippy/evals/skippy-competitive-benchmark.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for model in config["models"].as_array_mut().unwrap() {
            if model["key"] == "llama32-dense" {
                model["tokenizer_sha256"] = serde_json::json!(if mode == "pin-refusal" {
                    "a".repeat(64)
                } else {
                    source_pin.clone()
                });
            }
        }
        let config_bytes = serde_json::to_vec(&config).unwrap();
        fs::write(root.join("config.json"), &config_bytes).unwrap();
        let output = if mode == "source-output-overlap" {
            source.join("output")
        } else {
            root.join("output")
        };
        if mode == "existing-output" {
            fs::create_dir(&output).unwrap();
            fs::write(output.join("sentinel"), b"preserve").unwrap();
        }
        let request = serde_json::json!({"config":root.join("config.json"),"config_sha256":hash(&config_bytes),"model_keys":["llama32-dense"],"sources":[{"key":"llama32-dense","directory":source,"source_tree_sha256":source_pin,"kind":"supplied-derived-export"}],"output_directory":output,"timeout_seconds":5,"maximum_source_bytes":1024});
        fs::write(
            root.join("input.json"),
            serde_json::to_vec(&request).unwrap(),
        )
        .unwrap();
        let raw = process::supervise_raw(
            &process::ProcessSpec {
                executable: env!("CARGO_BIN_EXE_xtask").into(),
                cwd: root.clone(),
                environment: BTreeMap::new(),
                arguments: [
                    "automation",
                    "replay-matrix",
                    "competitive-inputs-local",
                    "--input",
                ]
                .map(|s| process::Value::Public(s.into()))
                .into_iter()
                .chain([process::Value::Public(
                    root.join("input.json").into_os_string(),
                )])
                .collect(),
            },
            &process::Limits {
                execution: Duration::from_secs(12),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: process::Readiness::None,
                completion: process::Completion::Exit,
            },
            &process::Cancellation::default(),
            process::RawCaptureOptions {
                stdout: NonZeroUsize::new(65536),
                stderr: NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        let report = raw.process;
        assert_eq!(report.outcome, process::Outcome::Exited);
        assert!(
            report.failure.is_none()
                && report.cleanup.complete
                && !report.cleanup.forced
                && !report.cleanup.graceful_signal_failed
                && report.cleanup.failure.is_none()
        );
        assert!(!report.stdout.truncated && !report.stderr.truncated);
        assert_eq!(
            raw.stdout.unwrap().as_bytes().len() as u64,
            report.stdout.bytes_seen
        );
        assert_eq!(
            raw.stderr.unwrap().as_bytes().len() as u64,
            report.stderr.bytes_seen
        );
        assert_eq!(report.status.unwrap().success(), mode == "success");
        if mode == "existing-output" {
            assert_eq!(fs::read(output.join("sentinel")).unwrap(), b"preserve");
            assert!(!output.join("materialization.json").exists());
        } else if mode == "source-output-overlap" {
            assert!(!output.exists());
            assert_eq!(tree(&source), source_pin);
            assert_eq!(fs::read_dir(&source).unwrap().count(), 1);
        } else {
            let receipt: serde_json::Value =
                serde_json::from_slice(&fs::read(output.join("materialization.json")).unwrap())
                    .unwrap();
            assert_eq!(
                receipt["status"],
                if mode == "success" {
                    "SUPPLIED_MATERIALIZED"
                } else {
                    "FAILED"
                }
            );
            assert_eq!(receipt["acquisition_performed"], false);
            assert_eq!(receipt["tokenizer_export_performed"], false);
            if mode == "success" {
                assert_eq!(
                    fs::read(output.join("tokenizers/llama32-dense/tokenizer.json")).unwrap(),
                    b"{\"fixture\":true}"
                );
            } else {
                assert!(receipt["families"].as_array().unwrap().is_empty());
            }
        }
        temp.close().unwrap();
    }
}
