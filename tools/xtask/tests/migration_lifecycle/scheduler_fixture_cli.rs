//! Native scheduler public interfaces; owned inert HF transport only, no corpus/model activity.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness,
    Value as Argument,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path, time::Duration};
fn catalog() -> Value {
    serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../skippy/evals/skippy-scheduler-fixtures.json"
    )))
    .unwrap()
}
fn cli(root: &Path, args: &[&str]) -> (bool, String, String) {
    let mut environment =
        BTreeMap::from([("PATH".into(), Argument::Public("/usr/bin:/bin".into()))]);
    for name in ["SYSTEMROOT", "WINDIR"] {
        if let Some(value) = std::env::var_os(name) {
            environment.insert(name.into(), Argument::Public(value));
        }
    }
    let raw = process::supervise_raw(
        &ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            arguments: ["automation", "agentic-prompt-manifest"]
                .into_iter()
                .chain(args.iter().copied())
                .map(|s| Argument::Public(s.into()))
                .collect(),
            cwd: root.into(),
            environment,
        },
        &Limits {
            execution: Duration::from_secs(10),
            graceful_shutdown: Duration::from_secs(2),
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
    let p = &raw.process;
    assert_eq!(p.outcome, process::Outcome::Exited);
    assert!(
        p.failure.is_none()
            && p.cleanup.complete
            && !p.cleanup.forced
            && !p.cleanup.graceful_signal_failed
            && p.cleanup.failure.is_none(),
        "{p:?}"
    );
    for (bytes, stream) in [
        (raw.stdout.as_ref().unwrap(), &p.stdout),
        (raw.stderr.as_ref().unwrap(), &p.stderr),
    ] {
        assert_eq!(bytes.as_bytes().len() as u64, stream.bytes_seen);
        assert!(!stream.truncated && stream.line_capture_complete);
    }
    // Raw byte completeness is separate from intentional diagnostic line redaction.
    std::fs::write(
        root.join("capture-diagnostics.json"),
        serde_json::to_vec(&json!({
            "stdout_suppressed_lines":p.stdout.suppressed_lines,
            "stderr_suppressed_lines":p.stderr.suppressed_lines,
            "stdout_bytes_seen":p.stdout.bytes_seen,"stderr_bytes_seen":p.stderr.bytes_seen
        }))
        .unwrap(),
    )
    .unwrap();
    (
        p.success(),
        String::from_utf8(raw.stdout.unwrap().as_bytes().to_vec()).unwrap(),
        String::from_utf8(raw.stderr.unwrap().as_bytes().to_vec()).unwrap(),
    )
}
#[test]
fn scheduler_fixture_actual_cli_preserves_catalog_profile_json_and_refuses_invalid_inputs() {
    let root = tempfile::tempdir().unwrap();
    let input = catalog();
    std::fs::write(
        root.path().join("catalog.json"),
        serde_json::to_vec(&input).unwrap(),
    )
    .unwrap();
    let (ok, stdout, stderr) = cli(root.path(), &["validate-fixtures", "catalog.json"]);
    assert!(ok, "{stderr}");
    let validation: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(validation["status"], "valid");
    let expected: Vec<_> = input["profiles"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();
    assert_eq!(validation["profiles"], json!(expected));
    for name in input["profiles"].as_object().unwrap().keys() {
        let (ok, stdout, stderr) = cli(root.path(), &["show-fixture", "catalog.json", name]);
        assert!(ok, "{stderr}");
        assert_eq!(
            serde_json::from_str::<Value>(&stdout).unwrap(),
            input["profiles"][name]
        );
    }
    let (ok, stdout, stderr) = cli(
        root.path(),
        &["show-fixture", "catalog.json", "missing-profile"],
    );
    assert!(!ok);
    assert!(stdout.is_empty() && stderr.contains("unknown"));
    let mut invalid = input.clone();
    invalid["profiles"]["warm-affinity"]["model"]
        .as_object_mut()
        .unwrap()
        .remove("sha256");
    std::fs::write(
        root.path().join("invalid.json"),
        serde_json::to_vec(&invalid).unwrap(),
    )
    .unwrap();
    let (ok, stdout, stderr) = cli(root.path(), &["validate-fixtures", "invalid.json"]);
    assert!(!ok);
    assert!(stdout.is_empty() && stderr.contains("sha256"));
    std::fs::write(root.path().join("existing.json"), b"preserve-existing").unwrap();
    for verb in ["materialize-fixture", "prepare-fixture"] {
        let mut args = vec![
            verb,
            "--catalog",
            "catalog.json",
            "--profile",
            "warm-affinity",
            "--output",
            "existing.json",
        ];
        if verb == "materialize-fixture" {
            args.extend(["--dataset-file", "missing.parquet"]);
        } else {
            args.extend(["--hf-bin", "/not-started-hf"]);
        }
        let (ok, stdout, stderr) = cli(root.path(), &args);
        assert!(!ok);
        assert!(
            stdout.is_empty()
                && stderr.contains(if verb == "materialize-fixture" {
                    "Hugging Face corpus"
                } else {
                    "HF corpus"
                })
        );
        assert_eq!(
            std::fs::read(root.path().join("existing.json")).unwrap(),
            b"preserve-existing"
        );
    }
    root.close().unwrap();
}
#[cfg(unix)]
fn quote(path: &Path) -> String {
    format!("'{}'", path.to_str().unwrap().replace('\'', "'\\''"))
}
#[cfg(unix)]
fn hf(root: &Path, fail: bool) -> std::path::PathBuf {
    use std::os::unix::fs::PermissionsExt as _;
    let binary = root.join("owned-hf");
    let script = format!(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" >> {}\nprintf 'end\\n' >> {}\nif [ \"$1\" = download ]; then printf '%s\\n' {}; elif [ {} = 1 ]; then printf 'owned verification refusal\\n' >&2; exit 23; fi\n",
        quote(&root.join("calls")),
        quote(&root.join("calls")),
        quote(&root.join("snapshot")),
        if fail { "1" } else { "0" }
    );
    std::fs::write(&binary, script).unwrap();
    std::fs::set_permissions(&binary, std::fs::Permissions::from_mode(0o700)).unwrap();
    binary
}
#[cfg(unix)]
#[test]
fn scheduler_fixture_actual_cli_fetches_verified_path_and_preserves_output_on_prepare_refusal() {
    for fail in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let input = catalog();
        std::fs::write(
            root.path().join("catalog.json"),
            serde_json::to_vec(&input).unwrap(),
        )
        .unwrap();
        std::fs::create_dir(root.path().join("snapshot")).unwrap();
        std::fs::write(
            root.path().join("snapshot/sessions.parquet"),
            b"inert verified-path transport fixture, not parquet proof",
        )
        .unwrap();
        let binary = hf(root.path(), fail);
        let cache = root.path().join("cache");
        let output = root.path().join("output.json");
        std::fs::write(&output, b"preserve-existing").unwrap();
        let mut args = vec![
            if fail {
                "prepare-fixture"
            } else {
                "fetch-fixture"
            },
            "--catalog",
            "catalog.json",
            "--profile",
            "agentic-eviction-pressure",
            "--hf-bin",
            binary.to_str().unwrap(),
            "--cache-dir",
            cache.to_str().unwrap(),
            "--timeout",
            "3",
        ];
        if fail {
            args.extend(["--output", output.to_str().unwrap()]);
        }
        let (ok, stdout, stderr) = cli(root.path(), &args);
        assert_eq!(ok, !fail, "{stderr}");
        if fail {
            assert!(stdout.is_empty() && stderr.contains("owned verification refusal"));
        } else {
            assert_eq!(
                stdout.trim(),
                root.path()
                    .join("snapshot/sessions.parquet")
                    .to_str()
                    .unwrap()
            );
        }
        let dataset = &input["datasets"]["agentic-coding-trajectories"];
        let mut download = vec![
            "download".to_owned(),
            dataset["repo_id"].as_str().unwrap().into(),
        ];
        download.extend(
            dataset["files"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_owned()),
        );
        download.extend([
            "--quiet".into(),
            "--repo-type".into(),
            "dataset".into(),
            "--revision".into(),
            dataset["revision"].as_str().unwrap().into(),
            "--cache-dir".into(),
            cache.to_str().unwrap().into(),
            "end".into(),
            "cache".into(),
            "verify".into(),
            dataset["repo_id"].as_str().unwrap().into(),
            "--repo-type".into(),
            "dataset".into(),
            "--revision".into(),
            dataset["revision"].as_str().unwrap().into(),
            "--fail-on-missing-files".into(),
            "--cache-dir".into(),
            cache.to_str().unwrap().into(),
            "end".into(),
        ]);
        assert_eq!(
            std::fs::read_to_string(root.path().join("calls")).unwrap(),
            download.join("\n") + "\n"
        );
        assert_eq!(std::fs::read(&output).unwrap(), b"preserve-existing");
        root.close().unwrap();
    }
}
