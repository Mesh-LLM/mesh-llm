//! Native generic materialization boundaries; no HF or real dataset download.
use crate::process;
use std::{
    collections::BTreeMap, fs, num::NonZeroUsize, os::unix::fs::PermissionsExt as _, path::Path,
    time::Duration,
};

fn invoke(root: &Path, body: &str, extra: &[(&str, String)]) -> (bool, String, String) {
    let mut environment = BTreeMap::from([
        (
            "PATH".into(),
            process::Value::Public("/usr/bin:/bin".into()),
        ),
        (
            "AUTOMATION".into(),
            process::Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
        ),
        ("ROOT".into(), process::Value::Public(root.into())),
    ]);
    for (key, value) in extra {
        environment.insert((*key).into(), process::Value::Public(value.as_str().into()));
    }
    let raw = process::supervise_raw(
        &process::ProcessSpec {
            executable: "/bin/bash".into(),
            cwd: root.into(),
            environment,
            arguments: ["-euo", "pipefail", "-c", body]
                .map(|v| process::Value::Public(v.into()))
                .into(),
        },
        &process::Limits {
            execution: Duration::from_secs(10),
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
            && report.cleanup.failure.is_none(),
        "{report:?}"
    );
    let out = raw.stdout.unwrap();
    let err = raw.stderr.unwrap();
    assert_eq!(out.as_bytes().len() as u64, report.stdout.bytes_seen);
    assert_eq!(err.as_bytes().len() as u64, report.stderr.bytes_seen);
    (
        report.status.unwrap().success(),
        String::from_utf8(out.as_bytes().to_vec()).unwrap(),
        String::from_utf8(err.as_bytes().to_vec()).unwrap(),
    )
}
fn source() -> String {
    fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/materialize-competitive-inputs.sh"),
    )
    .unwrap()
}

#[test]
fn materialize_canonical_path_uses_native_owner_with_spaces_and_no_python() {
    let temporary = tempfile::tempdir().unwrap();
    fs::create_dir(temporary.path().join("source with spaces")).unwrap();
    fs::write(temporary.path().join("source with spaces/data"), "fixture").unwrap();
    std::os::unix::fs::symlink("source with spaces/data", temporary.path().join("linked")).unwrap();
    let source = source();
    assert!(!source.contains("PROMPT_GENERATOR"));
    assert!(
        source.contains("if [[ \"$SKIP_TOKENIZERS\" -eq 0 ]]; then\n  command -v \"$PYTHON_BIN\"")
    );
    assert_eq!(
        source
            .matches("TRANSFORMERS_VERBOSITY=error \"$PYTHON_BIN\" -")
            .count(),
        1
    );
    let start = source.find("resolve_path() {").unwrap();
    let end = start + source[start..].find("\n}").unwrap() + 2;
    let function = &source[start..end];
    assert!(!function.contains("PYTHON_BIN"));
    let body = format!("automation=(\"$AUTOMATION\")\n{function}\nresolve_path \"$ROOT/linked\"");
    let (ok, out, error) = invoke(temporary.path(), &body, &[]);
    assert!(ok, "{error}");
    assert_eq!(
        out.trim(),
        temporary
            .path()
            .join("linked")
            .canonicalize()
            .unwrap()
            .to_str()
            .unwrap()
    );
    temporary.close().unwrap();
}

#[test]
fn materialize_prompt_call_uses_actual_native_frontend_and_keeps_reader_refusal() {
    for refusal in [false, true] {
        let temporary = tempfile::tempdir().unwrap();
        let root = temporary.path();
        fs::write(root.join("sessions.parquet"), b"owned inert reader input").unwrap();
        let response = serde_json::json!({"schema_version":1,"rows":[{"session_id":"session","source_dataset":"source","n_turns":20,"max_isl":9000,"total_tokens":9100,"messages_json":"[{\"role\":\"user\",\"content\":\"prefix-é\"}]"}]});
        let body = if refusal {
            "#!/bin/bash\nexit 37\n".into()
        } else {
            format!(
                "#!/bin/bash\nset -euo pipefail\n[[ $# == 4 && $1 == --request && $3 == --response ]] || exit 91\n/bin/cat > \"$4\" <<'RESPONSE'\n{response}\nRESPONSE\n"
            )
        };
        let reader = root.join("reader");
        fs::write(&reader, body).unwrap();
        fs::set_permissions(&reader, fs::Permissions::from_mode(0o700)).unwrap();
        let source = source();
        let needle = "\"${automation[@]}\" automation agentic-prompt-manifest \\\n";
        assert_eq!(source.matches(needle).count(), 1);
        let start = source.find(needle).unwrap();
        let end =
            start + source[start..].find("\"${sources[@]}\"").unwrap() + "\"${sources[@]}\"".len();
        let call = &source[start..end];
        // jq supplies fixed selection input only; actual maintained call and frontend own argv/policy.
        let prelude = "automation=(\"$AUTOMATION\")\nds_file=sessions.parquet\nds_rev=pinned\nCONFIG=owned-config\nsources=(--source-dataset source)\nmkdir -p \"$ROOT/thoughtworks\"\ncp \"$ROOT/sessions.parquet\" \"$ROOT/thoughtworks/sessions.parquet\"\njq() { case \"$2\" in .thoughtworks.selection.families|.thoughtworks.selection.requests_per_family) printf '1\\n' ;; .thoughtworks.selection.min_isl) printf '8192\\n' ;; .thoughtworks.selection.max_isl_exclusive) printf '12000\\n' ;; .thoughtworks.selection.min_turns) printf '20\\n' ;; *) return 92 ;; esac; }\n";
        let (ok, _, error) = invoke(
            root,
            &format!("{prelude}{call}"),
            &[(
                "MESH_LLM_TRAJECTORY_READER_BIN",
                reader.to_string_lossy().into_owned(),
            )],
        );
        let output = root.join("thoughtworks/manifest.json");
        if refusal {
            assert!(!ok);
            assert!(error.contains("trajectory reader failed"), "{error}");
            assert!(!output.exists());
        } else {
            assert!(ok, "{error}");
            let manifest: serde_json::Value =
                serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
            assert_eq!(manifest["metadata"]["dataset_revision"], "pinned");
            assert_eq!(manifest["metadata"]["rows"][0]["session_id"], "session");
            assert!(
                manifest["metadata"]["rows"][0]
                    .get("messages_json")
                    .is_none()
            );
            assert!(
                manifest["prompts"][0]["prompt"]
                    .as_str()
                    .unwrap()
                    .starts_with("<user>\nprefix-é")
            );
        }
        temporary.close().unwrap();
    }
}

#[test]
fn materializer_reader_preflight_refuses_before_downloads_and_skip_dataset_needs_no_reader() {
    let source = source();
    let start = source
        .find("# Reject missing or unusable native reader")
        .unwrap();
    let end = start + source[start..].find("\n# Pin the model list").unwrap();
    let gate = &source[start..end];
    assert!(end < source.find("hf download").unwrap_or(source.len()));
    for mode in ["absent", "refusal", "success", "skip"] {
        let temporary = tempfile::tempdir().unwrap();
        let reader = temporary.path().join("reader");
        if mode != "absent" {
            let body = if mode == "refusal" {
                "#!/bin/bash\nexit 37\n"
            } else {
                "#!/bin/bash\n[[ $# == 1 && $1 == --help ]] || exit 91\nprintf 'trajectory-reader --request FILE --response FILE\\n'\n"
            };
            fs::write(&reader, body).unwrap();
            fs::set_permissions(&reader, fs::Permissions::from_mode(0o700)).unwrap();
        }
        let body = format!(
            "automation=(\"$AUTOMATION\")\nSKIP_DATASET={}\n{gate}\nprintf downloaded > \"$ROOT/download-marker\"",
            usize::from(mode == "skip")
        );
        let (ok, _, error) = invoke(
            temporary.path(),
            &body,
            &[(
                "MESH_LLM_TRAJECTORY_READER_BIN",
                reader.to_string_lossy().into_owned(),
            )],
        );
        assert_eq!(ok, matches!(mode, "success" | "skip"), "{mode}: {error}");
        assert_eq!(temporary.path().join("download-marker").exists(), ok);
        temporary.close().unwrap();
    }
}
