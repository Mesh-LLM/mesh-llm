//! Offline HTTP acquisition fixtures. No native inference, paid Hub or model data.
mod fetch_server;
#[path = "../src/layer_package_inspection/fixtures.rs"]
mod fixtures;
use serde_json::Value;
use std::{fs, path::Path, process::Command};
fn fetch(cache: &Path, endpoint: &str, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_skippy-package-builder"))
        .args([
            "fetch-layer-package",
            "--reference",
            "hf://org/repo@release/étape#v2",
            "--cache-root",
        ])
        .arg(cache)
        .args(["--timeout-secs", "5"])
        .args(args)
        .env("HF_ENDPOINT", endpoint)
        .env("MESH_HF_RETRY_MAX_ATTEMPTS", "0")
        .env_remove("HF_TOKEN")
        .output()
        .unwrap()
}
#[test]
fn moving_requested_ref_resolves_once_and_all_full_parts_use_that_commit() {
    let payload = tempfile::tempdir().unwrap();
    fixtures::fixture(payload.path());
    let server = fetch_server::Server::start(payload.path(), "moving-branch");
    let cache = tempfile::tempdir().unwrap();
    let result = fetch(cache.path(), &server.endpoint, &[]);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["commit"], "a".repeat(40));
    assert_eq!(report["requested_revision"], "release/étape#v2");
    assert_eq!(report["artifacts_verified"], 6);
    let requests = server.requests.lock().unwrap();
    assert_eq!(
        requests
            .iter()
            .filter(|line| line.contains("/api/models/"))
            .count(),
        1
    );
    assert!(
        requests
            .iter()
            .any(|line| line.contains("release%2F%C3%A9tape%23v2"))
    );
    assert!(
        requests
            .iter()
            .filter(|line| line.contains("/resolve/"))
            .all(|line| line.contains(&format!("/resolve/{}/", "a".repeat(40))))
    );
}
#[test]
fn selected_stage_fetches_metadata_head_projector_and_selected_layers_only() {
    let payload = tempfile::tempdir().unwrap();
    fixtures::fixture(payload.path());
    let server = fetch_server::Server::start(payload.path(), "success");
    let cache = tempfile::tempdir().unwrap();
    let result = fetch(
        cache.path(),
        &server.endpoint,
        &["--stage-index", "0", "--stage-count", "2"],
    );
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let report: Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(report["layer_start"], 0);
    assert_eq!(report["layer_end"], 1);
    let requests = server.requests.lock().unwrap();
    assert!(
        requests
            .iter()
            .any(|line| line.ends_with("/projector.gguf HTTP/1.1"))
    );
    assert!(
        !requests
            .iter()
            .any(|line| line.ends_with("/output.gguf HTTP/1.1")
                || line.ends_with("/layers/1.gguf HTTP/1.1"))
    );
}
#[test]
fn wrong_or_malformed_commit_headers_refuse_before_blob_writes() {
    for mode in ["wrong-commit", "malformed-commit"] {
        let payload = tempfile::tempdir().unwrap();
        fixtures::fixture(payload.path());
        let server = fetch_server::Server::start(payload.path(), mode);
        let cache = tempfile::tempdir().unwrap();
        let result = fetch(cache.path(), &server.endpoint, &[]);
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
        assert_eq!(
            fs::read_dir(cache.path().join("models--org--repo/blobs"))
                .unwrap()
                .count(),
            0
        );
        assert!(
            !server
                .requests
                .lock()
                .unwrap()
                .iter()
                .any(|line| line.starts_with("GET /org/repo/resolve/"))
        );
    }
}
#[test]
fn integrity_geometry_and_unsafe_roster_refuse_without_success_output() {
    for mode in [
        "bad-size",
        "bad-digest",
        "missing-projector",
        "unsafe-manifest",
        "wrong-geometry",
        "metadata-stall",
    ] {
        let payload = tempfile::tempdir().unwrap();
        let mut manifest = fixtures::fixture(payload.path());
        if mode == "unsafe-manifest" {
            manifest["layers"][1]["path"] = serde_json::json!("../outside");
            fixtures::save(payload.path(), &manifest);
        }
        let server = fetch_server::Server::start(payload.path(), mode);
        let cache = tempfile::tempdir().unwrap();
        let extra = if mode == "wrong-geometry" {
            vec!["--expected-layer-count", "3"]
        } else {
            vec![]
        };
        let result = fetch(cache.path(), &server.endpoint, &extra);
        assert!(!result.status.success(), "{mode}");
        assert!(result.stdout.is_empty());
        if matches!(mode, "unsafe-manifest" | "wrong-geometry") {
            assert!(
                server
                    .requests
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|line| line.contains("/resolve/"))
                    .all(|line| line.contains("model-package.json"))
            );
        }
    }
}

#[cfg(unix)]
fn function(source: &str, name: &str) -> String {
    let start = source.find(&format!("{name}() {{")).unwrap();
    let end = source[start..].find("\n}\n").unwrap() + start + 3;
    source[start..end].into()
}
#[cfg(unix)]
#[test]
fn actual_host_caller_exports_exact_commit_and_actual_hub_mount_then_reuses_offline() {
    use std::os::unix::fs::PermissionsExt as _;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap();
    let source = fs::read_to_string(root.join("skippy/evals/wan-lab/up.sh")).unwrap();
    let functions = [
        "parse_hf_package_ref",
        "hf_home_dir",
        "verify_package_cache",
        "ensure_hf_package",
    ]
    .map(|name| function(&source, name))
    .join("\n");
    let payload = tempfile::tempdir().unwrap();
    fixtures::fixture(payload.path());
    let server = fetch_server::Server::start(payload.path(), "success");
    let cache = tempfile::tempdir().unwrap();
    let just = cache.path().join("just");
    fs::write(&just,"#!/usr/bin/env bash\nset -euo pipefail\n[[ $1 == --justfile ]]\nrecipe=$3; shift 3\ncase $recipe in skippy-package-reference) command=parse-package-reference;; skippy-layer-package-cache) command=resolve-layer-package-cache;; skippy-layer-package-fetch) command=fetch-layer-package;; skippy-layer-package-inspect) command=inspect-layer-package;; *) exit 64;; esac\nexec \"$REAL_BUILDER\" \"$command\" \"$@\"\n").unwrap();
    fs::set_permissions(&just, fs::Permissions::from_mode(0o755)).unwrap();
    let script = format!(
        "set -euo pipefail\n{functions}\nROOT=unused\nlog() {{ :; }}\nensure_hf_package\nprintf '%s\\n%s\\n' \"$MODEL_PACKAGE_REF\" \"$HF_CACHE_MOUNT\"\n"
    );
    let run = |reference: &str, endpoint: &str| {
        Command::new("bash")
            .args(["-c", &script])
            .env("REAL_BUILDER", env!("CARGO_BIN_EXE_skippy-package-builder"))
            .env("MODEL_PACKAGE_REF", reference)
            .env("HF_HOME", cache.path().join("unrelated-home"))
            .env("HF_HUB_CACHE", cache.path().join("actual-hub"))
            .env("HF_ENDPOINT", endpoint)
            .env("MESH_HF_RETRY_MAX_ATTEMPTS", "0")
            .env(
                "PATH",
                format!(
                    "{}:{}",
                    cache.path().display(),
                    std::env::var("PATH").unwrap()
                ),
            )
            .output()
            .unwrap()
    };
    let online = run("hf://org/repo@release/étape#v2", &server.endpoint);
    assert!(
        online.status.success(),
        "{}",
        String::from_utf8_lossy(&online.stderr)
    );
    let expected = format!(
        "hf://org/repo@{}\n{}\n",
        "a".repeat(40),
        cache.path().join("actual-hub").display()
    );
    assert_eq!(String::from_utf8(online.stdout).unwrap(), expected);
    drop(server);
    let offline = run(
        &format!("hf://org/repo@{}", "a".repeat(40)),
        "http://127.0.0.1:1",
    );
    assert!(
        offline.status.success(),
        "{}",
        String::from_utf8_lossy(&offline.stderr)
    );
    assert_eq!(String::from_utf8(offline.stdout).unwrap(), expected);
}

#[test]
fn actual_cli_overall_deadline_kills_and_reaps_pending_native_worker_without_output() {
    let payload = tempfile::tempdir().unwrap();
    fixtures::fixture(payload.path());
    let server = fetch_server::Server::start(payload.path(), "hard-timeout");
    let cache = tempfile::tempdir().unwrap();
    let started = std::time::Instant::now();
    let result = Command::new(env!("CARGO_BIN_EXE_skippy-package-builder"))
        .args([
            "fetch-layer-package",
            "--reference",
            "hf://org/repo@main",
            "--cache-root",
        ])
        .arg(cache.path())
        .args(["--timeout-secs", "2"])
        .env("HF_ENDPOINT", &server.endpoint)
        .env("MESH_HF_RETRY_MAX_ATTEMPTS", "0")
        .output()
        .unwrap();
    assert!(started.elapsed() < std::time::Duration::from_secs(4));
    assert!(!result.status.success());
    assert!(result.stdout.is_empty());
    let error = String::from_utf8(result.stderr).unwrap();
    assert!(
        error.contains("owned acquisition worker deadline expired"),
        "{error}"
    );
    assert!(error.contains("reaped=true"), "{error}");
    assert_eq!(server.requests.lock().unwrap().len(), 1);
    #[cfg(unix)]
    {
        let pid: libc::pid_t = error
            .split("worker_pid=")
            .nth(1)
            .unwrap()
            .split(';')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        // SAFETY: signal zero tests existence of the already reaped owned worker.
        assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
    }
}

#[cfg(unix)]
#[test]
fn actual_container_download_refuses_configured_geometry_before_exports_and_serving() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap();
    let source = fs::read_to_string(root.join("skippy/evals/wan-lab/entrypoint.sh")).unwrap();
    let functions = ["parse_hf_package_ref", "prepare_hf_layer_package"]
        .map(|name| function(&source, name))
        .join("\n")
        .replace("/usr/local/bin/skippy-package-builder", "\"$REAL_BUILDER\"");
    for case in ["valid", "wrong-layers", "wrong-width"] {
        let payload = tempfile::tempdir().unwrap();
        fixtures::fixture(payload.path());
        let server = fetch_server::Server::start(payload.path(), "success");
        let cache = tempfile::tempdir().unwrap();
        let journal = cache.path().join("environment");
        let serving = cache.path().join("serving");
        let script = format!(
            "set -euo pipefail\n{functions}\nlog() {{ :; }}\ntrap 'printf \"%s\\n%s\\n%s\\n\" \"$MODEL_PATH\" \"$LOAD_MODE\" \"$LAYER_START\" > \"$JOURNAL\"' EXIT\nprepare_hf_layer_package hf://org/repo 0 1 || exit $?\ntouch \"$SERVING\"\n"
        );
        let bash = if Path::new("/opt/homebrew/bin/bash").exists() {
            "/opt/homebrew/bin/bash"
        } else {
            "bash"
        };
        let result = Command::new(bash)
            .args(["-c", &script])
            .env("REAL_BUILDER", env!("CARGO_BIN_EXE_skippy-package-builder"))
            .env("HF_PACKAGE_SOURCE", "download")
            .env("HF_ENDPOINT", &server.endpoint)
            .env("MESH_HF_RETRY_MAX_ATTEMPTS", "0")
            .env("PACKAGE_CACHE_DIR", cache.path())
            .env(
                "LAYER_COUNT",
                if case == "wrong-layers" { "3" } else { "2" },
            )
            .env(
                "ACTIVATION_WIDTH",
                if case == "wrong-width" {
                    "4095"
                } else {
                    "4096"
                },
            )
            .env("MODEL_PATH", "original-path")
            .env("LOAD_MODE", "original-mode")
            .env("LAYER_START", "original-start")
            .env("MODEL_ID", "existing-model")
            .env("JOURNAL", &journal)
            .env("SERVING", &serving)
            .output()
            .unwrap();
        assert_eq!(
            result.status.success(),
            case == "valid",
            "{case}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(serving.exists(), case == "valid");
        if case != "valid" {
            assert!(result.stdout.is_empty());
            assert_eq!(
                fs::read_to_string(&journal).unwrap(),
                "original-path\noriginal-mode\noriginal-start\n"
            );
            assert!(
                server
                    .requests
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|line| line.contains("/resolve/"))
                    .all(|line| line.contains("model-package.json"))
            );
        } else {
            let state = fs::read_to_string(&journal).unwrap();
            assert!(state.contains(&format!(
                "/hub/models--org--repo/snapshots/{}",
                "a".repeat(40)
            )));
            assert!(state.ends_with("\nlayer-package\n0\n"));
        }
    }
}
