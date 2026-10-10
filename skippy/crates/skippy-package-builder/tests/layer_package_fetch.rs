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
