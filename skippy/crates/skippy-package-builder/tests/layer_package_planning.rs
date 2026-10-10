//! Actual product declaration planner; payloads need not be downloaded.
#[path = "../src/layer_package_inspection/fixtures.rs"]
mod fixtures;
use std::{fs, path::Path, process::Command};
fn builder() -> &'static str {
    env!("CARGO_BIN_EXE_skippy-package-builder")
}
fn plan(manifest: &Path) -> std::process::Output {
    Command::new(builder())
        .args(["plan-layer-package-artifacts", "--manifest"])
        .arg(manifest)
        .args([
            "--stage-index",
            "0",
            "--stage-count",
            "1",
            "--layer-start",
            "0",
            "--layer-end",
            "2",
        ])
        .output()
        .unwrap()
}
#[test]
fn real_cli_plans_without_payloads_and_refuses_unsafe_duplicate_declarations() {
    let temp = tempfile::tempdir().unwrap();
    let good = fixtures::fixture(temp.path());
    for name in [
        "metadata.gguf",
        "embeddings.gguf",
        "output.gguf",
        "projector.gguf",
    ] {
        fs::remove_file(temp.path().join(name)).unwrap();
    }
    fs::remove_dir_all(temp.path().join("layers")).unwrap();
    let manifest = temp.path().join("model-package.json");
    let result = plan(&manifest);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(String::from_utf8(result.stdout).unwrap().lines().count(), 6);
    for path in ["../escape", "metadata.gguf", "layers/x\ny"] {
        let mut bad = good.clone();
        bad["layers"][1]["path"] = serde_json::json!(path);
        fixtures::save(temp.path(), &bad);
        let result = plan(&manifest);
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
    }
    let result = Command::new(builder())
        .args(["even-layer-stage-range", "1", "3", "8"])
        .output()
        .unwrap();
    assert!(result.status.success());
    assert_eq!(result.stdout, b"3 6\n");
    for args in [
        ["0", "0", "8"],
        ["0", "9", "8"],
        ["3", "3", "8"],
        ["-1", "3", "8"],
    ] {
        let result = Command::new(builder())
            .arg("even-layer-stage-range")
            .args(args)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(result.stdout.is_empty());
    }
}
#[cfg(unix)]
fn function(source: &str, name: &str) -> String {
    let start = source.find(&format!("{name}() {{")).unwrap();
    let end = source[start..].find("\n}\n").unwrap() + start + 3;
    source[start..end].to_owned()
}
#[cfg(unix)]
mod fetch_server;
#[cfg(unix)]
#[test]
fn actual_remote_preparation_refuses_bad_ranges_and_rosters_before_artifact_downloads() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap();
    let source = fs::read_to_string(root.join("skippy/evals/wan-lab/entrypoint.sh")).unwrap();
    let functions = ["parse_hf_package_ref", "prepare_hf_layer_package"]
        .map(|name| function(&source, name))
        .join("\n")
        .replace("/usr/local/bin/skippy-package-builder", "\"$REAL_BUILDER\"");
    for case in ["valid", "bad-range", "bad-manifest"] {
        let payload = tempfile::tempdir().unwrap();
        let mut manifest = fixtures::fixture(payload.path());
        // Keep both layer HTTP transfers observable. The native Hub correctly reuses
        // one content-addressed blob when the default fixture layers have equal bytes.
        let layer_one = b"distinct-layer-one";
        fs::write(payload.path().join("layers/1.gguf"), layer_one).unwrap();
        use sha2::{Digest as _, Sha256};
        let digest: String = Sha256::digest(layer_one)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        manifest["layers"][1]["artifact_bytes"] = serde_json::json!(layer_one.len());
        manifest["layers"][1]["sha256"] = serde_json::json!(digest);
        fixtures::save(payload.path(), &manifest);
        if case == "bad-manifest" {
            manifest["projectors"][0]["path"] = serde_json::json!("../escape");
            fixtures::save(payload.path(), &manifest);
        }
        let server = fetch_server::Server::start(payload.path(), "success");
        let cache = tempfile::tempdir().unwrap();
        let script = format!(
            "set -euo pipefail\n{functions}\nlog() {{ :; }}\nprepare_hf_layer_package hf://org/repo 0 \"$STAGES\"\nprintf '%s\\n' \"$MODEL_PATH\"\n"
        );
        let bash = if Path::new("/opt/homebrew/bin/bash").exists() {
            "/opt/homebrew/bin/bash"
        } else {
            "bash"
        };
        let result = Command::new(bash)
            .args(["-c", &script])
            .env("REAL_BUILDER", builder())
            .env("HF_PACKAGE_SOURCE", "download")
            .env("PACKAGE_CACHE_DIR", cache.path())
            .env("MODEL_ID", "existing-model")
            .env("HF_ENDPOINT", &server.endpoint)
            .env("MESH_HF_RETRY_MAX_ATTEMPTS", "0")
            .env("STAGES", if case == "bad-range" { "3" } else { "1" })
            .output()
            .unwrap();
        assert_eq!(
            result.status.success(),
            case == "valid",
            "{case}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let requests = server.requests.lock().unwrap();
        let artifact_gets = requests
            .iter()
            .filter(|line| {
                line.starts_with("GET /org/repo/resolve/") && !line.contains("model-package.json")
            })
            .count();
        assert_eq!(artifact_gets, if case == "valid" { 6 } else { 0 });
        if case == "valid" {
            assert!(String::from_utf8_lossy(&result.stdout).contains(&format!(
                "/hub/models--org--repo/snapshots/{}",
                "a".repeat(40)
            )));
        }
    }
}

#[cfg(unix)]
#[test]
fn fifo_without_writer_refuses_finitely_and_regular_blob_symlink_remains_valid() {
    use std::{
        os::unix::fs::symlink,
        process::Stdio,
        time::{Duration, Instant},
    };
    let temp = tempfile::tempdir().unwrap();
    fixtures::fixture(temp.path());
    let manifest = temp.path().join("model-package.json");
    let link = temp.path().join("blob-pointer.json");
    symlink(&manifest, &link).unwrap();
    let valid = plan(&link);
    assert!(
        valid.status.success(),
        "{}",
        String::from_utf8_lossy(&valid.stderr)
    );
    let fifo = temp.path().join("no-writer.fifo");
    assert!(
        Command::new("mkfifo")
            .arg(&fifo)
            .status()
            .unwrap()
            .success()
    );
    let stdout = temp.path().join("stdout");
    let mut child = Command::new(builder())
        .args(["plan-layer-package-artifacts", "--manifest"])
        .arg(&fifo)
        .args([
            "--stage-index",
            "0",
            "--stage-count",
            "1",
            "--layer-start",
            "0",
            "--layer-end",
            "2",
        ])
        .stdout(fs::File::create(&stdout).unwrap())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= deadline {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("manifest FIFO admission blocked before regular-file refusal");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    assert!(!status.success());
    assert!(fs::read(stdout).unwrap().is_empty());
}
