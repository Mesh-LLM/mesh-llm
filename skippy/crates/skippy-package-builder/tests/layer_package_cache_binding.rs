use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
const COMMIT: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
fn fixture(home: &Path, revision: &str) -> PathBuf {
    let repo = home.join("hub/models--org--repo");
    let snapshot = repo.join("snapshots").join(COMMIT);
    fs::create_dir_all(&snapshot).unwrap();
    fs::write(
        snapshot.join("model-package.json"),
        br#"{"layer_count":2,"activation_width":4096,"model_id":"fixture"}"#,
    )
    .unwrap();
    let reference = repo.join("refs").join(revision);
    fs::create_dir_all(reference.parent().unwrap()).unwrap();
    fs::write(reference, COMMIT).unwrap();
    fs::create_dir_all(repo.join("snapshots/ffffffffffffffffffffffffffffffffffffffff")).unwrap();
    snapshot
}
fn builder() -> &'static str {
    env!("CARGO_BIN_EXE_skippy-package-builder")
}
fn command(home: &Path, revision: &str) -> Command {
    let mut cmd = Command::new(builder());
    cmd.args(["resolve-layer-package-cache", "--reference"])
        .arg(format!("hf://org/repo@{revision}"))
        .arg("--cache-root")
        .arg(home.join("hub"));
    cmd
}
#[test]
fn actual_cli_returns_exact_snapshot_and_missing_or_malformed_ref_has_no_output() {
    let home = tempfile::tempdir().unwrap();
    let snapshot = fixture(home.path(), "release/étape#v2%");
    let success = command(home.path(), "release/étape#v2%").output().unwrap();
    assert!(
        success.status.success(),
        "{}",
        String::from_utf8_lossy(&success.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&success.stdout).unwrap();
    assert_eq!(report["commit"], COMMIT);
    assert_eq!(report["requested_revision"], "release/étape#v2%");
    assert_eq!(
        report["snapshot_path"],
        snapshot.canonicalize().unwrap().to_str().unwrap()
    );
    for revision in ["missing", "malformed"] {
        if revision == "malformed" {
            fs::write(
                home.path().join("hub/models--org--repo/refs/malformed"),
                "../foreign",
            )
            .unwrap();
        }
        let result = command(home.path(), revision).output().unwrap();
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
#[test]
fn both_actual_wan_cache_helpers_preserve_home_hub_pairing_and_failure_status() {
    use std::os::unix::fs::PermissionsExt as _;
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap();
    for host in [false, true] {
        let source = fs::read_to_string(root.join(if host {
            "skippy/evals/wan-lab/up.sh"
        } else {
            "skippy/evals/wan-lab/entrypoint.sh"
        }))
        .unwrap();
        let helper = function(
            &source,
            if host {
                "hf_snapshot_dir"
            } else {
                "hf_cache_snapshot_dir"
            },
        );
        let helper = helper.replace("/usr/local/bin/skippy-package-builder", "\"$REAL_BUILDER\"");
        for valid in [false, true] {
            let home = tempfile::tempdir().unwrap();
            let snapshot = fixture(home.path(), "release=v2");
            let just = home.path().join("just");
            fs::write(&just, "#!/usr/bin/env bash\nset -euo pipefail\n[[ $# == 7 && $1 == --justfile && $3 == skippy-layer-package-cache ]]\nshift 3\nexec \"$REAL_BUILDER\" resolve-layer-package-cache \"$@\"\n").unwrap();
            fs::set_permissions(&just, fs::Permissions::from_mode(0o755)).unwrap();
            let invoke = if host {
                "hf_snapshot_dir org/repo \"$REVISION\" \"$HOME_ROOT\""
            } else {
                "hf_cache_snapshot_dir org/repo \"$REVISION\""
            };
            let script = format!(
                "set -euo pipefail\n{helper}\nROOT=unused\nresult=\"$({invoke})\" || exit $?\nprintf '%s\\n' \"$result\"\n"
            );
            let bash = if Path::new("/opt/homebrew/bin/bash").exists() {
                "/opt/homebrew/bin/bash"
            } else {
                "bash"
            };
            let result = Command::new(bash)
                .args(["-c", &script])
                .env("REAL_BUILDER", builder())
                .env("HOME_ROOT", home.path())
                .env("HF_CACHE_ROOT", home.path())
                .env("REVISION", if valid { "release=v2" } else { "missing" })
                .env(
                    "PATH",
                    format!(
                        "{}:{}",
                        home.path().display(),
                        std::env::var("PATH").unwrap()
                    ),
                )
                .output()
                .unwrap();
            assert_eq!(
                result.status.success(),
                valid,
                "host={host}: {}",
                String::from_utf8_lossy(&result.stderr)
            );
            if valid {
                assert_eq!(
                    String::from_utf8(result.stdout).unwrap().trim(),
                    snapshot.canonicalize().unwrap().to_str().unwrap()
                );
            } else {
                assert!(result.stdout.is_empty());
            }
        }
    }
}
#[cfg(unix)]
#[test]
fn unavailable_revision_stops_actual_host_cache_preparation_before_stage_checks() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)
        .unwrap();
    let source = fs::read_to_string(root.join("skippy/evals/wan-lab/entrypoint.sh")).unwrap();
    let functions = [
        "hf_cache_snapshot_dir",
        "prepare_hf_layer_package_from_host_cache",
    ]
    .map(|name| function(&source, name))
    .join("\n")
    .replace("/usr/local/bin/skippy-package-builder", "\"$REAL_BUILDER\"");
    let home = tempfile::tempdir().unwrap();
    fixture(home.path(), "main");
    let marker = home.path().join("stage-checks");
    let script = format!(
        "set -euo pipefail\n{functions}\nlog() {{ :; }}\n\
        even_stage_range() {{ touch \"$MARKER\"; echo '0 1'; }}\n\
        prepare_hf_layer_package_from_host_cache org/repo missing hf://org/repo@missing 0 1\n"
    );
    let result = Command::new("bash")
        .args(["-c", &script])
        .env("REAL_BUILDER", builder())
        .env("HF_CACHE_ROOT", home.path())
        .env("MARKER", &marker)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!marker.exists());
    assert!(result.stdout.is_empty());
}
#[cfg(unix)]
#[test]
fn no_writer_fifo_ref_refuses_with_finite_child_cleanup() {
    use std::{
        process::Stdio,
        time::{Duration, Instant},
    };
    let home = tempfile::tempdir().unwrap();
    fixture(home.path(), "main");
    let fifo = home.path().join("hub/models--org--repo/refs/fifo");
    assert!(Command::new("mkfifo").arg(fifo).status().unwrap().success());
    let stdout = home.path().join("stdout");
    let mut child = command(home.path(), "fifo")
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
            panic!("cache ref FIFO admission blocked");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    assert!(!status.success());
    assert!(fs::read(stdout).unwrap().is_empty());
}
