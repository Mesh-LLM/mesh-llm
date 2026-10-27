#[path = "../src/snapshot_promotion/fixtures.rs"]
mod fixtures;
#[path = "../src/snapshot_promotion/hub_fixture.rs"]
mod hub_fixture;

use hub_fixture::Server;
use std::{
    fs,
    io::{self, Read},
    process::{Command, Output, Stdio},
    sync::mpsc,
    thread,
    time::{Duration, Instant},
};

fn cli(endpoint: &str, args: &[&str]) -> Output {
    let home = tempfile::tempdir().unwrap();
    let child = Command::new(env!("CARGO_BIN_EXE_promote-layer-package-snapshot"))
        .env_clear()
        .envs(child_env::ca_certificates())
        .env("HF_ENDPOINT", endpoint)
        .env("HF_HOME", home.path())
        .env("HOME", home.path())
        .env("HF_TOKEN", "fixture-token")
        .args(args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    finish_fixture_child(child)
}

fn finish_fixture_child(child: std::process::Child) -> Output {
    finish_fixture_child_with_deadline(child, Duration::from_secs(30))
}

const CAPTURE_LIMIT: u64 = 16 * 1024 * 1024;

fn capture_stream<R: Read + Send + 'static>(
    pipe: Option<R>,
) -> mpsc::Receiver<io::Result<Vec<u8>>> {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let result = (|| {
            let mut bytes = Vec::new();
            if let Some(mut pipe) = pipe {
                pipe.by_ref()
                    .take(CAPTURE_LIMIT + 1)
                    .read_to_end(&mut bytes)?;
                // Keep draining after overflow so the owned child can exit; the
                // parent rejects the oversized capture instead of certifying it.
                if bytes.len() as u64 > CAPTURE_LIMIT {
                    io::copy(&mut pipe, &mut io::sink())?;
                }
            }
            Ok(bytes)
        })();
        let _ = sender.send(result);
    });
    receiver
}

fn finish_fixture_child_with_deadline(mut child: std::process::Child, budget: Duration) -> Output {
    let stdout = capture_stream(child.stdout.take());
    let stderr = capture_stream(child.stderr.take());
    let deadline = Instant::now() + budget;
    let mut timed_out = false;
    while child.try_wait().unwrap().is_none() {
        if Instant::now() >= deadline {
            child.kill().unwrap();
            timed_out = true;
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }
    let status = child.wait().unwrap();
    let collect = |receiver: mpsc::Receiver<io::Result<Vec<u8>>>| {
        let bytes = receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("fixture child output reader did not finish")
            .expect("fixture child output read failed");
        assert!(
            bytes.len() as u64 <= CAPTURE_LIMIT,
            "fixture child output exceeds capture limit"
        );
        bytes
    };
    let output = Output {
        status,
        stdout: collect(stdout),
        stderr: collect(stderr),
    };
    assert!(
        !timed_out,
        "snapshot CLI timed out: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

#[test]
#[ignore = "subprocess-only output fixture"]
fn large_output_fixture() {
    use std::io::Write;
    assert_eq!(
        std::env::var("MESH_PACKAGE_OUTPUT_FIXTURE").as_deref(),
        Ok("1")
    );
    std::io::stdout()
        .write_all(&vec![b'@'; 1024 * 1024])
        .unwrap();
    std::io::stderr()
        .write_all(&vec![b'%'; 1024 * 1024])
        .unwrap();
}

#[test]
fn fixture_child_drains_both_streams_before_waiting_for_exit() {
    let child = Command::new(std::env::current_exe().unwrap())
        .env_clear()
        .env("MESH_PACKAGE_OUTPUT_FIXTURE", "1")
        .args([
            "--exact",
            "large_output_fixture",
            "--ignored",
            "--nocapture",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let output = finish_fixture_child_with_deadline(child, Duration::from_secs(5));
    assert!(output.status.success());
    assert_eq!(
        output.stdout.iter().filter(|byte| **byte == b'@').count(),
        1024 * 1024
    );
    assert_eq!(
        output.stderr.iter().filter(|byte| **byte == b'%').count(),
        1024 * 1024
    );
}

fn info() -> Vec<u8> {
    serde_json::to_vec(&serde_json::json!({"id":"fixture/model","sha":"a".repeat(40)})).unwrap()
}

#[test]
fn prepare_preview_reads_main_but_never_creates_branch() {
    let server = Server::start(vec![(200, info())]);
    let output = cli(
        &server.endpoint,
        &[
            "prepare",
            "--repo",
            "fixture/model",
            "--source-revision",
            &"b".repeat(40),
            "--token",
            "run",
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let plan: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(plan["parent_commit"], "a".repeat(40));
    assert_eq!(
        plan["staging_revision"],
        "automation/republish-bbbbbbbbbbbb-run"
    );
    let requests = server.finish();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].starts_with(b"GET "));
}

#[test]
fn confirmed_prepare_emits_exact_consumed_lines_and_branches_from_parent() {
    let server = Server::start(vec![(200, info()), (200, b"{}".to_vec())]);
    let output = cli(
        &server.endpoint,
        &[
            "prepare",
            "--confirm",
            "--repo",
            "fixture/model",
            "--source-revision",
            &"b".repeat(40),
            "--token",
            "run",
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!(
            "automation/republish-bbbbbbbbbbbb-run\n{}\n",
            "a".repeat(40)
        )
    );
    let requests = server.finish();
    assert_eq!(requests.len(), 2);
    let branch = String::from_utf8_lossy(&requests[1]);
    assert!(branch.starts_with("POST "));
    assert!(branch.contains(&format!("\"startingPoint\":\"{}\"", "a".repeat(40))));
}

#[test]
fn promote_preview_validates_local_root_without_network_or_mutation() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model-package.json");
    fs::write(
        &path,
        fixtures::manifest("weights.gguf", 999, "b".repeat(64)),
    )
    .unwrap();
    // No listener: successful preview cannot have made an HTTP request.
    let output = cli(
        "http://127.0.0.1:1",
        &[
            "promote",
            "--repo",
            "fixture/model",
            "--manifest",
            path.to_str().unwrap(),
            "--staging-revision",
            "automation/republish-test",
            "--parent-commit",
            &"a".repeat(40),
        ],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let plan: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        plan["paths"],
        serde_json::json!(["weights.gguf", "model-package.json"])
    );
    fs::write(&path, b"{}").unwrap();
    let invalid = cli(
        "http://127.0.0.1:1",
        &[
            "promote",
            "--confirm",
            "--repo",
            "fixture/model",
            "--manifest",
            path.to_str().unwrap(),
            "--staging-revision",
            "automation/republish-test",
            "--parent-commit",
            &"a".repeat(40),
        ],
    );
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("snapshot package root"));
}

#[test]
fn confirmed_promote_commits_once_then_cleans_staging_and_warns_on_cleanup_failure() {
    for cleanup_status in [200, 403] {
        let directory = tempfile::tempdir().unwrap();
        let manifest = fixtures::manifest("weights.gguf", 999, "b".repeat(64));
        let path = directory.path().join("model-package.json");
        fs::write(&path, &manifest).unwrap();
        let paths = serde_json::json!([
            {"type":"file","oid":"e".repeat(40),"path":"weights.gguf","size":999,
                "lfs":{"size":999,"sha256":"b".repeat(64),"pointerSize":120}},
            {"type":"file","oid":"f".repeat(40),"path":"model-package.json","size":manifest.len()}
        ]);
        let server = Server::start(vec![
            (200, info()),
            (200, serde_json::to_vec(&paths).unwrap()),
            (200, manifest),
            (
                200,
                serde_json::to_vec(&serde_json::json!({"commitOid":"c".repeat(40)})).unwrap(),
            ),
            (cleanup_status, b"{}".to_vec()),
        ]);
        let output = cli(
            &server.endpoint,
            &[
                "promote",
                "--confirm",
                "--repo",
                "fixture/model",
                "--manifest",
                path.to_str().unwrap(),
                "--staging-revision",
                "automation/republish-test",
                "--parent-commit",
                &"a".repeat(40),
            ],
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stderr)
                .contains("promotion succeeded, staging cleanup failed"),
            cleanup_status == 403
        );
        let requests = server.finish();
        assert_eq!(requests.len(), 5);
        assert!(requests[3].starts_with(b"POST /api/models/fixture/model/commit/main "));
        assert!(requests[4].starts_with(b"DELETE /api/models/fixture/model/branch/"));
    }
}

#[cfg(unix)]
#[test]
fn embedded_prepare_stops_on_failed_child_even_with_two_plausible_state_lines() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let promoter = directory.path().join("failed-promoter");
    let arguments = directory.path().join("prepare arguments");
    fs::write(
        &promoter,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$ARGUMENT_CAPTURE\"\nprintf '%s\\n' 'automation/republish-fixture' '{}'\nexit 7\n",
            "a".repeat(40)
        ),
    )
    .unwrap();
    fs::set_permissions(&promoter, fs::Permissions::from_mode(0o755)).unwrap();
    let source = include_str!("../src/scripts/split-model-job.sh");
    let start = source.find("TARGET_UPLOAD_REVISION=\"main\"").unwrap();
    let end = source[start..]
        .find("export TARGET_UPLOAD_REVISION TARGET_MAIN_PARENT")
        .unwrap()
        + start;
    let script = format!(
        "set -euo pipefail\n{}\nprintf 'CALLER_CONTINUED\\n'\n",
        &source[start..end]
    );
    let child = Command::new(child_env::bash())
        .env_clear()
        .env("PATH", child_env::path())
        .env("REPUBLISH", "true")
        .env("SNAPSHOT_PROMOTER", promoter)
        .env("TARGET_REPO", "fixture/model")
        .env("SOURCE_REVISION", "b".repeat(40))
        .env("ARGUMENT_CAPTURE", &arguments)
        .args(["-c", &script])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let pid = child.id();
    let output = finish_fixture_child(child);
    assert_eq!(output.status.code(), Some(7));
    assert!(!String::from_utf8_lossy(&output.stdout).contains("CALLER_CONTINUED"));
    let captured = fs::read_to_string(arguments).unwrap();
    let words = captured.lines().collect::<Vec<_>>();
    assert!(words.contains(&"--confirm"));
    let nonce = words.windows(2).find(|pair| pair[0] == "--token").unwrap()[1];
    let (timestamp, nonce_pid) = nonce.split_once('-').unwrap();
    assert_eq!(timestamp.len(), 14);
    assert!(timestamp.bytes().all(|byte| byte.is_ascii_digit()));
    assert_eq!(
        nonce_pid,
        pid.to_string(),
        "staging nonce must retain the calling job PID"
    );
}

#[cfg(unix)]
#[test]
fn embedded_native_helper_survives_build_workspace_cleanup() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    let source = include_str!("../src/scripts/split-model-job.sh");
    assert!(
        source.find("just bootstrap-build-tools").unwrap()
            < source.find("just snapshot-promoter-release-build").unwrap()
    );
    let start = source
        .find("just snapshot-promoter-release-build\n")
        .unwrap();
    let end = source[start..]
        .find("SLICER=\"${TOOL_DIR}/skippy-package-builder\"")
        .unwrap()
        + start;
    let target = root.join("target");
    let tool_dir = root.join("tools");
    let fixtures = root.join("bin");
    fs::create_dir_all(target.join("release")).unwrap();
    fs::create_dir_all(&tool_dir).unwrap();
    fs::create_dir_all(&fixtures).unwrap();
    let original = env!("CARGO_BIN_EXE_promote-layer-package-snapshot");
    fs::copy(
        original,
        target.join("release/promote-layer-package-snapshot"),
    )
    .unwrap();
    let layer_helper = env!("CARGO_BIN_EXE_model-package-layer-job");
    fs::copy(layer_helper, target.join("release/model-package-layer-job")).unwrap();
    let just = fixtures.join("just");
    fs::write(
        &just,
        "#!/bin/sh\n[ \"$#\" -eq 1 ] || exit 9\ncase \"$1\" in snapshot-promoter-release-build|layer-job-helper-release-build) ;; *) exit 9 ;; esac\n",
    )
    .unwrap();
    fs::set_permissions(&just, fs::Permissions::from_mode(0o755)).unwrap();
    let cleanup = source
        .lines()
        .find(|line| {
            *line == "rm -rf \"$BUILD_DIR\" \"$CARGO_TARGET_DIR\" \"$CARGO_HOME\" \"$RUSTUP_HOME\""
        })
        .unwrap();
    let script = format!(
        "set -euo pipefail\n{}\n{}\n\"$SNAPSHOT_PROMOTER\" --help\n\"$LAYER_JOB\" --help\n",
        &source[start..end],
        cleanup
    );
    let output = Command::new(child_env::bash())
        .env_clear()
        .env("PATH", child_env::path_with(&fixtures))
        .env("CARGO_TARGET_DIR", &target)
        .env("TOOL_DIR", &tool_dir)
        .env(
            "SNAPSHOT_PROMOTER",
            tool_dir.join("promote-layer-package-snapshot"),
        )
        .env("BUILD_DIR", root.join("build"))
        .env("CARGO_HOME", root.join("cargo"))
        .env("RUSTUP_HOME", root.join("rustup"))
        .args(["-c", &script])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!target.exists());
    assert_eq!(
        fs::read(original).unwrap(),
        fs::read(tool_dir.join("promote-layer-package-snapshot")).unwrap()
    );
    assert_eq!(
        fs::read(layer_helper).unwrap(),
        fs::read(tool_dir.join("model-package-layer-job")).unwrap()
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("atomically publish"));
}

#[cfg(unix)]
#[path = "snapshot_promotion_cli/publisher_helper.rs"]
mod publisher_helper;

#[path = "snapshot_promotion_cli/child_env.rs"]
mod child_env;

#[path = "snapshot_promotion_cli/layer_job.rs"]
mod layer_job;

#[path = "snapshot_promotion_cli/competitive_inputs.rs"]
mod competitive_inputs;

#[path = "snapshot_promotion_cli/checkpoint_stitch.rs"]
mod checkpoint_stitch;

#[cfg(unix)]
#[path = "snapshot_promotion_cli/generic_jobs.rs"]
mod generic_jobs;
