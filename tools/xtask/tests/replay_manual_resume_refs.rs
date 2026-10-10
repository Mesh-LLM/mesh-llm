#![cfg(unix)]
use serde_json::json;
use std::{os::unix::fs::PermissionsExt, path::Path, process::Command};
fn executable(path: &Path, body: &str) {
    std::fs::write(path, body).unwrap();
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
}
#[test]
fn changed_requested_ref_or_label_rejects_retained_run_before_build_or_request_output() {
    for reference in ["main=changed", "renamed=original"] {
        let temporary = tempfile::tempdir().unwrap();
        let tools = temporary.path().join("tools");
        std::fs::create_dir(&tools).unwrap();
        executable(
            &tools.join("git"),
            "#!/bin/sh\nset -eu\ntest \"$GIT_MASTER\" = 1\ntest \"$1\" = rev-parse\ncase \"$4\" in changed*) printf 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\\n';; *) printf 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\\n';; esac\n",
        );
        let marker = temporary.path().join("must-not-build");
        executable(
            &tools.join("just"),
            &format!(
                "#!/bin/sh\nprintf 'unexpected' > '{}'\nexit 99\n",
                marker.display()
            ),
        );
        let output = temporary.path().join("artifact");
        std::fs::create_dir(&output).unwrap();
        let run = json!({"builds":[{"label":"main","ref":"original","commit":"a".repeat(40),"binary":"/fixture/host","binary_sha256":"c".repeat(64),"runtime_root":"/fixture/native","runtime":"/fixture/runtime","runtime_sha256":"d".repeat(64),"backend":"metal","worktree":"/fixture/worktree"}]});
        let run_bytes = serde_json::to_vec(&run).unwrap();
        std::fs::write(output.join("run.json"), &run_bytes).unwrap();
        std::fs::write(
            output.join("retained-requests.jsonl"),
            b"retained evidence\n",
        )
        .unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "replay-matrix", "run", "--repo"])
            .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
            .args([
                "--ref",
                reference,
                "--model",
                "hf://owner/model",
                "--trajectory-manifest",
            ])
            .arg(temporary.path().join("must-not-read-manifest.json"))
            .arg("--output")
            .arg(&output)
            .arg("--resume")
            .env("PATH", &tools)
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(
            String::from_utf8_lossy(&result.stderr)
                .contains("requested Mesh ref labels or resolved commits differ"),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(!marker.exists());
        assert_eq!(std::fs::read(output.join("run.json")).unwrap(), run_bytes);
        assert_eq!(
            std::fs::read(output.join("retained-requests.jsonl")).unwrap(),
            b"retained evidence\n"
        );
        assert_eq!(std::fs::read_dir(&output).unwrap().count(), 2);
    }
}
