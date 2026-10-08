#[cfg(unix)]
#[test]
fn build_arm_runs_exact_just_steps_and_binds_runtime_identity() {
    use std::{os::unix::fs::PermissionsExt, process::Command};
    let state = tempfile::tempdir().unwrap();
    let commit = "0123456789abcdef0123456789abcdef01234567";
    let worktrees = state.path().join("worktrees");
    let worktree = worktrees.join("base-0123456789");
    let runtime = worktree.join("dist/native-runtimes/cpu");
    std::fs::create_dir_all(&runtime).unwrap();
    std::fs::create_dir_all(worktree.join("target/release")).unwrap();
    std::fs::write(worktree.join("target/release/mesh-llm"), b"fixture host").unwrap();
    std::fs::write(
        runtime.join("manifest.json"),
        br#"{"runtime":{"backend":{"kind":"cpu"}}}"#,
    )
    .unwrap();
    std::fs::write(runtime.join("runtime.so"), b"fixture runtime").unwrap();
    let git = state.path().join("git-fixture");
    let just = state.path().join("just-fixture");
    let calls = state.path().join("calls.txt");
    std::fs::write(&git,format!("#!/bin/sh\ncase \"$1\" in rev-parse) printf '%s\\n' '{commit}';; status) exit 0;; *) exit 9;; esac\n")).unwrap();
    std::fs::write(
        &just,
        format!(
            "#!/bin/sh\nprintf '%s\\n' \"$*\" >> '{}'\n",
            calls.display()
        ),
    )
    .unwrap();
    for path in [&git, &just] {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let input = state.path().join("input.json");
    let output = state.path().join("build.json");
    std::fs::write(&input,serde_json::to_vec(&serde_json::json!({
        "repo":state.path(),"worktree_root":worktrees,"label":"base","ref":"base-ref","backend":"cpu",
        "git":git,"just":just,"timeout_seconds":5,"logs":state.path().join("logs")
    })).unwrap()).unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "build-arm", "--input"])
        .arg(&input)
        .arg("--output")
        .arg(&output)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        std::fs::read_to_string(calls).unwrap(),
        "release-host-build\nrelease-runtime-build cpu\n"
    );
    let build: serde_json::Value = serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
    assert_eq!(build["commit"], commit);
    assert_eq!(build["runtime"], runtime.to_str().unwrap());
    assert_eq!(build["binary_sha256"].as_str().unwrap().len(), 64);
    assert_eq!(build["runtime_sha256"].as_str().unwrap().len(), 64);
    std::fs::write(&just, "#!/bin/sh\nexit 7\n").unwrap();
    let mut document: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&input).unwrap()).unwrap();
    document["logs"] = serde_json::to_value(state.path().join("failed-logs")).unwrap();
    std::fs::write(&input, serde_json::to_vec(&document).unwrap()).unwrap();
    let denied = state.path().join("denied.json");
    let result = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "build-arm", "--input"])
        .arg(&input)
        .arg("--output")
        .arg(&denied)
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(!denied.exists());
}
