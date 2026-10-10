use super::*;

#[test]
fn selected_checkout_root_controls_default_manifest_and_source_identity() {
    let selected_root = temp_path("checkout");
    let manifest_dir = selected_root.join("ci/llama-canary");
    fs::create_dir_all(&manifest_dir).expect("selected checkout fixture");
    fs::write(selected_root.join("Cargo.toml"), b"").expect("workspace marker");
    fs::create_dir_all(selected_root.join("tools/xtask")).expect("xtask marker directory");
    fs::write(selected_root.join("tools/xtask/Cargo.toml"), b"").expect("xtask marker");
    let manifest = manifest_dir.join("family-certified.json");
    fs::write(&manifest, fixture("synthetic-manifest", "json")).expect("selected manifest");
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(root())
        .args([
            "--repo-root",
            selected_root.to_str().expect("UTF-8"),
            "ci",
            "family-plan",
        ])
        .output()
        .expect("xtask starts");
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let plan: Value = serde_json::from_slice(&output.stdout).expect("plan JSON");
    assert_eq!(plan["manifest"], "ci/llama-canary/family-certified.json");
    assert_eq!(
        plan["manifest_sha256"],
        "3643e767cfe6d547c009cb1b6954a66fc83e66b07ee2f2a2cb7a57ecf6530916"
    );
    assert_eq!(plan["selected_family_count"], 5);
    assert_eq!(plan["selected_models"][0]["family"], "zeta");

    let supplied = selected_root.join("plan.json");
    fs::write(&supplied, &output.stdout).expect("selected plan");
    let verified = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(root())
        .args([
            "--repo-root",
            selected_root.to_str().expect("UTF-8"),
            "ci",
            "family-plan",
            "--verify-plan",
            supplied.to_str().expect("UTF-8"),
        ])
        .output()
        .expect("xtask starts");
    assert_eq!(
        verified.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&verified.stderr)
    );
    assert!(verified.stdout.is_empty() && verified.stderr.is_empty());
    let wrong_root = run(&["--verify-plan", supplied.to_str().expect("UTF-8")]);
    assert_eq!(wrong_root.status.code(), Some(2));
    assert_eq!(wrong_root.stderr, fixture("tampered-plan", "stderr"));

    let external = run(&["--manifest", manifest.to_str().expect("UTF-8")]);
    assert_eq!(external.status.code(), Some(0));
    let external_plan: Value = serde_json::from_slice(&external.stdout).expect("external plan");
    assert_eq!(external_plan["manifest"], "family-certified.json");
    fs::remove_dir_all(selected_root).expect("cleanup selected checkout");
}

#[cfg(unix)]
#[test]
fn manifest_identity_preserves_literal_backslash_when_path_is_posix() {
    let directory = root().join(format!(
        "task24-path-{}-{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&directory).expect("test directory");
    let path = directory.join("review\\manifest.json");
    fs::write(&path, fixture("synthetic-manifest", "json")).expect("manifest");

    let output = run(&["--manifest", path.to_str().expect("UTF-8")]);

    assert_eq!(output.status.code(), Some(0));
    let plan: Value = serde_json::from_slice(&output.stdout).expect("plan");
    assert_eq!(
        plan["manifest"],
        format!(
            "{}/review\\manifest.json",
            directory
                .file_name()
                .expect("directory")
                .to_str()
                .expect("UTF-8")
        )
    );
    fs::remove_dir_all(directory).expect("cleanup");
}
