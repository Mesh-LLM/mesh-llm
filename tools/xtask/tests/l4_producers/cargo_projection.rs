use std::fs;
use std::io::Write;
use std::process::{Command, Stdio};

fn project(metadata: &[u8]) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["repository", "cargo-target-directory"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(metadata).unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn l4_cargo_projection_outputs_exact_path_when_metadata_has_configured_target() {
    let metadata = br#"{"target_directory":"/custom target/configured directory"}"#;
    let output = project(metadata);
    assert!(output.status.success());
    assert_eq!(output.stdout, b"/custom target/configured directory\n");
}

#[test]
fn l4_cargo_projection_preserves_real_metadata_when_cargo_config_overrides_target() {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join(".cargo")).unwrap();
    fs::create_dir(root.path().join("src")).unwrap();
    fs::write(root.path().join("Cargo.toml"), "[package]\nname = \"projection-fixture\"\nversion = \"0.1.0\"\nedition = \"2024\"\n[workspace]\n").unwrap();
    fs::write(root.path().join("src/lib.rs"), "").unwrap();
    fs::write(
        root.path().join(".cargo/config.toml"),
        "[build]\ntarget-dir = \"configured target directory\"\n",
    )
    .unwrap();
    let metadata = Command::new("cargo")
        .args([
            "metadata",
            "--offline",
            "--no-deps",
            "--format-version",
            "1",
        ])
        .current_dir(root.path())
        .env_remove("CARGO_TARGET_DIR")
        .output()
        .unwrap();
    assert!(
        metadata.status.success(),
        "{}",
        String::from_utf8_lossy(&metadata.stderr)
    );
    let output = project(&metadata.stdout);
    assert!(output.status.success());
    assert_eq!(
        output.stdout,
        format!(
            "{}\n",
            root.path()
                .canonicalize()
                .unwrap()
                .join("configured target directory")
                .display()
        )
        .as_bytes()
    );
}
