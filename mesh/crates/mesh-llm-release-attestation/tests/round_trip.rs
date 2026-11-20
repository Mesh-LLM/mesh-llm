//! Runs the built tool end to end on a stand-in binary.

use std::path::Path;
use std::process::Command;

const COMMIT: &str = "a12b535d7c4e1f09b8d3427a66c5e0f19d8a7b34";

fn tool(args: &[&str]) -> (bool, String) {
    let output = Command::new(env!("CARGO_BIN_EXE_mesh-llm-release-attestation"))
        .args(args)
        .output()
        .expect("run the attestation tool");
    let mut text = String::from_utf8_lossy(&output.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    (output.status.success(), text)
}

const TARGET: &str = "aarch64-unknown-linux-gnu";

fn inspect(binary: &Path, public_key: Option<&Path>) -> serde_json::Value {
    let binary = binary.to_str().unwrap();
    let mut args = vec!["inspect", "--json", "--binary", binary];
    if let Some(public_key) = public_key {
        args.extend(["--public-key-file", public_key.to_str().unwrap()]);
    }
    let (ok, text) = tool(&args);
    assert!(ok, "{text}");
    serde_json::from_str(&text).expect("inspect prints JSON")
}

fn status(binary: &Path, public_key: Option<&Path>) -> String {
    inspect(binary, public_key)["status"]
        .as_str()
        .unwrap()
        .to_string()
}

fn keypair(dir: &Path, name: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let private = dir.join(format!("{name}-private.json"));
    let public = dir.join(format!("{name}-public.json"));
    let (ok, text) = tool(&[
        "generate-keypair",
        "--private-key-out",
        private.to_str().unwrap(),
        "--public-key-out",
        public.to_str().unwrap(),
    ]);
    assert!(ok, "{text}");
    (private, public)
}

#[test]
fn a_stamped_binary_is_valid_until_it_or_the_key_changes() {
    let dir = std::env::temp_dir().join(format!("mesh-attestation-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let binary = dir.join("mesh-llm");
    std::fs::write(&binary, b"not really an executable").unwrap();
    let (private, public) = keypair(&dir, "release");
    let (_, other_public) = keypair(&dir, "other");

    assert_eq!(status(&binary, None), "missing");

    let (ok, text) = tool(&[
        "stamp",
        "--binary",
        binary.to_str().unwrap(),
        "--signing-key-file",
        private.to_str().unwrap(),
        "--require-source-commit",
        "--commit",
        COMMIT,
        "--target-triple",
        TARGET,
    ]);
    assert!(ok, "{text}");

    // What a verifier reads back is what was signed, not what this machine is.
    let signed = inspect(&binary, Some(&public));
    assert_eq!(signed["status"], "valid");
    assert_eq!(signed["commit"], COMMIT);
    assert_eq!(signed["target_triple"], TARGET);
    assert_eq!(signed["node_version"], env!("CARGO_PKG_VERSION"));
    assert_eq!(status(&binary, None), "invalid");
    assert_eq!(status(&binary, Some(&other_public)), "invalid");

    let mut bytes = std::fs::read(&binary).unwrap();
    bytes[0] ^= 1;
    std::fs::write(&binary, bytes).unwrap();
    assert_eq!(status(&binary, Some(&public)), "invalid");

    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_release_stamp_refuses_to_guess_its_commit_or_target() {
    let dir = std::env::temp_dir().join(format!("mesh-attestation-short-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let binary = dir.join("mesh-llm");
    std::fs::write(&binary, b"not really an executable").unwrap();
    let (private, _) = keypair(&dir, "release");

    let (ok, text) = tool(&[
        "stamp",
        "--binary",
        binary.to_str().unwrap(),
        "--signing-key-file",
        private.to_str().unwrap(),
        "--require-source-commit",
        "--commit",
        "a12b535",
        "--target-triple",
        TARGET,
    ]);
    assert!(!ok, "{text}");
    assert_eq!(std::fs::read(&binary).unwrap(), b"not really an executable");

    // Nor does it guess the target from the machine doing the signing.
    let (ok, text) = tool(&[
        "stamp",
        "--binary",
        binary.to_str().unwrap(),
        "--signing-key-file",
        private.to_str().unwrap(),
        "--require-source-commit",
        "--commit",
        COMMIT,
    ]);
    assert!(!ok, "{text}");
    assert!(text.contains("--target-triple"), "{text}");
    assert_eq!(std::fs::read(&binary).unwrap(), b"not really an executable");

    std::fs::remove_dir_all(&dir).unwrap();
}
