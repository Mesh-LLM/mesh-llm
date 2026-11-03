use serde_json::Value;
use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
};

pub type TestResult = Result<(), Box<dyn Error>>;

pub fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned()
}

pub fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(root())
        .arg("ci-governance")
        .args(args)
        .output()
        .expect("run actual governance CLI")
}

pub fn success(output: Output) -> Value {
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("machine-readable governance result")
}

pub fn denied(output: Output, reason: &str) {
    assert_eq!(output.status.code(), Some(1));
    assert!(
        output.stdout.is_empty(),
        "refusals must not emit success JSON"
    );
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(error.starts_with("error: "), "{error}");
    assert!(error.contains(reason), "expected {reason:?}, got {error}");
}

pub fn write_json(directory: &Path, name: &str, value: &Value) -> PathBuf {
    let path = directory.join(name);
    fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
    path
}
