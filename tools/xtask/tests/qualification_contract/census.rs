//! `repository cargo-census`: the batch execution census.
use super::support::{self, stderr};

fn census(
    temp: &std::path::Path,
    kind: &str,
    selected: &str,
    executed: &str,
) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(temp)
        .args([
            "repository",
            "cargo-census",
            "--kind",
            kind,
            "--batch-id",
            "batch-1",
            "--source-sha",
            support::SOURCE_SHA,
            "--selected",
            selected,
            "--executed",
            executed,
            "--out",
            temp.join("census.json").to_str().unwrap(),
        ])
        .output()
        .unwrap()
}

#[test]
fn accepts_actual_batch_in_any_execution_order() {
    let temp = tempfile::tempdir().unwrap();
    let result = census(temp.path(), "test", "[\"a\", \"b\"]", "[\"b\", \"a\"]");
    assert!(result.status.success(), "{}", stderr(&result));
    assert_eq!(
        String::from_utf8_lossy(&result.stdout),
        "test batch batch-1: 2 packages executed\n"
    );
    let text = std::fs::read_to_string(temp.path().join("census.json")).unwrap();
    assert_eq!(
        text,
        concat!(
            "{\"batch_id\":\"batch-1\",\"executed\":[\"a\",\"b\"],\"kind\":\"test\",",
            "\"schema\":\"mesh-ci-cargo-census-v1\",\"selected\":[\"a\",\"b\"],",
            "\"source_sha\":\"0123456789abcdef0123456789abcdef01234567\"}\n"
        )
    );
}

#[test]
fn rejects_missing_or_extra_package() {
    let temp = tempfile::tempdir().unwrap();
    let result = census(temp.path(), "clippy", "[\"a\", \"b\"]", "[\"a\", \"z\"]");
    assert!(!result.status.success());
    assert!(stderr(&result).contains("missing"), "{}", stderr(&result));
    assert!(
        stderr(&result).contains("unexpected"),
        "{}",
        stderr(&result)
    );
    assert!(!temp.path().join("census.json").exists());
}

#[test]
fn rejects_duplicate_execution() {
    let temp = tempfile::tempdir().unwrap();
    let result = census(
        temp.path(),
        "test",
        "[\"a\", \"b\"]",
        "[\"a\", \"b\", \"b\"]",
    );
    assert!(!result.status.success());
    assert!(stderr(&result).contains("duplicate"), "{}", stderr(&result));
}

#[test]
fn rejects_empty_names_and_unknown_kind() {
    let temp = tempfile::tempdir().unwrap();
    let result = census(temp.path(), "test", "[\"\"]", "[\"\"]");
    assert!(!result.status.success());
    assert!(stderr(&result).contains("nonempty"), "{}", stderr(&result));
    let result = census(temp.path(), "bench", "[\"a\"]", "[\"a\"]");
    assert!(!result.status.success());
    assert!(
        stderr(&result).contains("test or clippy"),
        "{}",
        stderr(&result)
    );
}
