use super::*;
use std::fs;

fn repository() -> tempfile::TempDir {
    let root = tempfile::tempdir().unwrap();
    text(root.path(), &["init", "--quiet"]).unwrap();
    root
}

#[test]
fn git_payload_preserves_model_names_long_lines_and_nul_source_inventory() {
    let root = repository();
    let payload = serde_json::to_vec(&serde_json::json!({
        "repo": "XHToken/Spark-X2.5-4B-GGUF",
        "notes": "x".repeat(9000),
        "token_generation": true
    }))
    .unwrap();
    fs::write(root.path().join("token-manifest.json"), &payload).unwrap();
    text(root.path(), &["add", "token-manifest.json"]).unwrap();
    let actual = git(
        root.path(),
        &["show".into(), ":token-manifest.json".into()],
        None,
    )
    .unwrap();
    assert_eq!(actual, payload);
    assert!(serde_json::from_slice::<serde_json::Value>(&actual).is_ok());
    let names = git(root.path(), &["ls-files".into(), "-z".into()], None).unwrap();
    assert_eq!(names, b"token-manifest.json\0");
}

#[test]
fn git_diff_capture_preserves_exact_source_bytes_and_refuses_existing_output() {
    let root = repository();
    fs::write(root.path().join("token.txt"), b"old\n").unwrap();
    text(root.path(), &["add", "token.txt"]).unwrap();
    fs::write(root.path().join("token.txt"), b"new token generation\n").unwrap();
    let scratch = tempfile::tempdir().unwrap();
    let capture = scratch.path().join("source.diff");
    let args = ["diff".into(), "--binary".into(), "--".into()];
    let bytes = git(root.path(), &args, Some(&capture)).unwrap();
    assert_eq!(fs::read(&capture).unwrap(), bytes);
    assert!(
        bytes
            .windows(b"+new token generation".len())
            .any(|part| part == b"+new token generation")
    );
    assert!(
        !bytes
            .windows(b"suppressed".len())
            .any(|part| part == b"suppressed")
    );
    assert!(git(root.path(), &args, Some(&capture)).is_err());
    assert_eq!(fs::read(&capture).unwrap(), bytes);
}

#[test]
fn git_payload_overflow_and_nonzero_exit_do_not_publish_partial_capture() {
    let root = repository();
    fs::write(root.path().join("large"), vec![b'x'; 4 * 1024 * 1024 + 1]).unwrap();
    text(root.path(), &["add", "large"]).unwrap();
    let scratch = tempfile::tempdir().unwrap();
    let capture = scratch.path().join("payload");
    for name in [":large", ":missing"] {
        assert!(git(root.path(), &["show".into(), name.into()], Some(&capture)).is_err());
        assert!(!capture.exists());
    }
}
