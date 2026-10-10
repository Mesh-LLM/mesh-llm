use std::io::Write;
use std::process::{Command, Stdio};

fn stdin_command(args: &[&str], input: &[u8]) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn chat_display_limits_unicode_content_and_renders_timings() {
    let response = serde_json::json!({
        "choices": [{"message": {"content": "é".repeat(201)}}],
        "timings": {"prompt_per_second": 12.34, "predicted_per_second": 5.67, "predicted_n": 9}
    });
    let output = stdin_command(
        &["ci-ops", "chat-display"],
        &serde_json::to_vec(&response).unwrap(),
    );
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        format!(
            "{}\n  prompt: 12.3 tok/s  gen: 5.7 tok/s (9 tok)\n",
            "é".repeat(200)
        )
    );
}

#[test]
fn chat_display_rejects_missing_choices() {
    let output = stdin_command(&["ci-ops", "chat-display"], br#"{"choices":[],"timings":{"prompt_per_second":1,"predicted_per_second":2,"predicted_n":3}}"#);
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
}

#[test]
fn raw_stream_projection_hashes_exact_bytes() {
    let output = stdin_command(&["artifact", "file-projection", "sha256", "-"], b"abc");
    assert!(output.status.success());
    assert_eq!(
        output.stdout,
        b"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n"
    );
}

#[test]
fn file_projection_resolves_real_file_and_hash() {
    let temporary = tempfile::tempdir().unwrap();
    let file = temporary.path().join("payload");
    std::fs::write(&file, b"abc").unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["artifact", "file-projection", "sha256"])
        .arg(&file)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(
        output.stdout,
        b"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n"
    );
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["artifact", "file-projection", "canonical-path"])
        .arg(&file)
        .output()
        .unwrap();
    assert!(output.status.success());
    assert_eq!(
        String::from_utf8(output.stdout).unwrap().trim(),
        file.canonicalize().unwrap().to_string_lossy()
    );
}
