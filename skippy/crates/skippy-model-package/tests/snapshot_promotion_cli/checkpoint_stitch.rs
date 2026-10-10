#![cfg(unix)]
use std::{
    fs,
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant},
};
fn run(path: &Path, root: &Path, label: &str) {
    let stdout = root.join(format!("{label}.out"));
    let stderr = root.join(format!("{label}.err"));
    let mut child = Command::new(env!("CARGO_BIN_EXE_model-package-mtp-checkpoint"))
        .env_clear()
        .args(["--input"])
        .arg(path)
        .stdout(Stdio::from(fs::File::create(&stdout).unwrap()))
        .stderr(Stdio::from(fs::File::create(&stderr).unwrap()))
        .spawn()
        .unwrap();
    let until = Instant::now() + Duration::from_secs(5);
    let (status, expired) = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break (s, false);
        }
        if Instant::now() >= until {
            let _ = child.kill();
            break (child.wait().unwrap(), true);
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    assert!(!expired);
    assert!(!status.success());
    assert!(fs::metadata(stdout).unwrap().len() < 65536);
    assert!(fs::metadata(stderr).unwrap().len() < 65536);
}
#[test]
fn actual_checkpoint_stage_cli_refuses_writerless_fifo_and_nonfresh_output_before_acquisition() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let fifo = root.join("writerless");
    let c = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap();
    // SAFETY: this path belongs to the fixture; no descriptor or process is affected.
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
    run(&fifo, &root, "fifo");
    let output = root.join("foreign");
    fs::create_dir(&output).unwrap();
    fs::write(output.join("sentinel"), b"preserve").unwrap();
    let profile = root.join("profile.json");
    fs::write(&profile, b"{}").unwrap();
    let input = serde_json::json!({"schema_version":1,"checkpoint":{"repo":"owner/checkpoint","revision":"a".repeat(40),"files":{"config.json":"a".repeat(64),"model.safetensors":"b".repeat(64)}},"tokenizer_source":{"repo":"owner/base","revision":"b".repeat(40),"files":{"tokenizer.json":"a".repeat(64),"tokenizer_config.json":"a".repeat(64),"special_tokens_map.json":"a".repeat(64)}},"tokenizer_profile":profile,"tokenizer_profile_sha256":"a".repeat(64),"output_directory":output,"credential_file":null,"timeout_seconds":5,"maximum_bytes":1024});
    let path = root.join("request.json");
    fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
    run(&path, &root, "existing");
    assert_eq!(fs::read(output.join("sentinel")).unwrap(), b"preserve");
    assert_eq!(fs::read_dir(output).unwrap().count(), 1);
    temp.close().unwrap();
}
