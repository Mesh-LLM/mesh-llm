use super::*;
use serde_json::{Value, json};
#[test]
fn mounted_worker_reads_exact_large_regular_bytes_and_refuses_pin_drift() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().canonicalize().unwrap().join("request.json");
    let bytes = vec![b' '; 100_000];
    std::fs::write(&path, &bytes).unwrap();
    let mut locator = Locator {
        schema_version: 1,
        path: path.to_str().unwrap().into(),
        sha256: admission::digest(&bytes),
        byte_size: bytes.len() as u64,
        repo: "fixture/request".into(),
        revision: "a".repeat(40),
    };
    assert_eq!(read(&locator).unwrap(), bytes);
    std::fs::write(&path, vec![b'x'; 100_000]).unwrap();
    assert!(read(&locator).is_err());
    locator.byte_size = MAX_REQUEST_BYTES as u64 + 1;
    assert!(read(&locator).is_err());
}
#[cfg(unix)]
#[test]
fn mounted_worker_refuses_symlink_and_writerless_fifo_before_open_wait() {
    use std::{ffi::CString, os::unix::fs::symlink};
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let file = root.join("file");
    std::fs::write(&file, b"{}").unwrap();
    let link = root.join("link");
    symlink(&file, &link).unwrap();
    let fifo = root.join("fifo");
    let name = CString::new(fifo.to_str().unwrap()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    for path in [link, fifo] {
        let locator = Locator {
            schema_version: 1,
            path: path.to_str().unwrap().into(),
            sha256: admission::digest(b"{}"),
            byte_size: 2,
            repo: "fixture/request".into(),
            revision: "a".repeat(40),
        };
        assert!(read(&locator).is_err());
    }
}

#[test]
fn composition_worker_mounted_reader_preserves_large_bytes_and_refuses_changed_or_oversized_input()
{
    let root = tempfile::tempdir().unwrap();
    let path = root.path().canonicalize().unwrap().join("request.json");
    let request = json!({"workflow":"default-mtp-composition","operator":{"checkpoint":{"files":(0..1024).map(|i|(format!("model-{i:04}.safetensors"),json!("1".repeat(64)))).collect::<serde_json::Map<String,Value>>()}}});
    let bytes = serde_json::to_vec(&request).unwrap();
    assert!(bytes.len() > 65536);
    std::fs::write(&path, &bytes).unwrap();
    let mut locator = Locator {
        schema_version: 1,
        path: path.to_str().unwrap().into(),
        sha256: admission::digest(&bytes),
        byte_size: bytes.len() as u64,
        repo: "fixture/request".into(),
        revision: "a".repeat(40),
    };
    assert_eq!(read(&locator).unwrap(), bytes);
    assert!(environment("UNREVIEWED_ENV").is_err());
    std::fs::write(&path, b"changed").unwrap();
    assert!(read(&locator).is_err());
    locator.byte_size = MAX_REQUEST_BYTES as u64 + 1;
    assert!(read(&locator).is_err());
    root.close().unwrap();
}
