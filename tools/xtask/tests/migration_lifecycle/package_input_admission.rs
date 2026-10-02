//! Public package JSON transports reject special inputs before domain admission.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{collections::BTreeMap, fs, path::Path, time::Duration};
fn reject(path: &Path, cwd: &Path, expected: &str) {
    for verb in [
        "verify-package-closure",
        "workload-manifest",
        "local-manifest-policy",
        "local-parity-inventory",
        "local-split-roster",
    ] {
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: env!("CARGO_BIN_EXE_xtask").into(),
                cwd: cwd.into(),
                arguments: ["automation", "canary-receipts", verb, "--input"]
                    .into_iter()
                    .map(|a| Value::Public(a.into()))
                    .chain([Value::Public(path.into())])
                    .collect(),
                environment: BTreeMap::new(),
            },
            &Limits {
                execution: Duration::from_secs(3),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: std::num::NonZeroUsize::new(65536),
                stderr: std::num::NonZeroUsize::new(65536),
            },
        )
        .unwrap();
        assert_eq!(
            report.process.outcome,
            process::Outcome::Exited,
            "{verb}: {:?}",
            report.process
        );
        assert!(!report.process.success());
        assert!(
            report.stdout.unwrap().as_bytes().is_empty(),
            "{verb}: success evidence on refusal"
        );
        assert!(
            String::from_utf8_lossy(report.stderr.unwrap().as_bytes()).contains(expected),
            "{verb}: wrong rejection"
        );
    }
}
#[test]
fn actual_public_package_transports_reject_fifo_without_a_writer() {
    use std::os::unix::ffi::OsStrExt;
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("input.pipe");
    let pipe = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(pipe.as_ptr(), 0o600) }, 0);
    reject(&path, temp.path(), "regular JSON file");
}
#[test]
fn actual_public_package_transports_reject_symlink_and_preserve_target() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("input.json");
    let link = temp.path().join("input-link.json");
    fs::write(&path, b"{}").unwrap();
    std::os::unix::fs::symlink(&path, &link).unwrap();
    reject(&link, temp.path(), "regular JSON file");
    assert_eq!(fs::read(path).unwrap(), b"{}");
}
#[test]
fn actual_public_package_transports_reject_oversize_before_schema_or_source_work() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("input.json");
    fs::write(&path, vec![b' '; 65537]).unwrap();
    reject(&path, temp.path(), "at most 64 KiB");
}
