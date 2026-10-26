//! Actual resolver route rejects nonregular input before any DNS or output publication.
use crate::process;
use std::{collections::BTreeMap, ffi::CString, fs, num::NonZeroUsize, path::Path, time::Duration};

fn invoke(root: &Path, input: &Path, output: &Path) -> String {
    let raw = process::supervise_raw(
        &process::ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            cwd: root.into(),
            arguments: ["models", "projector-download", "resolve-worker", "--input"]
                .into_iter()
                .map(|value| process::Value::Public(value.into()))
                .chain([
                    process::Value::Public(input.as_os_str().to_owned()),
                    process::Value::Public("--output".into()),
                    process::Value::Public(output.as_os_str().to_owned()),
                ])
                .collect(),
            environment: BTreeMap::new(),
        },
        &process::Limits {
            execution: Duration::from_secs(5),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 4096,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        &process::Cancellation::default(),
        process::RawCaptureOptions {
            stdout: NonZeroUsize::new(4096),
            stderr: NonZeroUsize::new(4096),
        },
    )
    .unwrap();
    let report = raw.process;
    assert_eq!(report.outcome, process::Outcome::Exited, "{report:?}");
    assert_eq!(report.status.unwrap().code(), Some(1));
    assert!(report.failure.is_none() && report.cleanup.complete && !report.cleanup.forced);
    assert!(!report.cleanup.graceful_signal_failed && report.cleanup.failure.is_none());
    let stdout = raw.stdout.unwrap();
    let stderr = raw.stderr.unwrap();
    assert_eq!(stdout.as_bytes().len() as u64, report.stdout.bytes_seen);
    assert_eq!(stderr.as_bytes().len() as u64, report.stderr.bytes_seen);
    assert!(stdout.as_bytes().is_empty());
    assert!(!output.exists());
    String::from_utf8(stderr.as_bytes().to_vec()).unwrap()
}

#[test]
fn hf_projector_resolver_cli_refuses_nonregular_and_bounded_inputs_before_dns() {
    for mode in [
        "fifo",
        "directory",
        "symlink",
        "oversized",
        "malformed",
        "schema",
        "typed",
    ] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let input = root.join("input.json");
        let output = root.join("receipt.json");
        let expected = match mode {
            "fifo" => {
                use std::os::unix::ffi::OsStrExt as _;
                let path = CString::new(input.as_os_str().as_bytes()).unwrap();
                // The CString is live, NUL-terminated, and names this owned fixture only.
                assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
                "projector input must be a regular file"
            }
            "directory" => {
                fs::create_dir(&input).unwrap();
                "projector input must be a regular file"
            }
            "symlink" => {
                let target = root.join("target.json");
                fs::write(&target, b"preserve target").unwrap();
                std::os::unix::fs::symlink(target, &input).unwrap();
                "projector input must be a regular file"
            }
            "oversized" => {
                fs::write(&input, vec![b' '; 16385]).unwrap();
                "bounded input exceeds limit"
            }
            "malformed" => {
                fs::write(&input, b"{").unwrap();
                "EOF while parsing"
            }
            "schema" => {
                fs::write(&input, br#"{"schema_version":2,"host":"hf.co"}"#).unwrap();
                "invalid resolver request"
            }
            "typed" => {
                // Valid closed Request shape reaches host policy, which refuses before DNS.
                fs::write(&input, br#"{"schema_version":1,"host":"localhost"}"#).unwrap();
                "untrusted projector URL host"
            }
            _ => unreachable!(),
        };
        let error = invoke(root, &input, &output);
        assert!(error.contains(expected), "{mode}: {error}");
        if mode == "symlink" {
            assert_eq!(
                fs::read(root.join("target.json")).unwrap(),
                b"preserve target"
            );
            assert!(
                fs::symlink_metadata(&input)
                    .unwrap()
                    .file_type()
                    .is_symlink()
            );
        }
        directory.close().unwrap();
    }
}
