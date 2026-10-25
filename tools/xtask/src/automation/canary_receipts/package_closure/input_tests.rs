use super::{input, process};
use crate::process::{
    self as supervisor, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions,
    Readiness, Value,
};
use std::{collections::BTreeMap, fs, time::Duration};
fn isolated(name: &str) -> bool {
    if std::env::var_os("PACKAGE_INPUT_FIXTURE_CHILD").is_some() {
        return false;
    }
    let target = format!("{}::{name}", module_path!().split_once("::").unwrap().1);
    let output = supervisor::supervise_raw(
        &ProcessSpec {
            executable: std::env::current_exe().unwrap(),
            cwd: std::env::current_dir().unwrap(),
            arguments: ["--exact", &target, "--nocapture", "--test-threads=1"]
                .into_iter()
                .map(|a| Value::Public(a.into()))
                .collect(),
            environment: BTreeMap::from([(
                "PACKAGE_INPUT_FIXTURE_CHILD".into(),
                Value::Public("1".into()),
            )]),
        },
        &Limits {
            execution: Duration::from_secs(5),
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
    assert!(output.process.success(), "{:?}", output.process);
    assert!(
        String::from_utf8_lossy(output.stdout.unwrap().as_bytes()).contains("1 passed; 0 failed")
    );
    true
}
#[test]
fn bounded_regular_transport_preserves_bytes_and_refuses_oversize() {
    if isolated("bounded_regular_transport_preserves_bytes_and_refuses_oversize") {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("input.json");
    for bytes in [
        Vec::new(),
        b"{\"source\":\"space and\\nnewline\"}\n".to_vec(),
        vec![b' '; input::MAX_INPUT_BYTES as usize],
    ] {
        fs::write(&path, &bytes).unwrap();
        assert_eq!(process::operation(|| input::read(&path)).unwrap(), bytes);
    }
    fs::write(&path, vec![b' '; input::MAX_INPUT_BYTES as usize + 1]).unwrap();
    assert!(
        process::operation(|| input::read(&path))
            .unwrap_err()
            .to_string()
            .contains("at most 64 KiB")
    );
}
#[cfg(unix)]
fn fifo(path: &std::path::Path) {
    use std::os::unix::ffi::OsStrExt;
    let path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
}
#[cfg(unix)]
#[test]
fn fifo_symlink_directory_and_device_input_fail_without_opening_a_writer() {
    if isolated("fifo_symlink_directory_and_device_input_fail_without_opening_a_writer") {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let regular = temp.path().join("regular");
    fs::write(&regular, b"{}").unwrap();
    let link = temp.path().join("link");
    std::os::unix::fs::symlink(&regular, &link).unwrap();
    let pipe = temp.path().join("pipe");
    fifo(&pipe);
    for path in [
        link.as_path(),
        pipe.as_path(),
        temp.path(),
        std::path::Path::new("/dev/null"),
    ] {
        assert!(
            process::operation(|| input::read(path))
                .unwrap_err()
                .to_string()
                .contains("regular JSON file")
        );
    }
    assert_eq!(fs::read(regular).unwrap(), b"{}");
}
#[cfg(unix)]
#[test]
fn cancellation_precedes_special_file_admission_and_restores_signal_scope() {
    if isolated("cancellation_precedes_special_file_admission_and_restores_signal_scope") {
        return;
    }
    let temp = tempfile::tempdir().unwrap();
    let pipe = temp.path().join("pipe");
    fifo(&pipe);
    assert!(
        process::operation(|| {
            unsafe {
                libc::raise(libc::SIGTERM);
            }
            let error = input::read(&pipe).unwrap_err();
            assert!(error.to_string().contains("interrupted"));
            Ok(())
        })
        .is_err()
    );
    let regular = temp.path().join("regular");
    fs::write(&regular, b"{}").unwrap();
    assert_eq!(process::operation(|| input::read(&regular)).unwrap(), b"{}");
}
