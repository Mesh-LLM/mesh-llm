//! Native execution of the actual Windows Just wrapper; no Cargo or product child.
use crate::process::{
    self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use std::{collections::BTreeMap, path::PathBuf, time::Duration};

fn just_executable() -> PathBuf {
    std::env::split_paths(&std::env::var_os("PATH").expect("Windows tool PATH"))
        .map(|path| path.join("just.exe"))
        .find(|path| path.is_file())
        .expect("existing Windows automation preparation must provide Just")
        .canonicalize()
        .unwrap()
}
fn invoke(code: u8, missing_sccache: bool) -> process::ProcessReport {
    let executable = just_executable();
    let mut environment: BTreeMap<_, _> = [
        "SystemRoot",
        "WINDIR",
        "PATH",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "HOME",
        "APPDATA",
        "LOCALAPPDATA",
    ]
    .into_iter()
    .filter_map(|key| std::env::var_os(key).map(|value| (key.into(), Value::Public(value))))
    .collect();
    if missing_sccache {
        let system = PathBuf::from(std::env::var_os("SystemRoot").expect("Windows SystemRoot"));
        let path = std::env::join_paths([
            system.join("System32"),
            system.join("System32/WindowsPowerShell/v1.0"),
        ])
        .unwrap();
        environment.insert("PATH".into(), Value::Public(path));
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let report = process::supervise(
        &ProcessSpec {
            executable,
            cwd: root,
            environment,
            arguments: ["with-lld", "cmd", "/d", "/c", "exit", &code.to_string()]
                .into_iter()
                .map(|value| Value::Public(value.into()))
                .collect(),
        },
        &Limits {
            execution: Duration::from_secs(10),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 4096,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap();
    assert_eq!(report.outcome, process::Outcome::Exited);
    assert!(report.failure.is_none());
    assert!(
        report.cleanup.complete
            && !report.cleanup.forced
            && !report.cleanup.graceful_signal_failed
            && report.cleanup.failure.is_none()
    );
    for stream in [&report.stdout, &report.stderr] {
        assert!(stream.line_capture_complete && !stream.truncated && stream.suppressed_lines == 0);
    }
    report
}
#[test]
fn windows_with_lld_propagates_native_failure_and_success() {
    let success = invoke(0, false);
    assert!(success.success());
    let failure = invoke(47, false);
    assert_eq!(failure.status.unwrap().code(), Some(47));
    let missing = invoke(0, true);
    assert!(!missing.status.unwrap().success());
    assert!(
        String::from_utf8_lossy(&missing.stderr.bytes_retained)
            .contains("sccache is required; install it and retry")
    );
}
