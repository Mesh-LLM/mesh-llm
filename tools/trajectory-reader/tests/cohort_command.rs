//! The actual native cohort executable and supervised xtask caller, no interpreter oracle.
#[path = "parquet_cohort_fixture.rs"]
mod fixture;
use super::{frontend, process};
use process::{Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Duration,
};
fn command(state: &Path, reader: bool, arguments: Vec<String>) -> process::ProcessReport {
    let bin = PathBuf::from(env!("CARGO_BIN_EXE_trajectory-reader"))
        .canonicalize()
        .unwrap();
    let mut args = if reader {
        vec!["cohorts".into()]
    } else {
        vec![
            "automation".into(),
            "replay-matrix".into(),
            "trajectory-reader".into(),
            "--reader".into(),
            bin.to_str().unwrap().into(),
            "--timeout".into(),
            "10".into(),
        ]
    };
    args.extend(arguments);
    let environment: [(&str, Option<std::ffi::OsString>); 2] = [
        ("SYSTEMROOT", std::env::var_os("SYSTEMROOT")),
        ("WINDIR", std::env::var_os("WINDIR")),
    ];
    process::supervise(
        &ProcessSpec {
            executable: if reader { bin } else { frontend() },
            cwd: state.to_owned(),
            arguments: args.into_iter().map(|s| Value::Public(s.into())).collect(),
            environment: environment
                .into_iter()
                .filter_map(|(k, v)| v.map(|v| (k.into(), Value::Public(v))))
                .collect::<BTreeMap<_, _>>(),
        },
        &Limits {
            execution: Duration::from_secs(15),
            graceful_shutdown: Duration::from_secs(2),
            forced_shutdown: Duration::from_secs(3),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        OutputFiles::default(),
    )
    .unwrap()
}
fn args(output: &str) -> Vec<String> {
    [
        "--dataset-file",
        "input.parquet",
        "--dataset-revision",
        &"a".repeat(40),
        "--output",
        output,
        "--cohort",
        "warmup",
        "--cohort",
        "1",
        "--framework",
        "z",
        "--framework",
        "a",
        "--sessions-per-cohort",
        "3",
        "--source-dataset",
        "source",
        "--min-turns",
        "2",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}
#[test]
fn native_prompt_command_cohort_and_frontend_preserve_identical_whole_session_document() {
    let state = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(state.path().join("tools/xtask")).unwrap();
    std::fs::write(state.path().join("Cargo.toml"), "[workspace]\n").unwrap();
    std::fs::write(state.path().join("tools/xtask/Cargo.toml"), "[package]\n").unwrap();
    let rows = (0..5)
        .flat_map(|i| ["z", "a"].map(|f| fixture::row(&format!("{f}-{i}"), f)))
        .collect::<Vec<_>>();
    fixture::write(
        &state.path().join("input.parquet"),
        &rows,
        parquet::basic::Compression::SNAPPY,
    );
    for (reader, output) in [(true, "direct.json"), (false, "frontend.json")] {
        let report = command(state.path(), reader, args(output));
        assert!(report.success(), "{report:?}");
        assert!(report.cleanup.complete);
    }
    let direct = std::fs::read(state.path().join("direct.json")).unwrap();
    assert_eq!(
        direct,
        std::fs::read(state.path().join("frontend.json")).unwrap()
    );
    let doc: serde_json::Value = serde_json::from_slice(&direct).unwrap();
    assert_eq!(
        doc["metadata"]["cohorts"]["warmup"]["framework_trajectories"]["z"],
        2
    );
    assert_eq!(
        doc["cohorts"]["warmup"][0]["messages"][2]["tool_call_id"],
        "call"
    );
    let report = command(state.path(), true, args("direct.json"));
    assert!(!report.success());
    assert_eq!(
        direct,
        std::fs::read(state.path().join("direct.json")).unwrap()
    );
}
#[test]
fn native_prompt_command_cohort_refuses_corrupt_source_and_bad_revision_before_output() {
    let state = tempfile::tempdir().unwrap();
    std::fs::write(state.path().join("input.parquet"), b"broken").unwrap();
    for bad_revision in [false, true] {
        let mut input = args("absent.json");
        if bad_revision {
            input[3] = "main".into();
        }
        let report = command(state.path(), true, input);
        assert!(!report.success(), "{report:?}");
        assert!(report.cleanup.complete);
        assert!(!state.path().join("absent.json").exists());
    }
}
#[cfg(unix)]
#[test]
fn native_prompt_command_cohort_refuses_fifo_without_waiting_for_a_writer() {
    use std::os::unix::ffi::OsStrExt;
    let state = tempfile::tempdir().unwrap();
    let file = state.path().join("input.parquet");
    let name = std::ffi::CString::new(file.as_os_str().as_bytes()).unwrap();
    // SAFETY: the live CString supplies a NUL-terminated owned fixture path.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let report = command(state.path(), true, args("absent.json"));
    assert!(!report.success());
    assert!(report.cleanup.complete);
    assert_ne!(report.outcome, process::Outcome::Deadline);
    assert!(!state.path().join("absent.json").exists());
}
