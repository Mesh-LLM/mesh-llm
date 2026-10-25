//! Execute supplied observations and the real wrapper with finite private tools only.
use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use std::{collections::BTreeMap, fs, num::NonZeroUsize, path::Path, time::Duration};
fn invoke(root: &Path, body: &str) -> process::RawProcessReport {
    let script = root.join("invoke.sh");
    fs::write(&script, body).unwrap();
    let environment = BTreeMap::from([
        (
            "PATH".into(),
            Value::Public(format!("{}:/usr/bin:/bin", root.join("bin").display()).into()),
        ),
        (
            "XTASK".into(),
            Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
        ),
        (
            "INPUT".into(),
            Value::Public(root.join("input.json").into_os_string()),
        ),
        (
            "FIXTURE_ROOT".into(),
            Value::Public(root.as_os_str().to_owned()),
        ),
        (
            "MESH_LLM_AUTOMATION_BIN".into(),
            Value::Public(env!("CARGO_BIN_EXE_xtask").into()),
        ),
    ]);
    let result = process::supervise_raw(
        &ProcessSpec {
            executable: "/bin/bash".into(),
            cwd: root.into(),
            environment,
            arguments: vec![Value::Public(script.into_os_string())],
        },
        &Limits {
            execution: Duration::from_secs(10),
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        &Cancellation::default(),
        RawCaptureOptions {
            stdout: NonZeroUsize::new(65536),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    let p = &result.process;
    assert_eq!(p.outcome, process::Outcome::Exited);
    assert!(p.failure.is_none() && p.cleanup.failure.is_none());
    assert!(p.cleanup.complete && !p.cleanup.forced && !p.cleanup.graceful_signal_failed);
    for (stream, raw) in [(&p.stdout, &result.stdout), (&p.stderr, &result.stderr)] {
        assert!(stream.line_capture_complete && !stream.truncated);
        assert_eq!(
            raw.as_ref().unwrap().as_bytes().len() as u64,
            stream.bytes_seen
        );
    }
    result
}
fn stdout(report: &process::RawProcessReport) -> &[u8] {
    report.stdout.as_ref().unwrap().as_bytes()
}
#[test]
fn actual_wan_cli_rtt_validates_units_refuses_nonfinite_and_closed_argv() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let success = invoke(
        &root,
        "exec \"$XTASK\" automation wan-observation delay 12.3456\n",
    );
    assert!(success.process.status.unwrap().success());
    assert_eq!(stdout(&success), b"6.173\n");
    for args in [
        "delay NaN",
        "delay -1",
        "delay 1 extra",
        "unknown",
        "bandwidth extra",
    ] {
        let failed = invoke(
            &root,
            &format!("exec \"$XTASK\" automation wan-observation {args}\n"),
        );
        assert!(!failed.process.status.unwrap().success());
        assert!(stdout(&failed).is_empty());
    }
    temp.close().unwrap();
}
#[test]
fn actual_wan_cli_bandwidth_fallback_optional_invalid_and_bound_are_observed() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    for (json, expected) in [
        (
            r#"{"end":{"sum":{"bits_per_second":3500000}}}"#,
            b"4\n".as_slice(),
        ),
        ("{}", b"".as_slice()),
    ] {
        fs::write(root.join("input.json"), json).unwrap();
        let report = invoke(
            &root,
            "exec \"$XTASK\" automation wan-observation bandwidth < \"$INPUT\"\n",
        );
        assert!(report.process.status.unwrap().success());
        assert_eq!(stdout(&report), expected);
    }
    for bytes in [
        br#"{"end":{"sum":{"bits_per_second":"private-invalid-value"}}}"#.to_vec(),
        vec![b' '; 1024 * 1024 + 1],
    ] {
        fs::write(root.join("input.json"), bytes).unwrap();
        let failed = invoke(
            &root,
            "exec \"$XTASK\" automation wan-observation bandwidth < \"$INPUT\"\n",
        );
        assert!(!failed.process.status.unwrap().success());
        assert!(stdout(&failed).is_empty());
        assert!(
            !String::from_utf8_lossy(failed.stderr.as_ref().unwrap().as_bytes())
                .contains("private-invalid-value")
        );
    }
    temp.close().unwrap();
}
