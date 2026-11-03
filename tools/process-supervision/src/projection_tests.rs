use super::*;
use std::{collections::BTreeMap, os::unix::fs::PermissionsExt, time::Duration};

fn fixture(body: &str) -> (tempfile::TempDir, ProcessSpec, Limits) {
    let dir = tempfile::tempdir().unwrap();
    let executable = dir.path().join("owned-projection-fixture");
    std::fs::write(&executable, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let spec = ProcessSpec {
        executable,
        arguments: Vec::new(),
        cwd: dir.path().to_path_buf(),
        environment: BTreeMap::from([(
            "PATH".into(),
            super::super::Value::Public("/usr/bin:/bin".into()),
        )]),
    };
    let limits = Limits {
        execution: Duration::from_secs(3),
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 0,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    (dir, spec, limits)
}

#[test]
fn exact_lines_including_eof_fragment_are_projected_but_never_retained() {
    let (_dir, spec, limits) =
        fixture("printf 'credential-private\\nfinal'; printf 'stderr-private' >&2");
    let mut facts = (0, 0, false, false);
    let report = supervise_projected(&spec, &limits, &Cancellation::default(), &mut |line| {
        if line.stream == super::super::Stream::Stdout {
            facts.0 += 1;
            facts.2 |= line.ending == super::super::LineEnding::Eof;
        } else {
            facts.1 += 1;
            facts.3 |= line.ending == super::super::LineEnding::Eof;
        }
    })
    .unwrap();
    assert_eq!(facts, (2, 1, true, true));
    assert!(report.success());
    assert!(report.stdout.line_capture_complete && report.stderr.line_capture_complete);
    assert!(report.stdout.bytes_retained.is_empty() && report.stderr.bytes_retained.is_empty());
    let debug = format!("{report:?}");
    assert!(!debug.contains("credential-private") && !debug.contains("stderr-private"));
}

#[test]
fn descendants_holding_output_and_late_failure_cannot_be_admitted_as_clean_success() {
    for body in [
        "printf 'typed-fact\\n'; exit 9",
        "trap '' TERM; sleep 10 & printf 'typed-fact\\n'; exit 0",
    ] {
        let (_dir, spec, limits) = fixture(body);
        let mut lines = 0;
        let report = supervise_projected(&spec, &limits, &Cancellation::default(), &mut |_| {
            lines += 1
        })
        .unwrap();
        assert_eq!(lines, 1);
        // Exit9 fails status; surviving TERM-ignoring descendant requires forced cleanup.
        assert!(!report.success());
        assert!(report.cleanup.complete);
        assert!(report.stdout.bytes_retained.is_empty());
    }
}

#[test]
fn projection_rejects_readiness_and_retention_specs_before_spawn() {
    let (_dir, mut spec, mut limits) = fixture("exit 0");
    spec.executable = spec.cwd.join("must-not-spawn-missing");
    limits.retained_bytes_per_stream = 1;
    assert!(matches!(
        supervise_projected(&spec, &limits, &Cancellation::default(), &mut |_| {}),
        Err(Failure::InvalidSpec(_))
    ));
    limits.retained_bytes_per_stream = 0;
    limits.completion = Completion::StopAfterReady;
    assert!(matches!(
        supervise_projected(&spec, &limits, &Cancellation::default(), &mut |_| {}),
        Err(Failure::InvalidSpec(_))
    ));
}
