use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde_json::{Value as Json, json};
use std::{collections::BTreeMap, fs, num::NonZeroUsize, path::Path, time::Duration};

fn evidence(path: &Path, hits: Json, misses: Json) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(
        path,
        serde_json::to_vec(&json!({"stats":{
        "cache_hits":{"counts":hits,"adv_counts":{"duplicate":9999}},
        "cache_misses":{"counts":misses,"adv_counts":{"duplicate":9999}}
    },"private_url":"https://user:secret@example.test/private"}))
        .unwrap(),
    )
    .unwrap();
}
fn invoke(root: &Path, arguments: &[String]) -> process::RawProcessReport {
    let arguments = ["ci-ops".to_owned(), "sccache-summary".to_owned()]
        .into_iter()
        .chain(arguments.iter().cloned())
        .map(|arg| Value::Public(arg.into()))
        .collect();
    let report = process::supervise_raw(
        &ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            cwd: root.to_owned(),
            environment: BTreeMap::new(),
            arguments,
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
            stdout: NonZeroUsize::new(65536),
            stderr: NonZeroUsize::new(65536),
        },
    )
    .unwrap();
    assert!(report.process.failure.is_none(), "{:?}", report.process);
    assert!(report.process.cleanup.complete);
    report
}
fn json_report(report: process::RawProcessReport, code: i32) -> Json {
    assert_eq!(report.process.status.unwrap().code(), Some(code));
    assert!(report.stderr.unwrap().as_bytes().is_empty());
    serde_json::from_slice(report.stdout.unwrap().as_bytes()).unwrap()
}
#[test]
fn sccache_summary_aggregates_recursive_counts_deduplicates_paths_and_ignores_advanced_counts() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path().canonicalize().unwrap();
    let a = root.join("evidence with spaces/sccache-stats-a.json");
    let b = root.join("evidence with spaces/nested/sccache-stats-b.json");
    evidence(&a, json!({"Rust":{"local":60}}), json!({"Rust":10}));
    evidence(&b, json!({"Rust":20}), json!({"Rust":10}));
    fs::write(
        root.join("evidence with spaces/unrelated.json"),
        "not evidence",
    )
    .unwrap();
    let before = fs::read(&a).unwrap();
    let args = vec![
        "--format".into(),
        "json".into(),
        "--minimum-hit-rate".into(),
        "0.80".into(),
        a.to_str().unwrap().into(),
        "evidence with spaces".into(),
        a.to_str().unwrap().into(),
    ];
    let summary = json_report(invoke(&root, &args), 0);
    assert_eq!(
        summary,
        json!({"file_count":2,"cache_hits":80,"cache_misses":20,
        "cache_requests":100,"hit_rate":0.8,"minimum_hit_rate":0.8,"passed":true})
    );
    assert_eq!(fs::read(&a).unwrap(), before);
    let reordered = vec![
        "evidence with spaces".into(),
        "--minimum-hit-rate".into(),
        "0.80".into(),
        "--format".into(),
        "json".into(),
        a.to_str().unwrap().into(),
    ];
    assert_eq!(json_report(invoke(&root, &reordered), 0), summary);
}
#[test]
fn sccache_summary_threshold_failure_and_zero_requests_remain_truthful() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    let path = root.join("sccache-stats.json");
    evidence(&path, json!({}), json!({}));
    for minimum in [None, Some("0"), Some("0.01")] {
        let mut args = vec![
            "--format".into(),
            "json".into(),
            "sccache-stats.json".into(),
        ];
        if let Some(minimum) = minimum {
            args.extend(["--minimum-hit-rate".into(), minimum.into()]);
        }
        let summary = json_report(invoke(root, &args), i32::from(minimum.is_some()));
        assert_eq!(summary["hit_rate"], Json::Null);
        assert_eq!(summary["passed"], minimum.is_none());
    }
    evidence(&path, json!({"Rust":1}), json!({"Rust":3}));
    let args = vec![
        "--format".into(),
        "json".into(),
        "--minimum-hit-rate".into(),
        "0.8".into(),
        "sccache-stats.json".into(),
    ];
    assert_eq!(json_report(invoke(root, &args), 1)["passed"], false);
}
#[test]
fn sccache_summary_invalid_counters_and_overflow_publish_no_summary() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    for bad in [
        json!({"Rust":true}),
        json!({"Rust":-1}),
        json!({"Rust":0.5}),
        json!({"Rust":[]}),
        json!({"first":u64::MAX,"second":1}),
    ] {
        evidence(&root.join("sccache-stats.json"), bad, json!({}));
        let report = invoke(root, &["sccache-stats.json".into()]);
        assert_eq!(report.process.status.unwrap().code(), Some(1));
        assert!(report.stdout.unwrap().as_bytes().is_empty());
        assert!(!String::from_utf8_lossy(report.stderr.unwrap().as_bytes()).contains("secret"));
    }
}
#[test]
fn sccache_summary_admission_limits_and_native_help_do_not_publish_success() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    for args in [
        vec![],
        vec!["--format", "xml", "missing"],
        vec!["--minimum-hit-rate", "NaN", "missing"],
        vec!["--minimum-hit-rate", "2", "missing"],
        vec!["--format", "json", "--format", "text", "missing"],
        vec![""],
    ] {
        let report = invoke(
            root,
            &args.iter().map(|arg| (*arg).into()).collect::<Vec<_>>(),
        );
        assert_eq!(report.process.status.unwrap().code(), Some(2));
        assert!(report.stdout.unwrap().as_bytes().is_empty());
    }
    let report = invoke(root, &["--help".into()]);
    assert!(report.process.success());
    assert!(
        String::from_utf8_lossy(report.stdout.unwrap().as_bytes())
            .contains("ci-ops sccache-summary")
    );
    fs::write(root.join("large.json"), vec![b' '; 1024 * 1024 + 1]).unwrap();
    for path in ["missing", "large.json"] {
        let report = invoke(root, &[path.into()]);
        assert_eq!(report.process.status.unwrap().code(), Some(1));
        assert!(report.stdout.unwrap().as_bytes().is_empty());
    }
}
#[cfg(unix)]
#[test]
fn sccache_summary_rejects_symlinks_and_named_pipes_without_blocking() {
    use std::{ffi::CString, os::unix::fs::symlink};
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    evidence(&root.join("valid.json"), json!({"Rust":1}), json!({}));
    symlink(root.join("valid.json"), root.join("linked.json")).unwrap();
    let pipe = CString::new(root.join("pipe").as_os_str().as_encoded_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(pipe.as_ptr(), 0o600) }, 0);
    for path in ["linked.json", "pipe"] {
        let report = invoke(root, &[path.into()]);
        assert_eq!(report.process.status.unwrap().code(), Some(1));
        assert!(report.stdout.unwrap().as_bytes().is_empty());
    }
}

#[test]
fn sccache_summary_empty_malformed_and_deep_discovery_are_bounded() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    fs::create_dir(root.join("empty")).unwrap();
    fs::write(root.join("broken.json"), b"{").unwrap();
    let mut deep = root.join("deep");
    for _ in 0..34 {
        deep = deep.join("level");
    }
    evidence(
        &deep.join("sccache-stats.json"),
        json!({"Rust":1}),
        json!({}),
    );
    for path in ["empty", "broken.json", "deep"] {
        let report = invoke(root, &[path.into()]);
        assert_eq!(report.process.status.unwrap().code(), Some(1));
        assert!(report.stdout.unwrap().as_bytes().is_empty());
    }
    evidence(
        &root.join("sccache-stats.json"),
        json!({"Rust":1}),
        json!({"Rust":3}),
    );
    let report = invoke(root, &["sccache-stats.json".into()]);
    assert!(report.process.success());
    let output = String::from_utf8(report.stdout.unwrap().as_bytes().to_vec()).unwrap();
    assert!(output.contains("Cache hits: 1\n") && output.contains("Hit rate: 25.00%\n"));
}
