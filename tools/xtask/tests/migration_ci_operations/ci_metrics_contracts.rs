use crate::process::{
    self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
};
use serde_json::{Value as Json, json};
use std::{collections::BTreeMap, fs, num::NonZeroUsize, path::Path, time::Duration};

fn invoke(root: &Path, arguments: &[&str]) -> process::RawProcessReport {
    invoke_environment(root, arguments, BTreeMap::new())
}
fn invoke_environment(
    root: &Path,
    arguments: &[&str],
    environment: BTreeMap<std::ffi::OsString, Value>,
) -> process::RawProcessReport {
    let arguments = ["ci-ops", "collect-metrics"]
        .into_iter()
        .chain(arguments.iter().copied())
        .map(|arg| Value::Public(arg.into()))
        .collect();
    let report = process::supervise_raw(
        &ProcessSpec {
            executable: env!("CARGO_BIN_EXE_xtask").into(),
            cwd: root.to_owned(),
            environment,
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
fn input(root: &Path, name: &str) {
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ci_operations/ci_metrics/inputs")
        .join(name);
    fs::copy(source, root.join("input.json")).unwrap();
}
#[test]
fn ci_metrics_actual_saved_input_publishes_both_formats_and_preserves_semantic_labels() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    input(root, "sample_runs.json");
    let report = invoke(
        root,
        &[
            "--markdown-out",
            "summary.md",
            "--label",
            "provider=first",
            "--input=input.json",
            "--json-out=report.json",
            "--label=provider=fixture",
        ],
    );
    assert!(
        report.process.success(),
        "{}",
        String::from_utf8_lossy(report.stderr.unwrap().as_bytes())
    );
    let json: Json = serde_json::from_slice(&fs::read(root.join("report.json")).unwrap()).unwrap();
    assert_eq!(json["schema_version"], 3);
    assert_eq!(json["benchmark_labels"], json!({"provider":"fixture"}));
    assert_eq!(json["selection"]["included_run_count"], 2);
    assert_eq!(json["workflow"]["wall_seconds"]["p50"], 900.0);
    let markdown = fs::read_to_string(root.join("summary.md")).unwrap();
    for section in [
        "# CI timing summary",
        "Runner dimensions",
        "Step timing",
        "Slow job families",
    ] {
        assert!(markdown.contains(section));
    }
}
#[test]
fn ci_metrics_native_bad_arguments_reject_before_any_existing_output_is_changed() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    input(root, "sample_runs.json");
    for bad in [
        vec!["--inp", "input.json"],
        vec!["--input", "input.json", "--input", "again"],
        vec!["--input", "input.json", "--limit", "1_0"],
        vec!["--input", "input.json", "--top", "-1"],
        vec!["--input", ""],
        vec!["--input", "input.json", "--compare-input="],
        vec!["--input", "input.json", "--help"],
    ] {
        fs::write(root.join("report.json"), b"preserved").unwrap();
        let mut args = bad;
        args.extend(["--json-out", "report.json", "--markdown-out", "summary.md"]);
        let report = invoke(root, &args);
        assert_eq!(report.process.status.unwrap().code(), Some(2));
        assert!(report.stdout.unwrap().as_bytes().is_empty());
        assert!(
            String::from_utf8_lossy(report.stderr.unwrap().as_bytes())
                .contains("cargo xtool ci-ops collect-metrics")
        );
        assert_eq!(fs::read(root.join("report.json")).unwrap(), b"preserved");
        assert!(!root.join("summary.md").exists());
    }
}
#[test]
fn ci_metrics_missing_job_details_reject_before_report_publication() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    fs::write(
        root.join("input.json"),
        serde_json::to_vec(&json!([{"status":"completed","conclusion":"success","id":1}])).unwrap(),
    )
    .unwrap();
    let report = invoke(
        root,
        &[
            "--input",
            "input.json",
            "--json-out",
            "report.json",
            "--markdown-out",
            "summary.md",
        ],
    );
    assert_eq!(report.process.status.unwrap().code(), Some(2));
    assert!(
        String::from_utf8_lossy(report.stderr.unwrap().as_bytes()).contains("has no jobs array")
    );
    assert!(!root.join("report.json").exists() && !root.join("summary.md").exists());
}

#[test]
fn ci_metrics_nonempty_small_baseline_keeps_provider_comparison_on_hold() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    input(root, "comparison_cohort.json");
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/ci_operations/ci_metrics/inputs");
    let baseline: Json =
        serde_json::from_slice(&fs::read(fixtures.join("compare_github_baseline.json")).unwrap())
            .unwrap();
    fs::write(
        root.join("baseline.json"),
        serde_json::to_vec(&json!([baseline[0]])).unwrap(),
    )
    .unwrap();
    let result = invoke(
        root,
        &[
            "--input",
            "input.json",
            "--compare-input",
            "baseline.json",
            "--json-out",
            "report.json",
        ],
    );
    assert!(
        result.process.success(),
        "{}",
        String::from_utf8_lossy(result.stderr.unwrap().as_bytes())
    );
    let report: Json =
        serde_json::from_slice(&fs::read(root.join("report.json")).unwrap()).unwrap();
    let comparison = &report["comparison"];
    assert_eq!(comparison["provider_cohort_separation"]["disjoint"], true);
    assert_eq!(comparison["sample_counts"]["baseline_jobs"], 2);
    assert_eq!(comparison["sample_counts"]["candidate_jobs"], 4);
    assert_eq!(comparison["sample_counts"]["minimum_each"], 3);
    assert_eq!(comparison["sample_counts"]["sufficient"], false);
    assert_eq!(comparison["recommendation"], "hold");
}

#[test]
fn ci_metrics_transport_actual_bad_inputs_preserve_existing_report_and_raw_outputs() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    input(root, "sample_runs.json");
    fs::write(root.join("bad.json"), b"{not json").unwrap();
    fs::create_dir(root.join("directory.json")).unwrap();
    fs::File::create(root.join("oversize.json"))
        .unwrap()
        .set_len(64 * 1024 * 1024 + 1)
        .unwrap();
    for name in ["bad.json", "directory.json", "oversize.json"] {
        for baseline in [false, true] {
            fs::write(root.join("report.json"), b"old report").unwrap();
            fs::write(root.join("raw.json"), b"old raw").unwrap();
            let mut args = vec![
                "--input",
                if baseline { "input.json" } else { name },
                "--json-out",
                "report.json",
                "--raw-out",
                "raw.json",
                "--markdown-out",
                "summary.md",
            ];
            if baseline {
                args.extend(["--compare-input", name]);
            }
            let report = invoke(root, &args);
            assert_eq!(report.process.outcome, process::Outcome::Exited);
            assert_eq!(report.process.status.unwrap().code(), Some(2));
            assert_eq!(fs::read(root.join("report.json")).unwrap(), b"old report");
            assert_eq!(fs::read(root.join("raw.json")).unwrap(), b"old raw");
            assert!(!root.join("summary.md").exists());
        }
    }
}

#[cfg(unix)]
#[test]
fn ci_metrics_transport_actual_fifo_socket_and_symlink_refuse_without_waiting_for_writer() {
    use std::{
        ffi::CString,
        os::unix::{ffi::OsStrExt, fs::symlink, net::UnixListener},
    };
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    input(root, "sample_runs.json");
    let fifo = root.join("fifo.json");
    let name = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let _socket = UnixListener::bind(root.join("socket.json")).unwrap();
    symlink(root.join("input.json"), root.join("link.json")).unwrap();
    for name in ["fifo.json", "socket.json", "link.json"] {
        fs::write(root.join("report.json"), b"old report").unwrap();
        let report = invoke(root, &["--input", name, "--json-out", "report.json"]);
        assert_eq!(report.process.outcome, process::Outcome::Exited);
        assert_eq!(report.process.status.unwrap().code(), Some(2));
        assert_eq!(fs::read(root.join("report.json")).unwrap(), b"old report");
    }
}

#[cfg(unix)]
#[test]
fn ci_metrics_transport_live_saved_baseline_cancels_owned_gh_and_preserves_output() {
    use std::os::unix::fs::PermissionsExt;
    for empty_input in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        fs::write(root.join("baseline.json"), b"[]").unwrap();
        fs::write(root.join("report.json"), b"old report").unwrap();
        let bin = root.join("bin");
        fs::create_dir(&bin).unwrap();
        let gh = bin.join("gh");
        // This finite tool signals its actual metrics parent after recording a
        // descendant. The inner live owner must reap its separate child group.
        fs::write(
            &gh,
            "#!/bin/sh\n/bin/sleep 20 &\nchild=$!\ntrap '/bin/kill \"$child\" 2>/dev/null; wait \"$child\" 2>/dev/null; exit 143' TERM INT\nprintf '%s\\n' \"$child\" > owned-child\n/bin/kill -TERM \"$PPID\"\nwait \"$child\"\n",
        )
        .unwrap();
        fs::set_permissions(&gh, fs::Permissions::from_mode(0o700)).unwrap();
        let mut arguments = vec![
            "--workflow",
            "fixture",
            "--compare-input",
            "baseline.json",
            "--json-out",
            "report.json",
        ];
        if empty_input {
            arguments.extend(["--input", ""]);
        }
        let report = invoke_environment(
            root,
            &arguments,
            BTreeMap::from([("PATH".into(), Value::Public(bin.into_os_string()))]),
        );
        assert_eq!(report.process.outcome, process::Outcome::Exited);
        assert_eq!(report.process.status.unwrap().code(), Some(2));
        let stderr = std::str::from_utf8(report.stderr.as_ref().unwrap().as_bytes()).unwrap();
        if empty_input {
            assert!(
                stderr.contains("--input requires a nonempty value"),
                "{stderr}"
            );
            assert!(!root.join("owned-child").exists());
            assert!(report.process.cleanup.complete);
            assert_eq!(fs::read(root.join("report.json")).unwrap(), b"old report");
            continue;
        }
        assert!(stderr.contains("cancelled"), "{stderr}");
        let child: i32 = fs::read_to_string(root.join("owned-child"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        // SAFETY: signal zero probes only the fixture-owned recorded descendant.
        assert_eq!(unsafe { libc::kill(child, 0) }, -1);
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
        assert!(report.process.cleanup.complete);
        assert_eq!(fs::read(root.join("report.json")).unwrap(), b"old report");
    }
}

#[test]
fn ci_metrics_saved_run_scalar_identity_preserves_outputs_and_containers_stay_json_only() {
    let temporary = tempfile::tempdir().unwrap();
    let root = temporary.path();
    let args = [
        "--input",
        "input.json",
        "--json-out",
        "report.json",
        "--raw-out",
        "raw.json",
        "--markdown-out",
        "summary.md",
    ];
    let record = |id: Json| {
        json!([{
            "id": id, "attempt": "01", "status": "completed", "conclusion": "success",
            "plan_profile": {"retained": [1, true, {"note": "saved metadata"}]},
            "jobs": [{"id": 1, "name": "fixture job", "conclusion": "success",
                "started_at": "2026-07-01T00:00:00Z", "completed_at": "2026-07-01T00:00:01Z"}]
        }])
    };
    for id in [
        Json::Null,
        json!(true),
        json!(17),
        json!(1.5),
        json!("saved-run"),
    ] {
        let saved = record(id.clone());
        fs::write(root.join("input.json"), serde_json::to_vec(&saved).unwrap()).unwrap();
        let result = invoke(root, &args);
        assert!(result.process.success(), "{:?}", result.process);
        let report: Json =
            serde_json::from_slice(&fs::read(root.join("report.json")).unwrap()).unwrap();
        let raw: Json = serde_json::from_slice(&fs::read(root.join("raw.json")).unwrap()).unwrap();
        assert_eq!(report["runs"][0]["id"], id);
        assert_eq!(report["runs"][0]["plan_profile"], saved[0]["plan_profile"]);
        assert_eq!(raw["runs"], saved);
        let shown = id.as_str().map_or_else(|| id.to_string(), str::to_owned);
        assert!(
            fs::read_to_string(root.join("summary.md"))
                .unwrap()
                .contains(&format!("| {shown} | fixture job |"))
        );
    }
    for id in [json!(["not", "a", "run"]), json!({"nested": "identity"})] {
        fs::write(
            root.join("input.json"),
            serde_json::to_vec(&record(id)).unwrap(),
        )
        .unwrap();
        for name in ["report.json", "raw.json", "summary.md"] {
            fs::write(root.join(name), b"preserved output").unwrap();
        }
        let result = invoke(root, &args);
        assert_eq!(result.process.status.unwrap().code(), Some(2));
        assert!(result.stdout.unwrap().as_bytes().is_empty());
        assert!(
            String::from_utf8_lossy(result.stderr.unwrap().as_bytes())
                .contains("CI run identity must be a scalar")
        );
        for name in ["report.json", "raw.json", "summary.md"] {
            assert_eq!(fs::read(root.join(name)).unwrap(), b"preserved output");
        }
    }
}
