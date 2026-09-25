//! `ci-ops collect-metrics` against `scripts/collect-ci-metrics.py` (the
//! offline JSON core). Each case runs in a temporary directory holding the
//! synthetic run JSON from `fixtures/ci_operations/ci_metrics/inputs/`
//! (derived from `scripts/tests/test_collect_ci_metrics.py`) and compares
//! the exit status, both streams and any written output file with a golden
//! in `fixtures/ci_operations/ci_metrics/`, captured from the legacy script
//! under Python 3.13 (`{root}` is the temp dir). `generated_at` is wall-clock
//! time, so its value is masked in both. With the legacy interpreter
//! configured, the legacy script also runs on identical inputs and the two
//! must agree byte for byte after that mask.
//!
//! Markdown cases also compare the `--markdown-out` file (or the summary on
//! stdout when no output path is given). `--compare-input` cases live in
//! `ci_metrics_compare.rs`; GitHub collection (`--workflow`/`--run-id`)
//! cases, run against a stub `gh`, live in `ci_metrics_github.rs`.

use crate::support::{CAPTURE_ENV, LEGACY_ENV, Stage, TestResult, fixture_dir, repo_root};
use serde_json::{Value, json};
use std::error::Error;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const SCRIPT: &str = "scripts/collect-ci-metrics.py";
pub(crate) const OUTPUT: &str = "out/nested/metrics.json";
pub(crate) const RAW_OUTPUT: &str = "raw/runs.json";
pub(crate) const MARKDOWN_OUTPUT: &str = "md/nested/summary.md";

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Observed {
    pub(crate) code: i32,
    pub(crate) stdout: String,
    pub(crate) stderr: String,
    pub(crate) output: Option<String>,
    pub(crate) raw_output: Option<String>,
    pub(crate) markdown_output: Option<String>,
}

/// Replaces every `"generated_at": "<...>"` value with a fixed marker.
pub(crate) fn mask(text: &str) -> String {
    const KEY: &str = "\"generated_at\": \"";
    let mut out = String::new();
    let mut rest = text;
    while let Some(start) = rest.find(KEY) {
        let value_start = start + KEY.len();
        out.push_str(&rest[..value_start]);
        let tail = &rest[value_start..];
        let end = tail.find('"').unwrap_or(tail.len());
        out.push_str("<generated_at>");
        rest = &tail[end..];
    }
    out.push_str(rest);
    out
}

struct Sandbox {
    stage: Stage,
}

impl Sandbox {
    fn new(label: &str) -> Result<Self, Box<dyn Error>> {
        let stage = Stage::empty(&format!("ci-metrics-{label}"))?;
        for entry in fs::read_dir(fixture_dir().join("ci_metrics/inputs"))? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_str().ok_or("non-UTF8 fixture name")?;
            stage.write(name, &fs::read(entry.path())?)?;
        }
        Ok(Self { stage })
    }

    fn reset(&self) {
        for name in ["out", "raw", "md"] {
            let _missing = fs::remove_dir_all(self.stage.path().join(name));
        }
    }

    fn optional(&self, relative: &str) -> Option<String> {
        let root = self.stage.root_arg();
        fs::read_to_string(self.stage.path().join(relative))
            .ok()
            .map(|text| mask(&text.replace(&root, "{root}")))
    }

    fn run(
        &self,
        program: &Path,
        prefix: &[PathBuf],
        args: &[&str],
        stdin: Option<&[u8]>,
    ) -> Result<Observed, Box<dyn Error>> {
        let root = self.stage.root_arg();
        let mut child = Command::new(program)
            .current_dir(self.stage.path())
            .args(prefix)
            .args(args.iter().map(|arg| arg.replace("{root}", &root)))
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .env_remove("COLUMNS")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let mut pipe = child.stdin.take().ok_or("no stdin")?;
        pipe.write_all(stdin.unwrap_or_default())?;
        drop(pipe);
        let output = child.wait_with_output()?;
        let clean = |bytes: &[u8]| mask(&String::from_utf8_lossy(bytes).replace(&root, "{root}"));
        Ok(Observed {
            code: output.status.code().unwrap_or(-1),
            stdout: clean(&output.stdout),
            stderr: clean(&output.stderr),
            output: self.optional(OUTPUT),
            raw_output: self.optional(RAW_OUTPUT),
            markdown_output: self.optional(MARKDOWN_OUTPUT),
        })
    }
}

fn golden_path(name: &str) -> PathBuf {
    fixture_dir()
        .join("ci_metrics")
        .join(format!("{name}.json"))
}

fn text(value: &Value) -> Option<String> {
    value.as_str().map(str::to_owned)
}

fn port(
    sandbox: &Sandbox,
    args: &[&str],
    stdin: Option<&[u8]>,
) -> Result<Observed, Box<dyn Error>> {
    sandbox.reset();
    let prefix = ["ci-ops", "collect-metrics"].map(PathBuf::from);
    sandbox.run(Path::new(env!("CARGO_BIN_EXE_xtask")), &prefix, args, stdin)
}

/// Runs the port, then (if configured) the legacy script on a reset copy;
/// both must match each other and the captured golden.
pub(crate) fn case_with_stdin(
    name: &str,
    args: &[&str],
    stdin: Option<&[u8]>,
) -> Result<Observed, Box<dyn Error>> {
    let sandbox = Sandbox::new(name)?;
    let actual = port(&sandbox, args, stdin)?;
    if let Some(python) = std::env::var_os(LEGACY_ENV).map(PathBuf::from) {
        sandbox.reset();
        let legacy = sandbox.run(&python, &[repo_root().join(SCRIPT)], args, stdin)?;
        if std::env::var_os(CAPTURE_ENV).is_some() {
            let mut golden = json!({
                "args": args,
                "code": legacy.code,
                "stdout": legacy.stdout,
                "stderr": legacy.stderr,
                "output": legacy.output,
                "raw_output": legacy.raw_output,
            });
            if let Some(summary) = &legacy.markdown_output {
                golden["markdown_output"] = json!(summary);
            }
            fs::create_dir_all(fixture_dir().join("ci_metrics"))?;
            fs::write(
                golden_path(name),
                serde_json::to_string_pretty(&golden)? + "\n",
            )?;
        }
        assert_eq!(
            actual,
            legacy,
            "{name}: Rust port differs from legacy {}",
            python.display()
        );
    }
    let golden: Value = serde_json::from_slice(&fs::read(golden_path(name))?)?;
    let expected = Observed {
        code: i32::try_from(golden["code"].as_i64().ok_or("golden code")?)?,
        stdout: text(&golden["stdout"]).ok_or("golden stdout")?,
        stderr: text(&golden["stderr"]).ok_or("golden stderr")?,
        output: text(&golden["output"]),
        raw_output: text(&golden["raw_output"]),
        markdown_output: text(&golden["markdown_output"]),
    };
    assert_eq!(
        actual, expected,
        "{name}: Rust port differs from captured golden"
    );
    Ok(actual)
}

pub(crate) fn case(name: &str, args: &[&str]) -> Result<Observed, Box<dyn Error>> {
    case_with_stdin(name, args, None)
}

/// A report case: `--input <file> --json-out <OUTPUT>` plus `extra`.
fn report(name: &str, input: &str, extra: &[&str]) -> Result<Observed, Box<dyn Error>> {
    let mut args = vec!["--input", input, "--json-out", OUTPUT];
    args.extend_from_slice(extra);
    case(name, &args)
}

pub(crate) fn report_json(observed: &Observed) -> Result<Value, Box<dyn Error>> {
    Ok(serde_json::from_str(
        observed.output.as_deref().ok_or("no report")?,
    )?)
}

#[test]
fn migration_ci_operations_ci_metrics_reports_sample_runs() -> TestResult {
    let observed = report(
        "report_sample",
        "sample_runs.json",
        &[
            "--label",
            "provider=fixture",
            "--label",
            "b=2",
            "--label",
            "provider=x",
        ],
    )?;
    let report = report_json(&observed)?;
    assert_eq!(report["schema_version"], 3);
    assert_eq!(
        report["benchmark_labels"],
        json!({"b": "2", "provider": "x"})
    );
    assert_eq!(report["workflow"]["wall_seconds"]["max"], 1200.0);
    Ok(())
}

#[test]
fn migration_ci_operations_ci_metrics_reports_timing_variants() -> TestResult {
    report(
        "report_wrapped_stdout",
        "wrapped_runs.json",
        &["--json-out", "-", "--top", "1"],
    )?;
    let gh = report_json(&report(
        "report_gh_run_view",
        "gh_run_view.json",
        &["--top", "5"],
    )?)?;
    assert_eq!(gh["jobs"]["by_name"][0]["name"], "build");
    assert_eq!(gh["jobs"]["by_name"][0]["queue_seconds"]["count"], 0);
    assert_eq!(gh["jobs"]["by_name"][0]["start_delay_seconds"]["p50"], 20.0);
    let reruns = report_json(&report("report_reruns", "reruns.json", &[])?)?;
    assert_eq!(reruns["selection"]["workflow_timing_excluded_reruns"], 1);
    let skipped = report_json(&report("report_skipped", "skipped_runs.json", &[])?)?;
    assert_eq!(
        skipped["selection"]["skipped_runs"],
        json!({"conclusion_failure": 1, "not_completed": 1})
    );
    report(
        "report_status_all",
        "skipped_runs.json",
        &["--status", "all"],
    )?;
    report(
        "report_status_failure",
        "skipped_runs.json",
        &["--status", "failure"],
    )?;
    report(
        "report_decomposition",
        "decomposition.json",
        &["--status", "completed"],
    )?;
    report("report_epoch_timestamps", "epoch_timestamps.json", &[])?;
    report("report_nan_float_id", "nan_and_float_id.json", &[])?;
    Ok(())
}

#[test]
fn migration_ci_operations_ci_metrics_reports_runner_dimensions() -> TestResult {
    let steps = report_json(&report("report_runner_steps", "runner_steps.json", &[])?)?;
    assert_eq!(steps["steps"]["sample_count"], 4);
    let contaminated = report_json(&report("report_contaminated", "contaminated.json", &[])?)?;
    assert_eq!(contaminated["heuristics"]["state"], "rollback");
    let capacity = report_json(&report(
        "report_capacity",
        "capacity_overlap.json",
        &["--status", "all"],
    )?)?;
    assert_eq!(capacity["capacity"]["peak_workers"]["total"], 2);
    report(
        "report_comparison_cohort",
        "comparison_cohort.json",
        &["--top", "2"],
    )?;
    report(
        "report_runner_dimensions",
        "runner_dimensions.json",
        &["--top", "20"],
    )?;
    Ok(())
}

/// Python object semantics the port must keep: exact big integers, `NaN`,
/// duplicate keys, codec errors, and the tracebacks (status 1) of
/// exceptions the legacy script does not catch.
#[test]
fn migration_ci_operations_ci_metrics_matches_python_values() -> TestResult {
    for (name, input) in [
        ("value_bom", "bom.json"),
        ("value_bad_utf8", "bad_utf8.json"),
        ("value_duplicate_keys", "duplicate_keys.json"),
        ("value_big_numbers", "big_numbers.json"),
        ("value_nan_wait", "nan_wait.json"),
        ("value_equal_job_ids", "equal_job_ids.json"),
        ("value_inert_list_job", "inert_list_job.json"),
        ("value_int_digit_limit", "int_digit_limit.json"),
    ] {
        report(name, input, &["--raw-out", RAW_OUTPUT])?;
    }
    for (name, input) in [
        ("uncaught_infinite_wait", "infinite_wait.json"),
        ("uncaught_infinite_attempt", "infinite_attempt.json"),
        ("uncaught_list_job_id", "list_job_id.json"),
        ("uncaught_string_job", "string_job.json"),
        ("uncaught_timestamp_overflow", "timestamp_overflow.json"),
    ] {
        uncaught(name, input)?;
    }
    Ok(())
}

/// An exception the legacy `main` does not catch: status 1 and a traceback
/// whose final line the port reproduces; the report may already be written.
fn uncaught(name: &str, input: &str) -> TestResult {
    let sandbox = Sandbox::new(name)?;
    let args = ["--input", input, "--json-out", OUTPUT];
    let mut actual = port(&sandbox, &args, None)?;
    let last_line = |text: &str| text.lines().last().unwrap_or_default().to_owned();
    actual.stderr = last_line(&actual.stderr);
    if let Some(python) = std::env::var_os(LEGACY_ENV).map(PathBuf::from) {
        sandbox.reset();
        let mut legacy = sandbox.run(&python, &[repo_root().join(SCRIPT)], &args, None)?;
        legacy.stderr = last_line(&legacy.stderr);
        if std::env::var_os(CAPTURE_ENV).is_some() {
            let golden = json!({
                "args": args,
                "code": legacy.code,
                "stdout": legacy.stdout,
                "stderr_last_line": legacy.stderr,
                "output": legacy.output,
                "raw_output": legacy.raw_output,
            });
            fs::write(
                golden_path(name),
                serde_json::to_string_pretty(&golden)? + "\n",
            )?;
        }
        assert_eq!(actual, legacy, "{name}: Rust port differs from legacy");
    }
    let golden: Value = serde_json::from_slice(&fs::read(golden_path(name))?)?;
    let expected = Observed {
        code: i32::try_from(golden["code"].as_i64().ok_or("golden code")?)?,
        stdout: text(&golden["stdout"]).ok_or("golden stdout")?,
        stderr: text(&golden["stderr_last_line"]).ok_or("golden stderr")?,
        output: text(&golden["output"]),
        raw_output: text(&golden["raw_output"]),
        markdown_output: text(&golden["markdown_output"]),
    };
    assert_eq!(
        actual, expected,
        "{name}: Rust port differs from captured golden"
    );
    assert_eq!(actual.code, 1, "{name}");
    Ok(())
}

#[test]
fn migration_ci_operations_ci_metrics_writes_raw_and_stdin() -> TestResult {
    let stdin = fs::read(fixture_dir().join("ci_metrics/inputs/reruns.json"))?;
    case_with_stdin(
        "report_stdin_raw_out",
        &[
            "--input",
            "-",
            "--json-out",
            OUTPUT,
            "--raw-out",
            RAW_OUTPUT,
        ],
        Some(&stdin),
    )?;
    case_with_stdin(
        "report_stdin_to_stdout",
        &["--input=-", "--json-out=-", "--raw-out", RAW_OUTPUT],
        Some(&stdin),
    )?;
    Ok(())
}

#[test]
fn migration_ci_operations_ci_metrics_reports_input_errors() -> TestResult {
    for (name, input) in [
        ("error_no_jobs", "no_jobs.json"),
        ("error_invalid_timestamp", "invalid_timestamp.json"),
        ("error_none_selected", "none_selected.json"),
        ("error_not_runs", "not_runs.json"),
        ("error_truncated", "truncated.json"),
        ("error_missing_file", "missing.json"),
        ("error_directory", "."),
    ] {
        let observed = report(name, input, &[])?;
        assert_eq!(observed.code, 2, "{name}");
        assert!(observed.stderr.starts_with("ci metrics error: "), "{name}");
    }
    report(
        "error_label",
        "sample_runs.json",
        &["--label", "no-separator"],
    )?;
    report(
        "error_label_empty_key",
        "sample_runs.json",
        &["--label", "=v"],
    )?;
    report(
        "error_output_parent_is_file",
        "sample_runs.json",
        &["--json-out", "sample_runs.json/x.json"],
    )?;
    Ok(())
}

#[test]
fn migration_ci_operations_ci_metrics_matches_argparse() -> TestResult {
    let cases: [(&str, &[&str]); 20] = [
        ("argv_help", &["-h"]),
        ("argv_help_bundled", &["--input", "x", "-hx"]),
        ("argv_help_explicit", &["--help=1"]),
        ("argv_empty", &[]),
        (
            "argv_unknown",
            &["--input", "x", "--json-out", "-", "extra", "--nope"],
        ),
        ("argv_double_dash", &["--input", "x", "--", "--json-out"]),
        ("argv_ambiguous", &["--input", "x", "--r", "y"]),
        (
            "argv_prefix",
            &["--inp", "sample_runs.json", "--json", OUTPUT, "--st", "all"],
        ),
        ("argv_expected_value", &["--input", "x", "--label"]),
        ("argv_invalid_int", &["--input", "x", "--limit", "ten"]),
        ("argv_invalid_run_id", &["--run-id", "1.5"]),
        (
            "argv_input_and_workflow",
            &["--input", "x", "--workflow", "w"],
        ),
        (
            "argv_input_and_run_id",
            &["--input", "x", "--run-id", "1_0"],
        ),
        (
            "argv_workflow_and_run_id",
            &["--workflow", "w", "--run-id", "3"],
        ),
        ("argv_top_zero", &["--input", "x", "--top", "0"]),
        ("argv_limit_negative", &["--input", "x", "--limit", "-1"]),
        (
            "argv_two_stdout",
            &["--input", "x", "--json-out", "-", "--raw-out", "-"],
        ),
        ("argv_empty_input_value", &["--input", ""]),
        (
            "argv_negative_number_value",
            &["--input", "x", "--top", "-5"],
        ),
        (
            "argv_error_before_later_option",
            &["--limit", "x", "--bogus", "--top"],
        ),
    ];
    for (name, args) in cases {
        case(name, args)?;
    }
    Ok(())
}

/// A markdown case: `--input <file> --markdown-out <MARKDOWN_OUTPUT>`.
fn markdown(name: &str, input: &str, extra: &[&str]) -> Result<Observed, Box<dyn Error>> {
    let mut args = vec!["--input", input, "--markdown-out", MARKDOWN_OUTPUT];
    args.extend_from_slice(extra);
    case(name, &args)
}

#[test]
fn migration_ci_operations_ci_metrics_renders_markdown_files() -> TestResult {
    let sample = markdown("markdown_sample", "sample_runs.json", &[])?;
    let summary = sample.markdown_output.as_deref().ok_or("no markdown")?;
    assert!(summary.starts_with("# CI timing summary\n"));
    assert!(sample.stdout.is_empty());
    let escaped = markdown("markdown_escape", "markdown_escape.json", &["--top", "2"])?;
    let summary = escaped.markdown_output.as_deref().ok_or("no markdown")?;
    assert!(summary.contains("build \\| linux x64"));
    for (name, input, extra) in [
        ("markdown_reruns", "reruns.json", &[][..]),
        (
            "markdown_runner_steps",
            "runner_steps.json",
            &["--top", "1"][..],
        ),
        ("markdown_contaminated", "contaminated.json", &[][..]),
        (
            "markdown_comparison_cohort",
            "comparison_cohort.json",
            &[][..],
        ),
        (
            "markdown_runner_dimensions",
            "runner_dimensions.json",
            &["--top", "20"][..],
        ),
        ("markdown_gh_run_view", "gh_run_view.json", &[][..]),
        ("markdown_nan_float_id", "nan_and_float_id.json", &[][..]),
        ("markdown_big_numbers", "big_numbers.json", &[][..]),
        (
            "markdown_status_all",
            "skipped_runs.json",
            &["--status", "all"][..],
        ),
    ] {
        markdown(name, input, extra)?;
    }
    Ok(())
}

#[test]
fn migration_ci_operations_ci_metrics_renders_markdown_streams() -> TestResult {
    let stdout = case("markdown_default_stdout", &["--input", "sample_runs.json"])?;
    assert!(stdout.stdout.starts_with("# CI timing summary\n"));
    case(
        "markdown_stdout_dash",
        &[
            "--input",
            "wrapped_runs.json",
            "--markdown-out",
            "-",
            "--json-out",
            OUTPUT,
        ],
    )?;
    case(
        "markdown_with_json_and_raw",
        &[
            "--input",
            "decomposition.json",
            "--json-out",
            OUTPUT,
            "--raw-out",
            RAW_OUTPUT,
            "--markdown-out",
            MARKDOWN_OUTPUT,
        ],
    )?;
    let stdin = fs::read(fixture_dir().join("ci_metrics/inputs/epoch_timestamps.json"))?;
    case_with_stdin("markdown_stdin_default", &["--input", "-"], Some(&stdin))?;
    case(
        "markdown_parent_is_file",
        &[
            "--input",
            "sample_runs.json",
            "--markdown-out",
            "sample_runs.json/x.md",
        ],
    )?;
    Ok(())
}
