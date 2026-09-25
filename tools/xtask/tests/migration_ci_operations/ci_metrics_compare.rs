//! `ci-ops collect-metrics --compare-input` against the legacy
//! `compare_reports` path of `scripts/collect-ci-metrics.py`: the baseline
//! file is analyzed like `--input` and the candidate report gains a
//! `comparison` section (and the Markdown its "Provider cohort comparison").
//! Goldens are `fixtures/ci_operations/ci_metrics/compare_*.json`, compared
//! and captured through the same harness as the other `collect-metrics`
//! cases.

use crate::ci_metrics::{MARKDOWN_OUTPUT, OUTPUT, Observed, case, case_with_stdin, report_json};
use crate::support::{TestResult, fixture_dir};
use serde_json::{Value, json};
use std::error::Error;
use std::fs;

/// `--input <candidate> --compare-input <baseline>` with both outputs.
fn compare(
    name: &str,
    candidate: &str,
    baseline: &str,
    extra: &[&str],
) -> Result<Observed, Box<dyn Error>> {
    let mut args = vec![
        "--input",
        candidate,
        "--compare-input",
        baseline,
        "--json-out",
        OUTPUT,
        "--markdown-out",
        MARKDOWN_OUTPUT,
    ];
    args.extend_from_slice(extra);
    case(name, &args)
}

fn comparison(observed: &Observed) -> Result<Value, Box<dyn Error>> {
    Ok(report_json(observed)?["comparison"].clone())
}

#[test]
fn migration_ci_operations_ci_metrics_compare_eligible_cohorts() -> TestResult {
    let observed = compare(
        "compare_eligible",
        "comparison_cohort.json",
        "compare_github_baseline.json",
        &[],
    )?;
    let result = comparison(&observed)?;
    assert_eq!(result["recommendation"], "eligible");
    assert_eq!(result["provider_cohort_separation"]["disjoint"], true);
    assert_eq!(
        result["provider_cohort_separation"]["candidate_providers"],
        json!(["depot"])
    );
    let summary = observed.markdown_output.as_deref().ok_or("no markdown")?;
    assert!(summary.contains("## Provider cohort comparison\n"));
    let reversed = comparison(&compare(
        "compare_reversed",
        "compare_github_baseline.json",
        "comparison_cohort.json",
        &["--top", "1"],
    )?)?;
    assert_eq!(reversed["provider_cohort_separation"]["status"], "pass");
    Ok(())
}

#[test]
fn migration_ci_operations_ci_metrics_compare_holds_and_rolls_back() -> TestResult {
    let same = comparison(&compare(
        "compare_same_provider",
        "sample_runs.json",
        "sample_runs.json",
        &[],
    )?)?;
    assert_eq!(same["recommendation"], "hold");
    assert_eq!(same["provider_cohort_separation"]["status"], "fail");
    let empty = comparison(&compare(
        "compare_empty_cohort",
        "comparison_cohort.json",
        "compare_orchestration_only.json",
        &[],
    )?)?;
    assert_eq!(empty["sample_counts"]["baseline_jobs"], 0);
    assert_eq!(empty["recommendation"], "hold");
    compare(
        "compare_empty_candidate_cohort",
        "compare_orchestration_only.json",
        "comparison_cohort.json",
        &[],
    )?;
    let rollback = comparison(&compare(
        "compare_rollback",
        "contaminated.json",
        "compare_github_baseline.json",
        &[],
    )?)?;
    assert_eq!(rollback["recommendation"], "rollback");
    let platforms = comparison(&compare(
        "compare_mismatched_platforms",
        "comparison_cohort.json",
        "compare_macos_baseline.json",
        &[],
    )?)?;
    assert_eq!(platforms["comparable_dimensions"], false);
    assert_eq!(platforms["recommendation"], "hold");
    compare(
        "compare_mismatched_dimensions",
        "runner_dimensions.json",
        "comparison_cohort.json",
        &["--status", "all"],
    )?;
    compare(
        "compare_status_all",
        "skipped_runs.json",
        "reruns.json",
        &["--status", "all", "--label", "provider=x"],
    )?;
    Ok(())
}

#[test]
fn migration_ci_operations_ci_metrics_compare_streams() -> TestResult {
    let stdout = case(
        "compare_default_stdout",
        &[
            "--input",
            "comparison_cohort.json",
            "--compare-input",
            "compare_github_baseline.json",
        ],
    )?;
    assert!(stdout.stdout.contains("## Provider cohort comparison\n"));
    let stdin = fs::read(fixture_dir().join("ci_metrics/inputs/compare_github_baseline.json"))?;
    case_with_stdin(
        "compare_stdin_baseline",
        &[
            "--input",
            "comparison_cohort.json",
            "--compare-input",
            "-",
            "--json-out",
            OUTPUT,
        ],
        Some(&stdin),
    )?;
    case_with_stdin(
        "compare_stdin_twice",
        &["--input", "-", "--compare-input", "-", "--json-out", OUTPUT],
        Some(&stdin),
    )?;
    case(
        "compare_empty_value",
        &[
            "--input",
            "sample_runs.json",
            "--compare-input=",
            "--json-out",
            OUTPUT,
        ],
    )?;
    Ok(())
}

#[test]
fn migration_ci_operations_ci_metrics_compare_reports_errors() -> TestResult {
    for (name, baseline) in [
        ("compare_error_missing_file", "missing.json"),
        ("compare_error_directory", "."),
        ("compare_error_truncated", "truncated.json"),
        ("compare_error_bom", "bom.json"),
        ("compare_error_not_runs", "not_runs.json"),
        ("compare_error_scalar", "compare_scalar.json"),
        ("compare_error_none_selected", "none_selected.json"),
        ("compare_error_no_jobs", "no_jobs.json"),
        ("compare_error_invalid_timestamp", "invalid_timestamp.json"),
    ] {
        let observed = compare(name, "sample_runs.json", baseline, &[])?;
        assert_eq!(observed.code, 2, "{name}");
        assert!(observed.stderr.starts_with("ci metrics error: "), "{name}");
        assert_eq!(observed.output, None, "{name}");
    }
    let candidate_first = compare(
        "compare_error_candidate_first",
        "truncated.json",
        "missing.json",
        &[],
    )?;
    assert_eq!(candidate_first.code, 2);
    Ok(())
}
