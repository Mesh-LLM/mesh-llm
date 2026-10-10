//! KV mode admission, owned execution and atomic evidence publication.
use super::{
    evidence, kv_certification,
    kv_native_logs::{Budget, Checkpoint, Scan},
    kv_options::{Options, USAGE},
    kv_plans,
    kv_reports::{self, Row},
    kv_transcripts::Transcripts,
    transport::Http,
};
use crate::{
    command::DynResult, command_interrupt::Interrupt, process::Cancellation,
    repository::check_report::CheckReport,
};
use std::{
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

pub(super) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        return CheckReport::success(format!("{USAGE}\n")).emit();
    }
    let mut options = match Options::parse(args) {
        Ok(options) => options,
        Err(error) => return CheckReport::usage(USAGE, &error).emit(),
    };
    if !options.output.is_absolute() {
        options.output = std::env::current_dir()?.join(&options.output);
    }
    let mut manifest = kv_plans::build(&options);
    if options.plan {
        return crate::command::print_json(&manifest);
    }
    let interrupt = Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let mut transcripts = Transcripts::create(&options.output)?;
    let mut scans = Vec::new();
    let checkpoint_started = Instant::now();
    let checkpoints = capture(&options, &cancellation);
    let mut rows = match checkpoints {
        Ok(checkpoints) => {
            let mut rows = execute(&options, &cancellation, &mut transcripts)?;
            if !checkpoints.is_empty() {
                let (row, observed) = scan(&checkpoints, &options, &cancellation);
                rows.push(row);
                scans = observed;
            }
            rows
        }
        Err(error) => vec![Row::new(
            ("native-log", 0, "native_log_scan"),
            checkpoint_started,
            None,
            None,
            Err(error),
        )],
    };
    let signal = super::finish(interrupt)?;
    let cancelled = cancellation.is_cancelled() || signal.is_some();
    if cancelled && rows.is_empty() {
        rows.push(Row::new(
            ("cohort", 0, "cancelled"),
            Instant::now(),
            None,
            None,
            Err("KV certification cancelled before its first request".into()),
        ));
    }
    let summary = kv_reports::summarize(&rows, cancelled);
    manifest["created_at"] = serde_json::json!(super::timestamp()?);
    manifest["transcript_files"] = serde_json::to_value(&transcripts.entries)?;
    publish(
        &options.output,
        &manifest,
        &rows,
        &summary,
        &scans,
        !options.native_logs.is_empty(),
    )?;
    CheckReport {
        stdout: format!(
            "kv-tool-loop stability: {}\nevidence: {}\n",
            if summary.ok { "PASS" } else { "FAIL" },
            options.output.display()
        ),
        stderr: String::new(),
        code: signal.unwrap_or(if summary.ok { 0 } else { 1 }),
    }
    .emit()
}

fn budget(options: &Options, cancellation: &Cancellation) -> Budget {
    Budget {
        deadline: Instant::now() + options.timeout,
        cancellation: cancellation.clone(),
    }
}
fn capture(options: &Options, cancellation: &Cancellation) -> Result<Vec<Checkpoint>, String> {
    let budget = budget(options, cancellation);
    options
        .native_logs
        .iter()
        .map(|path| {
            Checkpoint::capture(path, &budget)
                .map_err(|error| format!("native log checkpoint {}: {error}", path.display()))
        })
        .collect()
}
fn execute(
    options: &Options,
    cancellation: &Cancellation,
    transcripts: &mut Transcripts,
) -> DynResult<Vec<Row>> {
    let started = Instant::now();
    let http = match Http::new(
        options.base.clone(),
        options.timeout,
        cancellation.clone(),
        "mesh-llm-kv-stability",
    ) {
        Ok(http) => Arc::new(http),
        Err(error) => {
            return Ok(vec![Row::new(
                ("transport", 0, "transport_setup"),
                started,
                None,
                None,
                Err(error),
            )]);
        }
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let rows = runtime.block_on(kv_certification::run(http, options, transcripts));
    runtime.shutdown_timeout(Duration::from_secs(1));
    rows
}
fn scan(
    checkpoints: &[Checkpoint],
    options: &Options,
    cancellation: &Cancellation,
) -> (Row, Vec<Scan>) {
    let started = Instant::now();
    let budget = budget(options, cancellation);
    let mut scans = Vec::new();
    let mut failures = Vec::new();
    let mut total = 0u64;
    for (checkpoint, path) in checkpoints.iter().zip(&options.native_logs) {
        match checkpoint.scan(&budget) {
            Ok(scan) => {
                total = total.saturating_add(scan.total_findings);
                for finding in scan
                    .findings
                    .iter()
                    .take(5usize.saturating_sub(failures.len()))
                {
                    failures.push(format!(
                        "{}:{} {}: {}",
                        finding.path.display(),
                        finding.line_number,
                        finding.pattern,
                        finding.text
                    ));
                }
                scans.push(scan);
            }
            Err(error) => failures.push(format!(
                "native log scan {} failed: {error}",
                path.display()
            )),
        }
    }
    let result = if failures.is_empty() && total == 0 {
        Ok("no fatal KV log patterns appended during this run".into())
    } else {
        Err(format!(
            "{}; total fatal findings={total}",
            failures.join("; ")
        ))
    };
    (
        Row::new(
            ("native-log", 0, "native_log_scan"),
            started,
            None,
            None,
            result,
        ),
        scans,
    )
}
fn publish(
    output: &Path,
    manifest: &serde_json::Value,
    rows: &[Row],
    summary: &kv_reports::Summary,
    scans: &[Scan],
    has_logs: bool,
) -> DynResult<()> {
    evidence::json(&output.join("manifest.json"), manifest)?;
    evidence::jsonl(&output.join("results.jsonl"), rows)?;
    evidence::json(&output.join("summary.json"), summary)?;
    evidence::bytes(
        &output.join("summary.md"),
        kv_reports::markdown(summary, rows).as_bytes(),
    )?;
    if has_logs {
        evidence::json(&output.join("native-log-scan.json"), &scans)?;
    }
    Ok(())
}
