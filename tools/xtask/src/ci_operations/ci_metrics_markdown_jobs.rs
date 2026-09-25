//! The per-job tables of `render_markdown`: slow job families, terminal
//! (critical-path candidate) jobs and the slowest observations.

use crate::ci_operations::ci_metrics_markdown::timing;
use crate::ci_operations::ci_metrics_markdown_format::{
    escape, field, head, human, percent, truthy,
};
use crate::ci_operations::ci_metrics_normalize::Outcome;
use crate::ci_operations::ci_metrics_value::{Value, display};

pub(crate) fn job_families(lines: &mut Vec<String>, jobs: &Value, top: usize) -> Outcome<()> {
    lines.extend(
        [
            "",
            "## Slow job families",
            "",
            "| Job | Samples | Duration p50 | Duration p95 | Runner queue p50 | \
             Runner queue p95 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        .map(str::to_owned),
    );
    for item in head(jobs, "by_name", top) {
        lines.push(format!(
            "| {} | {} | {} | {} | {} | {} |",
            escape(field(item, "name")),
            display(field(item, "sample_count")),
            timing(item, "duration_seconds", "p50")?,
            timing(item, "duration_seconds", "p95")?,
            timing(item, "runner_queue_seconds", "p50")?,
            timing(item, "runner_queue_seconds", "p95")?,
        ));
    }
    Ok(())
}

pub(crate) fn terminal_jobs(lines: &mut Vec<String>, jobs: &Value, top: usize) -> Outcome<()> {
    lines.extend(
        [
            "",
            "## Terminal jobs (critical-path candidates)",
            "",
            "| Job | Runs finishing last | Share |",
            "| --- | ---: | ---: |",
        ]
        .map(str::to_owned),
    );
    for item in head(jobs, "critical_finish_candidates", top) {
        lines.push(format!(
            "| {} | {} | {} |",
            escape(field(item, "name")),
            display(field(item, "terminal_count")),
            percent(field(item, "share"))?,
        ));
    }
    Ok(())
}

pub(crate) fn slowest(lines: &mut Vec<String>, jobs: &Value, top: usize) -> Outcome<()> {
    lines.extend(
        [
            "",
            "## Slowest observations",
            "",
            "| Run | Job | Duration | Queue |",
            "| --- | --- | ---: | ---: |",
        ]
        .map(str::to_owned),
    );
    for item in head(jobs, "slowest_observations", top) {
        let run_id = display(field(item, "run_id"));
        let url = field(item, "run_url");
        let run = if truthy(url) {
            format!("[{run_id}]({})", display(url))
        } else {
            run_id
        };
        lines.push(format!(
            "| {run} | {} | {} | {} |",
            escape(field(item, "name")),
            human(field(item, "duration_seconds"))?,
            human(field(item, "queue_seconds"))?,
        ));
    }
    lines.push(String::new());
    Ok(())
}
