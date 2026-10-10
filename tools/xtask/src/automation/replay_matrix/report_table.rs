use super::report_escape::{cell, code, number, range};
use super::report_row::Row;
use crate::command::DynResult;
use std::io::Write;

pub(super) fn write(output: &mut impl Write, rows: &[Row]) -> DynResult<()> {
    writeln!(
        output,
        "\n## Result table\n\n| Arm | Identity | Offered C | Realized C | Slot use | Trajectories/pass | Measured requests | Prompt tokens | Failures | Output match | Decode tok/s | Pass range | vs baseline | E2E output tok/s | Pass range | TTFT p50 | Pass range | TTFT p95 | TTFT p50 vs baseline | Cached prompt | Budget exhausted |"
    )?;
    writeln!(
        output,
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )?;
    for row in rows {
        let prompt = match (row.prompt_tokens_min, row.prompt_tokens_max) {
            (Some(low), Some(high)) => format!("{low}\u{2013}{high}"),
            (Some(_) | None, None) | (None, Some(_)) => "\u{2014}".into(),
        };
        let fields = [
            cell(&row.label),
            code(&row.commit.chars().take(10).collect::<String>()),
            row.concurrency.to_string(),
            number(row.mean_in_flight, 2, ""),
            number(row.concurrency_utilization_pct, 1, "%"),
            row.trajectories_per_pass.to_string(),
            row.requests.to_string(),
            prompt,
            row.failed_requests.to_string(),
            if row.delta_comparable { "yes" } else { "no" }.into(),
            number(row.decode_tokens_per_second, 2, ""),
            range(
                (
                    row.decode_tokens_per_second_min,
                    row.decode_tokens_per_second_max,
                ),
                2,
            ),
            number(row.decode_tokens_per_second_delta_pct, 1, "%"),
            number(row.workload_output_tokens_per_second, 2, ""),
            range(
                (
                    row.workload_output_tokens_per_second_min,
                    row.workload_output_tokens_per_second_max,
                ),
                2,
            ),
            number(row.ttft_p50_seconds, 3, "s"),
            range((row.ttft_p50_seconds_min, row.ttft_p50_seconds_max), 3),
            number(row.ttft_p95_seconds, 3, "s"),
            number(row.ttft_p50_seconds_delta_pct, 1, "%"),
            number(row.cache_pct, 1, "%"),
            number(row.budget_exhausted_pct, 1, "%"),
        ];
        writeln!(output, "| {} |", fields.join(" | "))?;
    }
    Ok(())
}
