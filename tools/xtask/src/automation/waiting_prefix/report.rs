//! Human-readable measured comparisons, including unavailable measurements.
use super::acceptance::{Acceptance, Aggregate, Requirement, delta, pair};
use crate::command::DynResult;
use std::fmt::Write;

fn metric(value: Option<f64>) -> String {
    value
        .filter(|v| v.is_finite())
        .map_or_else(|| "n/a".into(), |value| format!("{value:.1}"))
}

fn chart(
    output: &mut String,
    title: &str,
    before: Option<f64>,
    after: Option<f64>,
) -> DynResult<()> {
    let (Some(before), Some(after)) = (before, after) else {
        return Ok(());
    };
    if !before.is_finite() || !after.is_finite() || before < 0.0 || after < 0.0 {
        return Err("A/B chart requires finite nonnegative measured values".into());
    }
    let ceiling = before.max(after).max(1.0) * 1.1;
    if !ceiling.is_finite() {
        return Err("A/B chart scale overflow".into());
    }
    writeln!(
        output,
        "```mermaid\nxychart-beta\n    title \"{title}\"\n    x-axis [\"Before\", \"After\"]\n    y-axis \"tokens\" 0 --> {:.0}\n    bar [{before:.0}, {after:.0}]\n```\n",
        ceiling.ceil()
    )?;
    Ok(())
}

pub(super) fn render(rows: &[Aggregate], acceptance: &Acceptance) -> DynResult<String> {
    render_optional(rows, Some(acceptance))
}

pub(super) fn render_optional(
    rows: &[Aggregate],
    acceptance: Option<&Acceptance>,
) -> DynResult<String> {
    let (before, after) = pair(rows)?;
    let mut output = String::new();
    chart(
        &mut output,
        "Waiting-prefix A/B: suffix tokens prefetched (lower is better)",
        before.suffix_prefill_tokens_median,
        after.suffix_prefill_tokens_median,
    )?;
    chart(
        &mut output,
        "Resident KV tokens evicted (lower is better)",
        before.resident_evicted_tokens_median,
        after.resident_evicted_tokens_median,
    )?;
    output.push_str("| Metric | Before | After | Delta |\n| --- | ---: | ---: | ---: |\n");
    for (label, old, new) in [
        (
            "Cache hits / round",
            before.cache_hits_median,
            after.cache_hits_median,
        ),
        (
            "Suffix prefill tokens / round",
            before.suffix_prefill_tokens_median,
            after.suffix_prefill_tokens_median,
        ),
        (
            "Capacity rejections",
            Some(before.capacity_rejections as f64),
            Some(after.capacity_rejections as f64),
        ),
        (
            "Resident KV evicted tokens / round",
            before.resident_evicted_tokens_median,
            after.resident_evicted_tokens_median,
        ),
        (
            "Resident KV evicted entries / round",
            before.resident_evicted_entries_median,
            after.resident_evicted_entries_median,
        ),
        (
            "Predicted recompute cost / round",
            before.predicted_recompute_cost_median,
            after.predicted_recompute_cost_median,
        ),
        (
            "Family switches / round",
            before.family_switches_median,
            after.family_switches_median,
        ),
        (
            "Client TTFT p50 ms",
            before.ttft_ms_p50_median,
            after.ttft_ms_p50_median,
        ),
        (
            "Client TTFT p95 ms",
            before.ttft_ms_p95_median,
            after.ttft_ms_p95_median,
        ),
        (
            "Makespan ms",
            before.makespan_ms_median,
            after.makespan_ms_median,
        ),
        (
            "Output tok/s",
            before.output_tokens_per_second_median,
            after.output_tokens_per_second_median,
        ),
    ] {
        let change = delta(old, new).map_or_else(|| "n/a".into(), |v| format!("{v:+.1}%"));
        writeln!(
            output,
            "| {label} | {} | {} | {change} |",
            metric(old),
            metric(new)
        )?;
    }
    let Some(acceptance) = acceptance else {
        output.push_str("\nAcceptance: not requested (manual workload).\n");
        return Ok(output);
    };
    writeln!(
        output,
        "\nFixture acceptance: **{}**\n",
        if acceptance.passed { "PASS" } else { "FAIL" }
    )?;
    output.push_str("| Check | Actual | Required | Result |\n| --- | ---: | ---: | ---: |\n");
    for check in &acceptance.checks {
        let required = match check.requirement {
            Requirement::Eq(count) => format!("= {count}"),
            Requirement::Max(max) => format!("≤ {max}"),
            Requirement::Min(min) => format!("≥ {min}"),
            Requirement::Range(range) => format!("{} to {}", range.min, range.max),
        };
        writeln!(
            output,
            "| {} | {} | {required} | {} |",
            check.label,
            metric(check.actual),
            if check.passed { "PASS" } else { "FAIL" }
        )?;
    }
    Ok(output)
}
