use super::report_escape::{cell, code};
use super::report_input::Document;
use super::report_row::Row;
use crate::command::DynResult;
use std::io::Write;

pub(super) fn render(document: &Document, rows: &[Row]) -> DynResult<Vec<u8>> {
    let mut output = Vec::new();
    writeln!(
        output,
        "# Agentic Replay\n\nGenerated: {}\n\n## Arm identities\n\n| Arm | Engine | Reported version or ref | Identity |\n|---|---|---|---|",
        code(document.completed_at.as_deref().unwrap_or("not recorded"))
    )?;
    for build in &document.builds {
        let version = build
            .version
            .as_deref()
            .filter(|value| !value.is_empty())
            .or(build.reference.as_deref())
            .unwrap_or(&build.label);
        let identity = build
            .commit
            .as_deref()
            .unwrap_or("unknown")
            .chars()
            .take(12)
            .collect::<String>();
        writeln!(
            output,
            "| {} | {} | {} | {} |",
            cell(&build.label),
            cell(build.engine.as_deref().unwrap_or("mesh")),
            code(version),
            code(&identity)
        )?;
    }
    writeln!(
        output,
        "\n## Trajectory selection\n\n| Client concurrency | Whole trajectories | Measured requests/pass | Recorded source steps | Framework trajectory / step breakdown |\n|---:|---:|---:|---:|---|"
    )?;
    for concurrency in &document.config.concurrency {
        let cohort = document
            .inputs
            .cohorts
            .get(&concurrency.to_string())
            .ok_or("missing measured cohort")?;
        let breakdown = cohort
            .framework_trajectories
            .iter()
            .map(|(framework, count)| {
                let turns = cohort
                    .framework_assistant_turns
                    .get(framework)
                    .ok_or("missing framework turn count")?;
                Ok(format!("{} {count} / {turns}", cell(framework)))
            })
            .collect::<DynResult<Vec<_>>>()?
            .join(" \u{00b7} ");
        writeln!(
            output,
            "| {concurrency} | {} | {} | {} | {breakdown} |",
            cohort.trajectory_count, cohort.assistant_turns, cohort.assistant_turns
        )?;
    }
    super::report_table::write(&mut output, rows)?;
    super::report_method::write(&mut output, document)?;
    if let Some(gates) = &document.gates {
        writeln!(
            output,
            "\n## Acceptance gates\n\nOverall: **{}**",
            match gates.passed {
                Some(true) => "PASS",
                Some(false) => "FAIL",
                None => "NOT EVALUATED",
            }
        )?;
        if !gates.evaluated {
            writeln!(
                output,
                "- **NOT EVALUATED**: no acceptance gates were configured."
            )?;
        } else {
            for check in &gates.checks {
                writeln!(
                    output,
                    "- **{}** {}: {}",
                    if check.passed { "PASS" } else { "FAIL" },
                    code(&check.name),
                    cell(&check.detail)
                )?;
            }
        }
        for failure in &gates.session_acceptance_failures {
            writeln!(
                output,
                "- Session acceptance **{}**: {}",
                if failure.passed { "PASS" } else { "FAIL" },
                cell(&failure.failures.join("; "))
            )?;
        }
    }
    Ok(output)
}
