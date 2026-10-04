use crate::command::DynResult;
use serde::Serialize;
use std::{fs, io::Write, path::Path};

pub(super) fn directory(path: &Path) -> DynResult<()> {
    fs::create_dir_all(path)?;
    if !fs::metadata(path)?.is_dir() {
        return Err("stability evidence ancestor is not a directory".into());
    }
    Ok(())
}

pub(super) fn bytes(path: &Path, bytes: &[u8]) -> DynResult<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    directory(parent)?;
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_file() => (),
        Ok(_) => return Err("stability evidence output is not a regular file".into()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => (),
        Err(error) => return Err(error.into()),
    }
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    temporary.write_all(bytes)?;
    temporary.flush()?;
    temporary.persist(path)?;
    Ok(())
}

pub(super) fn json(path: &Path, value: &impl Serialize) -> DynResult<()> {
    let mut output = serde_json::to_vec_pretty(value)?;
    output.push(b'\n');
    bytes(path, &output)
}

pub(super) fn jsonl(path: &Path, rows: &[impl Serialize]) -> DynResult<()> {
    let mut output = Vec::new();
    for row in rows {
        serde_json::to_writer(&mut output, row)?;
        output.push(b'\n');
    }
    bytes(path, &output)
}

pub(super) fn markdown(
    summary: &super::reports::Summary<'_>,
    probes: &[super::cases::Case],
    commands: &[super::reports::CommandRow],
) -> String {
    let status = if summary.counts.ok { "PASS" } else { "FAIL" };
    let mut text = format!(
        "# Nightly Stability Summary\n\nStatus: **{status}**\n\nPassed: {}. Failed: {}. Prerequisites: {}. Cancelled: {}.\n\n| Result | Model | Attempt | Phase | Detail |\n|---|---|---:|---|---|\n",
        summary.counts.passed, summary.counts.failed, summary.counts.prereq, summary.cancelled
    );
    for row in probes {
        text.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            if row.ok { "PASS" } else { "FAIL" },
            cell(row.model.as_deref().unwrap_or("-")),
            row.attempt.unwrap_or(0),
            row.phase,
            cell(&row.detail)
        ));
    }
    text.push_str("\n| Result | Command | Exit | Detail | Log |\n|---|---|---:|---|---|\n");
    for row in commands {
        text.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            row.status.name(),
            cell(&row.name),
            row.exit_code,
            cell(&row.detail),
            cell(&row.log)
        ));
    }
    text.push_str("\n## Timing snapshot\n\n| Group | Passed | Failed | Prerequisites | Elapsed ms |\n|---|---:|---:|---:|---:|\n");
    for (name, counts) in [
        ("OpenAI surface probes", &summary.probes),
        ("Command probes", &summary.commands),
        ("Release attestation", &summary.attestation.counts),
        ("Total", &summary.counts),
    ] {
        text.push_str(&format!(
            "| {name} | {} | {} | {} | {} |\n",
            counts.passed, counts.failed, counts.prereq, counts.elapsed_ms
        ));
    }
    let attestation = summary.release_attestation;
    text.push_str(&format!(
        "\nRelease attestation: {}. Expected: {}. {}\n",
        cell(&attestation.status),
        cell(
            attestation
                .expected_status
                .as_deref()
                .unwrap_or("not configured")
        ),
        cell(attestation.error.as_deref().unwrap_or(""))
    ));
    text
}

fn cell(value: &str) -> String {
    value.replace('|', "\\|").replace(['\r', '\n'], " ")
}
