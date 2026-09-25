use super::other_shard::source_lines;
use super::shard_rows::{ManualTsvRow, OutsideCall, OutsideKind};
use crate::command::DynResult;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

const SOURCES: [&str; 5] = [
    "evals/README.md",
    "crates/skippy-quantize/README.md",
    "tools/skippy-stage-rewriter/README.md",
    "docs/AGENTS.md",
    "docs/skippy/manual-smoke/manifest.tsv",
];
const MANUAL_TSV: &str = "docs/skippy/manual-smoke/manifest.tsv";
const MANUAL_HEADER: &str = "key_path\tfixture_path\tstartup_apply_command\tmodel_identifier\tverification_command\texpected_result\tactual_evidence_path\tpass_fail_status";
const SMOKE_RUNNER: &str = "docs/skippy/manual-smoke/runtime_smoke.py";

fn tsv_command(text: &str) -> Option<&str> {
    let mut columns = text.split('\t');
    let _key = columns.next()?;
    let _fixture = columns.next()?;
    let command = columns.next()?;
    if command.starts_with("python3 ") || command.starts_with("python ") {
        Some(command)
    } else {
        None
    }
}

fn is_manual_command(path: &str, text: &str) -> bool {
    let text = text.trim();
    if path.ends_with("manifest.tsv") {
        return tsv_command(text).is_some();
    }
    text.starts_with("python")
        || text.starts_with("uv run ")
        || text.starts_with("scripts/") && text.contains(".py")
}

fn validate_tsv_rows(
    root: &Path,
    commands: BTreeMap<String, String>,
    rows: Vec<ManualTsvRow>,
) -> DynResult<BTreeSet<(String, usize)>> {
    if commands.is_empty() && rows.is_empty() && !root.join(MANUAL_TSV).exists() {
        return Ok(BTreeSet::new());
    }
    let lines = source_lines(root, MANUAL_TSV)?;
    if lines.first().is_none_or(|header| header != MANUAL_HEADER) {
        return Err("other-edges.json: changed manual TSV header".into());
    }
    let mut seen = BTreeSet::new();
    for (line, key, fixture, model, command_id) in rows {
        let source = lines
            .get(
                line.checked_sub(1)
                    .ok_or("other-edges.json: invalid manual TSV line")?,
            )
            .ok_or_else(|| format!("other-edges.json: stale manual TSV line {line}"))?;
        let columns = source.split('\t').collect::<Vec<_>>();
        let command = commands
            .get(&command_id)
            .ok_or_else(|| format!("other-edges.json: missing manual TSV command {command_id}"))?;
        if columns.len() != 8
            || columns[0] != key
            || columns[1] != fixture
            || columns[2] != command
            || columns[3] != model
            || !command.starts_with(&format!("python3 {SMOKE_RUNNER} "))
            || !command.contains(&format!("--fixture {fixture} "))
            || !command.contains("--model-path $MESH_LLM_SMOKE_MODEL_PATH")
        {
            return Err(
                format!("other-edges.json: mismatched manual TSV command/target {line}").into(),
            );
        }
        if !seen.insert((MANUAL_TSV.to_owned(), line)) {
            return Err(
                format!("other-edges.json: duplicate instruction {MANUAL_TSV}:{line}").into(),
            );
        }
    }
    Ok(seen)
}

pub(super) fn validate_manual(
    root: &Path,
    calls: Vec<OutsideCall>,
    commands: BTreeMap<String, String>,
    rows: Vec<ManualTsvRow>,
) -> DynResult<()> {
    let mut expected = BTreeSet::new();
    let mut actual = BTreeSet::new();
    let tsv_rows = validate_tsv_rows(root, commands, rows)?;
    for call in calls {
        if !SOURCES.contains(&call.file.as_str())
            || call.line == 0
            || call.target.trim().is_empty()
            || call.target.starts_with("unresolved:")
            || call.boundary.trim().is_empty()
        {
            return Err(format!(
                "other-edges.json: incomplete instruction {}:{}",
                call.file, call.line
            )
            .into());
        }
        let lines = source_lines(root, &call.file)?;
        if lines
            .get(call.line - 1)
            .is_none_or(|line| line != &call.source_block)
            || !is_manual_command(&call.file, &call.source_block)
        {
            return Err(format!(
                "other-edges.json: stale instruction {}:{}",
                call.file, call.line
            )
            .into());
        }
        if call.file == MANUAL_TSV && call.target != SMOKE_RUNNER {
            return Err(format!(
                "other-edges.json: mismatched manual TSV target {}",
                call.line
            )
            .into());
        }
        if call.file == MANUAL_TSV && !tsv_rows.contains(&(call.file.clone(), call.line)) {
            return Err(format!(
                "other-edges.json: unbound manual TSV instruction {}",
                call.line
            )
            .into());
        }
        match (&call.kind, call.file.as_str()) {
            (OutsideKind::Data, "docs/skippy/manual-smoke/manifest.tsv")
            | (OutsideKind::Provisioning, "crates/skippy-quantize/README.md")
            | (OutsideKind::Instruction, _) => {}
            _ => {
                return Err(format!(
                    "other-edges.json: incorrect instruction disposition {}:{}",
                    call.file, call.line
                )
                .into());
            }
        }
        if !expected.insert((call.file.clone(), call.line)) {
            return Err(format!(
                "other-edges.json: duplicate instruction {}:{}",
                call.file, call.line
            )
            .into());
        }
    }
    expected.extend(tsv_rows);
    for path in SOURCES {
        if !root.join(path).is_file() {
            continue;
        }
        for (index, line) in source_lines(root, path)?.iter().enumerate() {
            if is_manual_command(path, line) {
                actual.insert((path.to_owned(), index + 1));
            }
        }
    }
    let missing_count = actual.difference(&expected).count();
    let stale_count = expected.difference(&actual).count();
    let missing = actual.difference(&expected).take(5).collect::<Vec<_>>();
    let stale = expected.difference(&actual).take(5).collect::<Vec<_>>();
    if !missing.is_empty() || !stale.is_empty() {
        return Err(format!("other-edges.json: {missing_count} missing instructions {missing:?}; {stale_count} stale instructions {stale:?}").into());
    }
    Ok(())
}
