//! Verify row coverage and declared evidence, without promoting blocked rows to passes.
use super::text;
use crate::{command::DynResult, repository::check_args::Grammar};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};
const HEADER: [&str; 8] = [
    "key_path",
    "fixture_path",
    "startup_apply_command",
    "model_identifier",
    "verification_command",
    "expected_result",
    "actual_evidence_path",
    "pass_fail_status",
];
const STATUSES: [&str; 6] = [
    "PASS",
    "EXPECTED_REJECTED_PASS",
    "BLOCKED_STAGED_LOCAL",
    "BLOCKED_NETWORK_LOCAL",
    "BLOCKED_ASSET_LOCAL",
    "BLOCKED_EVIDENCE_GAP",
];
const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation manual-smoke coverage [--root PATH] [--matrix PATH] [--manifest PATH] [--required-evidence PATH]...",
    values: &["--root", "--matrix", "--manifest", "--required-evidence"],
    flags: &["--help"],
};
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let parsed = GRAMMAR.parse(args).map_err(|e| format!("{e:?}"))?;
    if parsed.flag("--help") {
        println!("{}", GRAMMAR.usage);
        return Ok(());
    }
    if !parsed.positionals.is_empty() {
        return Err("coverage has no positional arguments".into());
    }
    let root = parsed
        .last("--root")
        .map(PathBuf::from)
        .unwrap_or(std::env::current_dir()?)
        .canonicalize()?;
    let matrix = root.join(
        parsed
            .last("--matrix")
            .unwrap_or("docs/skippy/CONFIGURATION.md"),
    );
    let manifest = root.join(
        parsed
            .last("--manifest")
            .unwrap_or("docs/skippy/manual-smoke/manifest.tsv"),
    );
    let required = parsed.all("--required-evidence");
    // Historical default coverage retains both task-11 receipts. Overrides are explicit.
    let required = if required.is_empty() {
        vec![
            ".sisyphus/evidence/task-11-manual-runtime-smoke.txt",
            ".sisyphus/evidence/task-11-manual-runtime-smoke-error.txt",
        ]
    } else {
        required
    };
    let count = check(&root, &text(&matrix)?, &text(&manifest)?, &required)?;
    println!(
        "Coverage OK: {count} matrix keys, {count} manifest rows; declared evidence, not new runtime qualification"
    );
    Ok(())
}
fn keys(matrix: &str) -> BTreeSet<String> {
    let mut keys = BTreeSet::new();
    for line in matrix.lines().filter(|line| {
        line.starts_with('|') && !line.starts_with("|---") && !line.contains("Config key path")
    }) {
        let cells: Vec<_> = line.split('|').collect();
        let Some(cell) = cells.get(3).map(|cell| cell.trim()) else {
            continue;
        };
        if cell == "—" || cell.starts_with("see ") || cell.starts_with("shared ") {
            continue;
        }
        for raw in cell.split('`').skip(1).step_by(2) {
            keys.extend(
                raw.split("<br>")
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned),
            );
        }
    }
    keys
}
fn columns(line: &str) -> DynResult<Vec<String>> {
    let mut values = Vec::new();
    let mut chars = line.chars().peekable();
    loop {
        let quoted = chars.peek() == Some(&'"');
        if quoted {
            chars.next();
        }
        let mut cell = String::new();
        let mut closed = !quoted;
        while let Some(ch) = chars.next() {
            if quoted && ch == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    cell.push('"');
                } else {
                    closed = true;
                    break;
                }
            } else if !quoted && ch == '\t' {
                break;
            } else {
                cell.push(ch);
            }
        }
        if !closed {
            return Err("unterminated/multiline TSV quote unsupported".into());
        }
        values.push(cell);
        if quoted && chars.peek().is_some() && chars.next() != Some('\t') {
            return Err("invalid quoted TSV delimiter".into());
        }
        if chars.peek().is_none() {
            break;
        }
    }
    Ok(values)
}
fn forbidden(value: &str) -> bool {
    let value = value.to_lowercase();
    value.starts_with("/users/")
        || value.starts_with("/home/")
        || [
            "/.cache/huggingface/",
            "/var/folders/",
            "/private/var/folders/",
        ]
        .iter()
        .any(|pattern| value.contains(pattern))
        || value.split("models--").skip(1).any(|rest| {
            rest.split_once("/snapshots/")
                .is_some_and(|(repo, _)| !repo.contains('/'))
        })
}
fn placeholder(value: &str) -> bool {
    value.split('<').skip(1).any(|rest| {
        rest.split_once('>')
            .is_some_and(|(inside, _)| !inside.is_empty())
    })
}
fn row(root: &Path, values: &[String]) -> DynResult<()> {
    let key = &values[0];
    let status = values[7].trim();
    if !STATUSES.contains(&status) {
        return Err(format!("invalid status for {key}").into());
    }
    if !std::fs::metadata(root.join(values[1].trim()))?.is_file() {
        return Err(format!("missing fixture for {key}").into());
    }
    for (index, value) in values.iter().enumerate().take(7).skip(1) {
        if value.trim().is_empty() || placeholder(value) || index != 6 && forbidden(value) {
            return Err(format!(
                "blank/placeholder/machine-local {} for {key}",
                HEADER[index]
            )
            .into());
        }
    }
    if text(&root.join(values[6].trim()))?.trim().is_empty() {
        return Err(format!("empty evidence for {key}").into());
    }
    if ["PASS", "EXPECTED_REJECTED_PASS"].contains(&status)
        && values[2].to_lowercase().contains("not-run locally")
    {
        return Err("non-executable command marked successful".into());
    }
    if status == "PASS" && !values[2].contains("--model-path $MESH_LLM_SMOKE_MODEL_PATH") {
        return Err("portable model env var missing for PASS".into());
    }
    if status == "PASS"
        && key.starts_with("speculative.")
        && !values[2].contains("--draft-path $MESH_LLM_SMOKE_DRAFT_PATH")
    {
        return Err("portable draft env var missing for speculative PASS".into());
    }
    if status != "PASS"
        && ["$MESH_LLM_SMOKE_MODEL_PATH", "$MESH_LLM_SMOKE_DRAFT_PATH"]
            .iter()
            .any(|name| values[3].contains(name))
    {
        return Err("env var leaked into non-PASS model identifier".into());
    }
    Ok(())
}
fn check(root: &Path, matrix: &str, manifest: &str, required: &[&str]) -> DynResult<usize> {
    let expected = keys(matrix);
    if expected.is_empty() {
        return Err("configuration matrix has no keys".into());
    }
    let mut lines = manifest.lines();
    if columns(lines.next().ok_or("empty manifest")?)? != HEADER {
        return Err("manual smoke TSV header mismatch".into());
    }
    let mut seen = BTreeMap::new();
    for line in lines.filter(|line| !line.trim().is_empty()) {
        let values = columns(line)?;
        if values.len() != HEADER.len() {
            return Err("manual smoke TSV requires eight columns".into());
        }
        if seen.insert(values[0].clone(), ()).is_some() {
            return Err("duplicate manifest key".into());
        }
        row(root, &values)?;
    }
    if seen.keys().cloned().collect::<BTreeSet<_>>() != expected {
        return Err("missing or extra manifest key paths".into());
    }
    for path in required {
        let evidence = text(&root.join(path))?;
        if evidence.trim().is_empty()
            || !evidence.contains("COMMAND:") && !evidence.contains("Coverage check")
        {
            return Err("required evidence missing command/result markers".into());
        }
    }
    Ok(expected.len())
}
