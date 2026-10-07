//! Read-only current bindings for review; this does not admit or rewrite ledgers.
use super::{ledger, scan};
use crate::command::{DynResult, print_json, run_command, trimmed_stderr_or_stdout};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Read, path::Path, process::Command};

const LEDGERS: [&str; 8] = [
    "github-edges.json",
    "script-edges.json",
    "test-edges.json",
    "other-edges.json",
    "invocations.json",
    "python-inventory.json",
    "python-exceptions.json",
    "instructions.json",
];

#[derive(Debug, Serialize)]
struct FilePin {
    path: String,
    sha256: String,
    bytes: usize,
    split_newline_count: usize,
}

#[derive(Debug, Serialize)]
struct Binding {
    id: String,
    path: String,
    line: usize,
    kind: String,
    normalized_source_hash: String,
    occurrence: usize,
    source_block: String,
    executable: bool,
}

#[derive(Debug, Serialize)]
struct Observations {
    schema_version: u32,
    scope: &'static str,
    acceptance: bool,
    head: String,
    ledger_preimages: Vec<FilePin>,
    source_files: Vec<FilePin>,
    scanner_source_tree_sha256: String,
    test_candidate_source_tree_sha256: String,
    candidates: Vec<Binding>,
}

fn head(root: &Path) -> DynResult<String> {
    let output = run_command(
        Command::new("git")
            .current_dir(root)
            .args(["rev-parse", "HEAD"]),
    )?;
    if !output.status.success() {
        return Err(format!(
            "binding observations: git HEAD failed: {}",
            trimmed_stderr_or_stdout(&output)
        )
        .into());
    }
    let revision = String::from_utf8(output.stdout)?.trim().to_owned();
    if !matches!(revision.len(), 40 | 64) || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("binding observations: invalid HEAD".into());
    }
    Ok(revision)
}

fn read(root: &Path, path: &str) -> DynResult<Vec<u8>> {
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let mut file = options
        .open(root.join(path))
        .map_err(|error| format!("binding observations: expected regular file {path}: {error}"))?;
    let before = file.metadata()?;
    if !before.is_file() {
        return Err(format!("binding observations: expected regular file {path}").into());
    }
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let after = file.metadata()?;
    if u64::try_from(bytes.len())? != before.len()
        || after.len() != before.len()
        || after.modified()? != before.modified()?
    {
        return Err(format!("binding observations: file changed during read {path}").into());
    }
    Ok(bytes)
}

fn pin(path: &str, bytes: &[u8]) -> DynResult<FilePin> {
    Ok(FilePin {
        path: path.to_owned(),
        sha256: hex::encode(Sha256::digest(bytes)),
        bytes: bytes.len(),
        split_newline_count: std::str::from_utf8(bytes)?.split('\n').count(),
    })
}

fn binding(located: scan::LocatedCandidate) -> DynResult<Binding> {
    let row = located.candidate;
    let (_, identity) = row
        .id
        .rsplit_once('#')
        .ok_or("binding observations: candidate identity")?;
    let parts = identity.split(':').collect::<Vec<_>>();
    let [kind, hash, occurrence] = parts.as_slice() else {
        return Err("binding observations: candidate identity fields".into());
    };
    Ok(Binding {
        kind: (*kind).to_owned(),
        normalized_source_hash: (*hash).to_owned(),
        occurrence: occurrence.parse()?,
        id: row.id,
        path: row.path,
        line: located.line,
        source_block: row.source_block,
        executable: row.executable,
    })
}

fn add_tree(digest: &mut Sha256, path: &str, bytes: &[u8]) {
    digest.update(path.as_bytes());
    digest.update([0]);
    digest.update(bytes);
    digest.update([0]);
}

fn paths(root: &Path) -> DynResult<Vec<String>> {
    let mut paths = ledger::tracked_paths(root)?;
    paths.sort();
    paths.dedup();
    Ok(paths)
}

fn observe(root: &Path, expected_head: Option<&str>) -> DynResult<Observations> {
    let revision = head(root)?;
    if expected_head.is_some_and(|expected| expected != revision) {
        return Err("binding observations: HEAD differs from expected revision".into());
    }
    let initial_paths = paths(root)?;
    let mut snapshot = BTreeMap::new();
    let mut ledger_preimages = Vec::new();
    for name in LEDGERS {
        let path = format!("ci/automation-migration/{name}");
        let bytes = read(root, &path)?;
        ledger_preimages.push(pin(&path, &bytes)?);
        snapshot.insert(path, bytes);
    }
    let mut source_files = Vec::new();
    let mut candidates = Vec::new();
    let mut source_tree = Sha256::new();
    let mut test_tree = Sha256::new();
    for path in initial_paths.iter().filter(|path| scan::scans_path(path)) {
        let bytes = read(root, path)?;
        let located = scan::scan_source_located(path, std::str::from_utf8(&bytes)?);
        add_tree(&mut source_tree, path, &bytes);
        if scan::is_script_test(path) && !located.is_empty() {
            // Same sorted path/NUL/bytes/NUL contract as test_shard::validate.
            add_tree(&mut test_tree, path, &bytes);
        }
        candidates.extend(
            located
                .into_iter()
                .map(binding)
                .collect::<DynResult<Vec<_>>>()?,
        );
        source_files.push(pin(path, &bytes)?);
        snapshot.insert(path.clone(), bytes);
    }
    if head(root)? != revision || paths(root)? != initial_paths {
        return Err("binding observations: repository identity or path census changed".into());
    }
    for (path, bytes) in snapshot {
        if read(root, &path)? != bytes {
            return Err(format!(
                "binding observations: source or ledger changed during observation: {path}"
            )
            .into());
        }
    }
    Ok(Observations {
        schema_version: 1,
        scope: "Current tracked and nonignored source bindings only; no ledger acceptance, classification, retirement, or qualification",
        acceptance: false,
        head: revision,
        ledger_preimages,
        source_files,
        scanner_source_tree_sha256: hex::encode(source_tree.finalize()),
        test_candidate_source_tree_sha256: hex::encode(test_tree.finalize()),
        candidates,
    })
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let expected = match args {
        [command, json] if command == "binding-observations" && json == "--json" => None,
        [command, json, flag, revision]
            if command == "binding-observations" && json == "--json" && flag == "--expect-head" =>
        {
            Some(revision.as_str())
        }
        _ => return Err(
            "usage: cargo xtool automation binding-observations --json [--expect-head <revision>]"
                .into(),
        ),
    };
    print_json(&observe(&crate::repo_consistency::repo_root()?, expected)?)
}

#[cfg(test)]
mod tests;
