use super::ledger::MigrationLedgers;
use crate::command::DynResult;
use std::collections::BTreeSet;

const SDK_CANDIDATES: [&str; 4] = [
    "scripts/ci-openai-python-smoke.py",
    "scripts/ci-langchain-openai-smoke.py",
    "scripts/ci-litellm-smoke.py",
    "scripts/ci-openai-embeddings-smoke.py",
];

pub(super) fn check_exceptions(paths: &[String], ledgers: &MigrationLedgers) -> DynResult<()> {
    let mut exception_paths = BTreeSet::new();
    for entry in &ledgers.exceptions.exceptions {
        let retained_reader = [
            "evals/agentic-trajectory-manifest.py",
            "scripts/generate-bench-corpus.py",
        ]
        .contains(&entry.path.as_str());
        if !exception_paths.insert(&entry.path)
            || !(SDK_CANDIDATES.contains(&entry.path.as_str()) || retained_reader)
        {
            return Err(format!(
                "automation policy: fabricated or duplicate Python exception {}",
                entry.path
            )
            .into());
        }
        let complete =
            entry.callers.as_ref().is_some_and(|items| {
                !items.is_empty() && items.iter().all(|item| !item.is_empty())
            }) && entry.local_dependency_files.as_ref().is_some_and(|items| {
                !items.is_empty() && items.iter().all(|item| !item.is_empty())
            }) && entry
                .cadence
                .as_deref()
                .is_some_and(|value| !value.is_empty())
                && entry
                    .purpose
                    .as_deref()
                    .is_some_and(|value| !value.is_empty())
                && entry
                    .isolation_test
                    .as_deref()
                    .is_some_and(|value| !value.is_empty());
        if !complete {
            return Err(format!(
                "automation policy: incomplete Python exception {}",
                entry.path
            )
            .into());
        }
        let source_recorded = paths.iter().any(|path| path == &entry.path)
            && ledgers
                .inventory
                .files
                .iter()
                .any(|file| file.path == entry.path);
        match entry.status.as_str() {
            "maintainer_retained" if retained_reader && source_recorded => {}
            // L8 retains required SDK cadence. A workflow filename is not qualification.
            // Qualification requires a separately reviewed execution-evidence contract.
            "conditional_unqualified" if SDK_CANDIDATES.contains(&entry.path.as_str()) => {}
            _ => {
                return Err(format!(
                    "automation policy: unqualified Python exception {} ({})",
                    entry.path, entry.status
                )
                .into());
            }
        }
    }
    Ok(())
}
