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
        if !exception_paths.insert(&entry.path) || !SDK_CANDIDATES.contains(&entry.path.as_str()) {
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
        let advisory_workflow = paths
            .iter()
            .any(|path| path == ".github/workflows/python-sdk-compatibility.yml");
        match entry.status.as_str() {
            "conditional_unqualified" if !advisory_workflow => {}
            "qualified" if source_recorded && advisory_workflow => {}
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
