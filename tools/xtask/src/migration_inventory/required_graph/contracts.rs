use super::super::scan::Candidate;
use crate::command::DynResult;
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct InvocationLedger {
    github_source_records: Vec<Invocation>,
    #[serde(default)]
    inline_source_records: Vec<Invocation>,
}

#[derive(Deserialize)]
struct ScriptLedger {
    groups: Vec<ScriptGroup>,
}

#[derive(Deserialize)]
struct ScriptGroup {
    file: String,
    members: Vec<(usize, String, usize, String, String, String)>,
}

#[derive(Deserialize)]
struct PythonInventory {
    files: Vec<PythonOwner>,
}

#[derive(Deserialize)]
struct PythonOwner {
    path: String,
    replacement_owner: String,
    deletion_condition: String,
}

#[derive(Deserialize)]
struct Invocation {
    id: String,
    source_block: String,
    disposition: String,
    argv: String,
    status_output_effects: String,
    transitive_boundary: String,
    replacement: String,
    deletion_condition: String,
    reason: String,
}

pub(super) struct Contract {
    pub(super) source: String,
    pub(super) argv: String,
    pub(super) effects: String,
}

pub(super) struct Contracts {
    root: PathBuf,
    records: Vec<Invocation>,
    inline_records: Vec<Invocation>,
    scripts: Vec<ScriptGroup>,
    owners: Vec<PythonOwner>,
}

impl Contracts {
    pub(super) fn load(root: &Path) -> DynResult<Self> {
        let file = root.join("ci/automation-migration/invocations.json");
        let (records, inline_records) = if file.is_file() {
            let ledger = serde_json::from_slice::<InvocationLedger>(&fs::read(file)?)?;
            (ledger.github_source_records, ledger.inline_source_records)
        } else {
            (Vec::new(), Vec::new())
        };
        let scripts = root.join("ci/automation-migration/script-edges.json");
        let scripts = if scripts.is_file() {
            serde_json::from_slice::<ScriptLedger>(&fs::read(scripts)?)?.groups
        } else {
            Vec::new()
        };
        let owners = root.join("ci/automation-migration/python-inventory.json");
        let owners = if owners.is_file() {
            serde_json::from_slice::<PythonInventory>(&fs::read(owners)?)?.files
        } else {
            Vec::new()
        };
        Ok(Self {
            root: root.to_owned(),
            records,
            inline_records,
            scripts,
            owners,
        })
    }

    pub(super) fn for_call(
        &self,
        path: &str,
        line: usize,
        block: &str,
        child: &str,
        observed: &[Candidate],
        validated: &BTreeSet<String>,
    ) -> Option<Contract> {
        if let Some(record) = self.match_call(path, line, block, observed, validated)
            && record
                .0
                .transitive_boundary
                .split(|character: char| {
                    !(character.is_ascii_alphanumeric()
                        || matches!(character, '/' | '.' | '_' | '-'))
                })
                .any(|target| target == child)
        {
            return Some(Self::contract(record.0, record.1));
        }
        let candidate = observed.iter().find(|row| {
            row.path == path
                && row.source_block == block
                && validated.contains(&row.id)
                && (row.executable || super::contexts::selected_interpreter_call(block))
        })?;
        let owner = self.owners.iter().find(|row| {
            row.path == child
                && !row.replacement_owner.is_empty()
                && !row.deletion_condition.is_empty()
        })?;
        let member = self
            .scripts
            .iter()
            .filter(|group| group.file == path)
            .flat_map(|group| &group.members)
            .find(|member| {
                member.0 == line
                    && member.3 == "script"
                    && member.4.split_whitespace().next() == Some(child)
                    && candidate.id == format!("{path}#candidate:{}:{}", member.1, member.2)
                    && !member.5.is_empty()
            })?;
        Some(Contract {
            source: format!(
                "ci/automation-migration/script-edges.json:{path}:{line}; ci/automation-migration/python-inventory.json:{child}"
            ),
            argv: block.to_owned(),
            effects: format!(
                "{}; replacement owner: {}; deletion condition: {}",
                member.5, owner.replacement_owner, owner.deletion_condition
            ),
        })
    }

    pub(super) fn inline_call(
        &self,
        path: &str,
        line: usize,
        block: &str,
        observed: &[Candidate],
        validated: &BTreeSet<String>,
    ) -> Option<Contract> {
        self.match_call(path, line, block, observed, validated)
            .map(|(record, source)| Self::contract(record, source))
    }

    fn match_call<'a>(
        &'a self,
        path: &str,
        line: usize,
        block: &str,
        observed: &[Candidate],
        validated: &BTreeSet<String>,
    ) -> Option<(&'a Invocation, &'static str)> {
        if line == 0 {
            return None;
        }
        let occurrence = fs::read_to_string(self.root.join(path))
            .ok()?
            .lines()
            .take(line)
            .filter(|source| source.trim() == block)
            .count();
        let candidate = observed
            .iter()
            .filter(|row| row.path == path && row.source_block == block)
            .nth(occurrence.checked_sub(1)?)?;
        if !candidate.executable {
            return None;
        }
        if !validated.contains(&candidate.id) {
            return None;
        }
        self.records
            .iter()
            .map(|record| (record, "github_source_records"))
            .chain(
                self.inline_records
                    .iter()
                    .map(|record| (record, "inline_source_records")),
            )
            .find(|(row, _)| {
                row.id == candidate.id
                    && row.source_block == block
                    && row.disposition == "execution"
                    && !row.argv.is_empty()
                    && !row.status_output_effects.is_empty()
                    && !row.transitive_boundary.is_empty()
                    && !row.replacement.is_empty()
                    && !row.deletion_condition.is_empty()
                    && !row.reason.is_empty()
            })
    }

    fn contract(record: &Invocation, source: &str) -> Contract {
        Contract {
            source: format!(
                "ci/automation-migration/invocations.json#{source}:{}",
                record.id
            ),
            argv: record.argv.clone(),
            effects: record.status_output_effects.clone(),
        }
    }
}
