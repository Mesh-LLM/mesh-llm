use super::shard_rows::ScriptSourceCall;
use crate::command::DynResult;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

const SOURCES: [&str; 5] = [
    "scripts/materialize-competitive-inputs.sh",
    "scripts/package-native-runtime.sh",
    "scripts/skippy-family-battery.sh",
    "scripts/verify-native-runtime-package.sh",
    "tools/skippy-stage-rewriter/CMakeLists.txt",
];

fn context(path: &str, line: &str) -> Option<&'static str> {
    let line = line.trim();
    if path.ends_with("CMakeLists.txt") {
        return line
            .starts_with("find_package(Python3 ")
            .then_some("CMake configure-time Python3 Interpreter discovery");
    }
    if line.starts_with("\"$candidate\" -c ") {
        return Some("interpreter version probe");
    }
    if line.starts_with("\"$PYTHON_BIN\" - ")
        || line.starts_with("\"$PYTHON_BIN\" \"$PROMPT_GENERATOR\" ")
        || line.starts_with("\"$PLANNER\" --manifest ")
        || line.contains("\"$PLANNER\" --inspect-gguf ")
    {
        return Some("selected shell interpreter or script invocation");
    }
    None
}

pub(super) fn check_script_source_calls(
    root: &Path,
    records: &[ScriptSourceCall],
) -> DynResult<()> {
    let mut observed = BTreeMap::new();
    for path in SOURCES {
        if !root.join(path).is_file() {
            continue;
        }
        let source = fs::read_to_string(root.join(path))?;
        let mut heredoc = None;
        let mut cmake_command = false;
        let mut cmake_target = false;
        for (index, line) in source.lines().enumerate() {
            let trimmed = line.trim();
            if let Some(end) = heredoc {
                if trimmed == end {
                    heredoc = None;
                }
                continue;
            }
            if let Some(kind) = context(path, trimmed) {
                let hash = hex::encode(Sha256::digest(
                    trimmed
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ")
                        .as_bytes(),
                ));
                let id = format!("{path}:{}:{}", index + 1, &hash[..16]);
                observed.insert(id, kind);
            }
            if path.ends_with("CMakeLists.txt") {
                if trimmed == "COMMAND" {
                    cmake_command = true;
                } else if cmake_command && trimmed == "${Python3_EXECUTABLE}" {
                    cmake_target = true;
                    cmake_command = false;
                } else if cmake_target {
                    if !trimmed.starts_with("${CMAKE_CURRENT_SOURCE_DIR}/tests/test_rewriter.py") {
                        return Err(format!(
                            "script-edges.json: changed CTest continued target {path}:{}",
                            index + 1
                        )
                        .into());
                    }
                    cmake_target = false;
                }
            }
            if trimmed.contains("<<'PY'") {
                heredoc = Some("PY");
            }
        }
        if cmake_target {
            return Err(format!("script-edges.json: missing CTest continued target {path}").into());
        }
    }
    let mut owned = BTreeSet::new();
    for record in records {
        if !owned.insert(record.id.as_str()) {
            return Err(format!(
                "script-edges.json: duplicate independent call {}",
                record.id
            )
            .into());
        }
        if observed
            .get(&record.id)
            .is_none_or(|kind| *kind != record.context)
            || [
                &record.target,
                &record.boundary,
                &record.owner,
                &record.replacement,
                &record.deletion_condition,
            ]
            .iter()
            .any(|value| value.trim().is_empty() || value.starts_with("unresolved:"))
        {
            return Err(format!(
                "script-edges.json: stale or incomplete independent call {}",
                record.id
            )
            .into());
        }
    }
    if let Some(id) = observed.keys().find(|id| !owned.contains(id.as_str())) {
        return Err(format!("script-edges.json: unowned independent call {id}").into());
    }
    Ok(())
}
