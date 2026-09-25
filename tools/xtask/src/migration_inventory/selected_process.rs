use super::ledger::SelectedProcessCall;
use crate::command::DynResult;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const SELECTED_SCRIPTS: [&str; 4] = [
    "scripts/ci-compose-product-input.sh",
    "scripts/package-native-runtime.sh",
    "scripts/verify-native-runtime-package.sh",
    "scripts/skippy-family-battery.sh",
];

fn indirect_launch(line: &str) -> bool {
    let line = line.trim();
    if line.starts_with('#') || line.starts_with("echo ") || line.starts_with("printf ") {
        return false;
    }
    line.starts_with("\"$candidate\" -c ")
        || line.starts_with("\"${plan_args[@]}\"")
        || line.starts_with("\"$PLANNER\" ")
        || line.contains("\"$PLANNER\" --inspect-gguf ")
}

pub(super) fn check_selected_processes(
    root: &Path,
    records: &[SelectedProcessCall],
) -> DynResult<()> {
    let mut recorded = BTreeMap::new();
    for record in records {
        if !SELECTED_SCRIPTS.contains(&record.caller.as_str())
            || record.line == 0
            || record.source_block.is_empty()
            || record.child.is_empty()
            || record.argv.is_empty()
            || record.status_streams_effects.is_empty()
            || record.replacement_owner.is_empty()
        {
            return Err(format!(
                "selected interpreter: incomplete record {}:{}",
                record.caller, record.line
            )
            .into());
        }
        if record.child_source_known
            && record.child.starts_with("scripts/")
            && !root.join(&record.child).is_file()
        {
            return Err(format!(
                "selected interpreter: child source mismatch {}",
                record.child
            )
            .into());
        }
        if record.child.ends_with(".py") {
            let child = fs::read_to_string(root.join(&record.child))?;
            if !child
                .lines()
                .next()
                .is_some_and(|line| line.starts_with("#!") && line.contains("python"))
            {
                return Err(format!(
                    "selected interpreter: missing Python shebang {}",
                    record.child
                )
                .into());
            }
        }
        if recorded
            .insert((record.caller.as_str(), record.line), record)
            .is_some()
        {
            return Err(format!(
                "selected interpreter: duplicate {}:{}",
                record.caller, record.line
            )
            .into());
        }
    }
    for path in SELECTED_SCRIPTS {
        if !root.join(path).is_file() {
            continue;
        }
        let text = fs::read_to_string(root.join(path))?;
        if path == "scripts/skippy-family-battery.sh"
            && !text
                .lines()
                .any(|line| line.trim() == "PLANNER=\"$ROOT/scripts/plan-family-battery.py\"")
        {
            return Err("selected interpreter: changed family planner binding".into());
        }
        for (index, line) in text.lines().enumerate() {
            if !indirect_launch(line) {
                continue;
            }
            let number = index + 1;
            let Some(record) = recorded.remove(&(path, number)) else {
                return Err(format!(
                    "selected interpreter: unowned required execution {path}:{number}: {}",
                    line.trim()
                )
                .into());
            };
            if record.source_block != line.trim() {
                return Err(format!("selected interpreter: changed source {path}:{number}").into());
            }
        }
    }
    if let Some(((path, line), _)) = recorded.into_iter().next() {
        return Err(format!("selected interpreter: stale source {path}:{line}").into());
    }
    Ok(())
}
