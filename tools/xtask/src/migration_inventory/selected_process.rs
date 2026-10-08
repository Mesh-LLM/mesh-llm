#[path = "selected_battery.rs"]
mod battery;
#[path = "selected_repair.rs"]
mod repair;
#[path = "selected_repair_producer.rs"]
mod repair_producer;
#[cfg(test)]
#[path = "selected_repair_tests.rs"]
mod repair_tests;
#[path = "selected_system_one.rs"]
mod system_one;
#[cfg(test)]
#[path = "selected_system_one_tests.rs"]
mod system_one_tests;
use super::ledger::SelectedProcessCall;
use crate::command::DynResult;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

const SELECTED_SCRIPTS: [&str; 14] = [
    "scripts/skippy-system-one-smoke.sh",
    "scripts/ci-compose-product-input.sh",
    "scripts/package-native-runtime.sh",
    "scripts/verify-native-runtime-package.sh",
    "scripts/skippy-family-battery.sh",
    "scripts/skippy-ci-smoke.sh",
    "scripts/skippy-workload-certify.sh",
    "scripts/llama-canary-agent-repair.sh",
    "skippy/scripts/skippy-system-one-smoke.sh",
    "skippy/scripts/package-native-runtime.sh",
    "skippy/scripts/verify-native-runtime-package.sh",
    "skippy/scripts/skippy-family-battery.sh",
    "skippy/scripts/skippy-ci-smoke.sh",
    "skippy/scripts/skippy-workload-certify.sh",
];

fn typed_caller(path: &str) -> bool {
    matches!(
        path.strip_prefix("skippy/").unwrap_or(path),
        "scripts/skippy-system-one-smoke.sh"
            | "scripts/skippy-ci-smoke.sh"
            | "scripts/skippy-workload-certify.sh"
            | "scripts/llama-canary-agent-repair.sh"
    )
}

fn indirect_launch(path: &str, line: &str) -> bool {
    let line = line.trim();
    if line.starts_with('#') || line.starts_with("echo ") || line.starts_with("printf ") {
        return false;
    }
    line.starts_with("\"$candidate\" -c ")
        || line.starts_with("\"${plan_args[@]}\"")
        || line.starts_with("\"$PLANNER\" ")
        || line.contains("\"$PLANNER\" --inspect-gguf ")
        || (matches!(
            path,
            "scripts/skippy-system-one-smoke.sh" | "skippy/scripts/skippy-system-one-smoke.sh"
        ) && line.starts_with("\"${case_command[@]}\""))
        || (typed_caller(path) && typed_launch(line))
}

fn typed_launch(line: &str) -> bool {
    [
        "automation",
        "workload_automation",
        "repair_workload_automation",
        "evidence_command",
    ]
    .iter()
    .any(|name| {
        let expansion = format!("\"${{{name}[@]}}\"");
        line.starts_with(&expansion)
            || line.starts_with(&format!("if {expansion}"))
            || line.contains(&format!("$({expansion}"))
    })
}

fn typed_source_argv(lines: &[&str], index: usize) -> DynResult<String> {
    if lines[index].trim() == "\"${evidence_command[@]}\"" {
        let start = lines[..index]
            .iter()
            .rposition(|line| {
                line.trim()
                    .starts_with("evidence_command=(\"${workload_automation[@]}\"")
            })
            .ok_or("selected process: missing evidence command declaration")?;
        return Ok(lines[start..index]
            .iter()
            .map(|line| line.trim())
            .collect::<Vec<_>>()
            .join("\n"));
    }
    let mut end = index;
    while lines[end].trim().ends_with('\\') {
        end += 1;
        if end >= lines.len() {
            return Err("selected process: incomplete source continuation".into());
        }
    }
    Ok(lines[index..=end]
        .iter()
        .map(|line| line.trim())
        .collect::<Vec<_>>()
        .join("\n"))
}

fn typed_owner(argv: &str) -> Option<&'static str> {
    [
        (
            "automation smoke-observation sdk-source",
            "tools/xtask/src/automation/python_sdk_source.rs",
        ),
        (
            "automation smoke-observation sdk-client",
            "tools/xtask/src/automation/smoke_observation/sdk_supervision/child.rs",
        ),
        (
            "automation system-one-smoke stage",
            "tools/xtask/src/automation/system_one_smoke.rs",
        ),
        (
            "automation system-one-smoke report",
            "tools/xtask/src/automation/system_one_smoke.rs",
        ),
        (
            "automation canary-receipts prepared-source",
            "tools/xtask/src/automation/canary_receipts/package_closure/prepared_source.rs",
        ),
        (
            "automation canary-receipts workload-manifest",
            "tools/xtask/src/automation/canary_receipts/package_closure/workload.rs",
        ),
        (
            "automation canary-timeout",
            "tools/xtask/src/automation/canary_timeout.rs",
        ),
        (
            "automation local-ports",
            "tools/xtask/src/automation/local_ports.rs",
        ),
        (
            "automation binary-stage-readiness",
            "tools/xtask/src/automation/binary_stage_readiness.rs",
        ),
        (
            "automation openai-smoke-config",
            "tools/xtask/src/automation/openai_smoke_config.rs",
        ),
        (
            "automation family-battery-policy",
            "tools/xtask/src/automation/family_battery_policy.rs",
        ),
        (
            "automation workload-smoke-config",
            "tools/xtask/src/automation/workload_smoke_config.rs",
        ),
        (
            "automation workload-media-oracle",
            "tools/xtask/src/automation/workload_smoke/media_comparison/mod.rs",
        ),
        (
            "automation workload-monolithic-oracle",
            "tools/xtask/src/automation/workload_smoke/comparison/mod.rs",
        ),
        (
            "automation workload-tts-oracle",
            "tools/xtask/src/automation/workload_smoke/tts_oracle/mod.rs",
        ),
        (
            "automation workload-oracle-evidence",
            "tools/xtask/src/automation/workload_oracle_evidence/mod.rs",
        ),
        (
            "automation workload-smoke",
            "tools/xtask/src/automation/workload_smoke/mod.rs",
        ),
        ("models resolve", "tools/xtask/src/model_registry/mod.rs"),
    ]
    .into_iter()
    .filter_map(|(command, owner)| argv.find(command).map(|index| (index, owner)))
    .min_by_key(|(index, _)| *index)
    .map(|(_, owner)| owner)
}

const SYSTEM_ONE_OWNER: &str =
    "tools/xtask/src/automation/system_one_cases/mod.rs; bounded explicit native executable driver";

fn check_system_one_binding(
    record: &SelectedProcessCall,
    lines: &[&str],
    index: usize,
) -> DynResult<()> {
    if !matches!(
        record.caller.as_str(),
        "scripts/skippy-system-one-smoke.sh" | "skippy/scripts/skippy-system-one-smoke.sh"
    ) || !lines[index].trim().starts_with("\"${case_command[@]}\"")
    {
        return Ok(());
    }
    let start = index
        .checked_sub(4)
        .ok_or("selected process: missing System One selector")?;
    let expected = [
        "local case_command=(\"${automation[@]}\" automation system-one-cases)",
        "if [[ \"${SYSTEMONE_SMOKE_DRIVER+set}\" == set ]]; then",
        "case_command=(\"${automation[@]}\" automation system-one-cases --driver-executable \"$CASES_DRIVER\" --driver-timeout \"$DRIVER_TIMEOUT_SECS\")",
        "fi",
    ];
    if lines[start..index]
        .iter()
        .map(|line| line.trim())
        .ne(expected)
    {
        return Err("selected process: changed System One default or explicit override".into());
    }
    let argv = format!(
        "{}\n{}",
        expected.join("\n"),
        typed_source_argv(lines, index)?
    );
    if record.argv != argv
        || record.replacement_owner != SYSTEM_ONE_OWNER
        || record.child != "tools/xtask default; bounded explicit native $CASES_DRIVER override"
    {
        return Err("selected process: changed System One mixed launch binding".into());
    }
    Ok(())
}

fn check_typed_binding(
    root: &Path,
    record: &SelectedProcessCall,
    lines: &[&str],
    index: usize,
) -> DynResult<()> {
    if let Some(binding) = repair_producer::binding(&record.caller, lines, index)?
        .or(repair::binding(&record.caller, lines, index)?)
    {
        if record.argv != binding.argv
            || record.replacement_owner != binding.owner
            || record.child != binding.child
        {
            return Err("selected process: changed repair launch binding".into());
        }
        return Ok(());
    }
    if !typed_caller(&record.caller) || !typed_launch(lines[index].trim()) {
        return Ok(());
    }
    let argv = typed_source_argv(lines, index)?;
    if argv.contains("automation smoke-observation sdk-client") {
        let text = lines.join("\n");
        let selection = if lines[index].trim().ends_with('\\') {
            index + 1
        } else {
            index
        };
        let child = super::sdk_calls::target(
            root,
            &record.caller,
            &text,
            selection + 1,
            lines[selection].trim(),
        )?
        .ok_or("selected process: missing closed SDK selection")?;
        if record.child != child || !record.child_source_known {
            return Err("selected process: changed SDK child binding".into());
        }
    }
    let owner = typed_owner(&argv).ok_or("selected process: unowned typed command")?;
    if record.argv != argv || record.replacement_owner != owner {
        return Err(format!(
            "selected process: changed typed source or owner {}:{}",
            record.caller, record.line
        )
        .into());
    }
    Ok(())
}

fn validate_record(root: &Path, record: &SelectedProcessCall) -> DynResult<()> {
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
    if record
        .argv
        .contains("automation smoke-observation sdk-client")
    {
        let text = fs::read_to_string(root.join(&record.caller))?;
        let lines: Vec<_> = text.lines().collect();
        let index = record.line.checked_sub(1).ok_or("SDK source line")?;
        let selection = if lines
            .get(index)
            .is_some_and(|line| line.trim().ends_with('\\'))
        {
            index + 1
        } else {
            index
        };
        let block = lines.get(selection).ok_or("SDK source selection")?.trim();
        if super::sdk_calls::target(root, &record.caller, &text, selection + 1, block)?
            != Some(record.child.as_str())
        {
            return Err("selected interpreter: external SDK selector changed".into());
        }
        return Ok(());
    }
    if record.child_source_known
        && super::scan::is_script(&record.child)
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
    Ok(())
}

pub(super) fn check_selected_processes(
    root: &Path,
    records: &[SelectedProcessCall],
) -> DynResult<()> {
    let mut recorded = BTreeMap::new();
    for record in records {
        validate_record(root, record)?;
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
        if matches!(
            path,
            "scripts/skippy-family-battery.sh" | "skippy/scripts/skippy-family-battery.sh"
        ) && text
            .lines()
            .any(|line| indirect_launch(path, line) && line.contains("$PLANNER"))
            && !text.lines().any(|line| {
                line.trim()
                    == format!(
                        "PLANNER=\"$ROOT/{}plan-family-battery.py\"",
                        if path.starts_with("skippy/") {
                            "skippy/scripts/"
                        } else {
                            "scripts/"
                        }
                    )
            })
        {
            return Err("selected interpreter: changed family planner binding".into());
        }
        let lines = text.lines().collect::<Vec<_>>();
        repair::check_shape(path, &lines)?;
        system_one::check_shape(path, &lines)?;
        for (index, line) in lines.iter().enumerate() {
            if !indirect_launch(path, line) && !repair::is_launch(path, &lines, index) {
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
            battery::check(record, &lines, index)?;
            check_system_one_binding(record, &lines, index)?;
            check_typed_binding(root, record, &lines, index)?;
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
