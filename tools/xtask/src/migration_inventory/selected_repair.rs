//! Closed normal-repair forwarding surfaces, not a general shell evaluator.
use crate::command::DynResult;
use std::ops::Range;

const CALLER: &str = "scripts/llama-canary-agent-repair.sh";
const PLAN_CALLS: [&str; 3] = [
    "repair_family_plan_step \"$log\" \"${repair_workload_automation[@]}\" --repo-root \"$ROOT\" ci family-plan \\",
    "repair_family_plan_step \"$log\" \"${repair_workload_automation[@]}\" --repo-root \"$ROOT\" ci family-plan \\",
    "repair_family_plan_step \"$log\" \"${repair_workload_automation[@]}\" automation family-battery-policy --cache \\",
];
const LOCAL_CALLS: [&str; 4] = [
    "repair_source_inspection local-manifest-policy > >(tee -a \"$MANIFEST_POLICY_LOG\") 2>&1",
    "repair_source_inspection local-parity-inventory \"\" \"$CERTIFY_LOG\" || return 1",
    "repair_source_inspection local-split-roster false",
    "repair_source_inspection local-split-roster true",
];
const PLAN_USES: [&str; 2] = [
    "repair_family_plan 256",
    "repair_family_plan 1 \"$CERTIFY_LOG\" || return 1",
];

pub(super) struct Binding {
    pub(super) argv: String,
    pub(super) owner: &'static str,
    pub(super) child: &'static str,
}
fn function(lines: &[&str], name: &str) -> DynResult<Option<Range<usize>>> {
    let marker = format!("{name}() {{");
    let starts = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l.trim() == marker)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    match starts.as_slice() {
        [] => Ok(None),
        [start] => {
            let end = lines[*start + 1..]
                .iter()
                .position(|l| *l == "}")
                .ok_or("selected repair: unclosed function")?
                + *start
                + 2;
            Ok(Some(*start..end))
        }
        _ => Err("selected repair: duplicate function".into()),
    }
}
fn calls<'a>(lines: &'a [&str], prefix: &str) -> Vec<&'a str> {
    lines
        .iter()
        .map(|l| l.trim())
        .filter(|l| l.starts_with(prefix))
        .collect()
}
fn check_presence(lines: &[&str], name: &str) -> DynResult<bool> {
    let present = function(lines, name)?.is_some();
    if !present && !calls(lines, &format!("{name} ")).is_empty() {
        return Err("selected repair: forwarding call has no owning helper".into());
    }
    Ok(present)
}
fn check_plan_arguments(lines: &[&str]) -> DynResult<()> {
    let expected = [
        r#"--manifest "$manifest" --shard-count "$shards" --output "$PLAN_PATH" || return 1"#,
        r#"--manifest "$manifest" --verify-plan "$PLAN_PATH" || return 1"#,
        r#""$ROOT" "$manifest" "$PLAN_PATH" "${HF_CACHE:?}" || return 1"#,
    ];
    let actual = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.trim().starts_with("repair_family_plan_step "))
        .map(|(index, _)| lines.get(index + 1).map(|line| line.trim()))
        .collect::<Vec<_>>();
    if actual != expected.map(Some) {
        return Err("selected repair: changed plan input authority".into());
    }
    Ok(())
}
fn check_local_arguments(lines: &[&str], range: Range<usize>) -> DynResult<()> {
    let body = lines[range]
        .iter()
        .map(|line| line.trim())
        .collect::<Vec<_>>();
    let declarations = body
        .iter()
        .filter(|line| line.starts_with("local verb=") || line.starts_with("verb="))
        .copied()
        .collect::<Vec<_>>();
    if declarations
        != [r#"local verb="$1" check="${2:-}" log="${3:-}" transaction_root input status"#]
    {
        return Err("selected repair: changed local verb input authority".into());
    }
    let launches = body
        .iter()
        .filter(|line| line.contains("automation canary-receipts"))
        .copied()
        .collect::<Vec<_>>();
    if launches
        != [
            r#""${repair_workload_automation[@]}" automation canary-receipts "$verb" --input "$input"; then"#,
            r#"elif "${repair_workload_automation[@]}" automation canary-receipts "$verb" --input "$input"; then"#,
        ]
    {
        return Err("selected repair: changed local launch ownership".into());
    }
    Ok(())
}
pub(super) fn check_shape(path: &str, lines: &[&str]) -> DynResult<()> {
    if path != CALLER {
        return Ok(());
    }
    if check_presence(lines, "repair_family_plan_step")? {
        check_plan_arguments(lines)?;
        if !check_presence(lines, "repair_family_plan")? {
            return Err("selected repair: absent plan projection owner".into());
        }
        if calls(lines, "repair_family_plan_step ") != PLAN_CALLS
            || calls(lines, "repair_family_plan ") != PLAN_USES
        {
            return Err("selected repair: changed plan forwarding ownership".into());
        }
        for (index, _) in lines
            .iter()
            .enumerate()
            .filter(|(_, l)| l.trim().starts_with("repair_family_plan "))
        {
            if index == 0 || lines[index - 1].trim() != "if [[ \"$HARNESS_MODE\" == repair ]]; then"
            {
                return Err("selected repair: plan call outside exact repair mode".into());
            }
        }
        let range = function(lines, "repair_family_plan_step")?
            .ok_or("selected repair: missing plan step")?;
        let body = lines[range].iter().map(|l| l.trim()).collect::<Vec<_>>();
        for required in [
            "repair_workload_controller_unchanged || return 1",
            "run_verification_logged \"full family certification plan\" \"$log\" \"$@\"",
            "\"$@\"",
        ] {
            if !body.contains(&required) {
                return Err("selected repair: changed bounded plan launch".into());
            }
        }
    }
    if check_presence(lines, "repair_source_inspection")? {
        if calls(lines, "repair_source_inspection ") != LOCAL_CALLS {
            return Err("selected repair: changed or expanded local verb ownership".into());
        }
        for (index, _) in lines
            .iter()
            .enumerate()
            .filter(|(_, l)| l.trim().starts_with("repair_source_inspection "))
        {
            if index == 0
                || !matches!(
                    lines[index - 1].trim(),
                    "if [[ \"$HARNESS_MODE\" == repair ]]; then"
                        | "elif [[ \"$HARNESS_MODE\" == repair ]]; then"
                )
            {
                return Err("selected repair: local call outside exact repair mode".into());
            }
        }
        let range = function(lines, "repair_source_inspection")?
            .ok_or("selected repair: missing local helper")?;
        check_local_arguments(lines, range.clone())?;
        if !lines[range]
            .iter()
            .any(|l| l.trim() == "repair_workload_controller_unchanged || return 1")
        {
            return Err("selected repair: missing frozen controller recheck".into());
        }
    }
    Ok(())
}
pub(super) fn is_launch(path: &str, lines: &[&str], index: usize) -> bool {
    if path != CALLER {
        return false;
    }
    for name in ["repair_family_plan_step", "repair_source_inspection"] {
        if let Ok(Some(range)) = function(lines, name)
            && range.contains(&index)
            && (lines[index].trim() == "\"$@\""
                || lines[index]
                    .trim()
                    .starts_with("run_verification_logged \"full family certification plan\"")
                || lines[index].trim().contains(
                    "\"${repair_workload_automation[@]}\" automation canary-receipts \"$verb\"",
                ))
        {
            return true;
        }
    }
    false
}
pub(super) fn binding(path: &str, lines: &[&str], index: usize) -> DynResult<Option<Binding>> {
    if !is_launch(path, lines, index) {
        return Ok(None);
    }
    check_shape(path, lines)?;
    let plan = function(lines, "repair_family_plan_step")?.is_some_and(|r| r.contains(&index));
    let name = if plan {
        "repair_family_plan_step"
    } else {
        "repair_source_inspection"
    };
    let range = function(lines, name)?.ok_or("selected repair: absent launch helper")?;
    let mut source = lines[range].iter().map(|l| l.trim()).collect::<Vec<_>>();
    if plan {
        let projection = function(lines, "repair_family_plan")?
            .ok_or("selected repair: missing plan projection")?;
        source.extend(lines[projection].iter().map(|l| l.trim()));
        source.extend(PLAN_USES);
    } else {
        source.extend(LOCAL_CALLS);
    }
    Ok(Some(Binding {
        argv: source.join("\n"),
        owner: if plan {
            "tools/xtask/src/ci_plan/family/mod.rs; tools/xtask/src/automation/family_battery_policy.rs"
        } else {
            "tools/xtask/src/automation/canary_receipts/package_closure/local_inspection.rs"
        },
        child: "pre-agent frozen normal-repair controller; closed source-owned command set",
    }))
}
