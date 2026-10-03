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
fn guarded_certification_plan(lines: &[&str], index: usize) -> DynResult<bool> {
    let Some(range) = function(lines, "run_certification")? else {
        return Ok(false);
    };
    let expected = [
        "run_certification() {",
        r#"if [[ "$HARNESS_MODE" != repair && "$HARNESS_MODE" != verify ]]; then"#,
        r#"echo "local certification requires repair or verify mode" >&2"#,
        "return 2",
        "fi",
    ];
    Ok(range.contains(&index)
        && lines[index].trim() == PLAN_USES[1]
        && lines[range]
            .iter()
            .take(expected.len())
            .map(|line| line.trim())
            .eq(expected))
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
            let directly_guarded = index > 0
                && matches!(
                    lines[index - 1].trim(),
                    "if [[ \"$HARNESS_MODE\" == repair ]]; then"
                        | "if [[ \"$HARNESS_MODE\" == repair || \"$HARNESS_MODE\" == verify ]]; then"
                );
            if !directly_guarded && !guarded_certification_plan(lines, index)? {
                return Err("selected repair: plan call outside exact repair mode".into());
            }
        }
        let range = function(lines, "repair_family_plan_step")?
            .ok_or("selected repair: missing plan step")?;
        let body = lines[range].iter().map(|l| l.trim()).collect::<Vec<_>>();
        if !body.contains(&"repair_workload_controller_unchanged || return 1") {
            return Err("selected repair: changed bounded plan launch".into());
        }
        let old = body
            .contains(&r#"run_verification_logged "full family certification plan" "$log" "$@""#)
            && body.contains(&r#""$@""#);
        let guarded = body.contains(&r#"if run_verification_logged "full family certification plan" "$log" "$@"; then status=0; else status=$?; fi"#)
            && body.contains(&r#"if "$@"; then status=0; else status=$?; fi"#)
            && body.iter().filter(|line| **line == "verification_candidate_unchanged || return 1").count() == 2;
        if (!old && !guarded)
            || (check_presence(lines, "verification_source_inspection")? && !guarded)
        {
            return Err("selected repair: changed bounded plan forwarding".into());
        }
    }
    check_verification(lines)?;
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
    for name in [
        "repair_family_plan_step",
        "repair_source_inspection",
        "verification_source_inspection",
    ] {
        if let Ok(Some(range)) = function(lines, name)
            && range.contains(&index)
            && (lines[index].trim() == "\"$@\""
                || lines[index].trim() == r#"if "$@"; then status=0; else status=$?; fi"#
                || lines[index]
                    .trim()
                    .contains("run_verification_logged \"full family certification plan\"")
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
    let verification =
        function(lines, "verification_source_inspection")?.is_some_and(|r| r.contains(&index));
    let name = if plan {
        "repair_family_plan_step"
    } else if verification {
        "verification_source_inspection"
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
        if let Some(candidate) = function(lines, "verification_candidate_unchanged")? {
            source.extend(lines[candidate].iter().map(|line| line.trim()));
        }
    } else if verification {
        source.extend(VERIFICATION_CALLS);
    } else {
        source.extend(LOCAL_CALLS);
    }
    Ok(Some(Binding {
        argv: source.join("\n"),
        owner: if plan {
            "tools/xtask/src/ci_plan/family/mod.rs; tools/xtask/src/automation/family_battery_policy.rs"
        } else if verification {
            "tools/xtask/src/automation/canary_receipts/package_closure/verification_source.rs"
        } else {
            "tools/xtask/src/automation/canary_receipts/package_closure/local_inspection.rs"
        },
        child: "pre-agent frozen controller; closed repair or independent verification command set",
    }))
}

const VERIFICATION_CALLS: [&str; 5] = [
    "verification_source_inspection verification-source-admit",
    "verification_source_inspection verification-manifest-policy > >(tee -a \"$MANIFEST_POLICY_LOG\") 2>&1",
    "verification_source_inspection verification-parity-inventory \"$CERTIFY_LOG\" || return 1",
    "verification_source_inspection verification-split-roster-check",
    "verification_source_inspection verification-source-admit || exit 1",
];
fn check_verification(lines: &[&str]) -> DynResult<()> {
    if !check_presence(lines, "verification_source_inspection")? {
        return Ok(());
    }
    if calls(lines, "verification_source_inspection ") != VERIFICATION_CALLS {
        return Err("selected verify: changed or expanded independent verb ownership".into());
    }
    let range = function(lines, "verification_source_inspection")?
        .ok_or("selected verify: missing owner")?;
    let actual = lines[range]
        .iter()
        .map(|line| line.trim())
        .collect::<Vec<_>>()
        .join("\n");
    if actual != VERIFICATION_HELPER {
        return Err("selected verify: changed independent launch or authority schema".into());
    }
    let candidate = function(lines, "verification_candidate_unchanged")?
        .ok_or("selected verify: missing candidate recheck")?;
    let candidate_body = lines[candidate]
        .iter()
        .map(|line| line.trim())
        .collect::<Vec<_>>()
        .join("\n");
    if candidate_body
        != "verification_candidate_unchanged() {\nif [[ \"$HARNESS_MODE\" == verify && -n \"$CERTIFIED_SHA\" ]]; then\nverification_source_inspection verification-source-admit\nfi\n}"
    {
        return Err("selected verify: changed candidate revalidation ownership".into());
    }
    for (index, _) in lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.trim().starts_with("verification_source_inspection "))
    {
        if index == 0
            || !matches!(
                lines[index - 1].trim(),
                "if [[ \"$HARNESS_MODE\" == verify && -n \"$CERTIFIED_SHA\" ]]; then"
                    | "if [[ \"$HARNESS_MODE\" == verify ]]; then"
                    | "elif [[ \"$HARNESS_MODE\" == verify ]]; then"
            )
        {
            return Err(
                "selected verify: caller outside exact independent verification mode".into(),
            );
        }
    }
    let freeze = lines
        .iter()
        .position(|line| line.trim() == "# Legacy workload automation selection begins.")
        .ok_or("selected verify: missing trusted freeze")?;
    let load = lines
        .iter()
        .position(|line| *line == "load_candidate_bundle")
        .ok_or("selected verify: missing candidate import")?;
    if freeze >= load {
        return Err("selected verify: trusted freeze after candidate import".into());
    }
    Ok(())
}

const VERIFICATION_HELPER: &str = r###"verification_source_inspection() {
local verb="$1" log="${2:-}" transaction_root input status
[[ "$HARNESS_MODE" == verify ]] || return 1
case "$verb" in
verification-source-admit|verification-manifest-policy|verification-parity-inventory|verification-split-roster-check) ;;
*) echo "unsupported independent verification inspection" >&2; return 1 ;;
esac
repair_workload_controller_unchanged || return 1
transaction_root="$(mktemp -d "${RUNNER_TEMP:-/tmp}/independent-verification.XXXXXXXX")" || return 1
input="$transaction_root/input.json"
if ! jq -n --arg controller_root "$TRUSTED_ROOT" --arg controller_revision "$BASE_HEAD" \
--arg controller_sha "$repair_workload_controller_sha" --arg root "$ROOT" \
--arg base "$CANDIDATE_BASE_HEAD" --arg candidate "$CERTIFIED_SHA" --arg tree "$VERIFICATION_TREE" \
'{authority:{controller:{root:$controller_root,revision:$controller_revision,executable_sha256:$controller_sha},
root:$root,base:$base,candidate:$candidate,tree:$tree}}' > "$input"; then
rm -rf "$transaction_root"
return 1
fi
if [[ -n "$log" ]]; then
if run_verification_logged "parity manifest validation" "$log" \
"${repair_workload_automation[@]}" automation canary-receipts "$verb" --input "$input"; then
status=0
else
status=$?
fi
elif "${repair_workload_automation[@]}" automation canary-receipts "$verb" --input "$input"; then
status=0
else
status=$?
fi
rm -rf "$transaction_root" || return 1
repair_workload_controller_unchanged || return 1
return "$status"
}"###;
