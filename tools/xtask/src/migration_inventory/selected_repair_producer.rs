//! Exact early Metal producer admission, independent of family-plan forwarding.
use super::repair::Binding;
use crate::command::DynResult;

const CALLER: &str = "scripts/llama-canary-agent-repair.sh";
const HEAD: &str =
    r#""${repair_workload_automation[@]}" automation canary-receipts workload-manifest verify \"#;
const BLOCK: &str = r#"repair_workload_controller_unchanged || return 1
verification_candidate_unchanged || return 1
run_verification_logged "verify exact workload producer" "$CERTIFY_LOG" \
"${repair_workload_automation[@]}" automation canary-receipts workload-manifest verify \
"$ROOT" "${LLAMA_STAGE_BUILD_DIR:?}-workloads/cargo/debug/skippy" \
"${LLAMA_STAGE_BUILD_DIR:?}-workloads/native" \
"${LLAMA_STAGE_BUILD_DIR:?}-workloads/producer.json" || return 1
verification_candidate_unchanged || return 1
repair_workload_controller_unchanged || return 1"#;

pub(super) fn binding(path: &str, lines: &[&str], index: usize) -> DynResult<Option<Binding>> {
    if path != CALLER {
        return Ok(None);
    }
    let starts = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.trim() == "run_early_metal_certification() {")
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    if starts.is_empty() {
        return Ok(None);
    }
    let [start] = starts.as_slice() else {
        return Err("selected producer: absent or duplicate early Metal owner".into());
    };
    let end = lines[*start + 1..]
        .iter()
        .position(|line| *line == "}")
        .ok_or("selected producer: unclosed early Metal owner")?
        + start
        + 1;
    if !(*start..end).contains(&index) {
        return Ok(None);
    }
    let launches = lines[*start..end]
        .iter()
        .enumerate()
        .filter(|(_, line)| {
            line.trim()
                .starts_with(r#""${repair_workload_automation[@]}""#)
        })
        .map(|(i, _)| start + i)
        .collect::<Vec<_>>();
    let [launch] = launches.as_slice() else {
        return Err("selected producer: changed early Metal launch count".into());
    };
    if *launch != index {
        return Ok(None);
    }
    let begin = index
        .checked_sub(3)
        .ok_or("selected producer: missing custody prefix")?;
    let finish = index + 6;
    let actual = lines
        .get(begin..finish)
        .ok_or("selected producer: incomplete producer inputs")?
        .iter()
        .map(|line| line.trim())
        .collect::<Vec<_>>()
        .join("\n");
    if lines[index].trim() != HEAD || actual != BLOCK {
        return Err("selected producer: changed arguments, custody, phase or refusal".into());
    }
    Ok(Some(Binding {
        argv: actual,
        owner: "tools/xtask/src/automation/canary_receipts/package_closure/workload_production.rs",
        child: "frozen trusted controller; exact early Metal native workload producer admission",
    }))
}
