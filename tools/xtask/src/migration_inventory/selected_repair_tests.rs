use super::super::ledger::SelectedProcessCall;
use super::{
    check_selected_processes, indirect_launch, repair, repair_producer, typed_owner,
    typed_source_argv,
};
use crate::command::DynResult;
use std::fs;
const CALLER: &str = "scripts/llama-canary-agent-repair.sh";
fn source() -> String {
    fs::read_to_string(crate::repo_consistency::repo_root().unwrap().join(CALLER)).unwrap()
}

#[test]
fn early_metal_producer_refuses_changed_inputs_and_custody_at_the_actual_call() {
    let text = source();
    let head = r#""${repair_workload_automation[@]}" automation canary-receipts workload-manifest verify \"#;
    let lines = text.lines().collect::<Vec<_>>();
    let index = lines
        .iter()
        .enumerate()
        .find(|(i, line)| *i > 700 && line.trim() == head)
        .unwrap()
        .0;
    let valid = repair_producer::binding(CALLER, &lines, index)
        .unwrap()
        .unwrap();
    assert_eq!(
        valid.owner,
        "tools/xtask/src/automation/canary_receipts/package_closure/workload_production.rs"
    );
    let block_start = index - 3;
    for (old, new) in [
        ("workload-manifest verify", "workload-manifest produce"),
        (
            r#""$ROOT" "${LLAMA_STAGE_BUILD_DIR:?}-workloads/cargo/debug/skippy""#,
            r#""$OTHER_ROOT" "${LLAMA_STAGE_BUILD_DIR:?}-workloads/cargo/debug/skippy""#,
        ),
        (
            "-workloads/cargo/debug/skippy",
            "-workloads/cargo/debug/foreign",
        ),
        ("-workloads/native", "-workloads/foreign-native"),
        ("-workloads/producer.json", "-workloads/foreign.json"),
        (
            r#"-workloads/producer.json" || return 1"#,
            r#"-workloads/producer.json" || true"#,
        ),
        (
            r#"-workloads/native" \"#,
            r#"-workloads/native" "$EXTRA_ARGUMENT" \"#,
        ),
        (
            "repair_workload_controller_unchanged || return 1",
            ": # custody omitted",
        ),
        (
            "verification_candidate_unchanged || return 1",
            ": # candidate omitted",
        ),
        ("verify exact workload producer", "unrelated phase"),
        ("|| return 1", "|| true"),
    ] {
        let mut changed = lines
            .iter()
            .map(|line| (*line).to_owned())
            .collect::<Vec<_>>();
        let block = changed[block_start..index + 6].join("\n");
        assert!(block.contains(old));
        let replacement = block.replacen(old, new, 1);
        changed.splice(
            block_start..index + 6,
            replacement.lines().map(str::to_owned),
        );
        let borrowed = changed.iter().map(String::as_str).collect::<Vec<_>>();
        assert!(
            repair_producer::binding(CALLER, &borrowed, index).is_err(),
            "accepted {old} -> {new}"
        );
    }
}
fn records(source: &str) -> Vec<SelectedProcessCall> {
    let lines = source.lines().collect::<Vec<_>>();
    lines
        .iter()
        .enumerate()
        .filter_map(|(index, line)| {
            if !indirect_launch(CALLER, line) && !repair::is_launch(CALLER, &lines, index) {
                return None;
            }
            let (argv, owner, child) = if let Some(binding) =
                repair_producer::binding(CALLER, &lines, index)
                    .unwrap()
                    .or(repair::binding(CALLER, &lines, index).unwrap())
            {
                (binding.argv, binding.owner, binding.child)
            } else {
                let argv = typed_source_argv(&lines, index).unwrap();
                let owner = typed_owner(&argv).unwrap();
                (argv, owner, "fixture typed controller")
            };
            Some(SelectedProcessCall {
                caller: CALLER.to_owned(),
                line: index + 1,
                source_block: line.trim().to_owned(),
                child: child.to_owned(),
                child_source_known: false,
                argv,
                replacement_owner: owner.to_owned(),
                status_streams_effects:
                    "Closed normal-repair input and real process/status ownership".to_owned(),
            })
        })
        .collect()
}
#[test]
fn normal_repair_closed_forwarding_and_local_verbs_are_bound_at_actual_launches() -> DynResult<()> {
    let text = source();
    let root = crate::command::unique_temp_dir("selected-normal-repair");
    fs::create_dir_all(root.join("scripts"))?;
    fs::write(root.join(CALLER), &text)?;
    let owned = records(&text);
    let narrow = owned
        .iter()
        .filter(|r| r.child.starts_with("pre-agent frozen") || r.child == "frozen trusted controller; exact early Metal native workload producer admission")
        .collect::<Vec<_>>();
    assert_eq!(
        narrow.len(),
        7,
        "two bounded plan paths, two local and two independent inspection paths, one separate early Metal producer admission"
    );
    check_selected_processes(&root, &owned)?;
    for missing in narrow {
        let partial = owned
            .iter()
            .filter(|r| r.line != missing.line)
            .cloned()
            .collect::<Vec<_>>();
        assert!(
            check_selected_processes(&root, &partial)
                .unwrap_err()
                .to_string()
                .contains("unowned required execution")
        );
    }
    let mut wrong = owned.clone();
    let record = wrong
        .iter_mut()
        .find(|r| r.child.starts_with("pre-agent frozen"))
        .unwrap();
    record.replacement_owner = "tools/xtask/src/automation/canary_timeout.rs".to_owned();
    assert!(
        check_selected_processes(&root, &wrong)
            .unwrap_err()
            .to_string()
            .contains("changed repair launch binding")
    );
    fs::remove_dir_all(root)?;
    Ok(())
}
#[test]
fn normal_repair_rejects_changed_omitted_or_expanded_callsite_authority() -> DynResult<()> {
    let text = source();
    for (old, new) in [
        (
            "repair_source_inspection local-manifest-policy",
            "repair_source_inspection manifest-policy",
        ),
        (
            "repair_source_inspection local-split-roster true",
            "repair_source_inspection local-split-roster true\n    repair_source_inspection unexpected-verb",
        ),
        (
            "repair_source_inspection local-split-roster false",
            "# omitted local inspection",
        ),
        (
            "repair_family_plan_step \"$log\" \"${repair_workload_automation[@]}\" automation family-battery-policy --cache",
            "repair_family_plan_step \"$log\" other-controller automation family-battery-policy --cache",
        ),
        ("repair_family_plan 256", "repair_family_plan 1"),
        (
            "--cache-descriptors \"$ROOT\"",
            "--cache-descriptors \"$OTHER_ROOT\"",
        ),
        (
            "\"$ROOT/ci/llama-canary/family-certified.json\" \"$PLAN_PATH\" \"${HF_CACHE:?}\" || return 1",
            "\"$ROOT/ci/llama-canary/family-certified.json\" \"$OTHER_PLAN\" \"${HF_CACHE:?}\" || return 1",
        ),
        (
            "local roster_mode=\"${1:-verify}\"",
            "local roster_mode=\"${1:-unowned}\"",
        ),
        (
            r#"local verb="$1" check="${2:-}""#,
            r#"local verb="unowned" check="${2:-}""#,
        ),
        ("repair_family_plan_step() {", "unowned_plan_step() {"),
        ("repair_family_plan() {", "unowned_plan_projection() {"),
        (
            "repair_source_inspection() {",
            "unowned_local_inspection() {",
        ),
        (
            r#"--manifest "$manifest" --verify-plan "$PLAN_PATH" || return 1"#,
            r#"--manifest "$manifest" --verify-plan "$UNOWNED_PLAN" || return 1"#,
        ),
        (
            r#"repair_source_inspection local-split-roster true"#,
            r#"repair_source_inspection "$dynamic_verb" true"#,
        ),
        (
            "if [[ \"$HARNESS_MODE\" == repair ]]; then\n    repair_source_inspection",
            "if [[ \"$HARNESS_MODE\" != repair ]]; then\n    repair_source_inspection",
        ),
        (
            "if [[ \"$HARNESS_MODE\" != repair && \"$HARNESS_MODE\" != verify ]]; then",
            "if [[ \"$HARNESS_MODE\" != repair ]]; then",
        ),
    ] {
        assert!(text.contains(old), "missing fixture mutation target {old}");
        let changed = text.replace(old, new);
        assert!(
            repair::check_shape(CALLER, &changed.lines().collect::<Vec<_>>()).is_err(),
            "mutation admitted: {old}"
        );
    }
    let root = crate::command::unique_temp_dir("selected-normal-repair-input");
    fs::create_dir_all(root.join("scripts"))?;
    let owned = records(&text);
    let changed = text.replace(
        "--arg base \"$CANDIDATE_BASE_HEAD\" --arg check",
        "--arg base \"$BASE_HEAD\" --arg check",
    );
    assert_ne!(changed, text);
    fs::write(root.join(CALLER), changed)?;
    assert!(
        check_selected_processes(&root, &owned)
            .unwrap_err()
            .to_string()
            .contains("changed repair launch binding")
    );
    fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn normal_verify_rejects_missing_expanded_or_rebound_independent_authority() {
    let text = source();
    for (old, new) in [
        (
            "verification_source_inspection() {",
            "unowned_verification() {",
        ),
        (
            "verification_source_inspection verification-split-roster-check",
            "verification_source_inspection verification-split-roster-write",
        ),
        (
            "verification_source_inspection verification-parity-inventory",
            "verification_source_inspection \"$dynamic_verb\"",
        ),
        (
            "verification_source_inspection verification-source-admit || exit 1",
            "verification_source_inspection verification-source-admit || exit 1\n  verification_source_inspection unowned",
        ),
        (
            "--arg controller_revision \"$BASE_HEAD\"",
            "--arg controller_revision \"$CANDIDATE_BASE_HEAD\"",
        ),
        (
            "--arg candidate \"$CERTIFIED_SHA\" --arg tree \"$VERIFICATION_TREE\"",
            "--arg candidate \"$CERTIFIED_SHA\" --arg tree \"$OTHER_TREE\"",
        ),
        (
            "verification_candidate_unchanged() {",
            "unowned_candidate_check() {",
        ),
    ] {
        assert!(text.contains(old), "missing fixture target {old}");
        assert!(
            repair::check_shape(CALLER, &text.replace(old, new).lines().collect::<Vec<_>>())
                .is_err(),
            "admitted {old}"
        );
    }
}
