//! Closed battery plan-array admission; other typed policy calls keep their own owners.
use super::super::ledger::SelectedProcessCall;
use crate::command::DynResult;
const CALLER: &str = "skippy/scripts/skippy-family-battery.sh";
const OWNER: &str = "tools/xtask/src/ci_plan/family; frozen controller typed family-plan and selected-root manifest/policy projection";
const ARGV: &str = "${automation[@]} --repo-root \"$ROOT\" ci family-plan --manifest \"$MANIFEST\" --output \"$POLICY_PLAN_COPY\" [--families \"$FAMILY_FILTER\"]; only without --plan";
const SHAPE: &str = r###"prepare_policy_plan() {
local plan_args=(
"${automation[@]}" --repo-root "$ROOT" ci family-plan
--manifest "$MANIFEST"
--output "$POLICY_PLAN_COPY"
)
if [[ -n "$POLICY_PLAN" ]]; then
if [[ ! -f "$POLICY_PLAN" ]]; then
echo "policy plan not found: $POLICY_PLAN" >&2
return 1
fi
if [[ "$POLICY_PLAN" != "$POLICY_PLAN_COPY" ]]; then
cp "$POLICY_PLAN" "$POLICY_PLAN_COPY"
fi
else
if [[ -n "$FAMILY_FILTER" ]]; then
plan_args+=(--families "$FAMILY_FILTER")
fi
"${plan_args[@]}"
fi
if [[ -n "${HF_CACHE:-}" ]]; then
"${automation[@]}" automation family-battery-policy --cache \
"$ROOT" "$MANIFEST" "$POLICY_PLAN_COPY" "$HF_CACHE"
fi
"${automation[@]}" automation family-battery-policy \
"$ROOT" "$MANIFEST" "$POLICY_PLAN_COPY" "$SHARD_INDEX"
}"###;
pub(super) fn check(record: &SelectedProcessCall, lines: &[&str], index: usize) -> DynResult<()> {
    if record.caller != CALLER || record.source_block != "\"${plan_args[@]}\"" {
        return Ok(());
    }
    let starts = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.trim() == "prepare_policy_plan() {")
        .map(|(at, _)| at)
        .collect::<Vec<_>>();
    let [start] = starts.as_slice() else {
        return Err("selected battery: missing or duplicate plan owner".into());
    };
    let end = lines[*start + 1..]
        .iter()
        .position(|line| *line == "}")
        .ok_or("selected battery: unclosed plan owner")?
        + *start
        + 2;
    let actual = lines[*start..end]
        .iter()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    if lines[index].trim() != "\"${plan_args[@]}\""
        || index < *start
        || index >= end
        || actual != SHAPE
        || record.argv != ARGV
        || record.replacement_owner != OWNER
    {
        return Err(
            "selected battery: changed frozen plan input or conditional launch binding".into(),
        );
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selected_battery_array_refuses_changed_authority_scope_and_cache_order() {
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../skippy/scripts/skippy-family-battery.sh"
        ));
        let binding = |source: &str| {
            let lines = source.lines().collect::<Vec<_>>();
            let index = lines
                .iter()
                .position(|line| line.trim().starts_with("\"${plan_args[@]}\""))
                .unwrap();
            let record = SelectedProcessCall {
                caller: CALLER.into(),
                line: index + 1,
                source_block: "\"${plan_args[@]}\"".into(),
                child: "tools/xtask".into(),
                child_source_known: true,
                argv: ARGV.into(),
                replacement_owner: OWNER.into(),
                status_streams_effects: "finite scope".into(),
            };
            check(&record, &lines, index)
        };
        binding(source).unwrap();
        for (old, new) in [
            ("--repo-root \"$ROOT\"", "--repo-root \"$OTHER\""),
            ("--manifest \"$MANIFEST\"", "--manifest \"$OTHER\""),
            ("--output \"$POLICY_PLAN_COPY\"", "--output \"$OTHER\""),
            (
                "plan_args+=(--families \"$FAMILY_FILTER\")",
                "plan_args+=(--families \"$OTHER\")",
            ),
            (
                "if [[ -n \"$POLICY_PLAN\" ]]; then",
                "if [[ -z \"$POLICY_PLAN\" ]]; then",
            ),
            ("\"${plan_args[@]}\"", "\"${plan_args[@]}\" --extra"),
        ] {
            assert!(source.contains(old));
            let changed = source.replace(old, new);
            assert!(binding(&changed).is_err(), "{old}");
        }
    }
}
