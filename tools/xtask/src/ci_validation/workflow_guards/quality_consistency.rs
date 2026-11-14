//! Candidate consistency must report independently of exhaustive contracts.
use super::{Node, field};
use crate::command::DynResult;
use std::collections::BTreeMap;

pub(super) fn check(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    let lane = workflows
        .get("ci-quality-lane.yml")
        .ok_or("missing Quality lane")?;
    let jobs = lane.get("jobs").ok_or("missing Quality jobs")?;
    let early = jobs
        .get("consistency")
        .ok_or("missing candidate consistency")?;
    if early.get("needs").is_some()
        || field(early, "runs-on") != Some("ubuntu-24.04")
        || field(early, "if")
            != Some("${{ contains(fromJson(inputs.lane_plan_json).required_slices, 'quality') }}")
        || early.get("continue-on-error").is_some()
    {
        return Err("candidate consistency must be an independent required hosted job".into());
    }
    let Some(Node::Seq(steps)) = early.get("steps") else {
        return Err("candidate consistency steps missing".into());
    };
    let checkout = steps.first().and_then(|step| step.get("with"));
    if checkout.and_then(|with| field(with, "ref")) != Some("${{ inputs.source_sha }}")
        || checkout.and_then(|with| field(with, "persist-credentials")) != Some("false")
        || steps
            .iter()
            .any(|step| step.get("if").is_some() || step.get("continue-on-error").is_some())
    {
        return Err("consistency must check candidate source without optional steps".into());
    }
    for command in [
        "\"$MESH_LLM_AUTOMATION_BIN\" repo-consistency ci-crate-lists",
        "\"$MESH_LLM_AUTOMATION_BIN\" repo-consistency publish-crates",
        "\"$MESH_LLM_AUTOMATION_BIN\" repo-consistency release-targets",
        "RUSTC_WRAPPER='' CARGO_BUILD_RUSTC_WRAPPER='' cargo tree -p mesh-llm-client --prefix=none --no-dedupe > \"$RUNNER_TEMP/mesh-client-deps.txt\"",
        "\"$MESH_LLM_AUTOMATION_BIN\" repo-consistency no-console-print",
        "\"$MESH_LLM_AUTOMATION_BIN\" repository env-mutation-census",
    ] {
        if !steps.iter().any(|step| {
            field(step, "run").is_some_and(|run| run.lines().any(|line| line.trim() == command))
        }) {
            return Err(
                format!("required candidate consistency command omitted: {command}").into(),
            );
        }
    }
    let full = jobs.get("quality").ok_or("missing full Quality call")?;
    if full.get("needs").map(Node::list).unwrap_or_default() != ["authority_linux_x64"]
        || field(full, "if")
            != Some(
                "${{ (contains(fromJson(inputs.lane_plan_json).required_slices, 'quality')) && needs.authority_linux_x64.result == 'success' }}",
            )
    {
        return Err("full Quality contracts must retain only their authority dependency".into());
    }
    let summary = jobs.get("summary").ok_or("missing Quality summary")?;
    if !summary
        .get("needs")
        .map(Node::list)
        .unwrap_or_default()
        .contains(&"consistency")
    {
        return Err("Quality result must require candidate consistency".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_consistency_and_exhaustive_contracts_remain_independent_and_required() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let source =
            std::fs::read_to_string(root.join(".github/workflows/ci-quality-lane.yml")).unwrap();
        let parse = |source: &str| {
            BTreeMap::from([(
                "ci-quality-lane.yml".to_owned(),
                super::super::workflow_yaml::parse(source).unwrap(),
            )])
        };
        check(&parse(&source)).unwrap();
        for (from, to) in [
            (
                "ref: ${{ inputs.source_sha }}",
                "ref: ${{ github.event.repository.default_branch }}",
            ),
            (
                "RUSTC_WRAPPER='' CARGO_BUILD_RUSTC_WRAPPER='' cargo tree -p mesh-llm-client",
                "cargo tree -p mesh-llm-client",
            ),
            (
                "  consistency:\n",
                "  consistency:\n    needs: authority_linux_x64\n",
            ),
            (
                "    needs: [authority_linux_x64]",
                "    needs: [authority_linux_x64, consistency]",
            ),
            ("needs: [consistency, quality,", "needs: [quality,"),
            (
                "repo-consistency ci-crate-lists'",
                "repo-consistency ci-crate-lists || true'",
            ),
            (
                "contains(fromJson(inputs.lane_plan_json).required_slices, 'quality')) &&",
                "contains(fromJson(inputs.lane_plan_json).required_slices, 'quality')) && needs.consistency.result == 'success' &&",
            ),
        ] {
            assert!(source.contains(from), "{from}");
            let error = check(&parse(&source.replace(from, to))).unwrap_err();
            assert!(!error.to_string().is_empty());
        }
    }
}
