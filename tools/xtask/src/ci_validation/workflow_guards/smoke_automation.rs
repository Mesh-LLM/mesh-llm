//! Product consumers restore before admission; standalone jobs own one preparation.
use super::{Node, field, handoffs as h, workflow_yaml};
use crate::command::DynResult;
use std::{collections::BTreeMap, path::Path};

pub(super) fn check(root: &Path, workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    let action = workflow_yaml::parse(&std::fs::read_to_string(
        root.join(".github/actions/restore-smoke-inputs/action.yml"),
    )?)?;
    shared(&action)?;
    for name in [
        "smoke.yml",
        "scripted-binary-smoke.yml",
        "sdk-smoke.yml",
        "native-sdk-artifact.yml",
        "swift-sdk-artifact.yml",
        "ci-linux-product-smoke-slice.yml",
        "ci-macos-product-smoke-slice.yml",
        "ci-windows-product-smoke-slice.yml",
    ] {
        consumers(workflows.get(name).ok_or("missing automation consumer")?)?;
    }
    forwarding(workflows)
}

fn shared(action: &Node) -> DynResult<()> {
    let steps = h::steps(h::member(action, "runs")?)?;
    for (name, condition) in [
        (
            "Prepare smoke automation",
            "(runner.os != 'Linux' || job.container.id != '') && env.MESH_LLM_AUTOMATION_BIN == ''",
        ),
        (
            "Prepare hosted smoke automation",
            "runner.os == 'Linux' && job.container.id == '' && runner.environment == 'github-hosted' && env.MESH_LLM_AUTOMATION_BIN == ''",
        ),
    ] {
        h::condition(h::step(steps, "name", name)?.1, condition)?;
    }
    Ok(())
}

fn consumers(workflow: &Node) -> DynResult<()> {
    for (_, job) in h::member(workflow, "jobs")?.entries() {
        let Ok(steps) = h::steps(job) else { continue };
        let managed = steps.iter().any(|step| {
            matches!(
                field(step, "uses"),
                Some(
                    "./.github/actions/restore-smoke-inputs"
                        | "./.github/actions/run-laya-product-smoke"
                        | "./.github/actions/prepare-native-sdk-input"
                )
            )
        }) || steps.iter().any(|step| {
            field(step, "name").is_some_and(|name| {
                name.starts_with("Build Swift target") || name.starts_with("Build host-only Swift")
            })
        });
        if !managed {
            continue;
        }
        let (restore, step) = h::step(steps, "uses", "./.github/actions/restore-automation")?;
        let inputs = h::member(step, "with")?;
        for (key, value) in [
            ("source_sha", "${{ inputs.source_sha }}"),
            ("require_product", "true"),
            (
                "product_source_sha",
                "${{ fromJson(inputs.product_automation).source_sha }}",
            ),
            (
                "source_profile",
                "${{ fromJson(inputs.product_automation).source_profile }}",
            ),
            (
                "artifact_id",
                "${{ fromJson(inputs.product_automation).artifact_id }}",
            ),
            (
                "binary_sha256",
                "${{ fromJson(inputs.product_automation).binary_sha256 }}",
            ),
            (
                "producer_os",
                "${{ fromJson(inputs.product_automation).producer_os }}",
            ),
            (
                "producer_arch",
                "${{ fromJson(inputs.product_automation).producer_arch }}",
            ),
        ] {
            h::binding(inputs, key, value)?;
        }
        if step.get("continue-on-error").is_some() {
            return Err("restore must refuse failures".into());
        }
        if step.get("if").is_some() {
            h::condition(step, "${{ inputs.product_automation != '' }}")?;
        }
        let mut preparations = 0;
        for (index, candidate) in steps.iter().enumerate() {
            if field(candidate, "uses") == Some("./.github/actions/prepare-automation") {
                h::condition(candidate, "${{ inputs.product_automation == '' }}")?;
                preparations += 1;
            }
            if matches!(
                field(candidate, "uses"),
                Some(
                    "./.github/actions/restore-smoke-inputs"
                        | "./.github/actions/run-laya-product-smoke"
                        | "./.github/actions/setup-canary-python"
                        | "./.github/actions/prepare-native-sdk-input"
                )
            ) || field(candidate, "name")
                == Some("Prepare source-bound native cleanup before managed work")
            {
                h::before(restore, index)?;
            }
            if field(candidate, "run").is_some_and(|run| run.contains("just automation-bootstrap"))
            {
                return Err("product consumer must not bootstrap cleanup".into());
            }
        }
        if preparations != usize::from(step.get("if").is_some()) {
            return Err("standalone consumer must own exactly one preparation".into());
        }
    }
    Ok(())
}

fn forwarding(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    for (name, workflow) in workflows {
        for (_, job) in h::member(workflow, "jobs")?.entries() {
            let callee = field(job, "uses").unwrap_or("");
            let eligible = callee.contains("product-smoke-slice.yml")
                || callee.contains("sdk-slice.yml")
                || (name.starts_with("ci-")
                    && matches!(
                        callee,
                        "./.github/workflows/native-sdk-artifact.yml"
                            | "./.github/workflows/swift-sdk-artifact.yml"
                    ));
            let smoke_forwarder = name.ends_with("-slice.yml")
                && matches!(
                    callee,
                    "./.github/workflows/smoke.yml"
                        | "./.github/workflows/scripted-binary-smoke.yml"
                        | "./.github/workflows/sdk-smoke.yml"
                );
            if !eligible && !smoke_forwarder {
                continue;
            }
            let inputs = h::member(job, "with")?;
            let identity = field(inputs, "product_automation")
                .ok_or("eligible caller missing product identity")?;
            if name.ends_with("-lane.yml") {
                let platform = name.trim_start_matches("ci-").trim_end_matches("-lane.yml");
                let suffix = if platform == "macos" { "arm64" } else { "x64" };
                let producer = format!("product_automation_{platform}_{suffix}");
                h::needs(job, &[&producer])?;
                h::binding(
                    inputs,
                    "product_automation",
                    &format!("${{{{ needs.{producer}.outputs.identity_json }}}}"),
                )?;
            } else if identity != "${{ inputs.product_automation }}" {
                return Err("slice must forward its product identity".into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
