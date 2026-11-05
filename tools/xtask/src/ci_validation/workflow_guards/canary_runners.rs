//! Canary self-hosted placement is platform independent. The caller's resolve
//! job picks the certification platform once per pass — macOS Metal hosts or
//! Linux CUDA hosts — emits managed runner label sets, and every self-hosted
//! job consumes them. The jobs never name a platform, pin a node, or fall back
//! to a raw runner pool, and the 256-tier labels always target macOS because
//! no other managed runner carries accelerator-memory-256plus.

use super::{Node, field, handoffs as h};
use crate::command::DynResult;
use std::collections::BTreeMap;

const CALLER: &str = "llama-upstream-canary.yml";
const PASS: &str = "llama-canary-family-pass.yml";

const PREFLIGHT_RUNNER: &str = "${{ fromJSON(needs.resolve.outputs.runner_labels_128) }}";
const PASS_128_RUNNER: &str = "${{ fromJSON(inputs.runner_labels_128) }}";
const PASS_TIERED_RUNNER: &str = "${{ fromJSON(matrix.memory_tier == 'accelerator-memory-256plus' && inputs.runner_labels_256 || inputs.runner_labels_128) }}";

pub(super) fn check(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    let caller = workflows.get(CALLER).ok_or("missing canary workflow")?;
    let pass = workflows
        .get(PASS)
        .ok_or("missing canary family pass workflow")?;

    h::binding(h::job(caller, "preflight")?, "runs-on", PREFLIGHT_RUNNER)?;
    h::binding(h::job(pass, "build")?, "runs-on", PASS_128_RUNNER)?;
    for name in ["family", "retry_family"] {
        h::binding(h::job(pass, name)?, "runs-on", PASS_TIERED_RUNNER)?;
    }

    // Resolve owns placement: the outputs exist and the select step constructs
    // the label sets only from the managed label vocabulary.
    let resolve = h::job(caller, "resolve")?;
    let outputs = h::member(resolve, "outputs")?;
    for name in [
        "platform",
        "runner_profile",
        "runner_labels_128",
        "runner_labels_256",
    ] {
        h::binding(
            outputs,
            name,
            &format!("${{{{ steps.select_platform.outputs.{name} }}}}"),
        )?;
    }
    let (_, select) = h::step(h::steps(resolve)?, "id", "select_platform")?;
    let script = field(select, "run").ok_or("canary platform selection must be authored")?;
    for managed in [
        "self-hosted",
        "ARM64",
        "jy-managed",
        "family-certify",
        "accelerator-memory-128plus",
        "accelerator-memory-256plus",
    ] {
        if !script.contains(managed) {
            return Err(format!(
                "canary platform selection must construct managed label {managed:?}"
            )
            .into());
        }
    }
    let Some(labels_256) = script
        .lines()
        .find(|line| line.contains("runner_labels_256="))
    else {
        return Err("canary platform selection must emit runner_labels_256".into());
    };
    if !labels_256.contains("macOS") || labels_256.contains("$os_label") {
        return Err("canary 256-tier runner labels must stay on macOS".into());
    }
    let Some(labels_128) = script
        .lines()
        .find(|line| line.contains("runner_labels_128="))
    else {
        return Err("canary platform selection must emit runner_labels_128".into());
    };
    if !labels_128.contains("$os_label") {
        return Err("canary 128-tier runner labels must follow the selected platform".into());
    }

    // The family pass receives placement from its caller; it cannot invent or
    // default it.
    let inputs = pass
        .get("on")
        .and_then(|on| on.get("workflow_call"))
        .and_then(|call| call.get("inputs"))
        .ok_or("canary family pass must declare workflow_call inputs")?;
    for name in ["runner_labels_128", "runner_labels_256", "runner_profile"] {
        let input = inputs
            .get(name)
            .ok_or_else(|| format!("canary family pass must require input {name:?}"))?;
        if field(input, "required") != Some("true") {
            return Err(format!("canary family pass input {name:?} must be required").into());
        }
    }

    // Every reusable call site passes resolve's placement through unchanged.
    for (_, job) in h::member(caller, "jobs")?.entries() {
        if field(job, "uses") == Some("./.github/workflows/llama-canary-family-pass.yml") {
            let with = h::member(job, "with")?;
            h::binding(
                with,
                "runner_labels_128",
                "${{ needs.resolve.outputs.runner_labels_128 }}",
            )?;
            h::binding(
                with,
                "runner_labels_256",
                "${{ needs.resolve.outputs.runner_labels_256 }}",
            )?;
            h::binding(
                with,
                "runner_profile",
                "${{ needs.resolve.outputs.runner_profile }}",
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci_validation::lane_results::workflow_yaml;

    const UPSTREAM: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.github/workflows/llama-upstream-canary.yml"
    ));
    const FAMILY: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../.github/workflows/llama-canary-family-pass.yml"
    ));

    fn workflows() -> BTreeMap<String, Node> {
        BTreeMap::from([
            (
                "llama-upstream-canary.yml".into(),
                workflow_yaml::parse_resolved_aliases(UPSTREAM).unwrap(),
            ),
            (
                "llama-canary-family-pass.yml".into(),
                workflow_yaml::parse(FAMILY).unwrap(),
            ),
        ])
    }

    fn member_mut<'a>(node: &'a mut Node, name: &str) -> &'a mut Node {
        let Node::Map(entries) = node else {
            panic!("expected mapping")
        };
        &mut entries.iter_mut().find(|(key, _)| key == name).unwrap().1
    }

    fn scalar_mut(node: &mut Node) -> &mut String {
        let Node::Scalar(text) = node else {
            panic!("expected scalar")
        };
        text
    }

    #[test]
    fn managed_platform_dispatch_rejects_pool_fallbacks_and_node_pinning() {
        check(&workflows()).unwrap();
        for (label, replacement) in [
            ("jy-managed", "other-pool"),
            ("family-certify", "named-node"),
            ("accelerator-memory-128plus", "accelerator-memory-64plus"),
        ] {
            let mut changed = workflows();
            let document = changed.get_mut("llama-upstream-canary.yml").unwrap();
            let resolve = member_mut(member_mut(document, "jobs"), "resolve");
            let steps = member_mut(resolve, "steps");
            let Node::Seq(steps) = steps else {
                panic!("expected steps")
            };
            let select = steps
                .iter_mut()
                .find(|step| super::field(step, "id") == Some("select_platform"))
                .unwrap();
            let script = scalar_mut(member_mut(select, "run"));
            *script = script.replace(label, replacement);
            assert!(check(&changed).is_err(), "select script: {label}");
        }
    }

    #[test]
    fn tiered_jobs_must_dispatch_on_the_matrix_tier() {
        let mut changed = workflows();
        let document = changed.get_mut("llama-canary-family-pass.yml").unwrap();
        let family = member_mut(member_mut(document, "jobs"), "family");
        *scalar_mut(member_mut(family, "runs-on")) = PASS_128_RUNNER.into();
        assert!(check(&changed).is_err());
    }

    #[test]
    fn macos_256_placement_cannot_follow_the_selected_platform() {
        let mut changed = workflows();
        let document = changed.get_mut("llama-upstream-canary.yml").unwrap();
        let resolve = member_mut(member_mut(document, "jobs"), "resolve");
        let steps = member_mut(resolve, "steps");
        let Node::Seq(steps) = steps else {
            panic!("expected steps")
        };
        let select = steps
            .iter_mut()
            .find(|step| super::field(step, "id") == Some("select_platform"))
            .unwrap();
        let script = scalar_mut(member_mut(select, "run"));
        *script = script
            .lines()
            .map(|line| {
                if line.contains("runner_labels_256=") {
                    line.replace("macOS", "Linux")
                } else {
                    line.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(check(&changed).is_err());
    }

    #[test]
    fn reusable_call_sites_must_forward_resolve_placement() {
        let mut changed = workflows();
        let document = changed.get_mut("llama-upstream-canary.yml").unwrap();
        let jobs = member_mut(document, "jobs");
        let Node::Map(jobs) = jobs else {
            panic!("expected jobs")
        };
        let call = jobs
            .iter_mut()
            .find(|(_, job)| {
                super::field(job, "uses")
                    == Some("./.github/workflows/llama-canary-family-pass.yml")
            })
            .map(|(_, job)| job)
            .unwrap();
        let Node::Scalar(value) = member_mut(member_mut(call, "with"), "runner_labels_256") else {
            panic!("expected scalar")
        };
        *value = "[]".into();
        assert!(check(&changed).is_err());
    }

    #[test]
    fn selfhosted_jobs_must_not_name_a_platform_directly() {
        let mut changed = workflows();
        let document = changed.get_mut("llama-upstream-canary.yml").unwrap();
        *scalar_mut(member_mut(
            member_mut(member_mut(document, "jobs"), "preflight"),
            "runs-on",
        )) = "[self-hosted, Linux, ARM64]".into();
        assert!(check(&changed).is_err());
    }
}
