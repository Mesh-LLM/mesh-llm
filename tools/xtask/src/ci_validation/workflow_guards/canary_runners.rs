//! Family jobs request memory only. Native producers request their OS and
//! memory; workers choose the matching payload after the scheduler places them.

use super::{Node, field, handoffs as h};
use crate::command::DynResult;
use std::collections::BTreeMap;

const CALLER: &str = "llama-upstream-canary.yml";
const PASS: &str = "llama-canary-family-pass.yml";

const PREFLIGHT_RUNNER: &str = "${{ fromJSON(needs.resolve.outputs.runner_labels_128) }}";
const PASS_MAC_RUNNER: &str = "${{ fromJSON(inputs.runner_labels_macos) }}";
const PASS_LINUX_RUNNER: &str = "${{ fromJSON(inputs.runner_labels_linux) }}";
const PASS_TIERED_RUNNER: &str = "${{ fromJSON(matrix.memory_tier == 'accelerator-memory-256plus' && inputs.runner_labels_256 || inputs.runner_labels_128) }}";
const CONTROL_RUNNER: &str = "${{ fromJSON(needs.resolve.outputs.runner_labels_128) }}";
const PASS_CONTROL_RUNNER: &str = "${{ fromJSON(inputs.runner_labels_128) }}";
const BOOTSTRAP_RUNNER: &str =
    "self-hosted,ARM64,jy-managed,family-certify,accelerator-memory-128plus";
const CONTROL_PROFILE: &str =
    "${{ runner.os == 'Linux' && 'preinstalled-linux' || 'preinstalled-macos' }}";

pub(super) fn check(workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    let caller = workflows.get(CALLER).ok_or("missing canary workflow")?;
    let pass = workflows
        .get(PASS)
        .ok_or("missing canary family pass workflow")?;

    h::binding(h::job(caller, "preflight")?, "runs-on", PREFLIGHT_RUNNER)?;
    for name in ["resolve", "result"] {
        let job = h::job(caller, name)?;
        if job.get("runs-on").map(Node::list) != Some(BOOTSTRAP_RUNNER.split(',').collect()) {
            return Err(format!("{name} must use the closed managed runner pool").into());
        }
    }
    for name in ["attempt_1", "attempt_2", "attempt_3"] {
        h::binding(h::job(caller, name)?, "runs-on", CONTROL_RUNNER)?;
    }
    for name in ["combine", "aggregate", "reconcile"] {
        h::binding(h::job(pass, name)?, "runs-on", PASS_CONTROL_RUNNER)?;
    }
    for (document, names) in [
        (
            caller,
            &["resolve", "attempt_1", "attempt_2", "attempt_3", "result"][..],
        ),
        (pass, &["combine", "aggregate", "reconcile"][..]),
    ] {
        for name in names {
            let (_, prepare) = h::step(
                h::steps(h::job(document, name)?)?,
                "uses",
                "./.github/actions/prepare-automation",
            )?;
            h::binding(
                h::member(prepare, "with")?,
                "runner-profile",
                CONTROL_PROFILE,
            )?;
        }
    }
    for name in ["publish-certified-canary", "alert-consecutive-failures"] {
        let job = h::job(caller, name)?;
        if job.get("runs-on").map(Node::list) != Some(BOOTSTRAP_RUNNER.split(',').collect()) {
            return Err(format!("{name} must use the closed managed runner pool").into());
        }
        let (_, prepare) = h::step(
            h::steps(job)?,
            "uses",
            "./.github/actions/prepare-automation",
        )?;
        h::binding(
            h::member(prepare, "with")?,
            "runner-profile",
            CONTROL_PROFILE,
        )?;
    }
    h::binding(h::job(pass, "build")?, "runs-on", PASS_MAC_RUNNER)?;
    h::binding(h::job(pass, "build_linux")?, "runs-on", PASS_LINUX_RUNNER)?;
    for name in ["family", "retry_family"] {
        h::binding(h::job(pass, name)?, "runs-on", PASS_TIERED_RUNNER)?;
    }

    // Resolve owns placement: the outputs exist and the select step constructs
    // the label sets only from the managed label vocabulary.
    let resolve = h::job(caller, "resolve")?;
    let outputs = h::member(resolve, "outputs")?;
    for name in [
        "runner_labels_128",
        "runner_labels_256",
        "runner_labels_macos",
        "runner_labels_linux",
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
    for name in ["runner_labels_128", "runner_labels_256"] {
        let line = script
            .lines()
            .find(|line| line.contains(&format!("{name}=")))
            .ok_or("missing memory label output")?;
        if line.contains("macOS") || line.contains("Linux") {
            return Err("family runner labels must select memory without an OS".into());
        }
    }
    let labels_mac = script
        .lines()
        .find(|line| line.contains("runner_labels_macos="))
        .ok_or("missing macOS producer labels")?;
    let labels_linux = script
        .lines()
        .find(|line| line.contains("runner_labels_linux="))
        .ok_or("missing Linux producer labels")?;
    if !labels_mac.contains("macOS") || !labels_linux.contains("Linux") {
        return Err("native producer labels must select their own OS".into());
    }

    // The family pass receives placement from its caller; it cannot invent or
    // default it.
    let inputs = pass
        .get("on")
        .and_then(|on| on.get("workflow_call"))
        .and_then(|call| call.get("inputs"))
        .ok_or("canary family pass must declare workflow_call inputs")?;
    for name in [
        "runner_labels_128",
        "runner_labels_256",
        "runner_labels_macos",
        "runner_labels_linux",
    ] {
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
                "runner_labels_macos",
                "${{ needs.resolve.outputs.runner_labels_macos }}",
            )?;
            h::binding(
                with,
                "runner_labels_linux",
                "${{ needs.resolve.outputs.runner_labels_linux }}",
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci_validation::lane_results::workflow_yaml;

    const PASS_128_RUNNER: &str = "${{ fromJSON(inputs.runner_labels_128) }}";

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
                workflow_yaml::parse_resolved_aliases(FAMILY).unwrap(),
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
    fn controller_jobs_reject_hosted_runner_fallbacks() {
        for (workflow, job_name) in [
            (CALLER, "resolve"),
            (CALLER, "result"),
            (CALLER, "publish-certified-canary"),
            (CALLER, "alert-consecutive-failures"),
            (PASS, "combine"),
            (PASS, "aggregate"),
            (PASS, "reconcile"),
        ] {
            let mut changed = workflows();
            let document = changed.get_mut(workflow).unwrap();
            let job = member_mut(member_mut(document, "jobs"), job_name);
            *member_mut(job, "runs-on") = Node::Scalar("ubuntu-24.04".into());
            assert!(check(&changed).is_err(), "{workflow}/{job_name}");
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
    fn memory_256_placement_cannot_select_an_os() {
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
                    line.replace("\"ARM64\"", "\"Linux\", \"ARM64\"")
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
