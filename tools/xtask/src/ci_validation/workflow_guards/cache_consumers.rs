//! Native cache consumers bind authorization at the operation, not whole-job admission.
use super::{Node, cache_predicate::requires};
use std::collections::BTreeMap;
const NATIVE: &str = "needs.runner_policy.outputs.allow_native_github_cache == 'true'";
fn text<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap_or("")
}
fn binding(node: &Node, key: &str, expected: &str) -> Result<(), String> {
    if text(node, key) != expected {
        return Err(format!("{key} must bind {expected}"));
    }
    Ok(())
}
fn gate(node: &Node, key: &str, clause: &str) -> Result<(), String> {
    if !requires(text(node, key), clause) {
        return Err(format!("{key} must require {clause}"));
    }
    Ok(())
}
fn consumer(name: &str, job_name: &str, step: &Node) -> Result<(), String> {
    let action = text(step, "uses");
    let empty = Node::Map(Vec::new());
    let inputs = step.get("with").unwrap_or(&empty);
    let cpu_lane = name == "ci-linux-runtime-slice.yml" && job_name == "linux_runtime";
    if action == "./.github/actions/configure-sccache-gha" && cpu_lane {
        binding(
            inputs,
            "allow_depot_remote_cache",
            super::cpu_runtime_cache::DEPOT,
        )?;
        binding(
            inputs,
            "allow_native_github_cache",
            super::cpu_runtime_cache::NATIVE,
        )?;
    } else if action == "./.github/actions/configure-sccache-gha" {
        for flag in ["allow_depot_remote_cache", "allow_native_github_cache"] {
            binding(
                inputs,
                flag,
                &format!("${{{{ needs.runner_policy.outputs.{flag} }}}}"),
            )?;
        }
    }
    if action.starts_with("Swatinem/rust-cache@") || action.starts_with("actions/cache") {
        let cpu_operation = cpu_lane
            && matches!(
                text(step, "name"),
                "Restore exact Linux CPU runtime"
                    | "Save exact Linux CPU runtime from trusted main"
            );
        gate(
            step,
            "if",
            if cpu_operation {
                super::cpu_runtime_cache::AUTHORITY
            } else {
                NATIVE
            },
        )?;
    }
    if action.starts_with("Swatinem/rust-cache@") {
        gate(inputs, "save-if", "github.ref == 'refs/heads/main'")?;
        if name == "swift-sdk-artifact.yml" {
            gate(inputs, "save-if", "github.event_name == 'push'")?;
        } else {
            for event in ["pull_request", "pull_request_target"] {
                gate(
                    inputs,
                    "save-if",
                    &format!("github.event.inputs.original_event_name != '{event}'"),
                )?;
            }
        }
    }
    if matches!(
        action,
        "./.github/actions/restore-windows-abi-cache" | "./.github/actions/setup-windows-rocm-sdk"
    ) {
        binding(
            inputs,
            "allow-native-github-cache",
            "${{ needs.runner_policy.outputs.allow_native_github_cache }}",
        )?;
    }
    if inputs.get("use-github-cache").is_some() {
        gate(inputs, "use-github-cache", NATIVE)?;
    }
    if action.starts_with("jakoch/install-vulkan-sdk-action@") {
        gate(inputs, "cache", NATIVE)?;
        for event in ["pull_request", "pull_request_target"] {
            gate(
                inputs,
                "cache_save_if",
                &format!("inputs.original_event_name != '{event}'"),
            )?;
        }
    }
    if name == "swift-sdk-artifact.yml" && action.starts_with("actions/setup-node@") {
        for key in ["cache", "package-manager-cache"] {
            gate(inputs, key, NATIVE)?;
        }
    }
    if action == "./.github/actions/restore-sccache-seed" {
        if name == "ci-linux-runtime-slice.yml" {
            binding(inputs, "allow_trusted_seed", "false")?;
        } else {
            let value = text(inputs, "allow_trusted_seed");
            if value != "${{ needs.runner_policy.outputs.allow_trusted_sccache_seed }}" {
                // Here 'false' is a string consumed by the action, not a truthy conditional.
                let value = value.replace("'false'", "false");
                if !requires(
                    &value,
                    "needs.runner_policy.outputs.allow_trusted_sccache_seed == 'true'",
                ) {
                    return Err("seed restore must bind central seed authority".into());
                }
            }
        }
    }
    Ok(())
}
fn projection(name: &str, workflow: &Node) -> Result<(), String> {
    let jobs = workflow.get("jobs").ok_or("jobs missing")?;
    let consumes = jobs.entries().iter().any(|(_, job)| {
        matches!(job.get("steps"), Some(Node::Seq(steps)) if steps.iter().any(|step| text(step, "uses") == "./.github/actions/configure-sccache-gha"))
    });
    if !consumes {
        return Ok(());
    }
    let policy = jobs
        .get("runner_policy")
        .ok_or("central cache producer missing")?;
    let outputs = policy
        .get("outputs")
        .ok_or("central cache producer outputs missing")?;
    for flag in ["allow_depot_remote_cache", "allow_native_github_cache"] {
        let producer = if name == "native-sdk-artifact.yml" {
            "resolve"
        } else {
            "policy"
        };
        binding(
            outputs,
            flag,
            &format!("${{{{ steps.{producer}.outputs.{flag} }}}}"),
        )?;
    }
    if name == "native-sdk-artifact.yml" {
        let Some(Node::Seq(steps)) = policy.get("steps") else {
            return Err("SDK resolver missing".into());
        };
        let resolve = steps
            .iter()
            .find(|step| text(step, "id") == "resolve")
            .ok_or("SDK resolver missing")?;
        let env = resolve
            .get("env")
            .ok_or("SDK resolver cache inputs missing")?;
        for (key, flag) in [
            ("ALLOW_DEPOT_REMOTE_CACHE", "allow_depot_remote_cache"),
            ("ALLOW_NATIVE_GITHUB_CACHE", "allow_native_github_cache"),
        ] {
            binding(env, key, &format!("${{{{ steps.policy.outputs.{flag} }}}}"))?;
        }
    }
    Ok(())
}
pub(super) fn check(workflows: &BTreeMap<String, Node>) -> Result<(), String> {
    for (name, workflow) in workflows {
        if !(name.starts_with("ci-") && name.ends_with("-slice.yml")
            || matches!(
                name.as_str(),
                "static-abi-artifact.yml" | "swift-sdk-artifact.yml" | "native-sdk-artifact.yml"
            ))
        {
            continue;
        }
        projection(name, workflow)?;
        for (job_name, job) in workflow.get("jobs").ok_or("jobs missing")?.entries() {
            if job_name == "authority_sentinel" {
                continue;
            }
            if text(job, "if").contains("allow_native_github_cache") {
                return Err(format!(
                    "{name}/{job_name}: cache denial cannot suppress required workload"
                ));
            }
            if let Some(Node::Seq(steps)) = job.get("steps") {
                for step in steps {
                    consumer(name, job_name, step)
                        .map_err(|e| format!("{name}/{job_name}: {e}"))?;
                }
            }
        }
    }
    Ok(())
}
