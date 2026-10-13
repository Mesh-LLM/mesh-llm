//! The hosted CPU cache has a separate authority and immutable runtime identity.
use super::Node;
use std::collections::BTreeMap;

pub(super) const AUTHORITY: &str =
    "needs.runner_policy.outputs.allow_native_github_cache_cpu == 'true'";
pub(super) const DEPOT: &str = "${{ matrix.runtime.backend != 'cpu' && needs.runner_policy.outputs.allow_depot_remote_cache }}";
pub(super) const NATIVE: &str = "${{ matrix.runtime.backend == 'cpu' && needs.runner_policy.outputs.allow_native_github_cache_cpu || needs.runner_policy.outputs.allow_native_github_cache }}";
const PIN: &str = "caa296126883cff596d87d8935842f9db880ef25";
const RESTORE: &str = "Restore exact Linux CPU runtime";
const SAVE: &str = "Save exact Linux CPU runtime from trusted main";
const RESTORE_GATE: &str = "${{ matrix.runtime.backend == 'cpu' && needs.runner_policy.outputs.allow_native_github_cache_cpu == 'true' && !startsWith(needs.runner_policy.outputs.runner_cpu, 'depot-') }}";
const SAVE_GATE: &str = "${{ matrix.runtime.backend == 'cpu' && github.event_name == 'push' && github.ref == 'refs/heads/main' && needs.runner_policy.outputs.allow_native_github_cache_cpu == 'true' && !startsWith(needs.runner_policy.outputs.runner_cpu, 'depot-') && steps.runtime_cache.outputs.cache-hit != 'true' }}";
const KEY: &str = "skippy-runtime-linux-${{ matrix.runtime.target }}-${{ matrix.runtime.toolchain_epoch }}-${{ hashFiles('skippy/**', 'Cargo.toml', 'Cargo.lock', '.cargo/**', 'Justfile', 'just/**', 'scripts/**', 'ci/slices.yml', '.github/actions/prepare-native-runtime-input/**', '.github/cache-version.txt') }}";

fn member<'a>(node: &'a Node, key: &str) -> Result<&'a Node, String> {
    node.get(key)
        .ok_or_else(|| format!("CPU cache {key} missing"))
}
fn bind(node: &Node, key: &str, value: &str) -> Result<(), String> {
    if node.get(key).and_then(Node::text) != Some(value) {
        return Err(format!("CPU cache {key} must bind {value}"));
    }
    Ok(())
}
fn step<'a>(steps: &'a [Node], name: &str) -> Result<(usize, &'a Node), String> {
    let mut matching = steps
        .iter()
        .enumerate()
        .filter(|(_, step)| step.get("name").and_then(Node::text) == Some(name));
    let found = matching
        .next()
        .ok_or_else(|| format!("CPU cache {name} missing"))?;
    if matching.next().is_some() {
        return Err(format!("CPU cache {name} duplicated"));
    }
    Ok(found)
}
fn operation(node: &Node, action: &str, condition: &str, key: &str) -> Result<(), String> {
    bind(node, "uses", &format!("actions/cache/{action}@{PIN}"))?;
    bind(node, "if", condition)?;
    if node.get("continue-on-error").is_some() {
        return Err("CPU cache failure cannot be ignored".into());
    }
    let inputs = member(node, "with")?;
    bind(inputs, "path", "runtime-input")?;
    bind(inputs, "key", key)?;
    if inputs.get("restore-keys").is_some() {
        return Err("CPU runtime cache must require an exact identity".into());
    }
    Ok(())
}
pub(super) fn check(workflows: &BTreeMap<String, Node>) -> Result<(), String> {
    let workflow = workflows
        .get("ci-linux-runtime-slice.yml")
        .ok_or("CPU cache workflow missing")?;
    let jobs = member(workflow, "jobs")?;
    let outputs = member(member(jobs, "runner_policy")?, "outputs")?;
    bind(
        outputs,
        "runner_cpu",
        "${{ steps.cpu_policy.outputs.runner_16 }}",
    )?;
    bind(
        outputs,
        "allow_native_github_cache_cpu",
        "${{ steps.cpu_policy.outputs.allow_native_github_cache }}",
    )?;
    let runtime = member(jobs, "linux_runtime")?;
    bind(
        runtime,
        "runs-on",
        "${{ matrix.runtime.backend == 'cpu' && needs.runner_policy.outputs.runner_cpu || needs.runner_policy.outputs.runner_16 }}",
    )?;
    let Node::Seq(steps) = member(runtime, "steps")? else {
        return Err("CPU cache runtime steps missing".into());
    };
    let (restore_index, restore) = step(steps, RESTORE)?;
    bind(restore, "id", "runtime_cache")?;
    operation(restore, "restore", RESTORE_GATE, KEY)?;
    let (verify, _) = step(steps, "Verify restored Linux CPU runtime")?;
    let (prepare, _) = step(steps, "Prepare immutable Linux native runtime")?;
    let (upload, _) = step(steps, "Upload immutable Linux runtime input")?;
    let (save_index, save) = step(steps, SAVE)?;
    operation(
        save,
        "save",
        SAVE_GATE,
        "${{ steps.runtime_cache.outputs.cache-primary-key }}",
    )?;
    if !(restore_index < verify && verify < prepare && prepare < upload && upload < save_index) {
        return Err("CPU runtime cache must restore, verify, prepare, upload, then save".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "cpu_runtime_cache_tests.rs"]
mod tests;
