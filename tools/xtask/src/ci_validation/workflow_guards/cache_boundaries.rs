//! Explicit current namespace and trusted release/provider exceptions.
use super::{Node, cache_predicate::requires};
use std::collections::BTreeMap;
fn text<'a>(node: &'a Node, key: &str) -> &'a str {
    node.get(key).and_then(Node::text).unwrap_or("")
}
fn steps(document: &Node) -> impl Iterator<Item = &Node> {
    document
        .get("jobs")
        .into_iter()
        .flat_map(Node::entries)
        .flat_map(|(_, job)| match job.get("steps") {
            Some(Node::Seq(steps)) => steps.as_slice(),
            _ => &[],
        })
}
pub(super) fn check(workflows: &BTreeMap<String, Node>) -> Result<(), String> {
    for name in [
        "ci-quality-slice.yml",
        "ci-rust-tests-slice.yml",
        "ci-linux-host-slice.yml",
        "ci-linux-runtime-slice.yml",
        "static-abi-artifact.yml",
    ] {
        let workflow = workflows.get(name).ok_or("cache policy workflow missing")?;
        let env = workflow.get("env").ok_or("cache namespace missing")?;
        if text(env, "CACHE_NAMESPACE") != "mesh-llm" {
            return Err(format!("{name}: cache namespace changed"));
        }
        if name != "static-abi-artifact.yml" && text(env, "SCCACHE_GHA_ENABLED") != "false" {
            return Err(format!(
                "{name}: implicit GHA compiler cache must remain disabled"
            ));
        }
    }
    let release = workflows.get("release.yml").ok_or("release missing")?;
    let outputs = release
        .get("jobs")
        .and_then(|j| j.get("metadata"))
        .and_then(|j| j.get("outputs"))
        .ok_or("release metadata outputs missing")?;
    if text(outputs, "allow_native_github_cache")
        != "${{ steps.runners.outputs.allow_native_github_cache }}"
    {
        return Err("release must project selected native cache decision".into());
    }
    for name in [
        "Cache native runtime ROCm backend build",
        "Cache native runtime Vulkan backend build",
    ] {
        let step = steps(release)
            .find(|s| text(s, "name") == name)
            .ok_or("release cache consumer missing")?;
        if !requires(
            text(step, "if"),
            "!startsWith(needs.metadata.outputs.runner_16, 'depot-')",
        ) {
            return Err(format!("{name}: native cache must exclude Depot provider"));
        }
    }
    let native_runtime = release
        .get("jobs")
        .and_then(|j| j.get("build_native_runtime"))
        .ok_or("release native runtime producer missing")?;
    let Some(Node::Seq(runtime_steps)) = native_runtime.get("steps") else {
        return Err("release runtime steps missing".into());
    };
    let configure = runtime_steps
        .iter()
        .find(|step| text(step, "uses") == "./.github/actions/configure-sccache-gha")
        .and_then(|step| step.get("with"))
        .ok_or("release runtime cache projection missing")?;
    let projection = "${{ ((matrix.target == 'x86_64-unknown-linux-gnu' && startsWith(needs.metadata.outputs.runner_8, 'depot-')) || (matrix.target == 'aarch64-unknown-linux-gnu' && startsWith(needs.metadata.outputs.runner_arm_8, 'depot-'))) && 'false' || 'true' }}";
    if text(configure, "allow_native_github_cache") != projection
        || text(configure, "allow_depot_remote_cache")
            != "${{ needs.metadata.outputs.allow_depot_remote_cache }}"
    {
        return Err("release platform/provider cache decision projection changed".into());
    }
    let warmer = workflows
        .get("windows-warm-caches.yml")
        .ok_or("trusted cache warmer missing")?;
    let mut restores = 0;
    for step in steps(warmer) {
        if matches!(
            text(step, "uses"),
            "./.github/actions/restore-windows-abi-cache"
                | "./.github/actions/setup-windows-rocm-sdk"
        ) {
            if text(
                step.get("with").ok_or("warmer inputs missing")?,
                "allow-native-github-cache",
            ) != "true"
            {
                return Err(
                    "trusted Windows warmer must retain explicit native cache opt-in".into(),
                );
            }
            restores += 1;
        }
    }
    if restores == 0 {
        return Err("Windows warmer cache consumers missing".into());
    }
    Ok(())
}
