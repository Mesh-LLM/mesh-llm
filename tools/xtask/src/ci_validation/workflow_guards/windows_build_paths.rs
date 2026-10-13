//! Managed Windows compilation jobs configure short paths before compiler caches.
use super::{Node, field, workflow_yaml};
use crate::command::DynResult;
use std::{fs, path::Path};
const ACTION: &str = "./.github/actions/setup-windows-short-paths";
const WINDOWS_MATRIX: &str = "${{ matrix.check.platform == 'windows' }}";

pub(super) fn check(root: &Path) -> DynResult<()> {
    for (workflow, jobs) in [
        ("ci-platform-checks-slice.yml", &["platform_checks"][..]),
        ("ci-windows-host-slice.yml", &["windows_host"][..]),
        ("ci-windows-runtime-slice.yml", &["windows_runtime"][..]),
        ("node-sdk-addon-artifact.yml", &["windows_addon"][..]),
        (
            "release.yml",
            &[
                "windows_host_input",
                "build_native_runtime_windows_cpu",
                "build_native_runtime_windows_gpu",
            ][..],
        ),
        (
            "windows-warm-caches.yml",
            &["warm_windows_cpu", "warm_windows_gpu"][..],
        ),
    ] {
        let source = fs::read_to_string(root.join(".github/workflows").join(workflow))?;
        let document = workflow_yaml::parse(&source)?;
        for job in jobs {
            let job = document
                .get("jobs")
                .and_then(|jobs| jobs.get(job))
                .ok_or_else(|| format!("managed Windows job missing: {workflow}/{job}"))?;
            job_paths(
                job,
                if workflow == "ci-platform-checks-slice.yml" {
                    Some(WINDOWS_MATRIX)
                } else {
                    None
                },
            )?;
        }
    }
    Ok(())
}
fn job_paths(job: &Node, condition: Option<&str>) -> DynResult<()> {
    let Some(Node::Seq(steps)) = job.get("steps") else {
        return Err("managed Windows compilation job has no steps".into());
    };
    let setups: Vec<_> = steps
        .iter()
        .enumerate()
        .filter(|(_, step)| field(step, "uses") == Some(ACTION))
        .collect();
    if setups.len() != 1 {
        return Err("managed Windows compilation requires one short-path setup".into());
    }
    let (position, setup) = setups[0];
    if field(setup, "if") != condition {
        return Err("Windows short-path setup may not be skipped on its compilation branch".into());
    }
    for step in &steps[..position] {
        if compiler_cache(step) || compiler_command(step) {
            return Err(
                "Windows short paths must precede compiler cache setup and compilation".into(),
            );
        }
    }
    Ok(())
}
fn compiler_cache(step: &Node) -> bool {
    field(step, "uses").is_some_and(|action| {
        action.starts_with("mozilla-actions/sccache-action@")
            || action.starts_with("Swatinem/rust-cache@")
            || action == "./.github/actions/configure-sccache-gha"
            || action == "./.github/actions/restore-windows-abi-cache"
    })
}
fn compiler_command(step: &Node) -> bool {
    field(step, "run").is_some_and(|run| {
        run.lines()
            .filter(|line| !line.trim_start().starts_with('#'))
            .any(|line| {
                line.split_whitespace().any(|word| {
                    matches!(word, "cargo" | "cmake")
                        || word.contains("scripts/build-windows.ps1")
                        || word.contains("scripts/build-llama.sh")
                })
            })
    })
}
#[cfg(test)]
#[path = "windows_build_paths_tests.rs"]
mod tests;
