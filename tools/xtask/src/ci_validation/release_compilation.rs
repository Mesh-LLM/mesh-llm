use super::lane_results::workflow_yaml::{self, Node};
use crate::command::DynResult;
use std::path::Path;

pub(super) const COMPOSITION_JOBS: &[&str] = &[
    "compose_cpu_products",
    "compose_linux_arm64_cpu",
    "compose_linux_aarch64_cuda",
    "compose_linux_cuda",
    "compose_linux_rocm",
    "compose_linux_vulkan",
    "compose_windows_cpu",
    "compose_windows_gpu",
];

fn field<'a>(node: &'a Node, name: &str) -> Option<&'a str> {
    node.get(name).and_then(Node::text)
}

fn steps(node: &Node) -> &[Node] {
    match node.get("steps") {
        Some(Node::Seq(items)) => items,
        Some(Node::Scalar(_) | Node::Map(_)) | None => &[],
    }
}

fn condition(node: &Node) -> String {
    field(node, "if")
        .unwrap_or("success()")
        .trim()
        .trim_start_matches("${{")
        .trim_end_matches("}}")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn covers(setup: &Node, caller: &Node) -> bool {
    let setup = condition(setup);
    let caller = condition(caller);
    setup == "always()"
        || setup == "success() || failure() || cancelled()"
        || setup == caller
        || (setup == "success()"
            && !["always()", "failure()", "cancelled()"]
                .iter()
                .any(|status| caller.contains(status)))
}

fn suppressed(node: &Node) -> bool {
    field(node, "continue-on-error") == Some("true")
}

fn package_bypass(job: &Node, step: &Node) -> bool {
    [
        "MESH_RELEASE_HOST_PRESTAMPED",
        "MESH_RELEASE_ATTESTATION_PREVERIFIED",
    ]
    .iter()
    .all(|key| {
        step.get("env")
            .and_then(|env| field(env, key))
            .or_else(|| job.get("env").and_then(|env| field(env, key)))
            == Some("1")
    })
}

fn cargo_step(job: &Node, step: &Node) -> bool {
    let run = field(step, "run").unwrap_or("");
    let executable = run
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join("\n");
    [
        "cargo ",
        "scripts/release-version.sh",
        "scripts/publish-crates.sh",
    ]
    .iter()
    .any(|marker| executable.contains(marker))
        || (executable.contains("scripts/package-release.sh") && !package_bypass(job, step))
}

fn check_setup(document: &Node, job: &Node, callers: &[(usize, &Node)]) -> DynResult<()> {
    if suppressed(document) || suppressed(job) {
        return Err("suppresses failures around required sccache initialization".into());
    }
    let hosted = field(job, "runs-on") == Some("ubuntu-24.04") && job.get("container").is_none();
    let markers = [
        "mozilla-actions/sccache-action@",
        "./.github/actions/configure-sccache-gha",
    ];
    let mut previous = None;
    let first = callers.first().ok_or("missing Cargo caller")?.0;
    let mut initialized = false;
    for marker in markers {
        let setup = steps(job).iter().enumerate().find(|(index, step)| {
            *index < first && field(step, "uses").is_some_and(|uses| uses.starts_with(marker))
        });
        let Some((index, setup)) = setup else {
            if hosted {
                return Err(
                    format!("Cargo before required sccache initialization {marker}").into(),
                );
            }
            continue;
        };
        if previous.is_some_and(|previous| hosted && index < previous) || suppressed(setup) {
            return Err(
                "sccache initialization order, condition or failure suppression is invalid".into(),
            );
        }
        for (_, caller) in callers {
            if !covers(setup, caller) {
                return Err(format!(
                    "sccache setup condition `{}` does not cover Cargo condition `{}` in {:?}",
                    condition(setup),
                    condition(caller),
                    field(caller, "run")
                )
                .into());
            }
        }
        previous = Some(index);
        initialized = true;
    }
    if !initialized {
        return Err("Cargo before any required sccache initialization".into());
    }
    Ok(())
}

fn check_composer(root: &Path, job: &Node) -> DynResult<()> {
    if !steps(job)
        .iter()
        .any(|step| field(step, "uses") == Some("./.github/actions/restore-automation"))
    {
        return Err(
            "composition-only job must restore immutable automation before adapters run".into(),
        );
    }
    for step in steps(job) {
        if cargo_step(job, step) {
            return Err("composition-only job invokes Cargo".into());
        }
        check_action(root, job, step, &mut std::collections::BTreeSet::new())?;
    }
    Ok(())
}

fn check_action(
    root: &Path,
    job: &Node,
    step: &Node,
    visited: &mut std::collections::BTreeSet<String>,
) -> DynResult<()> {
    let Some(action) = field(step, "uses").and_then(|uses| uses.strip_prefix("./")) else {
        return Ok(());
    };
    if !visited.insert(action.to_owned()) {
        return Ok(());
    }
    let source = std::fs::read_to_string(root.join(action).join("action.yml"))?;
    let document = workflow_yaml::parse(&source)?;
    let runs = document.get("runs").ok_or("local action missing runs")?;
    if source.contains("automation-bootstrap")
        || steps(runs).iter().any(|nested| cargo_step(job, nested))
    {
        return Err(format!("composition-only job reaches Cargo through {action}").into());
    }
    for nested in steps(runs) {
        check_action(root, job, nested, visited)?;
    }
    Ok(())
}

pub(super) fn check(root: &Path, source: &str) -> DynResult<()> {
    let document = workflow_yaml::parse(source)?;
    let jobs = document
        .get("jobs")
        .ok_or("release workflow missing jobs")?;
    for (name, job) in jobs.entries() {
        let result = if COMPOSITION_JOBS.contains(&name.as_str()) {
            let producer = if name.starts_with("compose_windows") {
                "windows_host_input"
            } else if name.contains("arm64") || name.contains("aarch64") {
                "build_linux_arm64"
            } else {
                "build"
            };
            if !job
                .get("needs")
                .is_some_and(|needs| needs.list().contains(&producer))
            {
                return Err(format!(
                    "release workflow `{name}`: automation producer {producer} missing from needs"
                )
                .into());
            }
            let producer_job = jobs.get(producer).ok_or("missing automation producer")?;
            if !steps(producer_job)
                .iter()
                .any(|step| field(step, "uses") == Some("./.github/actions/upload-automation"))
            {
                return Err(format!(
                    "release workflow `{name}`: host producer does not publish automation"
                )
                .into());
            }
            check_composer(root, job)
        } else {
            let callers = steps(job)
                .iter()
                .enumerate()
                .filter(|(_, step)| cargo_step(job, step))
                .collect::<Vec<_>>();
            if callers.is_empty() {
                Ok(())
            } else {
                check_setup(&document, job, &callers)
            }
        };
        result.map_err(|error| format!("release workflow `{name}`: {error}"))?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "release_compilation_tests.rs"]
mod tests;
