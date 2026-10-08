//! Repository cache and target-linker defaults in workflow and action definitions.
use super::{Node, field};
use crate::command::DynResult;
use std::{collections::BTreeSet, fs, path::Path};

const ISOLATED_STEP: &str = "Prepare isolated automation before measurement";

pub(super) fn check(root: &Path) -> DynResult<()> {
    definitions(root, &root.join(".github"))
}

fn definitions(root: &Path, directory: &Path) -> DynResult<()> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            definitions(root, &path)?;
        } else if matches!(
            path.extension().and_then(|s| s.to_str()),
            Some("yml" | "yaml")
        ) {
            let document = super::workflow_yaml::parse(&fs::read_to_string(&path)?)?;
            let relative = path
                .strip_prefix(root)?
                .to_string_lossy()
                .replace('\\', "/");
            check_node(&relative, &document)?;
        }
    }
    Ok(())
}

fn check_node(path: &str, node: &Node) -> DynResult<()> {
    let isolated =
        path == ".github/workflows/depot-canary.yml" && field(node, "name") == Some(ISOLATED_STEP);
    if isolated {
        isolated_automation(node)?;
    }
    if let Some(environment) = node.get("env") {
        environment_defaults(environment, isolated)?;
    }
    if let Some(run) = field(node, "run") {
        linker_defaults(run)?;
    }
    match node {
        Node::Seq(nodes) => {
            for child in nodes {
                check_node(path, child)?;
            }
        }
        Node::Map(entries) => {
            for (key, child) in entries {
                if key != "env" {
                    check_node(path, child)?;
                }
            }
        }
        Node::Scalar(_) => {}
    }
    Ok(())
}

fn environment_defaults(environment: &Node, isolated: bool) -> DynResult<()> {
    for (key, value) in environment.entries() {
        if matches!(key.as_str(), "RUSTC_WRAPPER" | "CARGO_BUILD_RUSTC_WRAPPER")
            && value.text() == Some("")
            && !isolated
        {
            return Err(format!("workflow disables repository compiler cache: {key}").into());
        }
        if key == "LLAMA_STAGE_USE_SCCACHE" && value.text() == Some("0") {
            return Err("workflow disables native compiler cache".into());
        }
        if key.starts_with("CARGO_TARGET_") && key.ends_with("_LINKER") {
            return Err("workflow overrides the repository target linker driver".into());
        }
    }
    Ok(())
}

fn linker_defaults(run: &str) -> DynResult<()> {
    for line in run
        .lines()
        .filter(|line| !line.trim_start().starts_with('#'))
    {
        let words: Vec<_> = line.split_whitespace().collect();
        if line.contains("fuse-ld=")
            || line.contains("-Clinker=")
            || words
                .windows(2)
                .any(|words| words[0] == "-C" && words[1].starts_with("linker="))
            || words
                .iter()
                .any(|word| word.starts_with("CARGO_TARGET_") && word.contains("_LINKER="))
        {
            return Err("workflow bypasses the repository linker probe and target policy".into());
        }
    }
    Ok(())
}

fn isolated_automation(step: &Node) -> DynResult<()> {
    let environment = step
        .get("env")
        .ok_or("isolated automation has no environment")?;
    for (key, expected) in [
        (
            "CARGO_TARGET_DIR",
            "${{ runner.temp }}/runtime-seed-automation-target",
        ),
        (
            "CARGO_HOME",
            "${{ runner.temp }}/runtime-seed-automation-cargo",
        ),
        ("RUSTC_WRAPPER", ""),
        ("CARGO_BUILD_RUSTC_WRAPPER", ""),
    ] {
        if field(environment, key) != Some(expected) {
            return Err(
                format!("isolated automation does not preserve its unmeasured {key}").into(),
            );
        }
    }
    let run = field(step, "run").ok_or("isolated automation has no command")?;
    let mut builds = 0;
    let mut publishes = 0;
    for line in run
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
    {
        if line == "set -euo pipefail" {
            continue;
        }
        if let Some(arguments) = line.strip_prefix("cargo ") {
            automation_build(arguments)?;
            builds += 1;
        } else if line
            == "echo \"MESH_LLM_AUTOMATION_BIN=$CARGO_TARGET_DIR/release/xtask\" >> \"$GITHUB_ENV\""
        {
            publishes += 1;
        } else {
            return Err("isolated automation exception contains another command".into());
        }
    }
    if builds != 1 || publishes != 1 {
        return Err(
            "isolated automation requires one locked release tool build and its published path"
                .into(),
        );
    }
    Ok(())
}

fn automation_build(arguments: &str) -> DynResult<()> {
    let words: Vec<_> = arguments.split_whitespace().collect();
    if words.first() != Some(&"build") {
        return Err("isolated automation may only build xtask".into());
    }
    let mut rest = &words[1..];
    let mut options = BTreeSet::new();
    while let Some((flag, following)) = rest.split_first() {
        if !options.insert(*flag) {
            return Err("duplicate isolated automation build option".into());
        }
        rest = match (*flag, following) {
            ("--locked" | "--release", rest) => rest,
            ("-p" | "--bin", [value, rest @ ..]) if *value == "xtask" => rest,
            _ => {
                return Err(
                    "isolated automation may not build a product, workspace or extra target".into(),
                );
            }
        };
    }
    if options != BTreeSet::from(["--locked", "--release", "-p", "--bin"]) {
        return Err("isolated automation must use its locked release xtask binary".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "build_acceleration_tests.rs"]
mod tests;
