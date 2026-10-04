//! Immutable external actions and explicit protected default-branch executors.
use super::super::lane_results::workflow_yaml::{self, ActionReference, Node};
use crate::command::DynResult;
use std::{fs, path::Path};

const PRE_CHECKOUT: &str = "Mesh-LLM/mesh-llm/.github/actions/audit-depot-pr-isolation@ed07043b84d720aab30e75ed2f038f7042576f16";
const PRE_CHECKOUT_CALLERS: &[&str] = &[
    "ci-linux-host-slice.yml",
    "ci-linux-product-slice.yml",
    "ci-linux-runtime-slice.yml",
    "ci-macos-host-slice.yml",
    "ci-macos-product-slice.yml",
    "ci-macos-runtime-slice.yml",
    "ci-platform-checks-slice.yml",
    "ci-quality-slice.yml",
    "ci-rust-tests-slice.yml",
    "ci-ui-artifact-slice.yml",
    "ci-web-slice.yml",
    "ci-windows-host-slice.yml",
    "ci-windows-product-slice.yml",
    "ci-windows-runtime-slice.yml",
    "native-sdk-artifact.yml",
    "static-abi-artifact.yml",
    "swift-sdk-artifact.yml",
];

pub(super) fn check(root: &Path) -> DynResult<()> {
    for entry in fs::read_dir(root.join(".github/workflows"))? {
        let path = entry?.path();
        if yaml(&path) {
            check_file(&path)?;
        }
    }
    for entry in fs::read_dir(root.join(".github/actions"))? {
        let directory = entry?.path();
        if directory.is_dir() {
            for name in ["action.yml", "action.yaml"] {
                let path = directory.join(name);
                if path.is_file() {
                    check_file(&path)?;
                }
            }
        }
    }
    Ok(())
}

fn yaml(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|v| v.to_str()),
        Some("yml" | "yaml")
    )
}

fn check_file(path: &Path) -> DynResult<()> {
    let name = path
        .file_name()
        .and_then(|v| v.to_str())
        .ok_or("invalid YAML filename")?;
    let (document, references) = workflow_yaml::parse_with_actions(&fs::read_to_string(path)?)?;
    execution_structure(&document)?;
    for reference in references {
        validate(name, &reference)
            .map_err(|error| format!("{}:{}: {error}", path.display(), reference.line))?;
    }
    Ok(())
}

fn execution_structure(document: &Node) -> Result<(), &'static str> {
    if let Some(jobs) = document.get("jobs") {
        let Node::Map(entries) = jobs else {
            return Err("pin audit requires a supported jobs mapping");
        };
        for (_, job) in entries {
            if !matches!(job, Node::Map(_)) {
                return Err("pin audit requires supported job mappings");
            }
            execution_steps(job)?;
        }
    }
    if let Some(runs) = document.get("runs") {
        if !matches!(runs, Node::Map(_)) {
            return Err("pin audit requires a supported action runs mapping");
        }
        execution_steps(runs)?;
    }
    Ok(())
}

fn execution_steps(container: &Node) -> Result<(), &'static str> {
    if let Some(steps) = container.get("steps") {
        let Node::Seq(entries) = steps else {
            return Err("pin audit requires a supported steps sequence");
        };
        if entries.iter().any(|step| !matches!(step, Node::Map(_))) {
            return Err("pin audit requires supported step mappings");
        }
    }
    Ok(())
}

fn validate(name: &str, action: &ActionReference) -> Result<(), &'static str> {
    let reference = action.reference.as_str();
    if reference.starts_with("./") {
        return Ok(());
    }
    if protected_lane(name, reference) {
        return Ok(());
    }
    if reference == PRE_CHECKOUT {
        return if PRE_CHECKOUT_CALLERS.contains(&name) {
            Ok(())
        } else {
            Err("protected pre-checkout action is outside its approved caller scope")
        };
    }
    let Some((identity, pin)) = reference.split_once('@') else {
        return Err("external action requires a full commit SHA and release comment");
    };
    if identity.is_empty()
        || identity.bytes().any(|byte| byte.is_ascii_whitespace())
        || pin.len() != 40
        || !pin
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        || action.provenance.is_none()
    {
        return Err("external action requires a full commit SHA and release comment");
    }
    Ok(())
}

fn protected_lane(name: &str, reference: &str) -> bool {
    ["quality", "website", "linux", "macos", "windows"]
        .iter()
        .any(|lane| {
            name == format!("pr_{lane}.yml")
                && reference
                    == format!("Mesh-LLM/mesh-llm/.github/workflows/ci-{lane}-lane.yml@main")
        })
        || (name == "pr_ci_canary.yml"
            && reference == "Mesh-LLM/mesh-llm/.github/workflows/ci-pr-canary-lane.yml@main")
}

#[cfg(test)]
#[path = "action_pins_tests.rs"]
mod tests;
