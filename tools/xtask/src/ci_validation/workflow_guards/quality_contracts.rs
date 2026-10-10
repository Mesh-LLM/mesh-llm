//! Required portable replacement coverage in local and hosted Quality gates.
use super::{Node, field};
use crate::command::DynResult;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
const REGISTRY: &str = "ci/quality-rust-contract-targets.json";

pub(super) fn check(root: &Path, workflows: &BTreeMap<String, Node>) -> DynResult<()> {
    let targets: Vec<String> = serde_json::from_slice(&std::fs::read(root.join(REGISTRY))?)?;
    if targets.is_empty() || targets.iter().collect::<BTreeSet<_>>().len() != targets.len() {
        return Err("Quality requires nonempty unique Cargo targets".into());
    }
    for target in &targets {
        if !target
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_')
            || !root
                .join(format!("tools/xtask/tests/{target}.rs"))
                .is_file()
        {
            return Err(format!("Quality target has no integration owner: {target}").into());
        }
    }
    integration_owners(root, &targets)?;
    let just = std::fs::read_to_string(root.join("just/ci.just"))?;
    recipe(&just, &targets)?;
    local(&just)?;
    quality(
        workflows
            .get("ci-quality-slice.yml")
            .ok_or("missing Quality slice")?,
    )
}
fn integration_owners(root: &Path, targets: &[String]) -> DynResult<()> {
    let selected: BTreeSet<_> = targets.iter().map(String::as_str).collect();
    let mut omitted = BTreeSet::new();
    for entry in std::fs::read_dir(root.join("tools/xtask/tests"))? {
        let path = entry?.path();
        if path.extension() != Some(std::ffi::OsStr::new("rs")) || !path.is_file() {
            continue;
        }
        let target = path
            .file_stem()
            .and_then(std::ffi::OsStr::to_str)
            .ok_or("Quality integration owner has an invalid target name")?;
        if !selected.contains(target) {
            omitted.insert(target.to_owned());
        }
    }
    if omitted.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Quality integration owners omitted from normal CI: {}",
            omitted.into_iter().collect::<Vec<_>>().join(", ")
        )
        .into())
    }
}
fn body<'a>(source: &'a str, name: &str) -> DynResult<Vec<&'a str>> {
    let header = format!("{name}:");
    let mut lines = source.lines().skip_while(|line| line.trim() != header);
    if lines.next().is_none() {
        return Err(format!("missing recipe {name}").into());
    }
    Ok(lines
        .take_while(|l| l.is_empty() || l.starts_with(char::is_whitespace))
        .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
        .map(str::trim)
        .collect())
}
fn position(commands: &[&str], prefix: &str) -> DynResult<usize> {
    let values: Vec<_> = commands
        .iter()
        .enumerate()
        .filter_map(|(i, l)| l.starts_with(prefix).then_some(i))
        .collect();
    if values.len() != 1 {
        return Err(format!("requires one direct {prefix}").into());
    }
    Ok(values[0])
}
fn recipe(source: &str, targets: &[String]) -> DynResult<()> {
    let normalized = body(source, "ci-automation-contracts")?
        .join("\n")
        .replace("\\\n", " ");
    let commands: Vec<_> = normalized.lines().map(str::trim).collect();
    if !commands.contains(&"set -euo pipefail")
        || commands.iter().any(|line| {
            line.contains("|| true")
                || line.starts_with("exit ")
                || line.starts_with("return ")
                || (line.starts_with("if ")
                    && *line != "if command -v cygpath >/dev/null 2>&1; then")
        })
    {
        return Err(
            "required coverage must not be masked or placed under an unrelated predicate".into(),
        );
    }
    if commands.iter().any(|line| {
        line.split_whitespace().any(|word| {
            word == "MESH_LLM_AUTOMATION_BIN" || word.starts_with("MESH_LLM_AUTOMATION_BIN=")
        })
    }) {
        return Err("Quality recipe must preserve configured controller admission and absent-override fallback".into());
    }
    if commands.iter().any(|line| {
        let first = line.split_whitespace().next().unwrap_or("");
        matches!(
            first,
            "while"
                | "until"
                | "for"
                | "select"
                | "function"
                | "{"
                | "("
                | "trap"
                | "exec"
                | "break"
                | "continue"
        ) || (first == "case"
            && *line != "case \"$binary\" in *.exe) fixture_suffix=\".exe\" ;; esac")
            || (first == "set" && *line != "set -euo pipefail")
            || line.contains("set +e")
            || line.contains("set +o errexit")
    }) {
        return Err(
            "Quality contract execution must retain errexit and direct control flow".into(),
        );
    }
    let bootstrap = position(&commands, "bootstrap=\"$(just automation-bootstrap)\"")?;
    let target = position(&commands, "target_directory=")?;
    let export = position(&commands, "export CARGO_TARGET_DIR=\"$target_directory\"")?;
    let build = position(&commands, "just with-lld cargo build --locked -p xtask ")?;
    let fixture = position(&commands, "export SPV_ADAPTER_FIXTURE=")?;
    let git = position(&commands, "export MIGRATION_TEST_GIT=\"$git_path\"")?;
    let test = position(&commands, "just with-lld cargo test --locked -p xtask ")?;
    if !(bootstrap < target
        && target < export
        && export < build
        && build < fixture
        && fixture < test
        && git < test)
        || !commands[target].contains("$1 == \"target_directory\"")
        || !commands[fixture]
            .contains("$target_directory/debug/examples/swift_privacy_lint_fixture$fixture_suffix")
    {
        return Err("fixtures must follow actual bootstrap target and inert build".into());
    }
    let words: BTreeSet<_> = commands[build].split_whitespace().collect();
    let required: BTreeSet<_> = [
        "just",
        "with-lld",
        "cargo",
        "build",
        "--locked",
        "-p",
        "xtask",
        "--bins",
        "--examples",
    ]
    .into_iter()
    .collect();
    if words != required {
        return Err("inert prebuild must not select a product/native package".into());
    }
    for word in ["--locked", "-p", "xtask", "--bins", "--examples"] {
        if !words.contains(word) {
            return Err("prebuild locked xtask bins/examples".into());
        }
    }
    if test + 1 != commands.len() {
        return Err("direct required Cargo test must be the terminal recipe statement".into());
    }
    invocation(commands[test], targets)?;
    reader_component(&commands, build, fixture, test)?;
    snapshot_component(source, &commands)
}

fn reader_component(commands: &[&str], build: usize, fixture: usize, test: usize) -> DynResult<()> {
    let required = [
        "just with-lld cargo build --locked -p trajectory-reader --features corpus-input --bin trajectory-reader",
        "just with-lld cargo test --locked -p trajectory-reader --features corpus-input --lib -- --test-threads=1",
        "MESH_LLM_TEST_XTASK_BIN=\"$binary\" just with-lld cargo test --locked -p trajectory-reader --features corpus-input --test prompt_command native_prompt_command_ -- --test-threads=1",
    ];
    let mut previous = build;
    for expected in required {
        let current = position(commands, expected)?;
        if commands[current] != expected || current <= previous || current >= fixture {
            return Err(
                "reader contracts must follow actual bootstrap and build before fixture admission"
                    .into(),
            );
        }
        previous = current;
    }
    for (index, line) in commands.iter().enumerate() {
        if (line.starts_with("just with-lld cargo build ")
            || line.starts_with("just with-lld cargo test "))
            && index != build
            && index != test
            && !required.contains(line)
        {
            return Err("unexpected Cargo invocation in required contracts".into());
        }
    }
    Ok(())
}

fn snapshot_component(source: &str, commands: &[&str]) -> DynResult<()> {
    if commands
        .iter()
        .filter(|line| **line == "just ci-snapshot-promotion-contracts")
        .count()
        != 1
    {
        return Err("required native snapshot component coverage omitted or masked".into());
    }
    if body(source, "ci-snapshot-promotion-contracts")?
        != [
            "just with-lld cargo test --locked -p skippy-model-package --lib --test snapshot_promotion_cli -- --test-threads=1",
        ]
    {
        return Err(
            "native snapshot library and actual CLI tests must be selected without filtering"
                .into(),
        );
    }
    Ok(())
}
fn invocation(command: &str, targets: &[String]) -> DynResult<()> {
    let mut words = command.split_whitespace();
    if words.by_ref().take(5).collect::<Vec<_>>()
        != ["just", "with-lld", "cargo", "test", "--locked"]
    {
        return Err("direct locked tests must use repository toolchain authority".into());
    }
    let mut observed = BTreeSet::new();
    let (mut package, mut unit) = (false, false);
    while let Some(word) = words.next() {
        match word {
            "-p" if !package => {
                package = words.next() == Some("xtask");
                if !package {
                    break;
                }
            }
            "--bin" if !unit => {
                unit = words.next() == Some("xtask");
                if !unit {
                    break;
                }
            }
            "--test" => {
                let value = words.next().ok_or("missing target")?;
                if !observed.insert(value) {
                    return Err("duplicate contract target".into());
                }
            }
            "--" => {
                if words.collect::<Vec<_>>() != ["--test-threads=1"] {
                    return Err(
                        "serial harness must have no test filter or ignored selection".into(),
                    );
                }
                if package && unit && observed == targets.iter().map(String::as_str).collect() {
                    return Ok(());
                }
                break;
            }
            _ => break,
        }
    }
    Err("select every reviewed integration target and owning unit harness".into())
}
fn local(source: &str) -> DynResult<()> {
    let calls = body(source, "ci-validate")?;
    let rust = calls
        .iter()
        .position(|l| *l == "just ci-automation-contracts")
        .ok_or("local Rust coverage omitted")?;
    let legacy = calls
        .iter()
        .position(|l| *l == "just ci-legacy-contracts")
        .ok_or("surviving legacy coverage omitted")?;
    if rust >= legacy {
        return Err("Rust contracts must precede legacy contracts".into());
    }
    Ok(())
}
fn quality(workflow: &Node) -> DynResult<()> {
    let steps = workflow
        .get("jobs")
        .and_then(|j| j.get("quality_contracts"))
        .and_then(|j| j.get("steps"))
        .ok_or("missing Quality steps")?;
    let Node::Seq(steps) = steps else {
        return Err("Quality steps must be sequence".into());
    };
    if steps.iter().any(|step| {
        field(step, "uses").is_some_and(|value| value.starts_with("actions/setup-python@"))
            || field(step, "run").is_some_and(|run| {
                run.contains("requirements-ci-python.txt")
                    || run
                        .split_whitespace()
                        .any(|word| matches!(word, "python" | "python3" | "pip" | "pip3"))
            })
    }) {
        return Err("Quality contracts require native tooling, not Python setup or install".into());
    }
    let selected: Vec<_> = steps
        .iter()
        .filter(|s| field(s, "name") == Some("Test CI, packaging, and SDK contracts"))
        .collect();
    if selected.len() != 1 {
        return Err("requires one contract execution step".into());
    }
    let step = selected[0];
    if step.get("if").is_some() || field(step, "continue-on-error").is_some_and(|v| v != "false") {
        return Err("contract execution must be unconditional and required".into());
    }
    let calls: Vec<_> = field(step, "run")
        .ok_or("missing execution")?
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    if calls != ["just ci-automation-contracts", "just ci-legacy-contracts"] {
        return Err("execute canonical Rust coverage then surviving legacy coverage".into());
    }
    Ok(())
}
#[cfg(test)]
#[path = "quality_contracts_tests.rs"]
mod tests;
