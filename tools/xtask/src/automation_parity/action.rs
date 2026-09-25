//! The protected `plan-ci` action as the comparison's oracle. Its output
//! derivation block is pinned by digest (the recorded `*.outputs.txt` goldens
//! were produced by exactly these bytes), its caller-contract checks must be
//! present, and when the legacy side runs, the block itself is executed with
//! bash and jq over the legacy planner's stdout.

use super::Run;
use super::process::run_bounded;
use super::stage::Tools;
use crate::command::DynResult;
use sha2::{Digest, Sha256};
use std::fs;
use std::process::Command;

const ACTION: &str = ".github/actions/plan-ci/action.yml";
const BLOCK_START: &str = "plan_json=$(jq -c . ci-plan.json)";
const BLOCK_END: &str = "} >> \"$GITHUB_OUTPUT\"";
const INDENT: &str = "        ";
/// SHA-256 of the dedented derivation block, trailing newline included.
const BLOCK_SHA256: &str = "ea54271346b506b1d3d5ece93d527e6f12b1e6efe0f49459026930b5640abd94";

/// Protected caller contract lines the inert manifest root depends on.
const CONTRACT: [&str; 6] = [
    "[[ \"$SOURCE_SHA\" =~ ^[0-9a-f]{40}$ ]] || { echo \"pull request source SHA is malformed\" >&2; exit 1; }",
    "if [[ \"$mode\" != \"100644\" || \"$type\" != \"blob\" || \"$path\" != \"$manifest\" ]]; then",
    "if ! git show \"$SOURCE_SHA:ci/slices.yml\" | cmp -s - ci/slices.yml; then",
    "if ! git show \"$SOURCE_SHA:ci/ownership.yml\" | cmp -s - ci/ownership.yml; then",
    "git archive --format=tar \"$SOURCE_SHA\" -- ci/ownership.yml ci/slices.yml \\",
    "| python3 scripts/plan-ci.py --manifest-root \"$manifest_root\" > ci-plan.json",
];

/// The dedented output derivation block of the protected action.
fn derivation_block(action: &str) -> Result<String, String> {
    let mut block = String::new();
    let mut inside = false;
    for line in action.lines() {
        let body = line.strip_prefix(INDENT).unwrap_or(line);
        inside |= body == BLOCK_START;
        if inside {
            block.push_str(body);
            block.push('\n');
            if body == BLOCK_END {
                return Ok(block);
            }
        }
    }
    Err(format!("{ACTION}: output derivation block not found"))
}

fn source_difference(action: &str) -> Option<String> {
    let block = match derivation_block(action) {
        Ok(block) => block,
        Err(error) => return Some(error),
    };
    let digest = hex::encode(Sha256::digest(block.as_bytes()));
    if digest != BLOCK_SHA256 {
        return Some(format!(
            "output derivation block digest {digest} != pinned {BLOCK_SHA256}; regenerate the *.outputs.txt goldens from the new block"
        ));
    }
    let lines = action.lines().map(str::trim).collect::<Vec<_>>();
    CONTRACT
        .iter()
        .find(|line| !lines.contains(line))
        .map(|line| format!("protected caller contract line is missing: {line}"))
}

/// Records whether the protected action still matches the pinned oracle.
pub(super) fn action_source(run: &mut Run<'_>) -> DynResult<()> {
    let action = fs::read_to_string(run.root.join(ACTION))?;
    run.ledger
        .record("action-source", ACTION, source_difference(&action));
    Ok(())
}

/// Runs the protected derivation block over `plan_stdout` and returns the
/// `$GITHUB_OUTPUT` it wrote.
pub(super) fn real_action_outputs(
    run: &Run<'_>,
    label: &str,
    plan_stdout: &[u8],
) -> DynResult<String> {
    let action = fs::read_to_string(run.root.join(ACTION))?;
    let block = derivation_block(&action)?;
    let directory = run.stage.directory(&format!("action-{label}"))?;
    fs::write(directory.join("ci-plan.json"), plan_stdout)?;
    fs::write(directory.join("derive.sh"), block)?;
    let output = directory.join("github-output.txt");
    fs::write(&output, b"")?;
    let mut command = Command::new("bash");
    command
        .current_dir(&directory)
        .args(["--noprofile", "--norc", "-euo", "pipefail", "derive.sh"])
        .env("PATH", run.stage.search_path(Tools::Real))
        .env("GITHUB_OUTPUT", &output);
    let captured = run_bounded(&mut command, b"")?;
    if captured.code != Some(0) {
        return Err(format!(
            "{label}: protected action derivation failed: {}",
            String::from_utf8_lossy(&captured.stderr)
        )
        .into());
    }
    Ok(fs::read_to_string(output)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_ci_shadow_pinned_block_detects_a_one_byte_edit() {
        let action = format!("x\n{INDENT}{BLOCK_START}\n{INDENT}echo\n{INDENT}{BLOCK_END}\n");
        let difference = source_difference(&action).unwrap_or_default();
        assert!(difference.contains("digest"), "{difference}");
        assert!(source_difference("no block").is_some());
    }
}
