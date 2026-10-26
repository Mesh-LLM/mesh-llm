//! Independent exact candidate inspection. This type cannot publish workflow receipts.
use super::input;
use super::{
    candidate_view,
    frozen_verifier::{self, FrozenVerifier},
    process, source,
};
use crate::command::DynResult;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct VerificationSource {
    pub(super) controller: FrozenVerifier,
    pub(super) root: PathBuf,
    pub(super) base: String,
    pub(super) candidate: String,
    pub(super) tree: String,
}
impl VerificationSource {
    pub(super) fn validate(&self) -> DynResult<PathBuf> {
        source::revision(&self.base)?;
        source::revision(&self.candidate)?;
        source::revision(&self.tree)?;
        if self.base == self.candidate || !self.root.is_absolute() {
            return Err("verification requires an independent direct-child candidate".into());
        }
        let controller_root = self.controller.validate()?;
        let root = self.root.canonicalize()?;
        if root == controller_root {
            return Err(
                "verification candidate must be separate from frozen controller checkout".into(),
            );
        }
        frozen_verifier::source_root(&root)?;
        if process::text(&root, &["rev-parse", "HEAD"])? != self.candidate {
            return Err("verification checkout HEAD differs from admitted candidate".into());
        }
        candidate_view::policy(&root, &self.base, &self.candidate)?;
        let tree = process::text(
            &root,
            &["rev-parse", &format!("{}^{{tree}}", self.candidate)],
        )?;
        if tree != self.tree {
            return Err("verification candidate tree differs from admitted tree".into());
        }
        frozen_verifier::clean_source(&root)?;
        process::check()?;
        Ok(root)
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    authority: VerificationSource,
}
pub(super) fn execute(bytes: &[u8]) -> DynResult<Value> {
    let input: Input = serde_json::from_slice(bytes)?;
    let root = input.authority.validate()?;
    input.authority.validate()?;
    process::check()?;
    Ok(json!({"status":"verification_source_admitted", "root":root,
        "base":input.authority.base, "candidate":input.authority.candidate,
        "tree":input.authority.tree}))
}
pub(super) fn inspect(bytes: &[u8], verb: &str) -> DynResult<Value> {
    let input: Input = serde_json::from_slice(bytes)?;
    let root = input.authority.validate()?;
    let result = match verb {
        "verification-manifest-policy" => {
            super::manifest_policy::admit(&root, &input.authority.base)
        }
        "verification-parity-inventory" => {
            super::parity_inventory::admit(&root, &input.authority.candidate)
        }
        "verification-split-roster-check" => super::split_roster::admit(&root, true),
        _ => Err("unknown independent verification inspection".into()),
    };
    if input.authority.validate()? != root {
        return Err("verification root changed during inspection".into());
    }
    process::check()?;
    result
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    run_for(args, "verification-source-admit")
}
pub(crate) fn run_inspection(args: &[String], verb: &str) -> DynResult<()> {
    run_for(args, verb)
}
fn run_for(args: &[String], verb: &str) -> DynResult<()> {
    use crate::repository::{check_args::Grammar, check_report::CheckReport};
    let grammar = Grammar {
        usage: match verb {
            "verification-source-admit" => {
                "cargo xtool automation canary-receipts verification-source-admit --input PATH"
            }
            "verification-manifest-policy" => {
                "cargo xtool automation canary-receipts verification-manifest-policy --input PATH"
            }
            "verification-parity-inventory" => {
                "cargo xtool automation canary-receipts verification-parity-inventory --input PATH"
            }
            "verification-split-roster-check" => {
                "cargo xtool automation canary-receipts verification-split-roster-check --input PATH"
            }
            _ => return Err("unknown independent verification verb".into()),
        },
        values: &["--input"],
        flags: &["--help"],
    };
    let parsed = match grammar.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", grammar.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return grammar.error("unexpected positional arguments").emit();
    }
    let Some(path) = parsed.last("--input") else {
        return grammar.error("missing --input").emit();
    };
    let result = process::operation(|| {
        let bytes = input::read(Path::new(path))?;
        if verb == "verification-source-admit" {
            execute(&bytes)
        } else {
            inspect(&bytes, verb)
        }
    });
    match result {
        Ok(value) => CheckReport::success(format!("{}\n", serde_json::to_string(&value)?)).emit(),
        Err(error) => CheckReport::failure(
            String::new(),
            format!("verification source rejected: {error}\n"),
        )
        .emit(),
    }
}
#[cfg(test)]
#[path = "verification_source/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "verification_source/inspection_tests.rs"]
mod inspection_tests;
