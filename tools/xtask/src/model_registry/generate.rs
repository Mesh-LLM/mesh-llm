//! `models generate [--registry <path>] [--check]`: the Rust owner of
//! `scripts/generate-test-model-manifests.py`. Writes, or with `--check`
//! compares, every projection of `ci/model-artifacts/registry.json`.

use super::fields::ModelError;
use super::projection::{Output, outputs};
use super::registry::validate;
use crate::ci_plan::catalog::{os_error_text, python_path_display};
use crate::ci_plan::document::Json;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use sha2::{Digest, Sha256};
use std::path::Path;

const PROGRAM: &str = "generate-test-model-manifests.py";
const GRAMMAR: Grammar = Grammar {
    usage: "generate-test-model-manifests.py [-h] [--registry REGISTRY] [--check]",
    values: &["--registry"],
    flags: &["--check"],
};
const REGISTRY: &str = "ci/model-artifacts/registry.json";

/// `root` is the checkout that owns the registry and every output path.
pub(super) fn run(root: &Path, args: &[String]) -> CheckReport {
    let parsed = match super::argv::parse(&GRAMMAR, PROGRAM, args) {
        Ok(parsed) => parsed,
        Err(report) => return report,
    };
    if !parsed.positionals.is_empty() {
        let message = format!("unrecognized arguments: {}", parsed.positionals.join(" "));
        return super::argv::usage(&GRAMMAR, PROGRAM, &message);
    }
    let registry = parsed
        .last("--registry")
        .map_or_else(|| root.join(REGISTRY), |path| Path::new(path).to_path_buf());
    match expected(&registry) {
        Ok(files) if parsed.flag("--check") => check(root, &files),
        Ok(files) => write(root, &files),
        Err(error) => CheckReport {
            stdout: String::new(),
            stderr: format!("test-model registry error: {error}\n"),
            code: 2,
        },
    }
}

fn io_error(error: &std::io::Error, path: &Path) -> ModelError {
    ModelError(os_error_text(error, &python_path_display(path)))
}

fn expected(registry: &Path) -> Result<Vec<Output>, ModelError> {
    let bytes = std::fs::read(registry).map_err(|error| io_error(&error, registry))?;
    let text = String::from_utf8(bytes).map_err(|error| ModelError(error.to_string()))?;
    let raw = Json::parse(text.as_bytes()).map_err(|error| ModelError(error.to_string()))?;
    let digest = hex::encode(Sha256::digest(text.as_bytes()));
    outputs(validate(&raw)?, &digest)
}

fn check(root: &Path, files: &[Output]) -> CheckReport {
    let stale = files
        .iter()
        .filter(|file| {
            std::fs::read(root.join(&file.path)).ok().as_deref() != Some(file.text.as_bytes())
        })
        .map(|file| format!("  {}\n", file.path))
        .collect::<String>();
    match stale.is_empty() {
        true => CheckReport::success(String::new()),
        false => CheckReport {
            stdout: String::new(),
            stderr: format!("generated test-model manifests are stale:\n{stale}"),
            code: 1,
        },
    }
}

fn write(root: &Path, files: &[Output]) -> CheckReport {
    for file in files {
        let path = root.join(&file.path);
        let written = path
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::write(&path, &file.text));
        if let Err(error) = written {
            return CheckReport {
                stdout: String::new(),
                stderr: format!("error: {}\n", io_error(&error, &path)),
                code: 1,
            };
        }
    }
    CheckReport::success(String::new())
}
