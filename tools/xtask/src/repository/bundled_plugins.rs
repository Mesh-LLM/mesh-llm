//! The default plugins a release bundles.
//!
//! `mesh-llm-plugin-manager` pins each default plugin in `DEFAULT_PLUGINS`
//! (`src/defaults.rs`): a version, the repository whose release publishes it,
//! and the SHA-256 of its archive for each target. A node installs a default
//! only from the archive its release carries in `plugins/` beside the host.
//!
//! - `repository bundled-plugins --target TRIPLE` reads those pins from the
//!   source and prints one line per default pinned for the target:
//!   `<name> <version> <sha256> <url>`, where the URL is the archive on the
//!   plugin's own GitHub release. `package-release.sh` fetches each one,
//!   checks it against the pin and places it under `plugins/`. A default with
//!   no pin for the target prints nothing: that platform bundles none.
//! - `repository bundled-plugins-check --bundle DIR` is the release check on
//!   a bundle's `plugins/`: the bundle's own `mesh-llm` installs its defaults
//!   into a scratch store, finding `plugins/` beside itself as an installed
//!   node does, and every archive there must come out installed as a default.
//!   An archive that does not hash to the host's pin fails the install; one
//!   the host pins nothing for installs nothing. Either way the check fails.

use super::{check_args::Grammar, check_report::CheckReport};
use crate::command::DynResult;
use std::path::{Path, PathBuf};

mod check;
mod pins;

pub(super) use check::bundle_check;

const LIST_GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool repository bundled-plugins --target <triple> [--defaults <path>]",
    values: &["--target", "--defaults"],
    flags: &[],
};

/// Where the default pins live, from the repository root.
const DEFAULTS_SOURCE: &str = "mesh/crates/mesh-llm-plugin-manager/src/defaults.rs";

pub(super) fn list(
    args: &[String],
    default_root: impl FnOnce() -> DynResult<PathBuf>,
) -> DynResult<()> {
    let parsed = match LIST_GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if !parsed.positionals.is_empty() {
        return LIST_GRAMMAR.error("no positional arguments").emit();
    }
    let Some(target) = parsed.last("--target") else {
        return LIST_GRAMMAR.error("--target is required").emit();
    };
    let defaults = match parsed.last("--defaults") {
        Some(path) => PathBuf::from(path),
        None => default_root()?.join(DEFAULTS_SOURCE),
    };
    let source = std::fs::read_to_string(&defaults)
        .map_err(|error| format!("read {}: {error}", defaults.display()))?;
    list_report(&source, &defaults, target).emit()
}

fn list_report(source: &str, path: &Path, target: &str) -> CheckReport {
    match pins::parse(source) {
        Ok(defaults) => CheckReport::success(
            defaults
                .iter()
                .filter_map(|default| default.bundle_line(target))
                .collect(),
        ),
        Err(error) => CheckReport::failure(
            String::new(),
            format!(
                "{}: no well-formed DEFAULT_PLUGINS: {error}\n",
                path.display()
            ),
        ),
    }
}

#[cfg(test)]
mod tests;
