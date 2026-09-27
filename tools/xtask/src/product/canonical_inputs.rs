//! `product canonical-inputs <workspace> <host> <runtime> <output>`: the
//! port of the path-canonicalization snippet in
//! `scripts/ci-compose-product-input.sh`. It resolves the host and runtime
//! producer inputs (which must be directories) and the product output inside
//! the workspace, rejects an escape, an output equal to the workspace or
//! overlapping a producer input, and prints the three resolved paths.
//!
//! A `SystemExit` message prints as-is; an uncaught exception prints its
//! traceback's last line. Both exit with status 1.

use super::posix_path::realpath;
use super::pure_path::PurePath;
use crate::repository::check_report::CheckReport;

pub(super) fn run(args: &[String]) -> CheckReport {
    match canonicalize(args) {
        Ok(paths) => CheckReport::success(paths.iter().map(|path| format!("{path}\n")).collect()),
        Err(message) => CheckReport::failure(String::new(), format!("{message}\n")),
    }
}

/// `sys.argv[index]` for the snippet, whose `argv[0]` is `-`.
fn argument(args: &[String], index: usize) -> Result<&str, String> {
    args.get(index - 1)
        .map(String::as_str)
        .ok_or_else(|| "IndexError: list index out of range".to_owned())
}

/// `left == right or left in right.parents or right in left.parents` for
/// absolute resolved paths.
fn overlaps(left: &str, right: &str) -> bool {
    let (left, right) = (PurePath::new(left), PurePath::new(right));
    left.relative_to(&right).is_ok() || right.relative_to(&left).is_ok()
}

fn is_dir(path: &str) -> bool {
    std::path::Path::new(path).is_dir()
}

/// `resolve_in_workspace`.
fn resolve_in_workspace(workspace: &str, raw: &str, require_dir: bool) -> Result<String, String> {
    let parsed = PurePath::new(raw).display();
    let candidate = if parsed.starts_with('/') {
        parsed
    } else {
        PurePath::new(&format!("{workspace}/{parsed}")).display()
    };
    let candidate = PurePath::new(&realpath(&candidate, false)?).display();
    if PurePath::new(&candidate)
        .relative_to(&PurePath::new(workspace))
        .is_err()
    {
        return Err(format!(
            "CI artifact path escapes GITHUB_WORKSPACE: {raw} -> {candidate}"
        ));
    }
    if require_dir && !is_dir(&candidate) {
        return Err(format!("CI producer input is not a directory: {candidate}"));
    }
    Ok(candidate)
}

fn canonicalize(args: &[String]) -> Result<[String; 3], String> {
    let workspace_arg = PurePath::new(argument(args, 1)?).display();
    let workspace = PurePath::new(&realpath(&workspace_arg, true)?).display();
    let host = resolve_in_workspace(&workspace, argument(args, 2)?, true)?;
    let runtime = resolve_in_workspace(&workspace, argument(args, 3)?, true)?;
    let output = resolve_in_workspace(&workspace, argument(args, 4)?, false)?;
    if output == workspace {
        return Err(format!(
            "product output cannot be GITHUB_WORKSPACE: {output}"
        ));
    }
    for (label, producer) in [("host", &host), ("runtime", &runtime)] {
        if overlaps(&output, producer) {
            return Err(format!(
                "product output overlaps {label} producer input: {output} and {producer}"
            ));
        }
    }
    Ok([host, runtime, output])
}

#[cfg(test)]
mod tests {
    use super::overlaps;

    #[test]
    fn migration_product_overlap_is_ancestry_not_prefix() {
        assert!(overlaps("/w/out", "/w/out"));
        assert!(overlaps("/w/out", "/w/out/host"));
        assert!(overlaps("/w/out/host", "/w/out"));
        assert!(!overlaps("/w/out", "/w/output"));
    }
}
