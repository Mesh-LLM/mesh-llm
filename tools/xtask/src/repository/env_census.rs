//! `repository env-mutation-census`: the Rust owner of
//! `scripts/check-env-mutation-contract.py`. Discovers every Rust process-env
//! mutation, holds it to the reviewable census in `registry`, and applies the
//! strict per-file contract to audited files.

mod file_contract;
mod registry;
mod source_scan;

use crate::command::DynResult;
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use crate::repository::python_text::splitlines;
use registry::{AUDITED_FILES, KNOWN_UNAUDITED_MUTATION_COUNTS};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool repository env-mutation-census [--root ROOT] [--file FILES]",
    values: &["--root", "--file"],
    flags: &[],
};

/// Totals for the success line.
struct Census {
    files: usize,
    sites: usize,
    audited: usize,
}

/// `default_root` supplies the checkout when `--root` is absent.
pub(crate) fn run(
    args: &[String],
    default_root: impl FnOnce() -> DynResult<PathBuf>,
) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if let Some(extra) = parsed.positionals.first() {
        return GRAMMAR
            .error(&format!("unrecognized arguments: {extra}"))
            .emit();
    }
    let root = match parsed.last("--root") {
        Some(root) => std::path::absolute(root)?,
        None => default_root()?,
    };
    let files = parsed.all("--file");
    let mut errors = Vec::new();
    let census = if files.is_empty() {
        census_repository(&root, &mut errors)?
    } else {
        census_files(&root, &files, &mut errors)?
    };
    report(&census, &errors).emit()
}

fn report(census: &Census, errors: &[String]) -> CheckReport {
    if errors.is_empty() {
        return CheckReport::success(format!(
            "environment mutation contract: discovered {} Rust files and {} mutation sites; {} contract-audited files; unresolved runtime sites remain explicit\n",
            census.files, census.sites, census.audited
        ));
    }
    let listed = errors
        .iter()
        .map(|error| format!("- {error}\n"))
        .collect::<String>();
    CheckReport::failure(
        String::new(),
        format!("environment mutation contract violations:\n{listed}"),
    )
}

fn census_files(root: &Path, files: &[&str], errors: &mut Vec<String>) -> DynResult<Census> {
    let mut sites = 0;
    for relative in files {
        let path = root.join(relative);
        if path.is_file() {
            sites += count_mutations(&path)?;
        }
        errors.extend(file_contract::check_file(root, relative)?);
    }
    Ok(Census {
        files: files.len(),
        sites,
        audited: files.len(),
    })
}

fn census_repository(root: &Path, errors: &mut Vec<String>) -> DynResult<Census> {
    let discovered = discover(root)?;
    let registered = |path: &str| {
        AUDITED_FILES.contains(&path)
            || KNOWN_UNAUDITED_MUTATION_COUNTS
                .iter()
                .any(|(file, _)| *file == path)
    };
    for (relative, count) in discovered.iter().filter(|(path, _)| !registered(path)) {
        errors.push(format!(
            "{relative}: unregistered process-environment mutation file ({count} sites)"
        ));
    }
    for (relative, expected) in KNOWN_UNAUDITED_MUTATION_COUNTS {
        if let Some(actual) = discovered
            .get(*relative)
            .filter(|actual| *actual != expected)
        {
            errors.push(format!(
                "{relative}: unaudited mutation census changed from {expected} to {actual} sites"
            ));
        }
    }
    for relative in AUDITED_FILES {
        if discovered.contains_key(*relative) || root.join(relative).is_file() {
            errors.extend(file_contract::check_file(root, relative)?);
        }
    }
    errors.extend(bootstrap_order(root)?);
    Ok(Census {
        files: discovered.len(),
        sites: discovered.values().sum(),
        audited: AUDITED_FILES
            .iter()
            .filter(|file| discovered.contains_key(**file))
            .count(),
    })
}

fn count_mutations(path: &Path) -> DynResult<usize> {
    let text = std::fs::read_to_string(path)?;
    Ok(source_scan::mutation_lines(&splitlines(&text)).len())
}

/// Every `*.rs` file with at least one mutation, keyed by `/`-separated
/// relative path; `.git` and `target` components are skipped at any depth.
fn discover(root: &Path) -> DynResult<BTreeMap<String, usize>> {
    let mut discovered = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => return Err(error.into()),
        };
        for entry in entries {
            let entry = entry?;
            let name = entry.file_name();
            if name == ".git" || name == "target" {
                continue;
            }
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") && path.is_file()
            {
                let count = count_mutations(&path)?;
                if count > 0 {
                    let relative = path.strip_prefix(root)?.components();
                    let relative = relative
                        .map(|part| part.as_os_str().to_string_lossy())
                        .collect::<Vec<_>>();
                    discovered.insert(relative.join("/"), count);
                }
            }
        }
    }
    Ok(discovered)
}

/// The Metal cache mutation must precede the application thread and the
/// Tokio runtime inside `fn main()` of the shipped binary.
fn bootstrap_order(root: &Path) -> DynResult<Option<String>> {
    const MAIN: &str = "crates/mesh-llm/src/main.rs";
    let path = root.join(MAIN);
    if !path.is_file() {
        return Ok(Some(format!("{MAIN}: bootstrap caller is missing")));
    }
    let text = std::fs::read_to_string(path)?;
    let Some(start) = text.find("fn main()") else {
        return Ok(Some(format!(
            "{MAIN}: synchronous Metal bootstrap markers are missing"
        )));
    };
    let after = |marker: &str| text[start..].find(marker).map(|offset| start + offset);
    let markers = (
        after("configure_metal_pipeline_cache();"),
        after("run_on_application_thread("),
        after("tokio::runtime::Builder::new_multi_thread()"),
    );
    let (Some(mutation), Some(thread), Some(runtime)) = markers else {
        return Ok(Some(format!(
            "{MAIN}: synchronous Metal bootstrap markers are missing"
        )));
    };
    Ok((!(start < mutation && mutation < thread && thread < runtime)).then(|| {
        format!(
            "{MAIN}: Metal cache environment mutation must run before application-thread and Tokio runtime construction"
        )
    }))
}
