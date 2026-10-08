//! `product canonical-inputs <workspace> <host> <runtime> <output>`: the
//! port of the path-canonicalization snippet in
//! `scripts/ci-compose-product-input.sh`. It resolves the host and runtime
//! producer inputs (which must be directories) and the product output inside
//! the workspace, rejects an escape, an output equal to the workspace or
//! overlapping a producer input, and prints the three resolved paths.
//!
use super::posix_path::{normpath, realpath};
use crate::repository::check_report::CheckReport;
use std::path::{Path, PathBuf};

pub(super) fn run(args: &[String]) -> CheckReport {
    match canonicalize(args) {
        Ok(paths) => CheckReport::success(paths.iter().map(|path| format!("{path}\n")).collect()),
        Err(message) => CheckReport::failure(String::new(), format!("{message}\n")),
    }
}

/// One-based input position; dispatch has already removed the operation name.
fn argument(args: &[String], index: usize) -> Result<&str, String> {
    args.get(index - 1)
        .map(String::as_str)
        .ok_or_else(|| "IndexError: list index out of range".to_owned())
}

/// Compare native, resolved path components, rather than string prefixes.
fn overlaps(left: &Path, right: &Path) -> bool {
    left.starts_with(right) || right.starts_with(left)
}

fn resolve(path: &Path, strict: bool) -> Result<PathBuf, String> {
    realpath(&normpath(&path.to_string_lossy()), strict).map(PathBuf::from)
}

/// `resolve_in_workspace`.
fn resolve_in_workspace(workspace: &Path, raw: &str, require_dir: bool) -> Result<PathBuf, String> {
    let input = Path::new(raw);
    // A drive-relative path (C:foo) depends on per-drive process state, not
    // GITHUB_WORKSPACE. Fully rooted drive, UNC and verbatim paths are resolved
    // normally and admitted only if they remain in the workspace.
    if input
        .components()
        .any(|part| matches!(part, std::path::Component::Prefix(_)))
        && !input.is_absolute()
    {
        return Err(format!("CI artifact path is drive-relative: {raw}"));
    }
    let candidate = resolve(&workspace.join(input), false)?;
    if !candidate.starts_with(workspace) {
        return Err(format!(
            "CI artifact path escapes GITHUB_WORKSPACE: {raw} -> {}",
            candidate.display()
        ));
    }
    if require_dir && !candidate.is_dir() {
        return Err(format!(
            "CI producer input is not a directory: {}",
            candidate.display()
        ));
    }
    Ok(candidate)
}

fn canonicalize(args: &[String]) -> Result<[String; 3], String> {
    let workspace = resolve(Path::new(argument(args, 1)?), true)?;
    let host = resolve_in_workspace(&workspace, argument(args, 2)?, true)?;
    let runtime = resolve_in_workspace(&workspace, argument(args, 3)?, true)?;
    let output = resolve_in_workspace(&workspace, argument(args, 4)?, false)?;
    if output == workspace {
        return Err(format!(
            "product output cannot be GITHUB_WORKSPACE: {}",
            output.display()
        ));
    }
    for (label, producer) in [("host", &host), ("runtime", &runtime)] {
        if overlaps(&output, producer) {
            return Err(format!(
                "product output overlaps {label} producer input: {} and {}",
                output.display(),
                producer.display()
            ));
        }
    }
    Ok([host, runtime, output].map(|path| path.to_string_lossy().into_owned()))
}

#[cfg(test)]
mod tests {
    use super::{canonicalize, overlaps};
    use std::fs;
    use std::path::Path;

    fn fixture() -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("host-input")).unwrap();
        fs::create_dir(root.path().join("runtime-input")).unwrap();
        root
    }

    fn arguments(root: &Path, output: &str) -> Vec<String> {
        vec![
            root.display().to_string(),
            "host-input".into(),
            "runtime-input".into(),
            output.into(),
        ]
    }

    #[test]
    fn migration_product_overlap_is_ancestry_not_prefix() {
        assert!(overlaps(Path::new("/w/out"), Path::new("/w/out")));
        assert!(overlaps(Path::new("/w/out"), Path::new("/w/out/host")));
        assert!(overlaps(Path::new("/w/out/host"), Path::new("/w/out")));
        assert!(!overlaps(Path::new("/w/out"), Path::new("/w/output")));
    }

    #[test]
    fn native_workspace_join_resolves_producers_and_missing_output_parents() {
        let root = fixture();
        // On Windows this root is verbatim (\\?\D:\...), exactly the form
        // returned by canonicalize in the hosted product composition failure.
        let workspace = root.path().canonicalize().unwrap();
        let paths = canonicalize(&arguments(&workspace, "new/product")).unwrap();
        #[cfg(windows)]
        assert!(
            paths.iter().all(|path| !path.contains('/')),
            "verbatim paths must contain native separators, not an appended POSIX suffix"
        );
        assert_eq!(Path::new(&paths[0]), workspace.join("host-input"));
        assert_eq!(Path::new(&paths[1]), workspace.join("runtime-input"));
        assert_eq!(Path::new(&paths[2]), workspace.join("new").join("product"));
        assert!(
            !workspace.join("new").exists(),
            "resolution must not create outputs"
        );
        let dotted = canonicalize(&arguments(&workspace, "new/../product")).unwrap();
        assert_eq!(Path::new(&dotted[2]), workspace.join("product"));
    }

    #[test]
    fn native_inputs_refuse_escape_overlap_and_non_directory_without_output() {
        let root = fixture();
        for (output, message) in [
            ("../outside", "escapes GITHUB_WORKSPACE"),
            (".", "cannot be GITHUB_WORKSPACE"),
            ("host-input/new", "overlaps host"),
            ("runtime-input", "overlaps runtime"),
        ] {
            assert!(
                canonicalize(&arguments(root.path(), output))
                    .unwrap_err()
                    .contains(message)
            );
        }
        let mut args = arguments(root.path(), "host-input-sibling/new");
        assert!(canonicalize(&args).is_ok());
        fs::write(root.path().join("file"), "producer must be a directory").unwrap();
        args[1] = "file".into();
        assert!(canonicalize(&args).unwrap_err().contains("not a directory"));
        args[1] = "missing".into();
        assert!(canonicalize(&args).unwrap_err().contains("not a directory"));
    }

    #[cfg(unix)]
    #[test]
    fn native_input_and_missing_output_ancestors_cannot_escape_through_symlinks() {
        let root = fixture();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), root.path().join("escape")).unwrap();
        let mut args = arguments(root.path(), "escape/new/product");
        assert!(
            canonicalize(&args)
                .unwrap_err()
                .contains("escapes GITHUB_WORKSPACE")
        );
        args[1] = "escape".into();
        args[3] = "new/product".into();
        assert!(
            canonicalize(&args)
                .unwrap_err()
                .contains("escapes GITHUB_WORKSPACE")
        );
        std::os::unix::fs::symlink("absent", root.path().join("broken")).unwrap();
        args[1] = "host-input".into();
        args[3] = "broken/new".into();
        assert!(canonicalize(&args).is_err());
    }

    #[cfg(windows)]
    #[test]
    fn native_windows_drive_unc_and_verbatim_ancestry_preserves_boundaries() {
        for root in [
            r"D:\workspace",
            r"\\server\share\workspace",
            r"\\?\D:\workspace",
            r"\\?\UNC\server\share\workspace",
        ] {
            let root = Path::new(root);
            assert!(overlaps(root, &root.join("host-input")));
            assert!(!overlaps(root, &root.with_file_name("workspace-other")));
        }
        assert!(!overlaps(
            Path::new(r"D:\workspace"),
            Path::new(r"E:\workspace")
        ));
        let fixture = fixture();
        let mut args = arguments(fixture.path(), "new/product");
        args[1] = "D:host-input".into();
        assert!(canonicalize(&args).unwrap_err().contains("drive-relative"));
        // Ordinary drive/UNC inputs and canonical verbatim inputs must resolve
        // to one identity, rather than being mistaken for workspace relatives.
        let canonical = fixture.path().canonicalize().unwrap();
        let text = canonical.display().to_string();
        let ordinary = if let Some(unc) = text.strip_prefix(r"\\?\UNC\") {
            format!(r"\\{unc}")
        } else {
            text.strip_prefix(r"\\?\").unwrap_or(&text).to_owned()
        };
        args = arguments(Path::new(&ordinary), "new/product");
        args[1] = Path::new(&ordinary)
            .join("host-input")
            .display()
            .to_string();
        assert_eq!(
            canonicalize(&args).unwrap(),
            canonicalize(&arguments(&canonical, "new/product")).unwrap()
        );
    }
}
