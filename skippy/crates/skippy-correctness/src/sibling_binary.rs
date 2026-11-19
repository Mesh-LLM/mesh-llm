//! Finds the Skippy binaries this tool starts.
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Where a tool that ships with this one is: next to this executable (an
/// installed copy, or the same Cargo profile directory), else `fallback`, its
/// path in a Cargo target directory under the working directory, when that
/// exists, else on `PATH`. With none of them it is `fallback`, so the error
/// names the path a build would produce.
pub(crate) fn sibling_binary(name: &str, fallback: &str) -> PathBuf {
    let directories = std::env::current_exe()
        .map(|executable| executable_directories(&executable))
        .unwrap_or_default();
    resolve_sibling(
        &directories,
        std::env::var_os("PATH").as_deref(),
        name,
        std::env::consts::EXE_SUFFIX,
        Path::new(fallback),
    )
}

/// The directory this executable was started from and, when it was started
/// through a link (macOS then reports the link), the directory of the file.
fn executable_directories(executable: &Path) -> Vec<PathBuf> {
    let started_from = executable.parent().map(Path::to_path_buf);
    let mut directories: Vec<PathBuf> = started_from.iter().cloned().collect();
    if let Ok(file) = std::fs::canonicalize(executable)
        && let Some(file_directory) = file.parent()
        && started_from
            .and_then(|directory| std::fs::canonicalize(directory).ok())
            .as_deref()
            != Some(file_directory)
    {
        directories.push(file_directory.to_path_buf());
    }
    directories
}

fn resolve_sibling(
    directories: &[PathBuf],
    path: Option<&OsStr>,
    name: &str,
    suffix: &str,
    fallback: &Path,
) -> PathBuf {
    let file_name = format!("{name}{suffix}");
    let beside = directories
        .iter()
        .map(|directory| directory.join(&file_name));
    // `fallback` is written without the platform's executable suffix.
    let mut built = fallback.as_os_str().to_owned();
    built.push(suffix);
    // An empty entry would give a bare file name, which `Command` looks up on
    // `PATH` again rather than in the working directory.
    let on_path = path
        .into_iter()
        .flat_map(std::env::split_paths)
        .filter(|directory| !directory.as_os_str().is_empty());
    beside
        .chain(std::iter::once(PathBuf::from(built)))
        .chain(on_path.map(|directory| directory.join(&file_name)))
        .find(|candidate| candidate.is_file())
        .unwrap_or_else(|| fallback.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::{executable_directories, resolve_sibling};
    use std::path::{Path, PathBuf};

    /// Removes its directory when dropped, also when an assertion fails.
    struct Scratch(PathBuf);

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn scratch(name: &str) -> Scratch {
        let root = std::env::temp_dir().join(format!(
            "skippy-correctness-sibling-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        Scratch(root)
    }

    fn program(directory: &Path, file_name: &str) -> PathBuf {
        std::fs::create_dir_all(directory).unwrap();
        let path = directory.join(file_name);
        std::fs::write(&path, b"").unwrap();
        path
    }

    #[test]
    fn the_tool_beside_the_executable_wins_then_the_build_path_then_path() {
        let dir = scratch("order");
        let root = &dir.0;
        let directories = [root.join("bin")];
        let fallback = root.join("target/tool");
        let search = std::env::join_paths([root.join("empty"), root.join("path")]).unwrap();
        let resolve = || resolve_sibling(&directories, Some(&search), "tool", "", &fallback);

        assert_eq!(resolve(), fallback, "nothing exists: the build path");
        let on_path = program(&root.join("path"), "tool");
        assert_eq!(resolve(), on_path);
        let built = program(&root.join("target"), "tool");
        assert_eq!(resolve(), built, "the build path comes before PATH");
        let beside = program(&root.join("bin"), "tool");
        assert_eq!(resolve(), beside);
        assert_eq!(
            resolve_sibling(&[], None, "tool", "", &fallback),
            built,
            "no executable path and no PATH"
        );
    }

    #[test]
    fn the_executable_suffix_applies_in_all_three_places() {
        let dir = scratch("suffix");
        let root = &dir.0;
        let directories = [root.join("bin")];
        let fallback = root.join("target/tool");
        let search = std::env::join_paths([root.join("path")]).unwrap();
        let resolve = || resolve_sibling(&directories, Some(&search), "tool", ".exe", &fallback);

        // Files without the suffix are not the tool.
        program(&root.join("bin"), "tool");
        program(&root.join("path"), "tool");
        program(&root.join("target"), "tool");
        assert_eq!(resolve(), fallback);
        let on_path = program(&root.join("path"), "tool.exe");
        assert_eq!(resolve(), on_path);
        let built = program(&root.join("target"), "tool.exe");
        assert_eq!(resolve(), built);
        let beside = program(&root.join("bin"), "tool.exe");
        assert_eq!(resolve(), beside);
    }

    #[test]
    fn a_directory_with_the_tool_name_is_not_a_tool() {
        let dir = scratch("directory");
        let root = &dir.0;
        let fallback = root.join("target/tool");
        std::fs::create_dir_all(root.join("bin/tool")).unwrap();
        std::fs::create_dir_all(root.join("target/tool")).unwrap();
        std::fs::create_dir_all(root.join("skipped/tool")).unwrap();
        let on_path = program(&root.join("path"), "tool");
        let search = std::env::join_paths([root.join("skipped"), root.join("path")]).unwrap();
        assert_eq!(
            resolve_sibling(&[root.join("bin")], Some(&search), "tool", "", &fallback),
            on_path
        );
    }

    #[test]
    fn an_empty_path_entry_is_not_the_working_directory() {
        let dir = scratch("empty-entry");
        let fallback = dir.0.join("target/tool");
        // Tests run in the package directory, which has a `Cargo.toml`.
        assert!(Path::new("Cargo.toml").is_file());
        let search = std::env::join_paths([PathBuf::new(), dir.0.join("path")]).unwrap();
        assert_eq!(
            resolve_sibling(&[], Some(&search), "Cargo", ".toml", &fallback),
            fallback
        );
    }

    #[test]
    fn a_tool_started_directly_looks_in_one_directory() {
        let dir = scratch("started-directly");
        let file = program(&dir.0.join("bin"), "self");
        assert_eq!(executable_directories(&file), [dir.0.join("bin")]);
    }
}
