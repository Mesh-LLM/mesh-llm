//! Finds the Skippy binaries this tool starts.
use std::ffi::OsStr;
use std::path::{Path, PathBuf};

/// Where a tool that ships with this one is: next to this executable (an
/// installed copy, or the same Cargo profile directory), else `fallback`, its
/// path in a Cargo target directory under the working directory, when that
/// exists, else on `PATH`. With none of them it is `fallback`, so the error
/// names the path a build would produce.
pub(crate) fn sibling_binary(name: &str, fallback: &str) -> PathBuf {
    resolve_sibling(
        std::env::current_exe().ok().as_deref(),
        std::env::var_os("PATH").as_deref(),
        name,
        std::env::consts::EXE_SUFFIX,
        Path::new(fallback),
    )
}

fn resolve_sibling(
    executable: Option<&Path>,
    path: Option<&OsStr>,
    name: &str,
    suffix: &str,
    fallback: &Path,
) -> PathBuf {
    let file_name = format!("{name}{suffix}");
    let beside = executable
        .and_then(Path::parent)
        .map(|directory| directory.join(&file_name));
    // `fallback` is written without the platform's executable suffix.
    let mut built = fallback.as_os_str().to_owned();
    built.push(suffix);
    let on_path = path.into_iter().flat_map(std::env::split_paths);
    beside
        .into_iter()
        .chain(std::iter::once(PathBuf::from(built)))
        .chain(on_path.map(|directory| directory.join(&file_name)))
        .find(|candidate| candidate.is_file())
        .unwrap_or_else(|| fallback.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::resolve_sibling;
    use std::path::{Path, PathBuf};

    fn scratch(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "skippy-correctness-sibling-{name}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn program(directory: &Path, file_name: &str) -> PathBuf {
        std::fs::create_dir_all(directory).unwrap();
        let path = directory.join(file_name);
        std::fs::write(&path, b"").unwrap();
        path
    }

    #[test]
    fn the_tool_beside_the_executable_wins_then_the_build_path_then_path() {
        let root = scratch("order");
        let executable = root.join("bin/self");
        let fallback = root.join("target/tool");
        let search = std::env::join_paths([root.join("empty"), root.join("path")]).unwrap();
        let resolve = || resolve_sibling(Some(&executable), Some(&search), "tool", "", &fallback);

        assert_eq!(resolve(), fallback, "nothing exists: the build path");
        let on_path = program(&root.join("path"), "tool");
        assert_eq!(resolve(), on_path);
        let built = program(&root.join("target"), "tool");
        assert_eq!(resolve(), built, "the build path comes before PATH");
        let beside = program(&root.join("bin"), "tool");
        assert_eq!(resolve(), beside);
        assert_eq!(
            resolve_sibling(None, None, "tool", "", &fallback),
            built,
            "no executable path and no PATH"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn the_executable_suffix_applies_in_all_three_places() {
        let root = scratch("suffix");
        let executable = root.join("bin/self.exe");
        let fallback = root.join("target/tool");
        let search = std::env::join_paths([root.join("path")]).unwrap();
        let resolve =
            || resolve_sibling(Some(&executable), Some(&search), "tool", ".exe", &fallback);

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
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn a_directory_with_the_tool_name_is_not_a_tool() {
        let root = scratch("directory");
        let fallback = root.join("target/tool");
        std::fs::create_dir_all(root.join("bin/tool")).unwrap();
        std::fs::create_dir_all(root.join("target/tool")).unwrap();
        std::fs::create_dir_all(root.join("skipped/tool")).unwrap();
        let on_path = program(&root.join("path"), "tool");
        let search = std::env::join_paths([root.join("skipped"), root.join("path")]).unwrap();
        assert_eq!(
            resolve_sibling(
                Some(&root.join("bin/self")),
                Some(&search),
                "tool",
                "",
                &fallback
            ),
            on_path
        );
        std::fs::remove_dir_all(root).unwrap();
    }
}
