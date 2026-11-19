use anyhow::{Context, ensure};
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const LLAMA_CLI_ENV: &str = "SKIPPY_QUANTIZE_LLAMA_CLI";
const LLAMA_CLI_CANDIDATES: &[&str] = &[
    ".deps/llama.cpp/build-cli/bin/llama-cli",
    ".deps/llama.cpp/build-cli/bin/llama",
    ".deps/llama.cpp/build-cli/bin/llama-simple",
    ".deps/llama.cpp/build/bin/llama-cli",
    ".deps/llama.cpp/build/bin/llama",
    ".deps/llama.cpp/build/bin/llama-simple",
    ".deps/llama.cpp/build/bin/Release/llama-cli",
    ".deps/llama.cpp/build/bin/Release/llama",
    ".deps/llama.cpp/build/bin/Release/llama-simple",
    "../../.deps/llama.cpp/build-cli/bin/llama-cli",
    "../../.deps/llama.cpp/build-cli/bin/llama",
    "../../.deps/llama.cpp/build-cli/bin/llama-simple",
    "../../.deps/llama.cpp/build/bin/llama-cli",
    "../../.deps/llama.cpp/build/bin/llama",
    "../../.deps/llama.cpp/build/bin/llama-simple",
    "../../.deps/llama.cpp/build/bin/Release/llama-cli",
    "../../.deps/llama.cpp/build/bin/Release/llama",
    "../../.deps/llama.cpp/build/bin/Release/llama-simple",
];

pub(crate) fn resolve_llama_cli(explicit: Option<&Path>) -> Option<PathBuf> {
    resolve_tool(explicit, LLAMA_CLI_ENV, LLAMA_CLI_CANDIDATES)
}

fn resolve_tool(
    explicit: Option<&Path>,
    env_var: &str,
    relative_candidates: &[&str],
) -> Option<PathBuf> {
    if let Some(explicit) = explicit {
        return Some(explicit.to_path_buf());
    }
    if let Some(from_env) = std::env::var_os(env_var).map(PathBuf::from)
        && from_env.is_file()
    {
        return Some(from_env);
    }
    for root in candidate_roots() {
        for relative in relative_candidates {
            let candidate = root.join(relative);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

fn candidate_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(current) = std::env::current_dir() {
        roots.push(current);
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    roots.push(manifest_dir.clone());
    if let Some(repo_root) = manifest_dir
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
    {
        roots.push(repo_root.to_path_buf());
    }
    roots.dedup();
    roots
}

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
        is_program,
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

/// A file `Command` can start: on Unix that needs an execute bit.
fn is_program(path: &Path) -> bool {
    std::fs::metadata(path).is_ok_and(|metadata| metadata.is_file() && executable(&metadata))
}

#[cfg(unix)]
fn executable(metadata: &std::fs::Metadata) -> bool {
    use std::os::unix::fs::PermissionsExt;
    metadata.permissions().mode() & 0o111 != 0
}

#[cfg(not(unix))]
fn executable(_metadata: &std::fs::Metadata) -> bool {
    true
}

/// Starts `program --help`, so a builder that cannot run fails before
/// `--replace-package` removes the existing package.
pub(crate) fn ensure_starts(program: &Path) -> anyhow::Result<()> {
    let status = Command::new(program)
        .arg("--help")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .with_context(|| format!("start {}", program.display()))?;
    ensure!(
        status.success(),
        "{} --help failed with status {status}",
        program.display()
    );
    Ok(())
}

fn resolve_sibling(
    directories: &[PathBuf],
    path: Option<&OsStr>,
    name: &str,
    suffix: &str,
    fallback: &Path,
    is_program: fn(&Path) -> bool,
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
        .find(|candidate| is_program(candidate))
        .unwrap_or_else(|| fallback.to_path_buf())
}

#[cfg(test)]
mod sibling_binary_tests {
    use super::{ensure_starts, resolve_sibling};
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
            "skippy-quantize-sibling-{name}-{}",
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
        let resolve = || {
            resolve_sibling(
                &directories,
                Some(&search),
                "tool",
                "",
                &fallback,
                Path::is_file,
            )
        };

        assert_eq!(resolve(), fallback, "nothing exists: the build path");
        let on_path = program(&root.join("path"), "tool");
        assert_eq!(resolve(), on_path);
        let built = program(&root.join("target"), "tool");
        assert_eq!(resolve(), built, "the build path comes before PATH");
        let beside = program(&root.join("bin"), "tool");
        assert_eq!(resolve(), beside);
        assert_eq!(
            resolve_sibling(&[], None, "tool", "", &fallback, Path::is_file),
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
        let resolve = || {
            resolve_sibling(
                &directories,
                Some(&search),
                "tool",
                ".exe",
                &fallback,
                Path::is_file,
            )
        };

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
            resolve_sibling(
                &[root.join("bin")],
                Some(&search),
                "tool",
                "",
                &fallback,
                Path::is_file,
            ),
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
            resolve_sibling(
                &[],
                Some(&search),
                "Cargo",
                ".toml",
                &fallback,
                Path::is_file
            ),
            fallback
        );
    }

    #[cfg(unix)]
    fn set_mode(path: &Path, mode: u32) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn a_file_without_execute_permission_is_not_a_tool() {
        let dir = scratch("not-executable");
        let root = &dir.0;
        let fallback = root.join("target/tool");
        let search = std::env::join_paths([root.join("early"), root.join("path")]).unwrap();
        let resolve = || {
            resolve_sibling(
                &[root.join("bin")],
                Some(&search),
                "tool",
                "",
                &fallback,
                super::is_program,
            )
        };

        for directory in ["bin", "target", "early"] {
            set_mode(&program(&root.join(directory), "tool"), 0o644);
        }
        let on_path = program(&root.join("path"), "tool");
        set_mode(&on_path, 0o755);
        assert_eq!(resolve(), on_path);
        let beside = root.join("bin/tool");
        set_mode(&beside, 0o755);
        assert_eq!(resolve(), beside);
    }

    #[cfg(unix)]
    #[test]
    fn a_tool_started_through_a_link_also_looks_beside_the_linked_file() {
        let dir = scratch("link");
        let root = &dir.0;
        let file = program(&root.join("real"), "self");
        std::fs::create_dir_all(root.join("links")).unwrap();
        let link = root.join("links/self");
        std::os::unix::fs::symlink(&file, &link).unwrap();

        assert_eq!(super::executable_directories(&file), [root.join("real")]);
        assert_eq!(
            super::executable_directories(&link),
            [
                root.join("links"),
                std::fs::canonicalize(root.join("real")).unwrap()
            ],
            "the link's directory first, then the file's"
        );
    }

    #[test]
    fn a_builder_that_cannot_start_fails_the_check() {
        let dir = scratch("starts");
        // The test binary itself answers `--help` and exits successfully.
        ensure_starts(&std::env::current_exe().unwrap()).unwrap();
        let empty = program(&dir.0, &format!("tool{}", std::env::consts::EXE_SUFFIX));
        assert!(ensure_starts(&empty).is_err());
        assert!(ensure_starts(&dir.0.join("missing")).is_err());
    }
}
