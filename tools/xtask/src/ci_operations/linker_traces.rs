//! Keep linker timing evidence local to the current CI attempt.

use crate::{command::DynResult, repository::check_report::CheckReport};
use std::{
    fs, io,
    path::{Path, PathBuf},
};

pub(super) fn run(args: &[String], root: impl FnOnce() -> DynResult<PathBuf>) -> DynResult<()> {
    const NAME: &str = "ci-ops reset-linker-traces";
    if matches!(args, [help] if help == "--help" || help == "-h") {
        return CheckReport::success(format!("usage: cargo xtool {NAME}\n")).emit();
    }
    if !args.is_empty() {
        return CheckReport::usage(NAME, "no arguments are accepted").emit();
    }
    match reset(&root()?.join("target")) {
        Ok(count) => CheckReport::success(format!("cleared_linker_traces={count}\n")),
        Err(error) => CheckReport::failure(String::new(), format!("{NAME}: {error}\n")),
    }
    .emit()
}

fn reset(target: &Path) -> io::Result<usize> {
    if let Err(error) = fs::symlink_metadata(target) {
        if error.kind() == io::ErrorKind::NotFound {
            return Ok(0);
        }
        return Err(error);
    }
    let mut traces = Vec::new();
    collect(target, &mut traces)?;
    for trace in &traces {
        fs::remove_file(trace)?;
    }
    Ok(traces.len())
}

fn collect(directory: &Path, traces: &mut Vec<PathBuf>) -> io::Result<()> {
    let metadata = fs::symlink_metadata(directory)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "trace target must be an unlinked directory",
        ));
    }
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        if kind.is_symlink() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "trace target contains a symbolic link",
            ));
        }
        if kind.is_dir() {
            collect(&entry.path(), traces)?;
        } else if entry
            .file_name()
            .to_string_lossy()
            .to_ascii_lowercase()
            .ends_with(".time-trace.json")
        {
            if !kind.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "linker trace must be a regular file",
                ));
            }
            traces.push(entry.path());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_removes_only_old_traces_and_preserves_new_attempt_outputs() {
        let repo = tempfile::tempdir().unwrap();
        let target = repo.path().join("target");
        let dependency = target.join("release/build/dependency-123/out");
        std::fs::create_dir_all(&dependency).unwrap();
        let old = dependency.join("build-script.time-trace.json");
        std::fs::write(&old, "old attempt").unwrap();
        let mixed_case = dependency.join("old.TIME-TRACE.JSON");
        std::fs::write(&mixed_case, "old Windows case variant").unwrap();
        let binary = dependency.join("build-script.exe");
        std::fs::write(&binary, "compiler output").unwrap();
        let other_json = dependency.join("metadata.json");
        std::fs::write(&other_json, "metadata").unwrap();
        let outside = repo.path().join("outside.time-trace.json");
        std::fs::write(&outside, "outside target").unwrap();

        assert_eq!(reset(&target).unwrap(), 2);
        assert!(!old.exists());
        assert!(!mixed_case.exists());
        assert_eq!(std::fs::read_to_string(binary).unwrap(), "compiler output");
        assert!(other_json.exists());
        assert!(outside.exists());
        assert_eq!(reset(&target).unwrap(), 0);

        let fresh = dependency.join("new-output.time-trace.json");
        std::fs::write(&fresh, "current attempt").unwrap();
        assert_eq!(std::fs::read_to_string(fresh).unwrap(), "current attempt");
    }

    #[test]
    fn cold_target_is_a_noop_and_a_file_is_not_a_target_tree() {
        let repo = tempfile::tempdir().unwrap();
        let target = repo.path().join("target");
        assert_eq!(reset(&target).unwrap(), 0);
        std::fs::write(&target, "not a directory").unwrap();
        assert!(reset(&target).is_err());
        assert_eq!(std::fs::read_to_string(target).unwrap(), "not a directory");
    }

    #[cfg(unix)]
    #[test]
    fn linked_cache_entries_are_refused_without_following_or_deleting_them() {
        let repo = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let trace = outside.path().join("old.time-trace.json");
        std::fs::write(&trace, "outside target").unwrap();
        let target = repo.path().join("target");
        std::fs::create_dir(&target).unwrap();
        std::os::unix::fs::symlink(outside.path(), target.join("linked")).unwrap();
        assert!(reset(&target).is_err());
        assert_eq!(std::fs::read_to_string(trace).unwrap(), "outside target");
    }
}
