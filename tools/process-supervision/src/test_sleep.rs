use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

/// The executable `sleep` the test process finds on its `PATH`, as an absolute
/// path. The supervisor clears a child's environment, so the shell fixtures
/// get it as their first argument (`"$1"`).
pub(crate) fn sleep_executable() -> PathBuf {
    let path = std::env::var_os("PATH").expect("the tests find sleep on PATH");
    let found = std::env::split_paths(&path)
        .map(|directory| directory.join("sleep"))
        .find(|candidate| is_executable(candidate))
        .expect("an executable sleep is on PATH");
    std::path::absolute(found).expect("sleep has an absolute path")
}

fn is_executable(path: &Path) -> bool {
    path.metadata()
        .is_ok_and(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
}

#[test]
fn only_an_executable_file_counts_as_sleep() {
    let directory = tempfile::tempdir().unwrap();
    let candidate = directory.path().join("sleep");
    std::fs::write(&candidate, "#!/bin/sh\n").unwrap();
    std::fs::set_permissions(&candidate, std::fs::Permissions::from_mode(0o600)).unwrap();
    assert!(!is_executable(&candidate));
    assert!(!is_executable(directory.path()));
    std::fs::set_permissions(&candidate, std::fs::Permissions::from_mode(0o700)).unwrap();
    assert!(is_executable(&candidate));
}
