use super::service_binary::service_binary_path;
use std::ffi::OsString;
use std::fs;
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::{Path, PathBuf};

fn executable(path: &Path) {
    fs::write(path, "#!/bin/sh\n").unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

/// A versioned binary and a link to it in a `bin` directory, as a package
/// manager installs it. Returns (bin dir, versioned binary).
fn linked_install(root: &Path) -> (PathBuf, PathBuf) {
    let versioned = root.join("cellar/mesh-llm-0.79.0/bin/mesh-llm");
    fs::create_dir_all(versioned.parent().unwrap()).unwrap();
    executable(&versioned);
    let bin = root.join("bin");
    fs::create_dir_all(&bin).unwrap();
    symlink(&versioned, bin.join("mesh-llm")).unwrap();
    (bin, versioned)
}

#[test]
fn a_bare_name_resolves_through_path_to_the_link_not_its_target() {
    let root = tempfile::tempdir().unwrap();
    let (bin, versioned) = linked_install(root.path());
    let path_var = std::env::join_paths([root.path().join("empty"), bin.clone()]).unwrap();

    let chosen = service_binary_path(
        Some(OsString::from("mesh-llm").as_os_str()),
        Some(&path_var),
        None,
        versioned,
    );

    assert_eq!(chosen, bin.join("mesh-llm"));
}

#[test]
fn an_absolute_invoked_path_is_kept_as_given() {
    let root = tempfile::tempdir().unwrap();
    let (bin, versioned) = linked_install(root.path());
    let invoked = bin.join("mesh-llm");

    let chosen = service_binary_path(Some(invoked.as_os_str()), None, None, versioned);

    assert_eq!(chosen, invoked);
}

#[test]
fn a_relative_invoked_path_is_joined_to_the_working_directory() {
    let root = tempfile::tempdir().unwrap();
    let (_, versioned) = linked_install(root.path());

    let chosen = service_binary_path(
        Some(OsString::from("bin/mesh-llm").as_os_str()),
        None,
        Some(root.path()),
        versioned,
    );

    assert_eq!(chosen, root.path().join("bin/mesh-llm"));
}

#[test]
fn falls_back_to_current_exe_when_the_invoked_name_is_not_found() {
    let root = tempfile::tempdir().unwrap();
    let (_, versioned) = linked_install(root.path());
    let path_var = std::env::join_paths([root.path().join("empty")]).unwrap();

    let chosen = service_binary_path(
        Some(OsString::from("mesh-llm").as_os_str()),
        Some(&path_var),
        None,
        versioned.clone(),
    );

    assert_eq!(chosen, versioned);
}

#[test]
fn skips_non_executable_files_and_relative_path_entries() {
    let root = tempfile::tempdir().unwrap();
    let (bin, versioned) = linked_install(root.path());
    let plain = root.path().join("plain");
    fs::create_dir_all(&plain).unwrap();
    fs::write(plain.join("mesh-llm"), "not a program").unwrap();
    let path_var = std::env::join_paths([PathBuf::from("bin"), plain, bin.clone()]).unwrap();

    let chosen = service_binary_path(
        Some(OsString::from("mesh-llm").as_os_str()),
        Some(&path_var),
        Some(root.path()),
        versioned,
    );

    assert_eq!(chosen, bin.join("mesh-llm"));
}

#[test]
fn falls_back_to_current_exe_without_an_invoked_name() {
    let current_exe = PathBuf::from("/opt/mesh-llm/bin/mesh-llm");

    let chosen = service_binary_path(None, None, None, current_exe.clone());

    assert_eq!(chosen, current_exe);
}
