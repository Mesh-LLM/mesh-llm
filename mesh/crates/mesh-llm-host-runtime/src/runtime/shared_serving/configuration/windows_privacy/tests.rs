use super::*;
use std::io::{Read, Seek, Write};
use windows_sys::Win32::Security::Authorization::{
    ConvertStringSecurityDescriptorToSecurityDescriptorW, GetNamedSecurityInfoW,
    SetNamedSecurityInfoW,
};
use windows_sys::Win32::Security::{
    GetSecurityDescriptorDacl, PROTECTED_DACL_SECURITY_INFORMATION,
};

// A deliberately public parent exercises creation without trusting inherited
// ACLs. Fixtures contain no credentials, and tempdir removes them afterwards.
fn public_dacl(path: &Path, directory: bool) {
    let sddl = if directory {
        "D:P(A;OICI;FA;;;WD)"
    } else {
        "D:P(A;;FA;;;WD)"
    };
    let wide: Vec<_> = sddl.encode_utf16().chain(Some(0)).collect();
    let mut descriptor = null_mut();
    assert_ne!(
        unsafe {
            ConvertStringSecurityDescriptorToSecurityDescriptorW(
                wide.as_ptr(),
                1,
                &mut descriptor,
                null_mut(),
            )
        },
        0
    );
    let _descriptor = SecurityDescriptor(descriptor);
    let mut present = 0;
    let mut defaulted = 0;
    let mut dacl = null_mut();
    assert_ne!(
        unsafe { GetSecurityDescriptorDacl(descriptor, &mut present, &mut dacl, &mut defaulted) },
        0
    );
    set_dacl(path, dacl);
}

fn set_dacl(path: &Path, dacl: *const ACL) {
    let wide = wide_path(path).unwrap();
    assert_eq!(
        unsafe {
            SetNamedSecurityInfoW(
                wide.as_ptr(),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                null_mut(),
                null_mut(),
                dacl,
                std::ptr::null(),
            )
        },
        0
    );
}

fn dacl_bytes(path: &Path) -> Option<Vec<u8>> {
    let wide = wide_path(path).unwrap();
    let mut dacl = null_mut();
    let mut descriptor = null_mut();
    assert_eq!(
        unsafe {
            GetNamedSecurityInfoW(
                wide.as_ptr(),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION,
                null_mut(),
                null_mut(),
                &mut dacl,
                null_mut(),
                &mut descriptor,
            )
        },
        0
    );
    let _descriptor = SecurityDescriptor(descriptor);
    if dacl.is_null() {
        None
    } else {
        Some(
            unsafe { std::slice::from_raw_parts(dacl.cast::<u8>(), (*dacl).AclSize as usize) }
                .to_vec(),
        )
    }
}

#[test]
fn creation_is_private_under_a_permissive_parent() {
    let temp = tempfile::tempdir().unwrap();
    public_dacl(temp.path(), true);
    let directory = temp.path().join("state").join("nested");
    private_directory(&directory).unwrap();
    private_directory(directory.parent().unwrap()).unwrap();
    for name in ["management.token", "host.lock", "node.key"] {
        let path = directory.join(name);
        let (mut file, created) = private_file(&path).unwrap();
        assert!(created);
        file.write_all(b"fixture").unwrap();
        drop(file);
        let (mut file, created) = private_file(&path).unwrap();
        assert!(!created);
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        assert_eq!(contents, "fixture");
    }
    // Files are independently protected even when created outside a state dir.
    private_file(&temp.path().join("private-file")).unwrap();
}

#[test]
fn unsafe_existing_state_is_rejected_without_repair_or_truncation() {
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("state");
    private_directory(&directory).unwrap();
    for name in ["management.token", "host.lock", "node.key"] {
        let path = directory.join(name);
        private_file(&path)
            .unwrap()
            .0
            .write_all(b"unchanged")
            .unwrap();
        public_dacl(&path, false);
        let before = dacl_bytes(&path);
        assert!(private_file(&path).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"unchanged");
        assert_eq!(dacl_bytes(&path), before);
    }
    public_dacl(&directory, true);
    let before = dacl_bytes(&directory);
    assert!(private_directory(&directory).is_err());
    assert_eq!(dacl_bytes(&directory), before);
}

#[test]
fn null_dacl_and_wrong_object_types_are_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let file = temp.path().join("file");
    private_file(&file)
        .unwrap()
        .0
        .write_all(b"unchanged")
        .unwrap();
    set_dacl(&file, std::ptr::null());
    assert!(dacl_bytes(&file).is_none());
    assert!(private_file(&file).is_err());
    assert!(dacl_bytes(&file).is_none());
    assert_eq!(std::fs::read(&file).unwrap(), b"unchanged");
    assert!(private_directory(&file).is_err());
    let directory = temp.path().join("directory");
    private_directory(&directory).unwrap();
    assert!(private_file(&directory).is_err());
}

#[test]
fn inherited_owner_only_access_still_requires_explicit_protection() {
    let temp = tempfile::tempdir().unwrap();
    let directory = temp.path().join("private");
    private_directory(&directory).unwrap();
    let path = directory.join("inherited.key");
    std::fs::write(&path, b"unchanged").unwrap();
    let before = dacl_bytes(&path);
    assert!(private_file(&path).is_err());
    assert_eq!(dacl_bytes(&path), before);
    assert_eq!(std::fs::read(&path).unwrap(), b"unchanged");
}

#[test]
fn hard_links_are_rejected_and_open_handles_prevent_replacement() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("key");
    let (mut file, _) = private_file(&path).unwrap();
    file.write_all(b"unchanged").unwrap();
    assert!(std::fs::rename(&path, temp.path().join("replaced")).is_err());
    file.rewind().unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    assert_eq!(contents, "unchanged");
    drop(file);
    std::fs::hard_link(&path, temp.path().join("link")).unwrap();
    assert!(private_file(&path).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), b"unchanged");
}

#[test]
fn directory_junctions_are_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let target = temp.path().join("target");
    private_directory(&target).unwrap();
    let junction = temp.path().join("junction");
    // Junction creation needs no Developer Mode or symbolic-link privilege.
    let status = std::process::Command::new("cmd")
        .args(["/D", "/C", "mklink", "/J"])
        .arg(&junction)
        .arg(&target)
        .output()
        .unwrap();
    assert!(
        status.status.success(),
        "junction fixture creation failed: {:?}",
        status.status
    );
    assert!(private_directory(&junction).is_err());
    assert!(private_file(&junction).is_err());
    std::fs::remove_dir(&junction).unwrap();
    assert!(target.is_dir());
}
