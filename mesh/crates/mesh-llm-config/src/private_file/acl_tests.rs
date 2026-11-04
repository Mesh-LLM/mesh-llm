//! A replaced file keeps the old file's ACL: it neither loses one set on it
//! nor picks up the directory's inherited one.

use super::replace_file_keeping_access;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::Path;

fn mode(path: &Path) -> u32 {
    fs::metadata(path).unwrap().permissions().mode() & 0o777
}

/// An old file with group read and no access for others.
fn old_file(dir: &Path) -> std::path::PathBuf {
    let target = dir.join("models.json");
    fs::write(&target, b"old").unwrap();
    fs::set_permissions(&target, fs::Permissions::from_mode(0o640)).unwrap();
    target
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use std::ffi::CString;
    use std::io;
    use std::os::unix::ffi::OsStrExt;

    const ACCESS: &str = "system.posix_acl_access";
    const DEFAULT: &str = "system.posix_acl_default";
    // A named entry for the current user: a sandbox that maps only one uid
    // rejects an ACL naming any other.
    fn reader() -> u32 {
        // SAFETY: geteuid has no arguments and no preconditions.
        unsafe { libc::geteuid() }
    }

    /// A POSIX ACL in the kernel's xattr format: owner read-write, `reader()`
    /// read, nothing for the group and others.
    fn acl_with_reader() -> Vec<u8> {
        const UNDEFINED: u32 = u32::MAX;
        let entries = [
            (0x01_u16, 6_u16, UNDEFINED),
            (0x02, 4, reader()),
            (0x04, 0, UNDEFINED),
            (0x10, 4, UNDEFINED),
            (0x20, 0, UNDEFINED),
        ];
        let mut acl = 2_u32.to_le_bytes().to_vec();
        for (tag, perm, id) in entries {
            acl.extend(tag.to_le_bytes());
            acl.extend(perm.to_le_bytes());
            acl.extend(id.to_le_bytes());
        }
        acl
    }

    fn c_string(bytes: &[u8]) -> CString {
        CString::new(bytes).unwrap()
    }

    fn set_xattr(path: &Path, name: &str, value: &[u8]) {
        let (path, name) = (
            c_string(path.as_os_str().as_bytes()),
            c_string(name.as_bytes()),
        );
        // SAFETY: both strings are NUL-terminated and `value` is valid for its length.
        let result = unsafe {
            libc::setxattr(
                path.as_ptr(),
                name.as_ptr(),
                value.as_ptr().cast(),
                value.len(),
                0,
            )
        };
        assert_eq!(
            result,
            0,
            "setxattr {name:?}: {}",
            io::Error::last_os_error()
        );
    }

    fn get_xattr(path: &Path, name: &str) -> io::Result<Vec<u8>> {
        let (path, name) = (
            c_string(path.as_os_str().as_bytes()),
            c_string(name.as_bytes()),
        );
        let mut value = vec![0_u8; 256];
        // SAFETY: both strings are NUL-terminated and `value` is valid for its length.
        let size = unsafe {
            libc::getxattr(
                path.as_ptr(),
                name.as_ptr(),
                value.as_mut_ptr().cast(),
                value.len(),
            )
        };
        if size < 0 {
            return Err(io::Error::last_os_error());
        }
        value.truncate(size as usize);
        Ok(value)
    }

    #[test]
    fn a_replaced_file_does_not_take_the_directory_default_acl_on_linux() {
        let temp = tempfile::TempDir::new().unwrap();
        let target = old_file(temp.path());
        set_xattr(temp.path(), DEFAULT, &acl_with_reader());
        replace_file_keeping_access(&target, b"new").unwrap();
        let error = get_xattr(&target, ACCESS).unwrap_err();
        assert_eq!(error.raw_os_error(), Some(libc::ENODATA), "{error}");
        assert_eq!(mode(&target), 0o640);
        assert_eq!(fs::read_to_string(&target).unwrap(), "new");
    }

    #[test]
    fn a_replaced_file_keeps_its_acl_on_linux() {
        let temp = tempfile::TempDir::new().unwrap();
        let target = old_file(temp.path());
        set_xattr(&target, ACCESS, &acl_with_reader());
        let before = get_xattr(&target, ACCESS).unwrap();
        replace_file_keeping_access(&target, b"new").unwrap();
        assert_eq!(get_xattr(&target, ACCESS).unwrap(), before);
        assert_eq!(fs::read_to_string(&target).unwrap(), "new");
    }
}

#[cfg(target_os = "macos")]
mod macos {
    use super::*;
    use std::process::Command;

    fn chmod_acl(path: &Path, entry: &str) {
        let status = Command::new("/bin/chmod")
            .args(["+a", entry])
            .arg(path)
            .status()
            .unwrap();
        assert!(status.success());
    }

    /// The ACL entries `ls -le` lists under the file, one per line.
    fn acl_entries(path: &Path) -> Vec<String> {
        let output = Command::new("/bin/ls")
            .arg("-led")
            .arg(path)
            .output()
            .unwrap();
        assert!(output.status.success());
        String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .skip(1)
            .map(str::trim)
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn a_replaced_file_does_not_take_the_directory_inherited_acl_on_macos() {
        let temp = tempfile::TempDir::new().unwrap();
        let target = old_file(temp.path());
        chmod_acl(temp.path(), "user:nobody allow read,file_inherit");
        replace_file_keeping_access(&target, b"new").unwrap();
        assert_eq!(acl_entries(&target), Vec::<String>::new());
        assert_eq!(mode(&target), 0o640);
        assert_eq!(fs::read_to_string(&target).unwrap(), "new");
    }

    #[test]
    fn a_replaced_file_keeps_its_acl_on_macos() {
        let temp = tempfile::TempDir::new().unwrap();
        let target = old_file(temp.path());
        chmod_acl(&target, "user:nobody allow read");
        let before = acl_entries(&target);
        assert_eq!(before.len(), 1);
        replace_file_keeping_access(&target, b"new").unwrap();
        assert_eq!(acl_entries(&target), before);
        assert_eq!(fs::read_to_string(&target).unwrap(), "new");
    }
}
