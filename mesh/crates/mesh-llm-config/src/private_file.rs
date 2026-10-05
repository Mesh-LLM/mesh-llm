//! Owner-only writes for the config file.
//!
//! The config can carry OTLP `Authorization` headers and plugin settings, so it
//! is written through a temporary file that only the current user can open and
//! then renamed over the target. Directories created on the way are owner-only
//! too. An existing parent directory is left as it is: a config path may point
//! anywhere, and its parent is not ours to tighten.
//!
//! The Windows half follows `mesh-llm-log-store`'s artifact privacy: a
//! protected DACL with a single full-control ACE for the current token user.
//! The temporary file is opened denying every other open, and that DACL is
//! applied and verified through the same handle, so no other principal can read
//! the config in the interval before it is restricted. Failures are reported,
//! never ignored.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub(crate) fn write_private_file(target: &Path, contents: &[u8]) -> io::Result<()> {
    reject_link_or_special(target)?;
    let parent = match target.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    create_private_dirs(parent)?;
    let (tmp, mut file) = create_private_temp(parent, target)?;
    let written = file.write_all(contents).and_then(|()| file.sync_all());
    drop(file);
    if let Err(error) = written {
        let _ = fs::remove_file(&tmp);
        return Err(error);
    }
    #[cfg(windows)]
    if target.exists()
        && let Err(error) = fs::remove_file(target)
    {
        let _ = fs::remove_file(&tmp);
        return Err(error);
    }
    if let Err(error) = fs::rename(&tmp, target) {
        let _ = fs::remove_file(&tmp);
        return Err(error);
    }
    Ok(())
}

fn reject_link_or_special(target: &Path) -> io::Result<()> {
    match fs::symlink_metadata(target) {
        Ok(metadata) if is_link_or_reparse_point(&metadata) || !metadata.is_file() => {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "config path must be a regular file, not a link or a directory",
            ))
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

fn is_link_or_reparse_point(metadata: &fs::Metadata) -> bool {
    if metadata.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        const FILE_ATTRIBUTE_REPARSE_POINT: u32 = 0x400;
        metadata.file_attributes() & FILE_ATTRIBUTE_REPARSE_POINT != 0
    }
    #[cfg(not(windows))]
    false
}

fn create_private_dirs(dir: &Path) -> io::Result<()> {
    let mut missing = Vec::new();
    let mut current = Some(dir);
    while let Some(path) = current {
        if path.as_os_str().is_empty() || path.exists() {
            break;
        }
        missing.push(path.to_path_buf());
        current = path.parent();
    }
    for path in missing.iter().rev() {
        match fs::create_dir(path) {
            Ok(()) => platform::restrict_directory(path)?,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn create_private_temp(parent: &Path, target: &Path) -> io::Result<(PathBuf, fs::File)> {
    let file_name = target
        .file_name()
        .unwrap_or(target.as_os_str())
        .to_string_lossy();
    let pid = std::process::id();
    for attempt in 0..16_u32 {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos();
        let tmp = parent.join(format!(".{file_name}.{pid}.{nanos}.{attempt}.tmp"));
        let mut options = fs::OpenOptions::new();
        // create_new never opens a file or link that already sits at `tmp`.
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            // Deny every other open (share mode 0) until `restrict_file` has
            // applied the owner-only DACL through this handle, so the file is
            // never readable while it still carries the parent's inherited ACL.
            options.share_mode(0);
        }
        match options.open(&tmp) {
            Ok(file) => {
                if let Err(error) = platform::restrict_file(&file) {
                    drop(file);
                    let _ = fs::remove_file(&tmp);
                    return Err(error);
                }
                return Ok((tmp, file));
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        "could not create a unique temporary config file",
    ))
}

#[cfg(unix)]
mod platform {
    use std::fs;
    use std::io;
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;

    pub(super) fn restrict_directory(path: &Path) -> io::Result<()> {
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
    }

    // The mode passed to open() is masked by the umask; set it explicitly too.
    pub(super) fn restrict_file(file: &fs::File) -> io::Result<()> {
        file.set_permissions(fs::Permissions::from_mode(0o600))
    }
}

#[cfg(windows)]
mod platform {
    use std::ffi::c_void;
    use std::fs;
    use std::io;
    use std::mem::{align_of, size_of};
    use std::os::windows::ffi::OsStrExt;
    use std::os::windows::io::AsRawHandle;
    use std::path::Path;
    use std::ptr::{null, null_mut};
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, LocalFree};
    use windows_sys::Win32::Security::Authorization::{
        GetNamedSecurityInfoW, GetSecurityInfo, SE_FILE_OBJECT, SetNamedSecurityInfoW,
        SetSecurityInfo,
    };
    use windows_sys::Win32::Security::{
        ACCESS_ALLOWED_ACE, ACL, ACL_REVISION, AddAccessAllowedAceEx, CONTAINER_INHERIT_ACE,
        DACL_SECURITY_INFORMATION, EqualSid, GetAce, GetLengthSid, GetSecurityDescriptorControl,
        GetTokenInformation, InitializeAcl, OBJECT_INHERIT_ACE, OWNER_SECURITY_INFORMATION,
        PROTECTED_DACL_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR, PSID, SE_DACL_PROTECTED,
        TOKEN_QUERY, TOKEN_USER, TokenUser,
    };
    use windows_sys::Win32::Storage::FileSystem::FILE_ALL_ACCESS;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    pub(super) fn restrict_directory(path: &Path) -> io::Result<()> {
        with_current_user_sid(|sid| {
            let acl = user_only_acl(sid, true)?;
            set_named(path, sid, &acl)?;
            verify_user_only_dacl(path, sid, ace_flags(true))
        })
    }

    // The file was opened with share mode 0, so no other open can reach it, and
    // the DACL is applied and verified through that same handle: there is no
    // interval in which it carries the ACL it inherited from the parent.
    pub(super) fn restrict_file(file: &fs::File) -> io::Result<()> {
        with_current_user_sid(|sid| {
            let acl = user_only_acl(sid, false)?;
            set_handle(file.as_raw_handle(), &acl)?;
            verify_handle(file.as_raw_handle(), sid, ace_flags(false))
        })
    }

    #[cfg(test)]
    pub(crate) fn verify_current_user_only(path: &Path, is_directory: bool) -> io::Result<()> {
        with_current_user_sid(|sid| verify_user_only_dacl(path, sid, ace_flags(is_directory)))
    }

    fn not_guaranteed() -> io::Error {
        io::Error::new(
            io::ErrorKind::PermissionDenied,
            "could not restrict the config path to the current user",
        )
    }

    fn ace_flags(is_directory: bool) -> u32 {
        if is_directory {
            OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE
        } else {
            0
        }
    }

    // Exactly one full-control ACE for `sid`, matching `ace_flags`.
    fn user_only_acl(sid: PSID, is_directory: bool) -> io::Result<Vec<u64>> {
        let acl_bytes = size_of::<ACL>() + size_of::<ACCESS_ALLOWED_ACE>() - size_of::<u32>()
            + unsafe { GetLengthSid(sid) as usize };
        let words = acl_bytes.div_ceil(size_of::<u64>());
        let mut storage = vec![0_u64; words];
        let acl = storage.as_mut_ptr().cast::<ACL>();
        let flags = ace_flags(is_directory);

        unsafe {
            if InitializeAcl(acl, acl_bytes as u32, ACL_REVISION) == 0
                || AddAccessAllowedAceEx(acl, ACL_REVISION, flags, FILE_ALL_ACCESS, sid) == 0
            {
                return Err(not_guaranteed());
            }
        }
        Ok(storage)
    }

    fn set_named(path: &Path, sid: PSID, acl: &[u64]) -> io::Result<()> {
        let path_wide = to_wide(path);
        let result = unsafe {
            SetNamedSecurityInfoW(
                path_wide.as_ptr(),
                SE_FILE_OBJECT,
                OWNER_SECURITY_INFORMATION
                    | DACL_SECURITY_INFORMATION
                    | PROTECTED_DACL_SECURITY_INFORMATION,
                sid,
                null_mut(),
                acl.as_ptr().cast::<ACL>(),
                null(),
            )
        };
        if result != 0 {
            return Err(not_guaranteed());
        }
        Ok(())
    }

    // The creator owns the file it just made, and an object's owner is
    // implicitly granted WRITE_DAC, so this handle can replace the DACL it
    // inherited without a second, separate writable open.
    fn set_handle(handle: HANDLE, acl: &[u64]) -> io::Result<()> {
        let result = unsafe {
            SetSecurityInfo(
                handle,
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                null_mut(),
                null_mut(),
                acl.as_ptr().cast::<ACL>(),
                null(),
            )
        };
        if result != 0 {
            return Err(not_guaranteed());
        }
        Ok(())
    }

    fn with_current_user_sid<T>(f: impl FnOnce(PSID) -> io::Result<T>) -> io::Result<T> {
        let mut token = null_mut();
        unsafe {
            if OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) == 0 {
                return Err(not_guaranteed());
            }
        }
        let _token = TokenHandle(token);

        let mut bytes = 0_u32;
        unsafe {
            let _ = GetTokenInformation(token, TokenUser, null_mut(), 0, &mut bytes);
        }
        if bytes == 0 {
            return Err(not_guaranteed());
        }

        let words = (bytes as usize).div_ceil(align_of::<usize>());
        let mut buffer = vec![0_usize; words];
        let ok = unsafe {
            GetTokenInformation(
                token,
                TokenUser,
                buffer.as_mut_ptr().cast::<c_void>(),
                bytes,
                &mut bytes,
            )
        };
        if ok == 0 {
            return Err(not_guaranteed());
        }

        let token_user = buffer.as_ptr().cast::<TOKEN_USER>();
        let sid = unsafe { (*token_user).User.Sid };
        if sid.is_null() {
            return Err(not_guaranteed());
        }
        f(sid)
    }

    fn verify_user_only_dacl(
        path: &Path,
        current_user: PSID,
        expected_ace_flags: u32,
    ) -> io::Result<()> {
        let path_wide = to_wide(path);
        let mut owner = null_mut();
        let mut dacl = null_mut();
        let mut descriptor: PSECURITY_DESCRIPTOR = null_mut();
        let result = unsafe {
            GetNamedSecurityInfoW(
                path_wide.as_ptr(),
                SE_FILE_OBJECT,
                OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
                &mut owner,
                null_mut(),
                &mut dacl,
                null_mut(),
                &mut descriptor,
            )
        };
        if result != 0 || owner.is_null() || dacl.is_null() || descriptor.is_null() {
            return Err(not_guaranteed());
        }
        let _descriptor = SecurityDescriptor(descriptor);
        check_user_only(owner, dacl, descriptor, current_user, expected_ace_flags)
    }

    fn verify_handle(
        handle: HANDLE,
        current_user: PSID,
        expected_ace_flags: u32,
    ) -> io::Result<()> {
        let mut owner = null_mut();
        let mut dacl = null_mut();
        let mut descriptor: PSECURITY_DESCRIPTOR = null_mut();
        let result = unsafe {
            GetSecurityInfo(
                handle,
                SE_FILE_OBJECT,
                OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
                &mut owner,
                null_mut(),
                &mut dacl,
                null_mut(),
                &mut descriptor,
            )
        };
        if result != 0 || owner.is_null() || dacl.is_null() || descriptor.is_null() {
            return Err(not_guaranteed());
        }
        let _descriptor = SecurityDescriptor(descriptor);
        check_user_only(owner, dacl, descriptor, current_user, expected_ace_flags)
    }

    fn check_user_only(
        owner: PSID,
        dacl: *mut ACL,
        descriptor: PSECURITY_DESCRIPTOR,
        current_user: PSID,
        expected_ace_flags: u32,
    ) -> io::Result<()> {
        if unsafe { EqualSid(owner, current_user) } == 0 {
            return Err(not_guaranteed());
        }

        let mut control = 0_u16;
        let mut revision = 0_u32;
        let control_ok =
            unsafe { GetSecurityDescriptorControl(descriptor, &mut control, &mut revision) != 0 };
        if !control_ok || control & SE_DACL_PROTECTED == 0 {
            return Err(not_guaranteed());
        }

        if unsafe { (*dacl).AceCount } != 1 {
            return Err(not_guaranteed());
        }

        let mut ace = null_mut();
        if unsafe { GetAce(dacl, 0, &mut ace) } == 0 || ace.is_null() {
            return Err(not_guaranteed());
        }
        let allowed = ace.cast::<ACCESS_ALLOWED_ACE>();
        let is_current_user = unsafe {
            (*allowed).Header.AceType == 0
                && (*allowed).Header.AceFlags as u32 == expected_ace_flags
                && EqualSid(
                    (&(*allowed).SidStart as *const u32)
                        .cast_mut()
                        .cast::<c_void>(),
                    current_user,
                ) != 0
        };

        if !is_current_user || unsafe { (*allowed).Mask } != FILE_ALL_ACCESS {
            return Err(not_guaranteed());
        }
        Ok(())
    }

    fn to_wide(path: &Path) -> Vec<u16> {
        path.as_os_str().encode_wide().chain(Some(0)).collect()
    }

    struct TokenHandle(*mut c_void);

    impl Drop for TokenHandle {
        fn drop(&mut self) {
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }

    struct SecurityDescriptor(PSECURITY_DESCRIPTOR);

    impl Drop for SecurityDescriptor {
        fn drop(&mut self) {
            unsafe {
                let _ = LocalFree(self.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn leftover_temp_files(dir: &Path) -> Vec<String> {
        fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".tmp"))
            .collect()
    }

    #[test]
    fn writes_then_replaces_the_target_without_leftovers() {
        let temp = tempfile::TempDir::new().unwrap();
        let target = temp.path().join("config.toml");
        write_private_file(&target, b"first = 1\n").unwrap();
        write_private_file(&target, b"second = 2\n").unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "second = 2\n");
        assert!(leftover_temp_files(temp.path()).is_empty());
    }

    #[test]
    fn rejects_a_directory_target() {
        let temp = tempfile::TempDir::new().unwrap();
        let target = temp.path().join("config.toml");
        fs::create_dir(&target).unwrap();
        let error = write_private_file(&target, b"x = 1\n").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert!(leftover_temp_files(temp.path()).is_empty());
    }

    #[cfg(unix)]
    fn mode(path: &Path) -> u32 {
        use std::os::unix::fs::PermissionsExt;
        fs::metadata(path).unwrap().permissions().mode() & 0o777
    }

    #[cfg(unix)]
    #[test]
    fn file_and_created_directories_are_owner_only_on_unix() {
        use std::os::unix::fs::PermissionsExt;
        let temp = tempfile::TempDir::new().unwrap();
        fs::set_permissions(temp.path(), fs::Permissions::from_mode(0o755)).unwrap();
        let target = temp.path().join("a").join("b").join("config.toml");
        write_private_file(&target, b"x = 1\n").unwrap();
        assert_eq!(mode(&target), 0o600);
        assert_eq!(mode(&temp.path().join("a")), 0o700);
        assert_eq!(mode(&temp.path().join("a").join("b")), 0o700);
        // The parent that already existed is not ours to tighten.
        assert_eq!(mode(temp.path()), 0o755);
    }

    #[cfg(unix)]
    #[test]
    fn an_existing_permissive_file_is_replaced_owner_only_on_unix() {
        use std::os::unix::fs::PermissionsExt;
        let temp = tempfile::TempDir::new().unwrap();
        let target = temp.path().join("config.toml");
        fs::write(&target, b"old = 1\n").unwrap();
        fs::set_permissions(&target, fs::Permissions::from_mode(0o644)).unwrap();
        write_private_file(&target, b"new = 1\n").unwrap();
        assert_eq!(mode(&target), 0o600);
        assert_eq!(fs::read_to_string(&target).unwrap(), "new = 1\n");
    }

    #[cfg(unix)]
    #[test]
    fn a_symlink_target_is_rejected_and_left_alone_on_unix() {
        let temp = tempfile::TempDir::new().unwrap();
        let elsewhere = temp.path().join("elsewhere.toml");
        fs::write(&elsewhere, b"untouched = 1\n").unwrap();
        let target = temp.path().join("config.toml");
        std::os::unix::fs::symlink(&elsewhere, &target).unwrap();
        let error = write_private_file(&target, b"x = 1\n").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
        assert_eq!(fs::read_to_string(&elsewhere).unwrap(), "untouched = 1\n");
        assert!(leftover_temp_files(temp.path()).is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn file_and_created_directories_are_current_user_only_on_windows() {
        let temp = tempfile::TempDir::new().unwrap();
        let target = temp.path().join("a").join("b").join("config.toml");
        write_private_file(&target, b"x = 1\n").unwrap();
        platform::verify_current_user_only(&target, false).unwrap();
        platform::verify_current_user_only(&temp.path().join("a"), true).unwrap();
        platform::verify_current_user_only(&temp.path().join("a").join("b"), true).unwrap();
        // The parent that already existed keeps its inherited ACL.
        assert!(platform::verify_current_user_only(temp.path(), true).is_err());
    }

    #[cfg(windows)]
    #[test]
    fn an_existing_file_is_replaced_current_user_only_on_windows() {
        let temp = tempfile::TempDir::new().unwrap();
        let target = temp.path().join("config.toml");
        fs::write(&target, b"old = 1\n").unwrap();
        assert!(platform::verify_current_user_only(&target, false).is_err());
        write_private_file(&target, b"new = 1\n").unwrap();
        platform::verify_current_user_only(&target, false).unwrap();
        assert_eq!(fs::read_to_string(&target).unwrap(), "new = 1\n");
    }
}
