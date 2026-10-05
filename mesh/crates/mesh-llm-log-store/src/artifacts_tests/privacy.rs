#[test]
fn artifact_privacy_prepares_root_tmp_request_and_files() {
    let db_root = tempfile::tempdir().expect("database root");
    let artifact_root = tempfile::tempdir().expect("artifact root");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(db_root.path(), clock.clone()).expect("open store");
    store
        .insert_summary(
            "req-privacy",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .expect("insert summary");

    let privacy = Arc::new(RecordingPrivacy::default());
    let afs = ArtifactFileStore::open_with_privacy_for_test(
        artifact_root.path().to_path_buf(),
        clock.clone(),
        store,
        privacy.clone(),
    )
    .expect("open artifact store");
    afs.write_artifact(
        "art-privacy",
        "req-privacy",
        "log",
        &clock.now(),
        b"privacy prepared content",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .expect("write artifact");

    let paths = privacy.paths();
    let canonical_root = artifact_root.path().canonicalize().expect("canonical root");
    let tmp = canonical_root.join("tmp");
    let request = canonical_root.join("req-privacy");
    let temp_file = tmp.join("art-privacy.part");
    let final_file = request.join("art-privacy");
    assert!(paths.iter().any(|(path, kind)| {
        *kind == PrivacyPathKind::Directory
            && (path == artifact_root.path() || path == &canonical_root)
    }));
    assert!(paths.contains(&(tmp, PrivacyPathKind::Directory)));
    assert!(paths.contains(&(request, PrivacyPathKind::Directory)));
    assert!(paths.contains(&(temp_file, PrivacyPathKind::File)));
    assert!(paths.contains(&(final_file, PrivacyPathKind::File)));
}

#[test]
fn artifact_privacy_failure_prevents_content_and_cleans_temp_file() {
    let db_root = tempfile::tempdir().expect("database root");
    let artifact_root = tempfile::tempdir().expect("artifact root");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(db_root.path(), clock.clone()).expect("open store");
    store
        .insert_summary(
            "req-privacy-failure",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .expect("insert summary");

    let privacy = Arc::new(RecordingPrivacy::rejecting_files());
    let afs = ArtifactFileStore::open_with_privacy_for_test(
        artifact_root.path().to_path_buf(),
        clock.clone(),
        store,
        privacy,
    )
    .expect("open artifact store");
    let result = afs.write_artifact(
        "art-privacy-failure",
        "req-privacy-failure",
        "log",
        &clock.now(),
        b"must never be written",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    );

    assert!(matches!(result, Err(LogStoreError::PrivacyNotGuaranteed)));
    assert!(
        !artifact_root
            .path()
            .join("tmp")
            .join("art-privacy-failure.part")
            .exists()
    );
    assert!(
        !artifact_root
            .path()
            .join("req-privacy-failure")
            .join("art-privacy-failure")
            .exists()
    );
}

#[cfg(windows)]
#[test]
fn windows_artifact_paths_have_current_owner_and_exact_user_only_dacl() {
    use std::ffi::c_void;
    use std::mem::align_of;
    use std::os::windows::ffi::OsStrExt;
    use std::ptr::null_mut;
    use windows_sys::Win32::Foundation::{CloseHandle, LocalFree};
    use windows_sys::Win32::Security::Authorization::{GetNamedSecurityInfoW, SE_FILE_OBJECT};
    use windows_sys::Win32::Security::{
        ACCESS_ALLOWED_ACE, DACL_SECURITY_INFORMATION, EqualSid, GetAce,
        GetSecurityDescriptorControl, GetTokenInformation, IsWellKnownSid,
        OWNER_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR, PSID, SE_DACL_PROTECTED, TOKEN_QUERY,
        TOKEN_USER, TokenUser, WinBuiltinUsersSid, WinWorldSid,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    struct Token(*mut c_void);

    impl Drop for Token {
        fn drop(&mut self) {
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }

    struct Descriptor(PSECURITY_DESCRIPTOR);

    impl Drop for Descriptor {
        fn drop(&mut self) {
            unsafe {
                let _ = LocalFree(self.0);
            }
        }
    }

    fn wide(path: &std::path::Path) -> Vec<u16> {
        path.as_os_str().encode_wide().chain(Some(0)).collect()
    }

    fn with_current_user_sid<T>(f: impl FnOnce(PSID) -> T) -> T {
        let mut token = null_mut();
        assert_ne!(
            unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) },
            0
        );
        let _token = Token(token);

        let mut bytes = 0_u32;
        unsafe {
            let _ = GetTokenInformation(token, TokenUser, null_mut(), 0, &mut bytes);
        }
        assert_ne!(bytes, 0);
        let mut buffer = vec![0_usize; (bytes as usize).div_ceil(align_of::<usize>())];
        assert_ne!(
            unsafe {
                GetTokenInformation(
                    token,
                    TokenUser,
                    buffer.as_mut_ptr().cast::<c_void>(),
                    bytes,
                    &mut bytes,
                )
            },
            0
        );
        let token_user = buffer.as_ptr().cast::<TOKEN_USER>();
        f(unsafe { (*token_user).User.Sid })
    }

    fn assert_current_owner_and_exact_dacl(path: &std::path::Path, expected_flags: u32) {
        with_current_user_sid(|current_user| {
            let path_wide = wide(path);
            let mut owner = null_mut();
            let mut dacl = null_mut();
            let mut descriptor: PSECURITY_DESCRIPTOR = null_mut();
            assert_eq!(
                unsafe {
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
                },
                0
            );
            let _descriptor = Descriptor(descriptor);
            assert!(!owner.is_null());
            assert!(!dacl.is_null());
            assert_eq!(unsafe { EqualSid(owner, current_user) }, 1);

            let mut control = 0_u16;
            let mut revision = 0_u32;
            assert_ne!(
                unsafe { GetSecurityDescriptorControl(descriptor, &mut control, &mut revision) },
                0
            );
            assert_ne!(control & SE_DACL_PROTECTED, 0);
            assert_eq!(unsafe { (*dacl).AceCount }, 1);

            let mut ace = null_mut();
            assert_ne!(unsafe { GetAce(dacl, 0, &mut ace) }, 0);
            let allowed = ace.cast::<ACCESS_ALLOWED_ACE>();
            let ace_sid = unsafe {
                (&(*allowed).SidStart as *const u32)
                    .cast_mut()
                    .cast::<c_void>()
            };
            assert_eq!(unsafe { (*allowed).Header.AceType }, 0);
            assert_eq!(unsafe { (*allowed).Header.AceFlags as u32 }, expected_flags);
            assert_ne!(unsafe { EqualSid(ace_sid, current_user) }, 0);
            assert_eq!(unsafe { IsWellKnownSid(ace_sid, WinWorldSid) }, 0);
            assert_eq!(unsafe { IsWellKnownSid(ace_sid, WinBuiltinUsersSid) }, 0);
        });
    }

    let db_root = tempfile::tempdir().expect("database root");
    let artifact_root = tempfile::tempdir().expect("artifact root");
    let clock: Arc<dyn ClockTrait> = Arc::new(TestClock::default());
    let store = LogStore::open(db_root.path(), clock.clone()).expect("open store");
    store
        .insert_summary(
            "req-windows-privacy",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .expect("insert summary");
    let afs = ArtifactFileStore::open(artifact_root.path().to_path_buf(), clock.clone(), store)
        .expect("open artifact store");
    afs.write_artifact(
        "art-windows-privacy",
        "req-windows-privacy",
        "log",
        &clock.now(),
        b"windows privacy artifact",
        None::<&str>,
        1,
        false,
        false,
        4096,
        8192,
    )
    .expect("write artifact");

    let canonical_root = artifact_root.path().canonicalize().expect("canonical root");
    assert_current_owner_and_exact_dacl(
        &canonical_root,
        windows_sys::Win32::Security::OBJECT_INHERIT_ACE
            | windows_sys::Win32::Security::CONTAINER_INHERIT_ACE,
    );
    assert_current_owner_and_exact_dacl(
        &canonical_root.join("tmp"),
        windows_sys::Win32::Security::OBJECT_INHERIT_ACE
            | windows_sys::Win32::Security::CONTAINER_INHERIT_ACE,
    );
    assert_current_owner_and_exact_dacl(
        &canonical_root.join("req-windows-privacy"),
        windows_sys::Win32::Security::OBJECT_INHERIT_ACE
            | windows_sys::Win32::Security::CONTAINER_INHERIT_ACE,
    );
    assert_current_owner_and_exact_dacl(
        &canonical_root
            .join("req-windows-privacy")
            .join("art-windows-privacy"),
        0,
    );
}
