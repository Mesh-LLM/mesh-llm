use super::{LocalAllocation, create, current_user_sid};
use std::os::windows::ffi::OsStrExt;
use std::ptr::{null_mut, slice_from_raw_parts};
use windows_sys::Win32::Security::Authorization::{
    ConvertSecurityDescriptorToStringSecurityDescriptorW, GetNamedSecurityInfoW, SDDL_REVISION_1,
    SE_FILE_OBJECT,
};
use windows_sys::Win32::Security::DACL_SECURITY_INFORMATION;

#[test]
fn migration_lifecycle_windows_directory_has_only_current_user_inheritable_dacl() {
    let parent = tempfile::tempdir().unwrap();
    let path = parent.path().join("private");
    let sid = current_user_sid().unwrap();

    create(&path).unwrap();

    let path: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let mut descriptor = null_mut();
    // SAFETY: terminated existing pathname and writable descriptor output; unused outputs are null.
    let status = unsafe {
        GetNamedSecurityInfoW(
            path.as_ptr(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            null_mut(),
            null_mut(),
            null_mut(),
            null_mut(),
            &mut descriptor,
        )
    };
    assert_eq!(status, 0);
    let descriptor = LocalAllocation(descriptor);
    let mut text = null_mut();
    let mut length = 0;
    // SAFETY: descriptor is owned and initialized; conversion supplies allocated UTF-16 and its size.
    assert_ne!(
        unsafe {
            ConvertSecurityDescriptorToStringSecurityDescriptorW(
                descriptor.0,
                SDDL_REVISION_1,
                DACL_SECURITY_INFORMATION,
                &mut text,
                &mut length,
            )
        },
        0
    );
    let _text = LocalAllocation(text.cast());
    // SAFETY: successful conversion returned length UTF-16 units including the terminator.
    let units = unsafe { &*slice_from_raw_parts(text, usize::try_from(length).unwrap()) };
    let actual = String::from_utf16(units.strip_suffix(&[0]).unwrap_or(units)).unwrap();
    assert_eq!(actual, format!("D:P(A;OICI;FA;;;{sid})"));
}
