//! Machine-local account home lookup for authored cache paths.
#[cfg(unix)]
pub(super) fn account_home(name: Option<&str>) -> Result<std::path::PathBuf, String> {
    let name = name
        .map(std::ffi::CString::new)
        .transpose()
        .map_err(|_| "invalid account name")?;
    let mut account = std::mem::MaybeUninit::<libc::passwd>::uninit();
    let mut result = std::ptr::null_mut();
    let mut buffer = vec![0_u8; 65536];
    // SAFETY: writable account/buffer/result storage outlives this reentrant lookup.
    let status = unsafe {
        match &name {
            Some(name) => libc::getpwnam_r(
                name.as_ptr(),
                account.as_mut_ptr(),
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                &mut result,
            ),
            None => libc::getpwuid_r(
                libc::getuid(),
                account.as_mut_ptr(),
                buffer.as_mut_ptr().cast(),
                buffer.len(),
                &mut result,
            ),
        }
    };
    if status != 0 || result.is_null() {
        return Err("account home is unavailable for cache path".into());
    }
    // SAFETY: successful lookup initialized the account and its buffer-backed directory.
    let directory = unsafe { (*result).pw_dir };
    if directory.is_null() {
        return Err("account home is unavailable for cache path".into());
    }
    // SAFETY: getpw*_r returned a terminated directory within the live buffer.
    let directory = unsafe { std::ffi::CStr::from_ptr(directory) }
        .to_str()
        .map_err(|_| "account home must be UTF-8")?;
    Ok(directory.into())
}

#[cfg(not(unix))]
pub(super) fn account_home(name: Option<&str>) -> Result<std::path::PathBuf, String> {
    if name.is_some() {
        return Err("named-account cache paths require a Unix account database".into());
    }
    std::env::var_os("USERPROFILE")
        .map(Into::into)
        .ok_or_else(|| "account home is unavailable for cache path".into())
}
