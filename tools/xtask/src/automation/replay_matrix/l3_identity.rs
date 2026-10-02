use crate::command::DynResult;
#[cfg(unix)]
pub(super) fn hostname() -> DynResult<String> {
    let mut bytes = [0_u8; 512];
    // SAFETY: bytes is writable for the exact bounded length supplied to libc.
    if unsafe { libc::gethostname(bytes.as_mut_ptr().cast(), bytes.len()) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let end = bytes
        .iter()
        .position(|byte| *byte == 0)
        .ok_or("unterminated host name")?;
    Ok(std::str::from_utf8(&bytes[..end])?.to_owned())
}
#[cfg(not(unix))]
pub(super) fn hostname() -> DynResult<String> {
    Ok(std::env::var("COMPUTERNAME")?)
}
