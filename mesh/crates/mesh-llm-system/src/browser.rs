//! Opening a URL in the person's default browser.

/// Hand `url` to the default browser through the Windows shell, which reports
/// whether a handler took it: `ShellExecuteW` returns a value above 32 when it
/// did, so a missing or broken HTTP handler comes back as an error instead of a
/// silent success.
#[cfg(windows)]
pub fn open_url(url: &str) -> std::io::Result<()> {
    use windows_sys::Win32::UI::Shell::ShellExecuteW;
    use windows_sys::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;

    let wide = |text: &str| -> Vec<u16> { text.encode_utf16().chain(std::iter::once(0)).collect() };
    let verb = wide("open");
    let file = wide(url);
    // SAFETY: both strings are NUL-terminated UTF-16 that outlive the call;
    // the window, parameters and directory are optional and passed as null.
    let result = unsafe {
        ShellExecuteW(
            std::ptr::null_mut(),
            verb.as_ptr(),
            file.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            SW_SHOWNORMAL,
        )
    } as isize;
    if result > 32 {
        Ok(())
    } else {
        Err(std::io::Error::other(format!(
            "the shell did not open {url} (ShellExecuteW returned {result})"
        )))
    }
}

#[cfg(not(windows))]
pub fn open_url(_url: &str) -> std::io::Result<()> {
    Err(std::io::Error::from(std::io::ErrorKind::Unsupported))
}
