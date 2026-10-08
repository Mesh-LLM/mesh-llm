use std::io;
use std::sync::atomic::{AtomicBool, Ordering};

static STOP: AtomicBool = AtomicBool::new(false);

#[cfg(unix)]
extern "C" fn stop(_: libc::c_int) {
    STOP.store(true, Ordering::SeqCst);
}

#[cfg(windows)]
unsafe extern "system" fn stop(_: u32) -> i32 {
    STOP.store(true, Ordering::SeqCst);
    1
}

pub fn install() -> io::Result<()> {
    #[cfg(unix)]
    {
        // SAFETY: expose the static C callback for libc's integer-encoded handler ABI;
        // the handler only stores a lock-free atomic and cannot unwind.
        let handler = stop as *const ();
        if unsafe { libc::signal(libc::SIGTERM, handler.expose_provenance()) } == libc::SIG_ERR {
            return Err(io::Error::last_os_error());
        }
    }
    #[cfg(windows)]
    {
        // SAFETY: static system ABI callback neither allocates nor unwinds.
        if unsafe { windows_sys::Win32::System::Console::SetConsoleCtrlHandler(Some(stop), 1) } == 0
        {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

pub fn stopped() -> bool {
    STOP.load(Ordering::SeqCst)
}

#[cfg(unix)]
pub fn close_stdout() -> io::Result<()> {
    // SAFETY: this disposable fixture closes its own stdout after completing all stdout writes.
    if unsafe { libc::close(libc::STDOUT_FILENO) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(windows)]
pub fn close_stdout() -> io::Result<()> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "EOF fixture requires Unix",
    ))
}
