//! Command interruption scope, owned by `process_supervision::interrupt`. Its
//! behavior tests stay here because they drive xtask's client readiness and
//! retained sessions.
pub(crate) use process_supervision::interrupt::{Interrupt, Reason};

#[cfg(all(test, unix))]
#[path = "../tests/migration_lifecycle/interrupt_scope.rs"]
mod tests;

#[cfg(all(test, unix))]
#[path = "../tests/migration_lifecycle/blocked_scope.rs"]
mod blocked_scope;

#[cfg(all(test, unix))]
fn action(signal: libc::c_int) -> std::io::Result<libc::sigaction> {
    let mut previous = std::mem::MaybeUninit::uninit();
    // SAFETY: null new action queries only; the OS initializes the correctly sized out parameter.
    if unsafe { libc::sigaction(signal, std::ptr::null(), previous.as_mut_ptr()) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: sigaction succeeded and initialized previous.
    Ok(unsafe { previous.assume_init() })
}

#[cfg(all(test, unix))]
fn replace(signal: libc::c_int, action: &libc::sigaction) -> std::io::Result<()> {
    // SAFETY: action is initialized, callback is static and cannot unwind; no out pointer is requested.
    if unsafe { libc::sigaction(signal, action, std::ptr::null_mut()) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}
