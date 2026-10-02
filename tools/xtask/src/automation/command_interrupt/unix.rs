use super::{INTERRUPTED, RECEIVED_SIGNAL, Reason};
use std::io;
use std::sync::atomic::Ordering;

#[cfg(test)]
#[path = "../../../tests/migration_lifecycle/blocked_scope.rs"]
mod tests;

extern "C" fn interrupt(signal: libc::c_int) {
    let _ = RECEIVED_SIGNAL.compare_exchange(0, signal, Ordering::SeqCst, Ordering::SeqCst);
    INTERRUPTED.store(true, Ordering::SeqCst);
}

pub(super) struct Registration {
    previous: Vec<(libc::c_int, libc::sigaction)>,
}

impl Registration {
    pub(super) fn install() -> Result<Self, Reason> {
        require_unblocked_signals()?;
        let mut registration = Self {
            previous: Vec::new(),
        };
        for signal in [libc::SIGINT, libc::SIGTERM] {
            let previous = action(signal)?;
            if previous.sa_sigaction != libc::SIG_DFL && previous.sa_sigaction != libc::SIG_IGN {
                return Err(Reason::ExistingHandler);
            }
            // SAFETY: sigaction is C POD; the handler and mask are initialized before registration.
            let mut next: libc::sigaction = unsafe { std::mem::zeroed() };
            next.sa_sigaction = (interrupt as *const ()).expose_provenance();
            next.sa_flags = libc::SA_RESTART;
            // SAFETY: next owns a writable sigset_t; the callback only stores a static lock-free atomic.
            if unsafe { libc::sigemptyset(&mut next.sa_mask) } < 0 {
                return Err(Reason::io(
                    "initialize signal mask",
                    io::Error::last_os_error(),
                ));
            }
            replace(signal, &next)?;
            registration.previous.push((signal, previous));
        }
        Ok(registration)
    }

    pub(super) fn unregister(&mut self) -> Result<(), Reason> {
        while let Some((signal, previous)) = self.previous.last() {
            if action(*signal)?.sa_sigaction == (interrupt as *const ()).expose_provenance() {
                replace(*signal, previous)?;
            }
            self.previous.pop();
        }
        Ok(())
    }
}

fn require_unblocked_signals() -> Result<(), Reason> {
    let mut mask = std::mem::MaybeUninit::<libc::sigset_t>::zeroed();
    // SAFETY: null set queries only the calling thread; mask is a correctly sized writable output.
    let result =
        unsafe { libc::pthread_sigmask(libc::SIG_BLOCK, std::ptr::null(), mask.as_mut_ptr()) };
    if result != 0 {
        return Err(Reason::io(
            "query signal mask",
            io::Error::from_raw_os_error(result),
        ));
    }
    // SAFETY: the query succeeded; zeroed backing storage also initializes any libc-unused words.
    let mask = unsafe { mask.assume_init() };
    for (signal, reason) in [
        (libc::SIGINT, Reason::BlockedSigint),
        (libc::SIGTERM, Reason::BlockedSigterm),
    ] {
        // SAFETY: mask is initialized and both signal numbers are valid POSIX signals.
        match unsafe { libc::sigismember(&mask, signal) } {
            0 => (),
            1 => return Err(reason),
            _ => {
                return Err(Reason::io(
                    "inspect signal mask",
                    io::Error::last_os_error(),
                ));
            }
        }
    }
    Ok(())
}

impl Drop for Registration {
    fn drop(&mut self) {
        if let Err(error) = self.unregister() {
            eprintln!("client readiness {error}");
        }
    }
}

pub(super) fn action(signal: libc::c_int) -> Result<libc::sigaction, Reason> {
    let mut previous = std::mem::MaybeUninit::uninit();
    // SAFETY: null new action queries only; the OS initializes the correctly sized out parameter.
    if unsafe { libc::sigaction(signal, std::ptr::null(), previous.as_mut_ptr()) } < 0 {
        return Err(Reason::io(
            "query signal handler",
            io::Error::last_os_error(),
        ));
    }
    // SAFETY: sigaction succeeded and initialized previous.
    Ok(unsafe { previous.assume_init() })
}

pub(super) fn replace(signal: libc::c_int, action: &libc::sigaction) -> Result<(), Reason> {
    // SAFETY: action is initialized, callback is static and cannot unwind; no out pointer is requested.
    if unsafe { libc::sigaction(signal, action, std::ptr::null_mut()) } < 0 {
        return Err(Reason::io("set signal handler", io::Error::last_os_error()));
    }
    Ok(())
}
