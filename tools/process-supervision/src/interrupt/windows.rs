use super::{INTERRUPTED, OWNED, Reason};
use std::io;
use std::sync::atomic::Ordering;
use windows_sys::Win32::System::Console::{CTRL_BREAK_EVENT, CTRL_C_EVENT, SetConsoleCtrlHandler};

unsafe extern "system" fn interrupt(event: u32) -> i32 {
    match event {
        CTRL_C_EVENT | CTRL_BREAK_EVENT if OWNED.load(Ordering::SeqCst) => {
            INTERRUPTED.store(true, Ordering::SeqCst);
            1
        }
        _ => 0,
    }
}

pub(super) struct Registration(bool);

impl Registration {
    pub(super) fn install() -> Result<Self, Reason> {
        // SAFETY: the static callback only accesses static atomics, consumes C/BREAK, and cannot unwind.
        if unsafe { SetConsoleCtrlHandler(Some(interrupt), 1) } == 0 {
            return Err(Reason::io(
                "register console handler",
                io::Error::last_os_error(),
            ));
        }
        Ok(Self(true))
    }

    pub(super) fn unregister(&mut self) -> Result<(), Reason> {
        if self.0 {
            // SAFETY: removes only our exact callback; other control handlers and ignore state are untouched.
            if unsafe { SetConsoleCtrlHandler(Some(interrupt), 0) } == 0 {
                return Err(Reason::io(
                    "remove console handler",
                    io::Error::last_os_error(),
                ));
            }
            self.0 = false;
        }
        Ok(())
    }
}
