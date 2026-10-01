use crate::process::Cancellation;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(unix)]
#[path = "unix.rs"]
mod platform;
#[cfg(windows)]
#[path = "windows.rs"]
mod platform;

static OWNED: AtomicBool = AtomicBool::new(false);
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

#[cfg(all(test, unix))]
#[path = "../../../tests/migration_lifecycle/interrupt_scope.rs"]
mod tests;

#[derive(Debug, thiserror::Error)]
pub(crate) enum Reason {
    #[error("signal scope already active")]
    ScopeBusy,
    #[error("cancelled by command interruption")]
    Interrupted,
    #[cfg(unix)]
    #[error("refuses an existing signal handler")]
    ExistingHandler,
    #[cfg(unix)]
    #[error("refuses blocked SIGINT")]
    BlockedSigint,
    #[cfg(unix)]
    #[error("refuses blocked SIGTERM")]
    BlockedSigterm,
    #[error("{operation} failed ({kind:?}, OS code {code:?})")]
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
        code: Option<i32>,
    },
}

impl Reason {
    fn io(operation: &'static str, error: io::Error) -> Self {
        Self::Io {
            operation,
            kind: error.kind(),
            code: error.raw_os_error(),
        }
    }
}

pub(crate) struct Interrupt {
    registration: Option<platform::Registration>,
}

impl Interrupt {
    pub(crate) fn install() -> Result<Self, Reason> {
        OWNED
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .map_err(|_| Reason::ScopeBusy)?;
        INTERRUPTED.store(false, Ordering::SeqCst);
        let mut scope = Self { registration: None };
        scope.registration = Some(platform::Registration::install()?);
        Ok(scope)
    }

    pub(crate) fn cancellation(&self) -> Cancellation {
        Cancellation::from_static(&INTERRUPTED)
    }

    pub(crate) fn finish(mut self) -> Result<(), Reason> {
        self.unregister()?;
        self.check()
    }

    pub(crate) fn check(&self) -> Result<(), Reason> {
        if INTERRUPTED.load(Ordering::SeqCst) {
            Err(Reason::Interrupted)
        } else {
            Ok(())
        }
    }

    fn unregister(&mut self) -> Result<(), Reason> {
        if let Some(registration) = self.registration.as_mut() {
            registration.unregister()?;
        }
        self.registration = None;
        Ok(())
    }
}

impl Drop for Interrupt {
    fn drop(&mut self) {
        if let Err(error) = self.unregister() {
            eprintln!("client readiness {error}");
        }
        OWNED.store(false, Ordering::SeqCst);
    }
}
