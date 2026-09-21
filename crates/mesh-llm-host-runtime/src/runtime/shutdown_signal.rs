//! Process-lifetime observation of termination signals.
//!
//! Termination signals are process-wide events, so the handlers are registered
//! once and every consumer observes the same delivery. A consumer that
//! registers its own signal stream per loop iteration can miss a signal that
//! arrives while no stream is registered, and it leaves the process unhandled
//! entirely until its first iteration — a SIGTERM delivered during startup is
//! then taken by the platform default disposition instead of shutting down
//! gracefully (#1812).
//!
//! Call [`install_shutdown_signals`] as early as possible from a runtime
//! entrypoint. A signal delivered before the handlers exist is handled by the
//! platform default, which terminates the process without a graceful shutdown.

use std::io;
use std::sync::{Mutex, OnceLock, PoisonError};
use tokio::sync::watch;

/// Signal name reported when no handler could be registered and only the
/// platform ctrl-c future is available.
#[cfg(windows)]
const FALLBACK_SIGNAL: &str = "CTRL-C";
#[cfg(not(windows))]
const FALLBACK_SIGNAL: &str = "SIGINT";

/// A delivery channel for the process termination signals, shared by every
/// waiter for the life of the process.
struct ShutdownDelivery {
    sender: watch::Sender<Option<&'static str>>,
    /// `watch::Sender::send` discards the value when the receiver count is
    /// zero, so a signal delivered before the first waiter subscribes would be
    /// lost. This receiver is retained for the process lifetime to keep the
    /// channel open, which is what makes delivery sticky.
    _retained_receiver: watch::Receiver<Option<&'static str>>,
}

impl ShutdownDelivery {
    fn new() -> Self {
        let (sender, retained_receiver) = watch::channel(None);
        Self {
            sender,
            _retained_receiver: retained_receiver,
        }
    }

    /// Wait for a termination signal, including one delivered before this call.
    async fn wait(&self) -> &'static str {
        let mut receiver = self.sender.subscribe();
        loop {
            if let Some(signal) = *receiver.borrow_and_update() {
                return signal;
            }
            if receiver.changed().await.is_err() {
                // Unreachable while the retained receiver keeps the channel
                // open. Parking is safer than spinning if that ever changes.
                std::future::pending::<()>().await;
            }
        }
    }
}

static DELIVERY: OnceLock<ShutdownDelivery> = OnceLock::new();
static INSTALL: Mutex<()> = Mutex::new(());

/// Register the process termination-signal handlers once.
///
/// Idempotent, and safe to call from any async context in the process. Failures
/// are reported rather than fatal: without a handler the platform default
/// disposition terminates the process, which is the outcome a graceful
/// shutdown reaches anyway.
pub(crate) fn install_shutdown_signals() {
    let _guard = INSTALL.lock().unwrap_or_else(PoisonError::into_inner);
    if DELIVERY.get().is_some() {
        return;
    }
    let signals = match TerminationSignals::register() {
        Ok(signals) => signals,
        Err(error) => {
            tracing::warn!(
                %error,
                "could not register termination-signal handlers; the platform default disposition applies"
            );
            return;
        }
    };
    let delivery = ShutdownDelivery::new();
    tokio::spawn(forward_shutdown_signals(signals, delivery.sender.clone()));
    let _ = DELIVERY.set(delivery);
}

/// Wait for a termination signal, including one delivered before this call.
pub(crate) async fn wait_for_shutdown_signal() -> &'static str {
    install_shutdown_signals();
    match DELIVERY.get() {
        Some(delivery) => delivery.wait().await,
        None => resolve_fallback_registration(tokio::signal::ctrl_c().await).await,
    }
}

/// Resolve the fallback wait from the result of registering the platform
/// ctrl-c handler.
///
/// A registration error is not a signal. Reporting the fallback name for one
/// would make every waiter start a shutdown that nothing requested, so the
/// error is logged and the wait stays pending: the platform default
/// disposition, which this module documents for that case, still applies.
async fn resolve_fallback_registration(result: io::Result<()>) -> &'static str {
    match result {
        Ok(()) => FALLBACK_SIGNAL,
        Err(error) => {
            tracing::warn!(
                %error,
                "fallback termination-signal handler unavailable; the platform default disposition applies"
            );
            std::future::pending::<&'static str>().await
        }
    }
}

async fn forward_shutdown_signals(
    mut signals: TerminationSignals,
    sender: watch::Sender<Option<&'static str>>,
) {
    loop {
        let signal = signals.recv().await;
        let _ = sender.send(Some(signal));
    }
}

/// Platform termination-signal streams, registered once per process.
struct TerminationSignals {
    #[cfg(unix)]
    interrupt: tokio::signal::unix::Signal,
    #[cfg(unix)]
    terminate: Option<tokio::signal::unix::Signal>,
    #[cfg(windows)]
    ctrl_c: tokio::signal::windows::CtrlC,
    #[cfg(windows)]
    ctrl_break: tokio::signal::windows::CtrlBreak,
}

impl TerminationSignals {
    fn register() -> io::Result<Self> {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            let interrupt = signal(SignalKind::interrupt())?;
            let terminate = match signal(SignalKind::terminate()) {
                Ok(terminate) => Some(terminate),
                Err(error) => {
                    tracing::warn!(%error, "SIGTERM handling unavailable; observing SIGINT only");
                    None
                }
            };
            Ok(Self {
                interrupt,
                terminate,
            })
        }
        #[cfg(windows)]
        {
            Ok(Self {
                ctrl_c: tokio::signal::windows::ctrl_c()?,
                ctrl_break: tokio::signal::windows::ctrl_break()?,
            })
        }
        #[cfg(not(any(unix, windows)))]
        {
            Ok(Self {})
        }
    }

    async fn recv(&mut self) -> &'static str {
        #[cfg(unix)]
        {
            let Self {
                interrupt,
                terminate,
            } = self;
            match terminate.as_mut() {
                Some(terminate) => tokio::select! {
                    _ = interrupt.recv() => "SIGINT",
                    _ = terminate.recv() => "SIGTERM",
                },
                None => {
                    let _ = interrupt.recv().await;
                    "SIGINT"
                }
            }
        }
        #[cfg(windows)]
        {
            let Self { ctrl_c, ctrl_break } = self;
            tokio::select! {
                _ = ctrl_c.recv() => "CTRL-C",
                _ = ctrl_break.recv() => "CTRL-BREAK",
            }
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = tokio::signal::ctrl_c().await;
            "CTRL-C"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// A signal delivered while nothing is awaiting the shutdown signal must
    /// still be observed by the next waiter.
    #[tokio::test]
    async fn a_delivery_that_precedes_the_waiter_is_still_observed() {
        let delivery = ShutdownDelivery::new();
        delivery
            .sender
            .send(Some("SIGTERM"))
            .expect("the retained receiver keeps the delivery channel open");

        let observed = tokio::time::timeout(Duration::from_secs(5), delivery.wait())
            .await
            .expect("a delivery that precedes the waiter must not be dropped");
        assert_eq!(observed, "SIGTERM");
    }

    /// A delivery made after a waiter started observing must reach it too, so
    /// the channel is not merely sticky about the past.
    #[tokio::test]
    async fn a_delivery_after_the_waiter_started_is_observed() {
        let delivery = ShutdownDelivery::new();
        let waiter = delivery.wait();
        let deliver = async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            delivery.sender.send(Some("SIGINT"))
        };
        let (observed, delivered) = tokio::join!(waiter, deliver);
        delivered.expect("the retained receiver keeps the delivery channel open");
        assert_eq!(observed, "SIGINT");
    }

    /// A fallback registration error must not be reported as a shutdown
    /// request: returning the fallback name there starts a shutdown nobody
    /// asked for, and every waiter acts on it (#1969 review).
    #[tokio::test]
    async fn a_fallback_registration_error_never_reports_a_signal() {
        let registration_error = io::Error::from(io::ErrorKind::PermissionDenied);
        let outcome = tokio::time::timeout(
            Duration::from_millis(100),
            resolve_fallback_registration(Err(registration_error)),
        )
        .await;
        assert!(
            outcome.is_err(),
            "a registration error must leave the wait pending instead of reporting a signal"
        );
    }

    /// The successful fallback still reports the platform ctrl-c signal.
    #[tokio::test]
    async fn a_registered_fallback_reports_the_platform_signal() {
        assert_eq!(resolve_fallback_registration(Ok(())).await, FALLBACK_SIGNAL);
    }

    /// The platform path must observe a raised signal that arrived before the
    /// waiter started, and must stay registered after the first delivery.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_raised_signal_is_observed_by_a_later_waiter() {
        let mut signals = TerminationSignals::register().expect("register termination signals");
        // SAFETY: `raise` only sends SIGTERM to this process, whose handler is
        // now registered.
        unsafe { libc::raise(libc::SIGTERM) };

        let observed = tokio::time::timeout(Duration::from_secs(10), signals.recv())
            .await
            .expect("a raised SIGTERM must not be dropped");
        assert_eq!(observed, "SIGTERM");
    }
}
