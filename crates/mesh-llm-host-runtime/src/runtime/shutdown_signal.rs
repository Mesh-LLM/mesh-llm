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
//! The registration is process-lifetime, so the forwarder that owns the
//! streams must be too. The platform keeps its handler installed for the life
//! of the process and never restores the default disposition, which means a
//! signal arriving after every observer is gone is captured and delivered to
//! nobody: the process stops being interruptible instead of being terminated.
//! A runtime that is dropped while the process lives on would leave that state
//! behind, so the forwarder runs on its own detached thread rather than on the
//! runtime that installs it. Installing therefore publishes the shared delivery
//! only once the forwarder reports its streams registered, because a delivery
//! published before that advertises an observer that does not exist yet and
//! reaches the same delivered-to-nothing state.
//!
//! Call [`install_shutdown_signals`] as early as possible from a runtime
//! entrypoint. A signal delivered before the handlers exist is handled by the
//! platform default, which terminates the process without a graceful shutdown.

use std::io;
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::Duration;
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

/// Name of the process-lifetime thread that owns the signal streams.
const FORWARDER_THREAD_NAME: &str = "mesh-llm-shutdown-forwarder";
const FORWARDER_START_TIMEOUT: Duration = Duration::from_secs(1);

static DELIVERY: OnceLock<Arc<ShutdownDelivery>> = OnceLock::new();
static INSTALL: Mutex<InstallState> = Mutex::new(InstallState {
    delivery: None,
    starting: None,
});

/// Keep one channel across startup attempts so waiters also observe a
/// forwarder that finishes registration after the synchronous wait times out.
struct InstallState {
    delivery: Option<Arc<ShutdownDelivery>>,
    starting: Option<mpsc::Receiver<Result<(), String>>>,
}

/// Register the process termination-signal handlers once.
///
/// Idempotent, and safe to call from any async context in the process. An
/// incomplete installation returns an error so startup cannot advertise
/// readiness before termination signals can be observed.
pub(crate) fn install_shutdown_signals() -> io::Result<()> {
    let mut installation = INSTALL.lock().unwrap_or_else(PoisonError::into_inner);
    if DELIVERY.get().is_some() {
        return Ok(());
    }
    if forwarder_is_still_starting(&mut installation) {
        return Err(forwarder_start_timeout(FORWARDER_START_TIMEOUT));
    }
    let delivery = installation
        .delivery
        .get_or_insert_with(|| Arc::new(ShutdownDelivery::new()))
        .clone();
    let (started_tx, started_rx) = mpsc::channel();
    let forwarder = std::thread::Builder::new()
        .name(FORWARDER_THREAD_NAME.to_owned())
        .spawn(move || run_shutdown_forwarder(delivery, started_tx));
    wait_for_forwarder_start(&mut installation, forwarder, started_rx)
}

fn forwarder_start_timeout(timeout: Duration) -> io::Error {
    io::Error::new(
        io::ErrorKind::TimedOut,
        format!("termination-signal forwarder did not start within {timeout:?}"),
    )
}

/// A timed-out start can still finish later. Avoid spawning another observer
/// until it reports success or failure.
fn forwarder_is_still_starting(installation: &mut InstallState) -> bool {
    let Some(started) = installation.starting.as_ref() else {
        return false;
    };
    match started.try_recv() {
        Err(mpsc::TryRecvError::Empty) => true,
        Ok(Ok(())) => {
            installation.starting = None;
            DELIVERY.get().is_some()
        }
        Ok(Err(cause)) => {
            tracing::warn!(%cause, "the termination-signal forwarder could not start");
            installation.starting = None;
            false
        }
        Err(mpsc::TryRecvError::Disconnected) => {
            tracing::warn!("the termination-signal forwarder exited before registering");
            installation.starting = None;
            false
        }
    }
}

/// Bound the synchronous install wait without abandoning late signal delivery.
fn wait_for_forwarder_start(
    installation: &mut InstallState,
    forwarder: io::Result<std::thread::JoinHandle<()>>,
    started: mpsc::Receiver<Result<(), String>>,
) -> io::Result<()> {
    // Detached on purpose: the forwarder observes signals for the life of the
    // process, and an unjoined thread does not hold the process open.
    let _forwarder = forwarder?;
    record_forwarder_start_result(installation, started, FORWARDER_START_TIMEOUT)
}

/// Retain a timed-out attempt so a later caller does not spawn a second one.
fn record_forwarder_start_result(
    installation: &mut InstallState,
    started: mpsc::Receiver<Result<(), String>>,
    timeout: Duration,
) -> io::Result<()> {
    match started.recv_timeout(timeout) {
        Ok(result) => result.map_err(io::Error::other),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(io::Error::new(
            io::ErrorKind::BrokenPipe,
            "termination-signal forwarder exited before registering its signal streams",
        )),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            installation.starting = Some(started);
            Err(forwarder_start_timeout(timeout))
        }
    }
}

/// Wait for a termination signal, including one delivered before this call.
pub(crate) async fn wait_for_shutdown_signal() -> &'static str {
    if let Err(error) = install_shutdown_signals() {
        tracing::warn!(%error, "termination-signal installation incomplete; retaining the fallback");
    }
    match DELIVERY.get() {
        Some(delivery) => delivery.wait().await,
        None => {
            let pending = INSTALL
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .delivery
                .clone();
            match pending {
                Some(delivery) => wait_for_pending_delivery_or_fallback(&delivery).await,
                None => resolve_fallback_registration(tokio::signal::ctrl_c().await).await,
            }
        }
    }
}

/// Keep observing a late forwarder while the ctrl-c fallback is available.
async fn wait_for_pending_delivery_or_fallback(delivery: &ShutdownDelivery) -> &'static str {
    tokio::select! {
        signal = delivery.wait() => signal,
        signal = async {
            resolve_fallback_registration(tokio::signal::ctrl_c().await).await
        } => signal,
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

/// Own the termination-signal streams on a runtime that lives for the life of
/// the process.
///
/// The streams cannot move to a runtime that outlives the one that created
/// them: each stream's waker is registered against its creating runtime's
/// signal driver, so a stream carried across would never be woken again. This
/// thread therefore registers its own, and the platform's process-wide handler
/// makes that registration equivalent to the first one.
fn run_shutdown_forwarder(
    delivery: Arc<ShutdownDelivery>,
    started: mpsc::Sender<Result<(), String>>,
) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = started.send(Err(format!("no runtime to observe signals with: {error}")));
            return;
        }
    };
    runtime.block_on(async move {
        let signals = match TerminationSignals::register() {
            Ok(signals) => signals,
            Err(error) => {
                let _ = started.send(Err(format!("could not register signal streams: {error}")));
                return;
            }
        };
        // Publish only after registration. A waiter using the pending channel
        // after an install timeout still receives signals from this forwarder.
        let sender = delivery.sender.clone();
        let _ = DELIVERY.set(delivery);
        let _ = started.send(Ok(()));
        forward_shutdown_signals(signals, &sender).await;
    });
}
async fn forward_shutdown_signals(
    mut signals: TerminationSignals,
    sender: &watch::Sender<Option<&'static str>>,
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
    terminate: tokio::signal::unix::Signal,
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
            let terminate = signal(SignalKind::terminate())?;
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
            tokio::select! {
                _ = interrupt.recv() => "SIGINT",
                _ = terminate.recv() => "SIGTERM",
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

    #[test]
    fn startup_times_out_when_the_forwarder_has_not_registered() {
        let mut installation = InstallState {
            delivery: None,
            starting: None,
        };
        let (_started_tx, started_rx) = mpsc::channel();
        let error =
            record_forwarder_start_result(&mut installation, started_rx, Duration::from_millis(10))
                .expect_err("startup must not continue without registered signal streams");
        assert_eq!(error.kind(), io::ErrorKind::TimedOut);
        assert!(installation.starting.is_some(), "retain the late forwarder");
    }

    #[test]
    fn startup_reports_a_signal_registration_failure() {
        let mut installation = InstallState {
            delivery: None,
            starting: None,
        };
        let (started_tx, started_rx) = mpsc::channel();
        started_tx
            .send(Err("could not register SIGTERM".to_owned()))
            .expect("the startup receiver is open");
        let error =
            record_forwarder_start_result(&mut installation, started_rx, Duration::from_millis(10))
                .expect_err("startup must not continue after signal registration fails");
        assert!(error.to_string().contains("could not register SIGTERM"));
        assert!(installation.starting.is_none());
    }

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

    /// A forwarder that registers after the bounded install wait must still
    /// reach a waiter that already entered the fallback path.
    #[tokio::test]
    async fn a_late_forwarder_delivery_reaches_a_pending_waiter() {
        let delivery = ShutdownDelivery::new();
        let deliver = async {
            tokio::time::sleep(Duration::from_millis(50)).await;
            delivery.sender.send(Some("SIGTERM"))
        };
        let (observed, delivered) = tokio::time::timeout(Duration::from_secs(5), async {
            tokio::join!(wait_for_pending_delivery_or_fallback(&delivery), deliver)
        })
        .await
        .expect("the pending waiter must observe a late forwarder delivery");
        delivered.expect("the retained receiver keeps the delivery channel open");
        assert_eq!(observed, "SIGTERM");
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

    /// A termination signal must still reach a waiter on a different runtime
    /// after the runtime that installed the handlers has been dropped.
    ///
    /// The handler registration is a process-lifetime one, so dropping a
    /// runtime must not take the observation of later signals with it. While
    /// the forwarder was a task on the installing runtime, dropping that
    /// runtime dropped the signal receivers while the platform handler stayed
    /// installed, so a later SIGTERM was captured and delivered to nobody and
    /// the daemon kept serving until its supervisor killed it (#1812).
    #[cfg(unix)]
    #[test]
    fn a_signal_raised_after_the_installing_runtime_drops_is_still_observed() {
        let installing = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime to install the handlers from");
        installing.block_on(async {
            super::install_shutdown_signals().expect("the forwarder to register its streams");
        });
        assert!(
            DELIVERY.get().is_some(),
            "the forwarder must register before this test raises SIGTERM"
        );
        drop(installing);

        // Raised as soon as installation returns, with nothing waited on in
        // between: installation only publishes once the forwarder has registered
        // its streams, so this is the earliest a signal can be observed and it
        // covers the publish/observe window rather than starting past it.
        //
        // SAFETY: `raise` sends SIGTERM to this process only, and the
        // process-lifetime handler for it is now registered.
        unsafe { libc::raise(libc::SIGTERM) };

        let waiter = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime for the waiter");
        let observed = waiter.block_on(async {
            tokio::time::timeout(Duration::from_secs(5), super::wait_for_shutdown_signal()).await
        });
        assert_eq!(
            observed.expect(
                "a signal raised after the installing runtime dropped must not be silently dropped (#1812)",
            ),
            "SIGTERM"
        );
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
