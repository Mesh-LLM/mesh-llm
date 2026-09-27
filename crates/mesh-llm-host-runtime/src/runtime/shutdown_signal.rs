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
const FORWARDER_STATUS_POLL_INTERVAL: Duration = Duration::from_millis(10);

static DELIVERY: OnceLock<Arc<ShutdownDelivery>> = OnceLock::new();
static INSTALL: Mutex<InstallState> = Mutex::new(InstallState {
    delivery: None,
    starting: None,
    terminal_failure: None,
    startup_gate_bypassed: false,
});

/// Keep one channel across startup attempts so waiters also observe a
/// forwarder that finishes registration after the synchronous wait times out.
struct InstallState {
    delivery: Option<Arc<ShutdownDelivery>>,
    starting: Option<mpsc::Receiver<Result<(), ForwarderStartFailure>>>,
    /// Registration and forwarder-channel failures are terminal for this
    /// process. Retain them so retries cannot spawn duplicate observers.
    terminal_failure: Option<StoredInstallFailure>,
    /// Failure to spawn the observer thread preserves the historical fallback
    /// behavior instead of refusing startup. Keep that decision sticky too, so
    /// readiness does not spin and later waiters do not retry in a tight loop.
    startup_gate_bypassed: bool,
}

struct ForwarderStartFailure {
    cause: String,
    observer_active: bool,
}

struct StoredInstallFailure {
    kind: io::ErrorKind,
    cause: String,
}

/// Register the process termination-signal handlers once.
///
/// Idempotent, and safe to call from any async context in the process. A
/// genuine registration failure returns an error so startup cannot advertise
/// readiness without the expected termination handlers. A slow forwarder is
/// retained and allowed to finish in the background.
pub(crate) fn install_shutdown_signals() -> io::Result<()> {
    let mut installation = INSTALL.lock().unwrap_or_else(PoisonError::into_inner);
    if DELIVERY.get().is_some() {
        installation.starting = None;
        return Ok(());
    }
    if let Some(error) = retained_terminal_failure(&installation) {
        return Err(error);
    }
    if installation.startup_gate_bypassed {
        return Ok(());
    }
    match forwarder_start_state(&mut installation)? {
        ForwarderStartState::Idle => {}
        ForwarderStartState::Starting => {
            tracing::debug!(
                "termination-signal forwarder is still starting; retaining the late observer"
            );
            return Ok(());
        }
        ForwarderStartState::Installed => return Ok(()),
    }
    let delivery = installation
        .delivery
        .get_or_insert_with(|| Arc::new(ShutdownDelivery::new()))
        .clone();
    let (started_tx, started_rx) = mpsc::channel();
    let forwarder = std::thread::Builder::new()
        .name(FORWARDER_THREAD_NAME.to_owned())
        .spawn(move || run_shutdown_forwarder(delivery, started_tx));
    let forwarder = match forwarder {
        Ok(forwarder) => forwarder,
        Err(error) => {
            record_nonfatal_spawn_failure(&mut installation, error);
            return Ok(());
        }
    };
    wait_for_forwarder_start(&mut installation, forwarder, started_rx)
}

/// Wait for a slow installation attempt to report success or failure before
/// startup publishes readiness.
///
/// The initial bounded install wait lets unrelated startup work continue. This
/// later gate preserves that behavior while ensuring a late registration
/// failure is still returned to the runtime entrypoint.
pub(crate) async fn wait_for_shutdown_signal_installation() -> io::Result<()> {
    wait_for_shutdown_signal_installation_with(
        || {
            install_shutdown_signals()?;
            let installation = INSTALL.lock().unwrap_or_else(PoisonError::into_inner);
            Ok(DELIVERY.get().is_some() || installation.startup_gate_bypassed)
        },
        FORWARDER_STATUS_POLL_INTERVAL,
    )
    .await
}

async fn wait_for_shutdown_signal_installation_with(
    mut observe: impl FnMut() -> io::Result<bool>,
    poll_interval: Duration,
) -> io::Result<()> {
    loop {
        if observe()? {
            return Ok(());
        }
        tokio::time::sleep(poll_interval).await;
    }
}

#[derive(Debug)]
enum ForwarderStartState {
    Idle,
    Starting,
    Installed,
}

/// A timed-out start can still finish later. Avoid spawning another observer
/// until it reports success or failure.
fn forwarder_start_state(installation: &mut InstallState) -> io::Result<ForwarderStartState> {
    let Some(started) = installation.starting.as_ref() else {
        return Ok(ForwarderStartState::Idle);
    };
    match started.try_recv() {
        Err(mpsc::TryRecvError::Empty) => Ok(ForwarderStartState::Starting),
        Ok(Ok(())) => {
            installation.starting = None;
            Ok(ForwarderStartState::Installed)
        }
        Ok(Err(cause)) => {
            installation.starting = None;
            Err(retain_forwarder_failure(installation, cause))
        }
        Err(mpsc::TryRecvError::Disconnected) => {
            installation.starting = None;
            Err(retain_install_failure(
                installation,
                io::ErrorKind::BrokenPipe,
                "termination-signal forwarder exited before registering its signal streams",
                false,
            ))
        }
    }
}

/// Bound the synchronous install wait without abandoning late signal delivery.
fn wait_for_forwarder_start(
    installation: &mut InstallState,
    forwarder: std::thread::JoinHandle<()>,
    started: mpsc::Receiver<Result<(), ForwarderStartFailure>>,
) -> io::Result<()> {
    // Detached on purpose: the forwarder observes signals for the life of the
    // process, and an unjoined thread does not hold the process open.
    let _forwarder = forwarder;
    record_forwarder_start_result(installation, started, FORWARDER_START_TIMEOUT)
}

/// Retain a timed-out attempt so a later caller does not spawn a second one.
fn record_forwarder_start_result(
    installation: &mut InstallState,
    started: mpsc::Receiver<Result<(), ForwarderStartFailure>>,
    timeout: Duration,
) -> io::Result<()> {
    match started.recv_timeout(timeout) {
        Ok(Ok(())) => Ok(()),
        Ok(Err(failure)) => Err(retain_forwarder_failure(installation, failure)),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(retain_install_failure(
            installation,
            io::ErrorKind::BrokenPipe,
            "termination-signal forwarder exited before registering its signal streams",
            false,
        )),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            installation.starting = Some(started);
            tracing::warn!(
                ?timeout,
                "timed out waiting for the termination-signal forwarder; retaining the late observer"
            );
            Ok(())
        }
    }
}

fn retain_forwarder_failure(
    installation: &mut InstallState,
    failure: ForwarderStartFailure,
) -> io::Error {
    retain_install_failure(
        installation,
        io::ErrorKind::Other,
        failure.cause,
        failure.observer_active,
    )
}

fn retain_install_failure(
    installation: &mut InstallState,
    kind: io::ErrorKind,
    cause: impl Into<String>,
    observer_active: bool,
) -> io::Error {
    let cause = cause.into();
    if observer_active {
        tracing::debug!(
            "retaining the partial termination-signal observer after registration failure"
        );
    }
    installation.terminal_failure = Some(StoredInstallFailure {
        kind,
        cause: cause.clone(),
    });
    io::Error::new(kind, cause)
}

fn record_nonfatal_spawn_failure(installation: &mut InstallState, error: io::Error) {
    installation.startup_gate_bypassed = true;
    tracing::warn!(
        %error,
        "could not start the termination-signal forwarder; retaining the platform fallback"
    );
}

fn retained_terminal_failure(installation: &InstallState) -> Option<io::Error> {
    installation
        .terminal_failure
        .as_ref()
        .map(|failure| io::Error::new(failure.kind, failure.cause.clone()))
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
    wait_for_pending_delivery_or(delivery, async {
        resolve_fallback_registration(tokio::signal::ctrl_c().await).await
    })
    .await
}

async fn wait_for_pending_delivery_or(
    delivery: &ShutdownDelivery,
    fallback: impl std::future::Future<Output = &'static str>,
) -> &'static str {
    tokio::select! {
        signal = delivery.wait() => signal,
        signal = fallback => signal,
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
    started: mpsc::Sender<Result<(), ForwarderStartFailure>>,
) {
    let runtime = match tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(error) => {
            let _ = started.send(Err(ForwarderStartFailure {
                cause: format!("no runtime to observe signals with: {error}"),
                observer_active: false,
            }));
            return;
        }
    };
    runtime.block_on(async move {
        let signals = match TerminationSignals::register() {
            Ok(signals) => signals,
            Err(failure) => {
                let (failure, signals) = classify_registration_failure(failure);
                let _ = started.send(Err(failure));
                let Some(signals) = signals else {
                    return;
                };
                // A successfully registered SIGINT listener must stay alive
                // even when SIGTERM registration fails. Keep the partial
                // observer on the pending delivery channel while reporting the
                // failure to the startup path so it can refuse readiness.
                let sender = delivery.sender.clone();
                forward_shutdown_signals(signals, &sender).await;
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

fn classify_registration_failure(
    failure: SignalRegistrationFailure,
) -> (ForwarderStartFailure, Option<TerminationSignals>) {
    let cause = format!("could not register signal streams: {}", failure.error);
    let observer_active = failure.signals.is_some();
    (
        ForwarderStartFailure {
            cause,
            observer_active,
        },
        failure.signals,
    )
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
    terminate: Option<tokio::signal::unix::Signal>,
    #[cfg(windows)]
    ctrl_c: tokio::signal::windows::CtrlC,
    #[cfg(windows)]
    ctrl_break: tokio::signal::windows::CtrlBreak,
}

struct SignalRegistrationFailure {
    error: io::Error,
    signals: Option<TerminationSignals>,
}

impl std::fmt::Debug for SignalRegistrationFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SignalRegistrationFailure")
            .field("error", &self.error)
            .field("has_registered_signals", &self.signals.is_some())
            .finish()
    }
}

impl From<io::Error> for SignalRegistrationFailure {
    fn from(error: io::Error) -> Self {
        Self {
            error,
            signals: None,
        }
    }
}

impl TerminationSignals {
    fn register() -> Result<Self, SignalRegistrationFailure> {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            let interrupt = signal(SignalKind::interrupt())?;
            let terminate = match signal(SignalKind::terminate()) {
                Ok(terminate) => terminate,
                Err(error) => {
                    return Err(SignalRegistrationFailure {
                        error,
                        signals: Some(Self {
                            interrupt,
                            terminate: None,
                        }),
                    });
                }
            };
            Ok(Self {
                interrupt,
                terminate: Some(terminate),
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

    #[test]
    fn a_slow_forwarder_does_not_abort_startup() {
        let mut installation = InstallState {
            delivery: None,
            starting: None,
            terminal_failure: None,
            startup_gate_bypassed: false,
        };
        let (_started_tx, started_rx) = mpsc::channel();
        record_forwarder_start_result(&mut installation, started_rx, Duration::from_millis(10))
            .expect("a slow forwarder must retain the previous warn-and-continue behavior");
        assert!(installation.starting.is_some(), "retain the late forwarder");
    }

    #[test]
    fn a_late_success_is_not_reported_as_a_timeout() {
        let mut installation = InstallState {
            delivery: None,
            starting: None,
            terminal_failure: None,
            startup_gate_bypassed: false,
        };
        let (started_tx, started_rx) = mpsc::channel();
        started_tx.send(Ok(())).expect("the receiver is open");
        installation.starting = Some(started_rx);

        assert!(matches!(
            forwarder_start_state(&mut installation).expect("late success is valid"),
            ForwarderStartState::Installed
        ));
        assert!(installation.starting.is_none());
    }

    #[test]
    fn startup_reports_a_signal_registration_failure() {
        let mut installation = InstallState {
            delivery: None,
            starting: None,
            terminal_failure: None,
            startup_gate_bypassed: false,
        };
        let (started_tx, started_rx) = mpsc::channel();
        started_tx
            .send(Err(ForwarderStartFailure {
                cause: "could not register SIGTERM".to_owned(),
                observer_active: false,
            }))
            .expect("the startup receiver is open");
        let error =
            record_forwarder_start_result(&mut installation, started_rx, Duration::from_millis(10))
                .expect_err("startup must not continue after signal registration fails");
        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert!(error.to_string().contains("could not register SIGTERM"));
        assert!(installation.starting.is_none());
        let retry = retained_terminal_failure(&installation)
            .expect("a hard registration failure must remain sticky");
        assert_eq!(retry.kind(), io::ErrorKind::Other);
    }

    #[test]
    fn a_partial_registration_failure_remains_sticky_across_retries() {
        let mut installation = InstallState {
            delivery: None,
            starting: None,
            terminal_failure: None,
            startup_gate_bypassed: false,
        };
        let (started_tx, started_rx) = mpsc::channel();
        started_tx
            .send(Err(ForwarderStartFailure {
                cause: "could not register SIGTERM".to_owned(),
                observer_active: true,
            }))
            .expect("the startup receiver is open");
        installation.starting = Some(started_rx);

        let first = forwarder_start_state(&mut installation)
            .expect_err("the partial registration failure must reach startup");
        assert_eq!(first.kind(), io::ErrorKind::Other);
        let retained = installation
            .terminal_failure
            .as_ref()
            .expect("the partial failure is retained");
        assert_eq!(retained.cause, "could not register SIGTERM");
        assert!(installation.starting.is_none());

        let retry = retained_terminal_failure(&installation)
            .expect("a retry must reuse the retained partial observer failure");
        assert_eq!(retry.kind(), io::ErrorKind::Other);
        assert!(retry.to_string().contains("could not register SIGTERM"));
    }

    #[test]
    fn a_thread_spawn_failure_keeps_startup_on_the_platform_fallback() {
        let mut installation = InstallState {
            delivery: None,
            starting: None,
            terminal_failure: None,
            startup_gate_bypassed: false,
        };
        record_nonfatal_spawn_failure(
            &mut installation,
            io::Error::from(io::ErrorKind::WouldBlock),
        );

        assert!(installation.startup_gate_bypassed);
        assert!(installation.terminal_failure.is_none());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_sigint_only_registration_failure_retains_and_forwards_the_partial_observer() {
        const CHILD_ENV: &str = "MESH_LLM_SIGINT_ONLY_REGISTRATION_CHILD";
        if std::env::var_os(CHILD_ENV).is_none() {
            let status = std::process::Command::new(
                std::env::current_exe().expect("resolve the current test executable"),
            )
            .args([
                "--exact",
                "runtime::shutdown_signal::tests::a_sigint_only_registration_failure_retains_and_forwards_the_partial_observer",
                "--nocapture",
            ])
            .env(CHILD_ENV, "1")
            .status()
            .expect("run the live SIGINT check in an isolated process");
            assert!(status.success(), "the isolated live SIGINT check failed");
            return;
        }

        let interrupt = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
            .expect("register the test SIGINT stream");
        let (failure, signals) = classify_registration_failure(SignalRegistrationFailure {
            error: io::Error::new(io::ErrorKind::PermissionDenied, "SIGTERM unavailable"),
            signals: Some(TerminationSignals {
                interrupt,
                terminate: None,
            }),
        });

        assert!(failure.observer_active);
        assert!(failure.cause.contains("SIGTERM unavailable"));
        let mut signals = signals.expect("the registered SIGINT stream remains owned");
        assert!(signals.terminate.is_none());

        // SAFETY: `raise` sends SIGINT to this process only, whose handler was
        // registered above. This exercises the SIGINT-only `recv` branch that
        // remains alive after the synthetic SIGTERM registration failure.
        unsafe { libc::raise(libc::SIGINT) };
        let observed = tokio::time::timeout(Duration::from_secs(5), signals.recv())
            .await
            .expect("the retained SIGINT-only observer must forward its signal");
        assert_eq!(observed, "SIGINT");
    }

    #[tokio::test]
    async fn readiness_wait_propagates_a_late_registration_failure() {
        let mut polls = 0;
        let error = wait_for_shutdown_signal_installation_with(
            || {
                polls += 1;
                if polls == 1 {
                    Ok(false)
                } else {
                    Err(io::Error::new(
                        io::ErrorKind::PermissionDenied,
                        "late registration failure",
                    ))
                }
            },
            Duration::ZERO,
        )
        .await
        .expect_err("readiness must remain gated by a late registration failure");
        assert_eq!(error.kind(), io::ErrorKind::PermissionDenied);
        assert_eq!(polls, 2);
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
            tokio::join!(
                wait_for_pending_delivery_or(&delivery, std::future::pending()),
                deliver
            )
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
