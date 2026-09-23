#![forbid(unsafe_code)]
//! Anonymous, opt-out product analytics for mesh-llm.
//!
//! # What this is
//!
//! A narrow PostHog reporter for questions we cannot answer any other way:
//! which platforms mesh-llm runs on, which commands people actually use,
//! whether `serve` sessions survive, and how big real meshes get.
//!
//! # What it is not
//!
//! It is not the `[telemetry]` config section. That one is operator-facing
//! OTLP: the operator points it at *their own* collector and nothing leaves
//! their network. This crate is the only code in mesh-llm that reports to a
//! vendor, and it is deliberately kept in one reviewable place.
//!
//! # Design constraints
//!
//! - **Closed vocabulary.** [`Event`] and [`Properties`] admit no free-form
//!   `String`. Text only enters through [`Label::sanitize`], whose grammar
//!   rejects paths, prose, and anything over-long. A prompt cannot reach the
//!   wire through this API even by mistake.
//! - **No key, no reporting.** The project key is compiled in by the release
//!   pipeline. Source and development builds have none and are inert.
//! - **Never load-bearing.** Capture is non-blocking, delivery is best
//!   effort, and no failure here changes a command's behavior or exit code.
//! - **Anonymous by construction.** The identifier is a random UUID with no
//!   derivation from the machine, and it is not the published mesh identity.

mod client;
mod consent;
mod event;
mod install_id;
mod notice;
mod properties;

pub use consent::{
    ConfigPreference, ConsentInputs, DEFAULT_POSTHOG_HOST, Disposition, ENV_ANALYTICS,
    ENV_DO_NOT_TRACK, ENV_POSTHOG_HOST, ENV_POSTHOG_KEY, ingestion_host, project_key,
};
pub use event::{
    Event, Label, Properties, Value, bucket_count, bucket_duration_secs, bucket_gigabytes,
};
pub use install_id::{INSTALL_ID_FILE, InstallId, load, load_or_create, state_dir};
pub use notice::{NOTICE, NOTICE_MARKER_FILE};
pub use properties::{BuildChannel, LIB_NAME, base_properties};

use chrono::Utc;
use client::Envelope;
use std::path::PathBuf;
use std::sync::OnceLock;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

/// Queue depth. Capture drops events rather than applying backpressure, so
/// this only has to absorb a burst between flushes.
const QUEUE_CAPACITY: usize = 256;

/// Events per ingestion request.
const BATCH_SIZE: usize = 20;

/// How long the flusher waits for more events before sending a partial batch.
const FLUSH_INTERVAL: std::time::Duration = std::time::Duration::from_secs(10);

/// Budget for delivering queued events at process exit.
///
/// A one-shot command exits in well under a second, so without a drain at
/// exit almost nothing would ever be delivered. The budget is short on
/// purpose: a slow or captive network must not hold the command open, and
/// losing the event is the correct trade.
const SHUTDOWN_BUDGET: std::time::Duration = std::time::Duration::from_millis(1_500);

/// The process-global reporter. `None` once consent resolves to disabled.
static REPORTER: OnceLock<Option<Reporter>> = OnceLock::new();

struct Reporter {
    /// Held in an `Option` so [`shutdown`] can drop it. Dropping the last
    /// sender is what tells the flush loop the process is going away.
    sender: std::sync::Mutex<Option<mpsc::Sender<Envelope>>>,
    worker: std::sync::Mutex<Option<JoinHandle<()>>>,
}

impl Reporter {
    fn sender(&self) -> Option<mpsc::Sender<Envelope>> {
        self.sender.lock().ok().and_then(|slot| slot.clone())
    }
}

/// How [`init`] resolved, for `mesh-llm analytics status`.
#[derive(Clone, Debug)]
pub struct Status {
    /// Why reporting is on or off.
    pub disposition: Disposition,
    /// The install identifier, when one could be read.
    pub install_id: Option<String>,
    /// Where events are sent, when reporting is on.
    pub endpoint: Option<String>,
    /// Where the install identifier lives.
    pub install_id_path: Option<PathBuf>,
}

/// Initialize analytics for this process.
///
/// `config` is what the mesh-llm config file says about analytics. Call once,
/// early; later calls return the first result.
///
/// On a first run with reporting enabled, this prints the disclosure notice
/// and captures [`Event::InstallFirstRun`].
///
/// Fail closed on the disclosure: if the notice cannot be delivered, this run
/// reports nothing. On-by-default reporting is defensible only because the
/// disclosure actually happens, so a notice that never reached stderr must not
/// be recorded as shown.
pub fn init(config: ConfigPreference) -> Status {
    let inputs = ConsentInputs::from_env(config);
    let disposition = inputs.resolve();
    let dir = state_dir().ok();

    // Nothing is written while reporting is off: no identifier, no marker.
    // An opted-out machine should not accumulate analytics state at all.
    if !disposition.is_enabled() {
        let _ = REPORTER.set(None);
        return Status {
            disposition,
            install_id: dir
                .as_deref()
                .and_then(load)
                .map(|id| id.as_str().to_owned()),
            endpoint: None,
            install_id_path: dir.map(|dir| dir.join(INSTALL_ID_FILE)),
        };
    }

    let Some(dir) = dir else {
        // Without a readable state directory there is no stable identifier,
        // and reporting every run as a new install would be worse than not
        // reporting at all.
        let _ = REPORTER.set(None);
        return Status {
            disposition,
            install_id: None,
            endpoint: None,
            install_id_path: None,
        };
    };

    // Disclose before anything is queued, and independently of whether the
    // identifier already exists. The marker is what makes this once-only.
    //
    // Fail closed. Reporting is justified by the disclosure having actually
    // happened, so a run that cannot deliver the notice reports nothing at all.
    // `print_notice` records the marker only once the bytes are written, which
    // means the next run retries rather than the disclosure being quietly
    // spent and never shown.
    if !notice::was_shown(&dir) && !notice::print_notice(&dir) {
        tracing::debug!("analytics disabled: first-run disclosure not delivered");
        let _ = REPORTER.set(None);
        return Status {
            disposition,
            install_id: None,
            endpoint: None,
            install_id_path: Some(dir.join(INSTALL_ID_FILE)),
        };
    }

    let Some(install) = load_or_create(&dir).ok() else {
        let _ = REPORTER.set(None);
        return Status {
            disposition,
            install_id: None,
            endpoint: None,
            install_id_path: Some(dir.join(INSTALL_ID_FILE)),
        };
    };

    let endpoint = client::batch_endpoint(&ingestion_host());
    let status = Status {
        disposition,
        install_id: Some(install.as_str().to_owned()),
        endpoint: Some(endpoint.clone()),
        install_id_path: Some(dir.join(INSTALL_ID_FILE)),
    };

    let Some(api_key) = project_key() else {
        // Unreachable: `resolve` already returned `DisabledNoKey`. Handled
        // rather than unwrapped so a future refactor cannot turn this into a
        // panic in the shipped binary.
        let _ = REPORTER.set(None);
        return Status {
            endpoint: None,
            ..status
        };
    };

    // `tokio::spawn` panics outside a runtime. Every call site in this binary
    // is under `block_on`, but this is a public API whose stated contract is
    // that it is never load-bearing, so an embedder gets a clean disable
    // rather than a panic.
    if tokio::runtime::Handle::try_current().is_err() {
        tracing::debug!("analytics disabled: no tokio runtime at init");
        let _ = REPORTER.set(None);
        return Status {
            endpoint: None,
            ..status
        };
    }

    let (sender, receiver) = mpsc::channel(QUEUE_CAPACITY);
    let worker = tokio::spawn(flush_loop(
        receiver,
        api_key,
        install.as_str().to_owned(),
        endpoint,
    ));
    let _ = REPORTER.set(Some(Reporter {
        sender: std::sync::Mutex::new(Some(sender)),
        worker: std::sync::Mutex::new(Some(worker)),
    }));

    if install.is_first_run() {
        capture(Event::InstallFirstRun, Properties::new());
    }

    status
}

/// Queue one event. Never blocks; drops the event if the queue is full.
pub fn capture(event: Event, properties: Properties) {
    let Some(Some(reporter)) = REPORTER.get() else {
        return;
    };
    let Some(sender) = reporter.sender() else {
        return;
    };
    let envelope = Envelope {
        event,
        properties,
        captured_at: Utc::now(),
    };
    if sender.try_send(envelope).is_err() {
        tracing::debug!(
            event = event.name(),
            "analytics event dropped: queue full or already shut down"
        );
    }
}

/// Deliver queued events, giving up after [`SHUTDOWN_BUDGET`].
///
/// Safe to call when analytics is disabled or was never initialized.
pub async fn shutdown() {
    let Some(Some(reporter)) = REPORTER.get() else {
        return;
    };
    let mut worker = reporter.worker.lock().ok().and_then(|mut slot| slot.take());
    let Some(worker) = worker.as_mut() else {
        // Already shut down. A second call is a no-op rather than a second
        // wait, so repeated shutdowns on an error path stay free.
        return;
    };

    // Drop the reporter's sender. The flush loop ends once every sender is
    // gone, sends what it has accumulated, and returns.
    if let Ok(mut slot) = reporter.sender.lock() {
        slot.take();
    }

    if tokio::time::timeout(SHUTDOWN_BUDGET, &mut *worker)
        .await
        .is_err()
    {
        // `timeout` cancels only its own wait. Dropping the handle would
        // detach the flush task instead of ending it, so an embedder that
        // keeps the runtime alive could still see a batch go out after
        // `shutdown` returned. Abort it: the budget is the promise.
        worker.abort();
        tracing::debug!("analytics flush exceeded its shutdown budget; batch aborted");
    }
}

/// Drain the queue, batching events and posting them.
async fn flush_loop(
    mut receiver: mpsc::Receiver<Envelope>,
    api_key: String,
    distinct_id: String,
    endpoint: String,
) {
    let http = match reqwest::Client::builder()
        .timeout(client::REQUEST_TIMEOUT)
        .build()
    {
        Ok(http) => http,
        Err(error) => {
            tracing::debug!(%error, "analytics disabled: no HTTP client");
            return;
        }
    };
    let base = base_properties();
    let mut pending: Vec<Envelope> = Vec::new();

    loop {
        match tokio::time::timeout(FLUSH_INTERVAL, receiver.recv()).await {
            Ok(Some(envelope)) => {
                pending.push(envelope);
                if pending.len() >= BATCH_SIZE {
                    flush(
                        &http,
                        &endpoint,
                        &api_key,
                        &distinct_id,
                        &base,
                        &mut pending,
                    )
                    .await;
                }
            }
            // The channel closed: the process is going away.
            Ok(None) => break,
            // Idle for a full interval: send whatever has accumulated.
            Err(_) => {
                flush(
                    &http,
                    &endpoint,
                    &api_key,
                    &distinct_id,
                    &base,
                    &mut pending,
                )
                .await
            }
        }
    }

    flush(
        &http,
        &endpoint,
        &api_key,
        &distinct_id,
        &base,
        &mut pending,
    )
    .await;
}

async fn flush(
    http: &reqwest::Client,
    endpoint: &str,
    api_key: &str,
    distinct_id: &str,
    base: &Properties,
    pending: &mut Vec<Envelope>,
) {
    if pending.is_empty() {
        return;
    }
    let body = client::batch_body(api_key, distinct_id, base, pending);
    // Cleared regardless of the result: retrying analytics is not worth
    // holding memory or delaying exit for.
    pending.clear();
    client::send_batch(http, endpoint, &body).await;
}

#[cfg(test)]
#[path = "lib/tests.rs"]
mod tests;
