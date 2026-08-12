//! Runtime scheduling for durable webhook delivery attempts.
//!
//! The scheduler owns no delivery policy and has no request-path hook. It
//! drives the existing one-at-a-time delivery worker from the logging runtime
//! after terminal records are durable, then bounds shutdown so an interrupted
//! attempt remains recoverable through the store's stale-lease claim rules.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::sync::watch;
use tokio::task::JoinHandle;

use super::{WebhookDeliveryWorker, WebhookWorkerOutcome};

const WEBHOOK_DELIVERY_CADENCE: Duration = Duration::from_secs(1);
const WEBHOOK_DELIVERY_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

/// One cancellable scheduler task per logging runtime state.
pub(crate) struct WebhookDeliveryScheduler {
    shutdown_tx: watch::Sender<bool>,
    task: Mutex<Option<JoinHandle<()>>>,
    shutdown_timeout: Arc<Mutex<Duration>>,
}

impl WebhookDeliveryScheduler {
    pub(crate) fn start(worker: WebhookDeliveryWorker) -> Self {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let shutdown_timeout = Arc::new(Mutex::new(WEBHOOK_DELIVERY_SHUTDOWN_TIMEOUT));
        let task = tokio::spawn(run_scheduler(
            worker,
            shutdown_rx,
            Arc::clone(&shutdown_timeout),
        ));
        Self {
            shutdown_tx,
            task: Mutex::new(Some(task)),
            shutdown_timeout,
        }
    }

    /// Stop accepting new scheduling ticks, then use the fixed runtime bound
    /// to flush currently eligible records. An HTTP attempt that does not fit
    /// the remaining bound is cancelled; its durable in-flight lease remains
    /// reclaimable after a restart.
    pub(crate) async fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
        let Some(mut task) = self
            .task
            .lock()
            .expect("webhook delivery scheduler mutex poisoned")
            .take()
        else {
            return;
        };
        let shutdown_timeout = *self
            .shutdown_timeout
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if tokio::time::timeout(shutdown_timeout, &mut task)
            .await
            .is_err()
        {
            task.abort();
            let _ = task.await;
        }
    }

    #[cfg(test)]
    fn with_shutdown_timeout_for_test(self, timeout: Duration) -> Self {
        *self
            .shutdown_timeout
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = timeout;
        self
    }
}

impl Drop for WebhookDeliveryScheduler {
    fn drop(&mut self) {
        let _ = self.shutdown_tx.send(true);
        if let Some(task) = self
            .task
            .lock()
            .expect("webhook delivery scheduler mutex poisoned")
            .take()
        {
            task.abort();
        }
    }
}

async fn run_scheduler(
    worker: WebhookDeliveryWorker,
    mut shutdown_rx: watch::Receiver<bool>,
    shutdown_timeout: Arc<Mutex<Duration>>,
) {
    let mut ticker =
        tokio::time::interval_at(tokio::time::Instant::now(), WEBHOOK_DELIVERY_CADENCE);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            biased;
            changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                    let timeout = *shutdown_timeout
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    drain_eligible_deliveries(&worker, timeout).await;
                    return;
                }
            }
            _ = ticker.tick() => {
                match worker.process_next().await {
                    Ok(WebhookWorkerOutcome::Idle) => {}
                    Ok(_) => ticker.reset_immediately(),
                    Err(_) => tracing::warn!("webhook delivery worker step failed; continuing"),
                }
            }
        }
    }
}

async fn drain_eligible_deliveries(worker: &WebhookDeliveryWorker, timeout: Duration) {
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        match tokio::time::timeout_at(deadline, worker.process_next()).await {
            Ok(Ok(WebhookWorkerOutcome::Idle)) | Ok(Err(_)) | Err(_) => return,
            Ok(Ok(_)) => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use mesh_llm_config::LoggingWebhookConfig;
    use mesh_llm_log_store::{LogStore, RealClock, WebhookDeliveryState};
    use tokio::sync::Notify;

    use super::*;
    use crate::logging::webhook_delivery::WebhookTerminalPayload;
    use crate::logging::{
        RandomWebhookJitter, SystemWebhookWorkerClock, WebhookJitter, WebhookTransport,
        WebhookTransportError, WebhookWorkerClock,
    };

    const CREATED_AT: &str = "2020-01-01T00:00:00Z";
    const CLAIMED_AT: &str = "2020-01-01T00:00:01Z";
    const LEASE_EXPIRES_AT: &str = "2020-01-01T00:00:02Z";

    struct SuccessfulTransport {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl WebhookTransport for SuccessfulTransport {
        async fn post_terminal(
            &self,
            _: &url::Url,
            _: &WebhookTerminalPayload,
            _: Duration,
        ) -> Result<u16, WebhookTransportError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(204)
        }
    }

    struct BlockingTransport {
        entered: Notify,
    }

    #[async_trait]
    impl WebhookTransport for BlockingTransport {
        async fn post_terminal(
            &self,
            _: &url::Url,
            _: &WebhookTerminalPayload,
            _: Duration,
        ) -> Result<u16, WebhookTransportError> {
            self.entered.notify_one();
            std::future::pending().await
        }
    }

    fn open_store() -> (Arc<LogStore>, tempfile::TempDir) {
        let root = tempfile::tempdir().expect("temporary log store");
        let store = LogStore::open(root.path(), Arc::new(RealClock)).expect("open log store");
        (Arc::new(store), root)
    }

    fn enqueue_terminal_delivery(store: &LogStore, delivery_id: &str) {
        store
            .insert_summary(
                "request-terminal",
                None,
                None,
                None,
                None,
                CREATED_AT,
                None,
                None,
                None,
            )
            .expect("durable request summary");
        store
            .write_terminal_event(
                "request-terminal",
                "event-terminal",
                r#"{"type":"completed"}"#,
                "completed",
                None,
                CREATED_AT,
            )
            .expect("durable terminal event");
        store
            .enqueue_webhook_delivery(delivery_id, "request-terminal", CREATED_AT, 3)
            .expect("enqueue delivery");
    }

    fn worker(store: Arc<LogStore>, transport: Arc<dyn WebhookTransport>) -> WebhookDeliveryWorker {
        WebhookDeliveryWorker::from_config(
            store,
            &LoggingWebhookConfig {
                enabled: true,
                url: Some("http://127.0.0.1:9444/webhook".to_string()),
                max_attempts: 3,
                timeout_secs: 60,
                dead_letter_retention_secs: 3_600,
            },
            transport,
            Arc::new(SystemWebhookWorkerClock) as Arc<dyn WebhookWorkerClock>,
            Arc::new(RandomWebhookJitter) as Arc<dyn WebhookJitter>,
        )
        .expect("valid delivery worker")
    }

    async fn wait_for_state(store: &LogStore, delivery_id: &str, expected: WebhookDeliveryState) {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if store
                    .webhook_delivery(delivery_id)
                    .expect("load delivery")
                    .is_some_and(|record| record.state == expected)
                {
                    return;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("scheduler reaches expected durable state");
    }

    #[tokio::test]
    async fn startup_scheduler_reclaims_and_delivers_a_stale_in_flight_lease() {
        let (store, _root) = open_store();
        enqueue_terminal_delivery(&store, "stale-delivery");
        let claimed = store
            .claim_next_webhook_delivery(CLAIMED_AT, LEASE_EXPIRES_AT)
            .expect("claim stale delivery")
            .expect("delivery claimed");
        assert_eq!(claimed.state, WebhookDeliveryState::InFlight);

        let transport = Arc::new(SuccessfulTransport {
            calls: AtomicUsize::new(0),
        });
        let worker_transport: Arc<dyn WebhookTransport> = transport.clone();
        let scheduler =
            WebhookDeliveryScheduler::start(worker(Arc::clone(&store), worker_transport));

        wait_for_state(&store, "stale-delivery", WebhookDeliveryState::Succeeded).await;
        assert_eq!(transport.calls.load(Ordering::Relaxed), 1);
        scheduler.shutdown().await;
    }

    #[tokio::test]
    async fn shutdown_is_bounded_when_an_http_attempt_is_still_in_flight() {
        let (store, _root) = open_store();
        enqueue_terminal_delivery(&store, "blocked-delivery");
        let transport = Arc::new(BlockingTransport {
            entered: Notify::new(),
        });
        let worker_transport: Arc<dyn WebhookTransport> = transport.clone();
        let scheduler =
            WebhookDeliveryScheduler::start(worker(Arc::clone(&store), worker_transport))
                .with_shutdown_timeout_for_test(Duration::ZERO);

        transport.entered.notified().await;
        tokio::time::timeout(Duration::from_millis(100), scheduler.shutdown())
            .await
            .expect("bounded scheduler shutdown");
        assert_eq!(
            store
                .webhook_delivery("blocked-delivery")
                .expect("load blocked delivery")
                .expect("delivery remains durable")
                .state,
            WebhookDeliveryState::InFlight
        );
    }
}
