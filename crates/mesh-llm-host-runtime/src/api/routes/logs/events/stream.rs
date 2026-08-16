use std::sync::Arc;
use std::time::Duration;

use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::sync::broadcast;

use super::protocol::{audit_reconcile_error_frame, heartbeat_frame};
use super::session::{ConnectionQueue, QueueError, ReplaySession};
use crate::logging::{LoggingQueryFacade, ReplayBus, ReplayUpdate};

const CONNECTION_QUEUE_CAPACITY: usize = 64;
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);
const AUDIT_RECONCILE_INTERVAL: Duration = Duration::from_secs(1);
const AUDIT_RECONCILE_LIMIT: usize = 100;
const WRITE_TIMEOUT: Duration = Duration::from_millis(250);
const SSE_HEADER: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n";

/// Run the already-validated stream through a bounded socket adapter.
pub(in crate::api::routes::logs) async fn stream(
    stream: &mut TcpStream,
    subscription: super::query::Subscription,
    bus: Arc<ReplayBus>,
    query_facade: LoggingQueryFacade,
    recovery_cursor: Option<String>,
) -> anyhow::Result<()> {
    // Subscribe before the response becomes observable by a client. Otherwise
    // a producer can publish between the successful header write and the
    // asynchronous producer task's subscription, losing a live update that
    // the client is entitled to receive after it sees `200 OK`.
    let updates = bus.subscribe_updates();
    tokio::time::timeout(WRITE_TIMEOUT, stream.write_all(SSE_HEADER))
        .await
        .map_err(|_| anyhow::anyhow!("logs SSE header write timed out"))??;

    run(
        stream,
        bus,
        query_facade,
        subscription,
        recovery_cursor,
        updates,
    )
    .await;
    Ok(())
}

async fn run(
    stream: &mut TcpStream,
    bus: Arc<ReplayBus>,
    query_facade: LoggingQueryFacade,
    subscription: super::query::Subscription,
    recovery_cursor: Option<String>,
    updates: broadcast::Receiver<ReplayUpdate>,
) {
    let (queue, mut receiver) = ConnectionQueue::new(CONNECTION_QUEUE_CAPACITY);
    let producer = tokio::spawn(produce_frames(
        Arc::clone(&bus),
        query_facade,
        subscription,
        recovery_cursor,
        queue.clone(),
        updates,
    ));

    while let Some(frame) = receiver.recv().await {
        let write = tokio::time::timeout(WRITE_TIMEOUT, stream.write_all(frame.as_bytes())).await;
        if !matches!(write, Ok(Ok(()))) {
            queue.cancel();
            break;
        }
    }

    producer.abort();
    let _ = producer.await;
}

async fn produce_frames(
    bus: Arc<ReplayBus>,
    query_facade: LoggingQueryFacade,
    subscription: super::query::Subscription,
    recovery_cursor: Option<String>,
    queue: ConnectionQueue,
    mut updates: broadcast::Receiver<ReplayUpdate>,
) -> Result<(), ()> {
    let mut session = ReplaySession::new(subscription);
    let initial_frames = current_frames(
        &mut session,
        bus.as_ref(),
        &query_facade,
        recovery_cursor.clone(),
    )
    .await;
    enqueue_frames_or_error(&queue, &session, initial_frames).await?;

    let mut heartbeat = tokio::time::interval(HEARTBEAT_INTERVAL);
    heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    heartbeat.tick().await;
    let mut audit_reconcile = tokio::time::interval(AUDIT_RECONCILE_INTERVAL);
    audit_reconcile.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    audit_reconcile.tick().await;
    loop {
        tokio::select! {
            _ = heartbeat.tick() => {
                require_enqueued(&queue, vec![heartbeat_frame().to_owned()]).await?;
            }
            _ = audit_reconcile.tick(), if session.is_audit() => {
                let frames = reconcile_durable_audit(&mut session, query_facade.clone()).await;
                enqueue_frames_or_error(&queue, &session, frames).await?;
            }
            update = updates.recv() => match update {
                Ok(update) => {
                    let frames = update_frames(
                        &mut session,
                        bus.as_ref(),
                        &query_facade,
                        &update,
                        recovery_cursor.clone(),
                    ).await;
                    enqueue_frames_or_error(&queue, &session, frames).await?;
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    let frames = current_frames(
                        &mut session,
                        bus.as_ref(),
                        &query_facade,
                        recovery_cursor.clone(),
                    ).await;
                    enqueue_frames_or_error(&queue, &session, frames).await?;
                }
                Err(broadcast::error::RecvError::Closed) => return Ok(()),
            },
        }
    }
}

async fn require_enqueued(queue: &ConnectionQueue, frames: Vec<String>) -> Result<(), ()> {
    enqueue_frames(queue, frames).await.then_some(()).ok_or(())
}

/// Enqueue frames produced by a replay phase. A durable-audit reconcile
/// failure enqueues EXACTLY ONE typed `stream_error` frame and aborts the
/// producer instead of silently replaying empty frames every tick.
async fn enqueue_frames_or_error(
    queue: &ConnectionQueue,
    session: &ReplaySession,
    frames: Result<Vec<String>, ()>,
) -> Result<(), ()> {
    match frames {
        Ok(frames) => require_enqueued(queue, frames).await,
        Err(()) => {
            let sequence = session
                .durable_audit_query()
                .map(|(sequence, _)| sequence)
                .unwrap_or_default();
            require_enqueued(queue, vec![audit_reconcile_error_frame(sequence)]).await?;
            Err(())
        }
    }
}

async fn current_frames(
    session: &mut ReplaySession,
    bus: &ReplayBus,
    query_facade: &LoggingQueryFacade,
    recovery_cursor: Option<String>,
) -> Result<Vec<String>, ()> {
    if session.is_audit() {
        reconcile_durable_audit(session, query_facade.clone()).await
    } else {
        Ok(session.next_frames(bus, recovery_cursor))
    }
}

async fn update_frames(
    session: &mut ReplaySession,
    bus: &ReplayBus,
    query_facade: &LoggingQueryFacade,
    update: &ReplayUpdate,
    recovery_cursor: Option<String>,
) -> Result<Vec<String>, ()> {
    if session.is_audit() {
        reconcile_durable_audit(session, query_facade.clone()).await
    } else {
        Ok(session.next_update_frames(bus, update, recovery_cursor))
    }
}

async fn reconcile_durable_audit(
    session: &mut ReplaySession,
    query_facade: LoggingQueryFacade,
) -> Result<Vec<String>, ()> {
    let Some((cursor, filters)) = session.durable_audit_query() else {
        return Ok(Vec::new());
    };
    let records = match tokio::task::spawn_blocking(move || {
        query_facade.audit_entries_after_sequence(cursor, AUDIT_RECONCILE_LIMIT, filters)
    })
    .await
    {
        Ok(Ok(records)) => records,
        Ok(Err(error)) => {
            tracing::warn!("logs audit reconcile failed: {error}");
            return Err(());
        }
        Err(join_error) => {
            tracing::error!("logs audit reconcile task failed: {join_error}");
            return Err(());
        }
    };
    Ok(session.durable_audit_frames(records))
}

async fn enqueue_frames(queue: &ConnectionQueue, frames: Vec<String>) -> bool {
    for frame in frames {
        if !enqueue(queue, frame).await {
            return false;
        }
    }
    true
}

async fn enqueue(queue: &ConnectionQueue, frame: String) -> bool {
    match queue.send_with_timeout(frame, WRITE_TIMEOUT).await {
        Ok(()) => true,
        Err(QueueError::SlowConsumer | QueueError::Cancelled) => {
            queue.cancel();
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::query::{AuditCursor, AuditSelection, Cursor, LedgerFilters, Subscription};
    use super::*;

    fn runtime() -> (tempfile::TempDir, crate::logging::LoggingRuntimeState) {
        let temp = tempfile::tempdir().expect("temporary application state root");
        let root = temp.path().join("logging-state");
        let foundation = crate::logging::LoggingFoundation::init(true, Some(&root));
        let config = mesh_llm_config::LoggingConfig {
            application_state_root: Some(root),
            ..Default::default()
        };
        (
            temp,
            crate::logging::LoggingRuntimeState::initialize(&foundation, &config),
        )
    }

    fn audit_subscription(sequence: u64) -> Subscription {
        Subscription {
            channels: Vec::new(),
            filters: LedgerFilters::default(),
            cursor: Cursor::default(),
            audit: Some(AuditSelection {
                cursor: AuditCursor(sequence),
                source: None,
                severity: None,
            }),
        }
    }

    #[tokio::test]
    async fn audit_reconcile_failure_emits_one_error_frame_and_terminates() {
        // A cursor beyond the i64 range makes the durable query reject the
        // sequence, exercising the reconcile failure path against the real store.
        let (_temp, state) = runtime();
        let bus = ReplayBus::new(64);
        let updates = bus.subscribe_updates();
        let (queue, mut receiver) = ConnectionQueue::new(CONNECTION_QUEUE_CAPACITY);

        let result = produce_frames(
            Arc::new(bus),
            state.query_facade().expect("query facade"),
            audit_subscription(u64::MAX),
            None,
            queue,
            updates,
        )
        .await;

        assert!(
            result.is_err(),
            "reconcile failure must terminate the producer"
        );
        let frame = receiver
            .recv()
            .await
            .expect("exactly one error frame is enqueued");
        assert!(frame.contains("event: stream_error"));
        assert!(frame.contains("\"code\":\"audit_reconcile_failed\""));
        assert!(!frame.contains("invalid_event"));
        assert!(
            receiver.recv().await.is_none(),
            "no further frames may follow the single error frame"
        );
    }
}
