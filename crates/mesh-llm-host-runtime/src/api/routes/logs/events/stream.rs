use std::sync::Arc;
use std::time::Duration;

use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::sync::broadcast;

use super::protocol::heartbeat_frame;
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
    require_enqueued(&queue, initial_frames).await?;

    let mut heartbeat = tokio::time::interval(HEARTBEAT_INTERVAL);
    heartbeat.tick().await;
    let mut audit_reconcile = tokio::time::interval(AUDIT_RECONCILE_INTERVAL);
    audit_reconcile.tick().await;
    loop {
        tokio::select! {
            _ = heartbeat.tick() => {
                require_enqueued(&queue, vec![heartbeat_frame().to_owned()]).await?;
            }
            _ = audit_reconcile.tick(), if session.is_audit() => {
                let frames = reconcile_durable_audit(&mut session, query_facade.clone()).await;
                require_enqueued(&queue, frames).await?;
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
                    require_enqueued(&queue, frames).await?;
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    let frames = current_frames(
                        &mut session,
                        bus.as_ref(),
                        &query_facade,
                        recovery_cursor.clone(),
                    ).await;
                    require_enqueued(&queue, frames).await?;
                }
                Err(broadcast::error::RecvError::Closed) => return Ok(()),
            },
        }
    }
}

async fn require_enqueued(queue: &ConnectionQueue, frames: Vec<String>) -> Result<(), ()> {
    enqueue_frames(queue, frames).await.then_some(()).ok_or(())
}

async fn current_frames(
    session: &mut ReplaySession,
    bus: &ReplayBus,
    query_facade: &LoggingQueryFacade,
    recovery_cursor: Option<String>,
) -> Vec<String> {
    if session.is_audit() {
        reconcile_durable_audit(session, query_facade.clone()).await
    } else {
        session.next_frames(bus, recovery_cursor)
    }
}

async fn update_frames(
    session: &mut ReplaySession,
    bus: &ReplayBus,
    query_facade: &LoggingQueryFacade,
    update: &ReplayUpdate,
    recovery_cursor: Option<String>,
) -> Vec<String> {
    if session.is_audit() {
        reconcile_durable_audit(session, query_facade.clone()).await
    } else {
        session.next_update_frames(bus, update, recovery_cursor)
    }
}

async fn reconcile_durable_audit(
    session: &mut ReplaySession,
    query_facade: LoggingQueryFacade,
) -> Vec<String> {
    let Some((cursor, filters)) = session.durable_audit_query() else {
        return Vec::new();
    };
    let records = tokio::task::spawn_blocking(move || {
        query_facade.audit_entries_after_sequence(cursor, AUDIT_RECONCILE_LIMIT, filters)
    })
    .await
    .ok()
    .and_then(Result::ok)
    .unwrap_or_default();
    session.durable_audit_frames(records)
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
