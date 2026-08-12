use std::sync::Arc;
use std::time::Duration;

use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::sync::broadcast;

use super::protocol::heartbeat_frame;
use super::session::{ConnectionQueue, QueueError, ReplaySession};
use crate::logging::{ReplayBus, ReplayUpdate};

const CONNECTION_QUEUE_CAPACITY: usize = 64;
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);
const WRITE_TIMEOUT: Duration = Duration::from_millis(250);
const SSE_HEADER: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n";

/// Run the already-validated stream through a bounded socket adapter.
pub(in crate::api::routes::logs) async fn stream(
    stream: &mut TcpStream,
    subscription: super::query::Subscription,
    bus: Arc<ReplayBus>,
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

    run(stream, bus, subscription, recovery_cursor, updates).await;
    Ok(())
}

async fn run(
    stream: &mut TcpStream,
    bus: Arc<ReplayBus>,
    subscription: super::query::Subscription,
    recovery_cursor: Option<String>,
    updates: broadcast::Receiver<ReplayUpdate>,
) {
    let (queue, mut receiver) = ConnectionQueue::new(CONNECTION_QUEUE_CAPACITY);
    let producer = tokio::spawn(produce_frames(
        Arc::clone(&bus),
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
    subscription: super::query::Subscription,
    recovery_cursor: Option<String>,
    queue: ConnectionQueue,
    mut updates: broadcast::Receiver<ReplayUpdate>,
) {
    let mut session = ReplaySession::new(subscription);
    if !enqueue_frames(
        &queue,
        session.next_frames(bus.as_ref(), recovery_cursor.clone()),
    )
    .await
    {
        return;
    }

    let mut heartbeat = tokio::time::interval(HEARTBEAT_INTERVAL);
    heartbeat.tick().await;
    loop {
        tokio::select! {
            _ = heartbeat.tick() => {
                if !enqueue(&queue, heartbeat_frame().to_owned()).await {
                    return;
                }
            }
            update = updates.recv() => match update {
                Ok(update) => {
                    let frames = session.next_update_frames(
                        bus.as_ref(),
                        &update,
                        recovery_cursor.clone(),
                    );
                    if !enqueue_frames(&queue, frames).await {
                        return;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => {
                    let frames = session.next_frames(bus.as_ref(), recovery_cursor.clone());
                    if !enqueue_frames(&queue, frames).await {
                        return;
                    }
                }
                Err(broadcast::error::RecvError::Closed) => return,
            },
        }
    }
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
