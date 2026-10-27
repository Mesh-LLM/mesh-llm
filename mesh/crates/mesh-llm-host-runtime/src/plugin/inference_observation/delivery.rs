//! Sends one subscriber's queued events to its plugin connection.

use super::hub::{ObservationHub, SubscriberId};
use super::proto;
use crate::plugin::PROTOCOL_VERSION;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Notify, mpsc};

/// Batches go out at least this often, empty when nothing happened, so the
/// watermark keeps advancing.
pub(crate) const HEARTBEAT: Duration = Duration::from_secs(1);
/// Events in one batch at most.
const BATCH_EVENTS: usize = 256;

/// Deliver subscriber `id` to `outbound_tx` until the plugin connection closes
/// or a later subscription replaces this one; then unsubscribe it. Waits on
/// the connection, never on the inference path.
pub(crate) fn spawn(
    hub: Arc<ObservationHub>,
    id: SubscriberId,
    notify: Arc<Notify>,
    plugin_name: String,
    outbound_tx: mpsc::Sender<proto::Envelope>,
) {
    tokio::spawn(async move {
        loop {
            let _ = tokio::time::timeout(HEARTBEAT, notify.notified()).await;
            let mut more = true;
            while more {
                let Some(drained) = hub.drain(id, BATCH_EVENTS) else {
                    return;
                };
                more = drained.events.len() == BATCH_EVENTS;
                let batch = proto::InferenceObservationBatch {
                    host_generation: hub.host_generation(),
                    events: drained.events,
                    watermark_unix_ns: drained.watermark_unix_ns,
                    loss: drained.loss,
                    resync: drained.resync,
                };
                let envelope = proto::Envelope {
                    protocol_version: PROTOCOL_VERSION,
                    plugin_id: plugin_name.clone(),
                    request_id: 0,
                    payload: Some(proto::envelope::Payload::InferenceObservationBatch(batch)),
                };
                if outbound_tx.send(envelope).await.is_err() {
                    hub.unsubscribe(id);
                    return;
                }
            }
        }
    });
}
