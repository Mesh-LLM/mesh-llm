//! The local inference observation feed.
//!
//! A plugin declares [`inference_observation`] in its manifest, its operator
//! sets `allow_inference_observation = true` for it, and it calls
//! [`crate::PluginContext::subscribe_inference_observation`]. It gets a
//! snapshot of what this node is executing, then batches of events in order:
//! executions starting, producing first output, finishing compute, reporting
//! token usage and ending, and local models becoming ready or unloaded. Only
//! work this node executes appears, requests from peers included; work it
//! only forwards never does. See `InferenceObservationBatch` for the timing and
//! loss rules.

use crate::manifest::ManifestEntry;
use crate::proto;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// The feed contract this SDK reads.
pub const CONTRACT_VERSION: u32 = 1;
/// Batches the SDK holds for a plugin that reads its feed slowly. When it is
/// full, the SDK stops delivering and the feed reports
/// [`InferenceObservationItem::Overflowed`].
const FEED_CAPACITY: usize = 1024;

/// Declare that this plugin reads the local inference observation feed.
pub fn inference_observation() -> InferenceObservationBuilder {
    InferenceObservationBuilder {
        inner: proto::InferenceObservationManifest {
            contract_version: CONTRACT_VERSION,
            required: false,
        },
    }
}

#[derive(Clone, Debug)]
pub struct InferenceObservationBuilder {
    inner: proto::InferenceObservationManifest,
}

impl InferenceObservationBuilder {
    /// Refuse to initialize on a host that does not list
    /// [`crate::host_capabilities::LOCAL_INFERENCE_OBSERVATION`].
    pub fn required(mut self, required: bool) -> Self {
        self.inner.required = required;
        self
    }

    pub fn build(self) -> proto::InferenceObservationManifest {
        self.inner
    }
}

impl From<InferenceObservationBuilder> for ManifestEntry {
    fn from(value: InferenceObservationBuilder) -> Self {
        ManifestEntry::InferenceObservation(value.build())
    }
}

/// What [`InferenceObservationFeed::next`] returns.
#[derive(Debug)]
pub enum InferenceObservationItem {
    Batch(proto::InferenceObservationBatch),
    /// The plugin read so slowly that this SDK dropped batches. Nothing after
    /// the last batch can be trusted: subscribe again for a fresh snapshot.
    Overflowed,
}

/// A subscription's snapshot and the batches that follow it.
pub struct InferenceObservationFeed {
    snapshot: proto::InferenceObservationSnapshot,
    rx: mpsc::Receiver<proto::InferenceObservationBatch>,
    overflowed: Arc<AtomicBool>,
}

impl InferenceObservationFeed {
    /// Where the feed starts.
    pub fn snapshot(&self) -> &proto::InferenceObservationSnapshot {
        &self.snapshot
    }

    /// The next batch, in order. `None` once the host connection is gone or a
    /// later subscription replaced this one. Events from before the snapshot
    /// (a replaced subscription's last batch) are left out.
    pub async fn next(&mut self) -> Option<InferenceObservationItem> {
        let batch = match self.rx.try_recv() {
            Ok(batch) => batch,
            Err(_) if self.overflowed.swap(false, Ordering::AcqRel) => {
                return Some(InferenceObservationItem::Overflowed);
            }
            Err(_) => self.rx.recv().await?,
        };
        Some(InferenceObservationItem::Batch(self.after_snapshot(batch)))
    }

    fn after_snapshot(
        &self,
        mut batch: proto::InferenceObservationBatch,
    ) -> proto::InferenceObservationBatch {
        if batch.host_generation == self.snapshot.host_generation {
            let start = self.snapshot.next_sequence;
            batch.events.retain(|event| event.sequence >= start);
        }
        batch
    }
}

/// A subscription waiting for the host's answer. Batches that arrive before
/// the answer is read are kept, so none is lost between the two.
pub(crate) struct PendingFeed {
    rx: mpsc::Receiver<proto::InferenceObservationBatch>,
    overflowed: Arc<AtomicBool>,
}

impl PendingFeed {
    pub(crate) fn start(
        self,
        snapshot: proto::InferenceObservationSnapshot,
    ) -> InferenceObservationFeed {
        InferenceObservationFeed {
            snapshot,
            rx: self.rx,
            overflowed: self.overflowed,
        }
    }
}

struct Delivery {
    tx: mpsc::Sender<proto::InferenceObservationBatch>,
    overflowed: Arc<AtomicBool>,
}

/// Where the runtime's read loop hands batches to the current subscription.
#[derive(Clone, Default)]
pub(crate) struct FeedSlot(Arc<Mutex<Option<Delivery>>>);

impl FeedSlot {
    /// Replace the current subscription with a new one, before asking the host
    /// for it.
    pub(crate) fn open(&self) -> PendingFeed {
        let (tx, rx) = mpsc::channel(FEED_CAPACITY);
        let overflowed = Arc::new(AtomicBool::new(false));
        *self.0.lock().expect("feed slot poisoned") = Some(Delivery {
            tx,
            overflowed: overflowed.clone(),
        });
        PendingFeed { rx, overflowed }
    }

    /// The host refused the subscription: stop delivering.
    pub(crate) fn close(&self) {
        *self.0.lock().expect("feed slot poisoned") = None;
    }

    /// Hand a batch to the subscription without waiting, so a slow plugin
    /// never stalls the connection.
    pub(crate) fn deliver(&self, batch: proto::InferenceObservationBatch) {
        let slot = self.0.lock().expect("feed slot poisoned");
        let Some(delivery) = slot.as_ref() else {
            return;
        };
        if delivery.overflowed.load(Ordering::Acquire) {
            return;
        }
        if let Err(mpsc::error::TrySendError::Full(_)) = delivery.tx.try_send(batch) {
            delivery.overflowed.store(true, Ordering::Release);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(watermark_unix_ns: u64) -> proto::InferenceObservationBatch {
        proto::InferenceObservationBatch {
            watermark_unix_ns,
            ..Default::default()
        }
    }

    #[test]
    fn the_declaration_lands_in_the_manifest() {
        let manifest = crate::plugin_manifest()
            .item(inference_observation().required(true))
            .build();
        assert_eq!(
            manifest.inference_observation,
            Some(proto::InferenceObservationManifest {
                contract_version: CONTRACT_VERSION,
                required: true,
            })
        );
    }

    #[tokio::test]
    async fn batches_arrive_in_order_and_a_new_subscription_ends_the_old_one() {
        let slot = FeedSlot::default();
        slot.deliver(batch(1)); // no subscription yet: dropped
        let mut first = slot
            .open()
            .start(proto::InferenceObservationSnapshot::default());
        slot.deliver(batch(2));
        slot.deliver(batch(3));
        for expected in [2, 3] {
            let Some(InferenceObservationItem::Batch(batch)) = first.next().await else {
                panic!("expected a batch");
            };
            assert_eq!(batch.watermark_unix_ns, expected);
        }
        let _second = slot.open();
        assert!(first.next().await.is_none(), "replaced");
    }

    #[tokio::test]
    async fn a_plugin_that_reads_too_slowly_is_told_after_the_batches_it_has() {
        let slot = FeedSlot::default();
        let mut feed = slot
            .open()
            .start(proto::InferenceObservationSnapshot::default());
        for watermark in 0..=FEED_CAPACITY as u64 {
            slot.deliver(batch(watermark));
        }
        // Later batches are dropped too: the stream has a hole now.
        slot.deliver(batch(u64::MAX));
        for expected in 0..FEED_CAPACITY as u64 {
            let Some(InferenceObservationItem::Batch(batch)) = feed.next().await else {
                panic!("expected batch {expected}");
            };
            assert_eq!(batch.watermark_unix_ns, expected);
        }
        assert!(matches!(
            feed.next().await,
            Some(InferenceObservationItem::Overflowed)
        ));
    }

    #[tokio::test]
    async fn batches_before_the_answer_are_kept_and_events_before_the_snapshot_left_out() {
        let event = |sequence| proto::InferenceObservationEvent {
            sequence,
            ..Default::default()
        };
        let slot = FeedSlot::default();
        let pending = slot.open();
        // A replaced subscription's batch, then one sent after the snapshot,
        // both before the plugin reads the answer.
        slot.deliver(proto::InferenceObservationBatch {
            host_generation: 9,
            events: vec![event(3), event(4)],
            ..Default::default()
        });
        slot.deliver(proto::InferenceObservationBatch {
            host_generation: 9,
            events: vec![event(5)],
            ..Default::default()
        });
        let mut feed = pending.start(proto::InferenceObservationSnapshot {
            host_generation: 9,
            next_sequence: 4,
            ..Default::default()
        });
        let mut seen = Vec::new();
        for _ in 0..2 {
            let Some(InferenceObservationItem::Batch(batch)) = feed.next().await else {
                panic!("expected a batch");
            };
            seen.extend(batch.events.iter().map(|event| event.sequence));
        }
        assert_eq!(seen, [4, 5]);
    }

    #[test]
    fn a_required_declaration_needs_the_host_capability() {
        let manifest = crate::plugin_manifest()
            .item(inference_observation().required(true))
            .build();
        let message = crate::host_capabilities::unsupported_requirement(Some(&manifest), |_| false)
            .expect("refused");
        assert!(
            message.contains("local_inference_observation.v1"),
            "{message}"
        );
        assert!(
            crate::host_capabilities::unsupported_requirement(Some(&manifest), |capability| {
                capability == crate::host_capabilities::LOCAL_INFERENCE_OBSERVATION
            })
            .is_none()
        );
        let optional = crate::plugin_manifest()
            .item(inference_observation())
            .build();
        assert!(
            crate::host_capabilities::unsupported_requirement(Some(&optional), |_| false).is_none()
        );
    }
}
