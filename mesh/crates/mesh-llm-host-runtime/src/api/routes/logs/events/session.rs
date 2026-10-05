use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use mesh_llm_events::logging::envelope::CanonicalEnvelope;
use mesh_llm_events::logging::replay::ReplayChannel;
use tokio::sync::mpsc;

use super::protocol::{
    GapData, audit_entry_frame, audit_gap_frame, durable_audit_entry_frame, error_frame,
    event_frame, gap_frame,
};
use super::query::{AuditCursor, AuditSelection, Cursor, Subscription};
use crate::logging::{
    AuditReplayRecord, ReplayBus, ReplayCursor, ReplayRecord, ReplayUpdate, ReplayWindow,
    RequestSummaryEventSnapshots,
};

#[cfg(test)]
thread_local! {
    static LIFECYCLE_RECORDS_INSPECTED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Build the deterministic initial replay in bus insertion order. The caller
/// supplies an opaque REST cursor created by the durable query layer; this
/// protocol never treats the in-memory window as recovery authority.
#[cfg(test)]
pub(in crate::api::routes::logs) fn replay_frames(
    bus: &ReplayBus,
    subscription: &Subscription,
    recovery_cursor: Option<String>,
) -> Vec<String> {
    if subscription.audit.is_some() {
        replay_audit_frames(bus, subscription, recovery_cursor)
    } else {
        let window = bus.replay_window();
        replay_window_frames(&window, subscription, recovery_cursor)
    }
}

#[cfg(test)]
fn replay_audit_frames(
    bus: &ReplayBus,
    subscription: &Subscription,
    recovery_cursor: Option<String>,
) -> Vec<String> {
    let sel = subscription.audit.as_ref().expect("audit subscription");
    let window = bus.audit_replay_window();
    let mut frames = Vec::new();
    let cursor = sel.cursor;

    if cursor.sequence() < window.evicted_through {
        let gap = audit_gap_frame(
            cursor.sequence().saturating_add(1),
            window.evicted_through,
            recovery_cursor.clone(),
        );
        if let Ok(frame) = gap {
            frames.push(frame);
        }
    }

    for record in &window.records {
        if record.sequence <= cursor.sequence() {
            continue;
        }
        if !audit_matches_filter(&record.entry.payload, sel) {
            continue;
        }
        match audit_entry_frame(record) {
            Ok(frame) => frames.push(frame),
            Err(()) => {
                frames.push(format!(
                    "event: stream_error\nid: a1:{}\ndata: {{\"code\":\"invalid_event\"}}\n\n",
                    record.sequence
                ));
            }
        }
    }

    frames
}

/// Per-connection replay state. It advances over every selected-channel
/// record, including filtered or invalid records, so a live snapshot is never
/// re-emitted after an update notification.
pub(in crate::api::routes::logs) struct ReplaySession {
    subscription: Subscription,
    cursor: Cursor,
    audit_cursor: AuditCursor,
}

impl ReplaySession {
    pub(super) fn new(subscription: Subscription) -> Self {
        let cursor = subscription.cursor;
        let audit_cursor = subscription
            .audit
            .as_ref()
            .map(|a| a.cursor)
            .unwrap_or_default();
        Self {
            subscription,
            cursor,
            audit_cursor,
        }
    }

    pub(super) fn is_audit(&self) -> bool {
        self.subscription.audit.is_some()
    }

    pub(super) fn durable_audit_query(
        &self,
    ) -> Option<(u64, mesh_llm_log_store::AuditEntryFilters)> {
        let selection = self.subscription.audit.as_ref()?;
        let filters = selection.durable_filters().ok()?;
        Some((self.audit_cursor.sequence(), filters))
    }

    pub(super) fn durable_audit_frames(
        &mut self,
        records: Vec<mesh_llm_log_store::AuditEntryDetail>,
    ) -> Vec<String> {
        let mut frames = Vec::with_capacity(records.len());
        for record in records {
            let Ok(sequence) = u64::try_from(record.entry.sequence) else {
                continue;
            };
            if sequence <= self.audit_cursor.sequence() {
                continue;
            }
            match durable_audit_entry_frame(record) {
                Ok(frame) => frames.push(frame),
                Err(()) => frames.push(format!(
                    "event: stream_error\nid: a1:{sequence}\ndata: {{\"code\":\"invalid_event\"}}\n\n"
                )),
            }
            self.audit_cursor.advance(sequence);
        }
        frames
    }

    pub(super) fn next_frames(
        &mut self,
        bus: &ReplayBus,
        recovery_cursor: Option<String>,
    ) -> Vec<String> {
        if self.subscription.audit.is_some() {
            self.next_audit_frames(bus, recovery_cursor)
        } else {
            self.next_lifecycle_frames(bus, recovery_cursor)
        }
    }

    /// Project one normal live-delivery delta. A lagged broadcast receiver must
    /// use `next_frames` instead so the bounded snapshot can report any gap and
    /// recover all records still retained after this session's cursor.
    pub(super) fn next_update_frames(
        &mut self,
        bus: &ReplayBus,
        update: &ReplayUpdate,
        recovery_cursor: Option<String>,
    ) -> Vec<String> {
        match update {
            ReplayUpdate::Lifecycle {
                record,
                evicted_through,
                latest,
            } if self.subscription.audit.is_none() => self.next_lifecycle_update_frames(
                bus,
                record,
                *evicted_through,
                *latest,
                recovery_cursor,
            ),
            ReplayUpdate::Audit {
                record,
                evicted_through,
            } if self.subscription.audit.is_some() => {
                self.next_audit_update_frames(bus, record, *evicted_through, recovery_cursor)
            }
            ReplayUpdate::Lifecycle { .. } | ReplayUpdate::Audit { .. } => Vec::new(),
        }
    }

    fn next_lifecycle_frames(
        &mut self,
        bus: &ReplayBus,
        recovery_cursor: Option<String>,
    ) -> Vec<String> {
        let window = bus.replay_window();
        let subscription = Subscription {
            channels: self.subscription.channels.clone(),
            filters: self.subscription.filters.clone(),
            cursor: self.cursor,
            audit: None,
        };
        bus.record_replay_gaps(replay_gap_count(window.evicted_through, &subscription));
        let frames = replay_window_frames(&window, &subscription, recovery_cursor);
        // A gap acknowledges every selected channel's evicted prefix. Advance
        // even when no retained record exists for that channel, otherwise the
        // same gap is emitted on every live notification forever.
        for channel in &self.subscription.channels {
            self.cursor
                .advance(*channel, window.evicted_through.sequence(*channel));
        }
        for record in window.records {
            if self.subscription.channels.contains(&record.replay.channel) {
                self.cursor
                    .advance(record.replay.channel, record.replay.sequence);
            }
        }
        frames
    }

    fn next_lifecycle_update_frames(
        &mut self,
        bus: &ReplayBus,
        record: &ReplayRecord,
        evicted_through: ReplayCursor,
        latest: ReplayCursor,
        recovery_cursor: Option<String>,
    ) -> Vec<String> {
        let subscription = Subscription {
            channels: self.subscription.channels.clone(),
            filters: self.subscription.filters.clone(),
            cursor: self.cursor,
            audit: None,
        };
        let gap_count = replay_gap_count(evicted_through, &subscription);
        bus.record_replay_gaps(gap_count);
        let mut frames = gap_frames(evicted_through, latest, &subscription, recovery_cursor);
        for channel in &self.subscription.channels {
            self.cursor
                .advance(*channel, evicted_through.sequence(*channel));
        }
        if self.subscription.channels.contains(&record.replay.channel)
            && record.replay.sequence > self.cursor.sequence(record.replay.channel)
        {
            if matches_filter(record, &self.subscription) {
                match event_frame(record) {
                    Ok(frame) => frames.push(frame),
                    Err(()) => frames.push(error_frame(cursor_from_replay(record.cursor))),
                }
            }
            self.cursor
                .advance(record.replay.channel, record.replay.sequence);
        }
        frames
    }

    fn next_audit_frames(
        &mut self,
        bus: &ReplayBus,
        recovery_cursor: Option<String>,
    ) -> Vec<String> {
        let sel = self.subscription.audit.as_ref().expect("audit mode");
        let window = bus.audit_replay_window();
        let mut frames = Vec::new();

        if self.audit_cursor.sequence() < window.evicted_through {
            let gap = audit_gap_frame(
                self.audit_cursor.sequence().saturating_add(1),
                window.evicted_through,
                recovery_cursor.clone(),
            );
            if let Ok(frame) = gap {
                frames.push(frame);
            }
            bus.record_replay_gaps(
                window
                    .evicted_through
                    .saturating_sub(self.audit_cursor.sequence()),
            );
            self.audit_cursor.advance(window.evicted_through);
        }

        for record in &window.records {
            if record.sequence <= self.audit_cursor.sequence() {
                continue;
            }
            if !audit_matches_filter(&record.entry.payload, sel) {
                continue;
            }
            match audit_entry_frame(record) {
                Ok(frame) => frames.push(frame),
                Err(()) => {
                    frames.push(format!(
                        "event: stream_error\nid: a1:{}\ndata: {{\"code\":\"invalid_event\"}}\n\n",
                        record.sequence
                    ));
                }
            }
        }

        for record in window.records {
            self.audit_cursor.advance(record.sequence);
        }

        frames
    }

    fn next_audit_update_frames(
        &mut self,
        bus: &ReplayBus,
        record: &AuditReplayRecord,
        evicted_through: u64,
        recovery_cursor: Option<String>,
    ) -> Vec<String> {
        let sel = self.subscription.audit.as_ref().expect("audit mode");
        let mut frames = Vec::new();
        if self.audit_cursor.sequence() < evicted_through {
            bus.record_replay_gaps(evicted_through.saturating_sub(self.audit_cursor.sequence()));
            if let Ok(frame) = audit_gap_frame(
                self.audit_cursor.sequence().saturating_add(1),
                evicted_through,
                recovery_cursor,
            ) {
                frames.push(frame);
            }
            self.audit_cursor.advance(evicted_through);
        }
        if record.sequence > self.audit_cursor.sequence() {
            if audit_matches_filter(&record.entry.payload, sel) {
                match audit_entry_frame(record) {
                    Ok(frame) => frames.push(frame),
                    Err(()) => frames.push(format!(
                        "event: stream_error\nid: a1:{}\ndata: {{\"code\":\"invalid_event\"}}\n\n",
                        record.sequence
                    )),
                }
            }
            self.audit_cursor.advance(record.sequence);
        }
        frames
    }
}

fn replay_gap_count(evicted_through: ReplayCursor, subscription: &Subscription) -> u64 {
    subscription
        .channels
        .iter()
        .filter(|channel| {
            subscription.cursor.sequence(**channel) < evicted_through.sequence(**channel)
        })
        .count() as u64
}

fn replay_window_frames(
    window: &ReplayWindow,
    subscription: &Subscription,
    recovery_cursor: Option<String>,
) -> Vec<String> {
    let mut frames = gap_frames(
        window.evicted_through,
        window.latest,
        subscription,
        recovery_cursor,
    );
    for record in &window.records {
        if !subscription.channels.contains(&record.replay.channel)
            || record.replay.sequence <= subscription.cursor.sequence(record.replay.channel)
            || !matches_filter(record, subscription)
        {
            continue;
        }
        match event_frame(record) {
            Ok(frame) => frames.push(frame),
            Err(()) => frames.push(error_frame(cursor_from_replay(record.cursor))),
        }
    }
    frames
}

fn gap_frames(
    evicted_through: ReplayCursor,
    latest: ReplayCursor,
    subscription: &Subscription,
    recovery_cursor: Option<String>,
) -> Vec<String> {
    subscription
        .channels
        .iter()
        .filter_map(|channel| {
            let requested = subscription.cursor.sequence(*channel);
            let channel_evicted_through = evicted_through.sequence(*channel);
            (requested < channel_evicted_through).then(|| {
                let gap = GapData::new(
                    *channel,
                    requested.saturating_add(1),
                    channel_evicted_through,
                    recovery_cursor.clone(),
                );
                gap_frame(cursor_from_replay(latest), &gap)
                    .expect("bounded replay-gap data fits the SSE frame cap")
            })
        })
        .collect()
}

fn matches_filter(record: &crate::logging::ReplayRecord, subscription: &Subscription) -> bool {
    #[cfg(test)]
    LIFECYCLE_RECORDS_INSPECTED.with(|count| count.set(count.get().saturating_add(1)));
    let Ok(payload) = serde_json::from_str::<serde_json::Value>(&record.entry.payload) else {
        return false;
    };
    let Some(envelope) = payload
        .get("canonical_envelope")
        .and_then(|value| CanonicalEnvelope::from_json_str(&value.to_string()).ok())
    else {
        return false;
    };
    let summary_snapshots = payload
        .get("request_summary_snapshots")
        .cloned()
        .and_then(|value| serde_json::from_value::<RequestSummaryEventSnapshots>(value).ok());
    subscription
        .filters
        .matches(&envelope, summary_snapshots.as_ref())
}

fn audit_matches_filter(payload: &str, sel: &AuditSelection) -> bool {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) else {
        return false;
    };
    if let Some(ref source) = sel.source
        && value.get("source").and_then(|v| v.as_str()) != Some(source.as_str())
    {
        return false;
    }
    if let Some(ref severity) = sel.severity
        && value.get("severity").and_then(|v| v.as_str()) != Some(severity.as_str())
    {
        return false;
    }
    true
}

fn cursor_from_replay(cursor: ReplayCursor) -> Cursor {
    Cursor::from_sequences(
        cursor.sequence(ReplayChannel::Requests),
        cursor.sequence(ReplayChannel::Operations),
        cursor.sequence(ReplayChannel::System),
    )
}

/// Bounded per-connection hand-off between a replay/live producer and the
/// socket writer. A full queue is a deliberate slow-consumer disconnect, not
/// an unbounded allocation or a blocked logging producer.
#[derive(Clone)]
pub(in crate::api::routes::logs) struct ConnectionQueue {
    sender: mpsc::Sender<String>,
    cancelled: Arc<AtomicBool>,
}

pub(super) struct ConnectionReceiver {
    receiver: mpsc::Receiver<String>,
    cancelled: Arc<AtomicBool>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::api::routes::logs) enum QueueError {
    SlowConsumer,
    Cancelled,
}

impl ConnectionQueue {
    pub(super) fn new(capacity: usize) -> (Self, ConnectionReceiver) {
        let (sender, receiver) = mpsc::channel(capacity.max(1));
        let cancelled = Arc::new(AtomicBool::new(false));
        (
            Self {
                sender,
                cancelled: Arc::clone(&cancelled),
            },
            ConnectionReceiver {
                receiver,
                cancelled,
            },
        )
    }

    #[cfg(test)]
    pub(super) fn try_send(&self, frame: String) -> Result<(), QueueError> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(QueueError::Cancelled);
        }
        match self.sender.try_send(frame) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(_)) => Err(QueueError::SlowConsumer),
            Err(mpsc::error::TrySendError::Closed(_)) => Err(QueueError::Cancelled),
        }
    }

    pub(super) async fn send_with_timeout(
        &self,
        frame: String,
        timeout: std::time::Duration,
    ) -> Result<(), QueueError> {
        if self.cancelled.load(Ordering::Acquire) {
            return Err(QueueError::Cancelled);
        }
        match tokio::time::timeout(timeout, self.sender.send(frame)).await {
            Ok(Ok(())) => Ok(()),
            Ok(Err(_)) => Err(QueueError::Cancelled),
            Err(_) => Err(QueueError::SlowConsumer),
        }
    }

    pub(super) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }
}

impl ConnectionReceiver {
    pub(super) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub(super) async fn recv(&mut self) -> Option<String> {
        if self.cancelled.load(Ordering::Acquire) {
            return None;
        }
        self.receiver.recv().await
    }
}

#[cfg(test)]
mod tests;
