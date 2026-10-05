use iroh::EndpointId;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tokio::sync::watch;

const MAX_CACHED_LIFECYCLE_RESPONSES: usize = 256;
const MAX_CACHED_LIFECYCLE_RESPONSES_PER_REQUESTER: usize = 64;

type OwnerControlEnvelope = crate::proto::node::OwnerControlEnvelope;
type PendingLifecycleSender = watch::Sender<Option<OwnerControlEnvelope>>;

#[derive(Clone, Default)]
pub(crate) struct OwnerLifecycleResponseCache {
    entries: Arc<Mutex<VecDeque<OwnerLifecycleResponseEntry>>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OwnerLifecycleCacheKey {
    requester: EndpointId,
    request_id: u64,
}

enum OwnerLifecycleResponseEntry {
    Pending {
        key: OwnerLifecycleCacheKey,
        tx: PendingLifecycleSender,
        expires_at: Instant,
    },
    Ready {
        key: OwnerLifecycleCacheKey,
        envelope: Box<OwnerControlEnvelope>,
        expires_at: Instant,
    },
}

impl OwnerLifecycleResponseEntry {
    fn key(&self) -> OwnerLifecycleCacheKey {
        match self {
            Self::Pending { key, .. } | Self::Ready { key, .. } => *key,
        }
    }

    fn is_live_ready_at(&self, now: Instant) -> bool {
        match self {
            Self::Pending { expires_at, .. } | Self::Ready { expires_at, .. } => *expires_at > now,
        }
    }
}

pub(crate) enum OwnerLifecycleResponseReservation {
    Leader(OwnerLifecycleResponseLeader),
    Follower(OwnerLifecycleResponseFollower),
    Ready(OwnerControlEnvelope),
}

pub(crate) struct OwnerLifecycleResponseLeader {
    cache: OwnerLifecycleResponseCache,
    key: OwnerLifecycleCacheKey,
    reservation: PendingLifecycleSender,
    fallback: OwnerControlEnvelope,
    published: bool,
}

pub(crate) struct OwnerLifecycleResponseFollower {
    rx: watch::Receiver<Option<OwnerControlEnvelope>>,
    fallback: OwnerControlEnvelope,
}

impl OwnerLifecycleResponseCache {
    #[cfg(test)]
    pub(crate) fn reserve(
        &self,
        requester: EndpointId,
        request_id: u64,
    ) -> OwnerLifecycleResponseReservation {
        self.reserve_with_fallback(
            requester,
            request_id,
            lifecycle_unavailable_envelope(request_id),
            Duration::from_secs(60),
        )
    }

    pub(crate) fn reserve_with_fallback(
        &self,
        requester: EndpointId,
        request_id: u64,
        fallback: OwnerControlEnvelope,
        cache_duration: Duration,
    ) -> OwnerLifecycleResponseReservation {
        let key = OwnerLifecycleCacheKey {
            requester,
            request_id,
        };
        let mut entries = self.lock_entries();
        let now = Instant::now();
        entries.retain(|entry| entry.is_live_ready_at(now));

        if let Some(entry) = entries.iter().find(|entry| entry.key() == key) {
            return match entry {
                OwnerLifecycleResponseEntry::Ready { envelope, .. } => {
                    OwnerLifecycleResponseReservation::Ready(envelope.as_ref().clone())
                }
                OwnerLifecycleResponseEntry::Pending { tx, .. } => {
                    let rx = tx.subscribe();
                    if let Some(envelope) = rx.borrow().clone() {
                        OwnerLifecycleResponseReservation::Ready(envelope)
                    } else {
                        OwnerLifecycleResponseReservation::Follower(
                            OwnerLifecycleResponseFollower { rx, fallback },
                        )
                    }
                }
            };
        }

        if !enforce_lifecycle_cache_bound(&mut entries, requester) {
            return OwnerLifecycleResponseReservation::Ready(fallback);
        }

        let (tx, _rx) = watch::channel(None);
        entries.push_back(OwnerLifecycleResponseEntry::Pending {
            key,
            tx: tx.clone(),
            expires_at: now + cache_duration,
        });

        OwnerLifecycleResponseReservation::Leader(OwnerLifecycleResponseLeader {
            cache: self.clone(),
            key,
            reservation: tx,
            fallback,
            published: false,
        })
    }

    fn complete_key(
        &self,
        key: OwnerLifecycleCacheKey,
        reservation: &PendingLifecycleSender,
        envelope: OwnerControlEnvelope,
    ) -> bool {
        let mut entries = self.lock_entries();
        let Some(entry) = entries.iter_mut().find(|entry| entry.key() == key) else {
            return false;
        };
        let (notify, expires_at) = match entry {
            OwnerLifecycleResponseEntry::Pending { tx, expires_at, .. }
                if tx.same_channel(reservation) =>
            {
                (tx.clone(), *expires_at)
            }
            OwnerLifecycleResponseEntry::Pending { .. } => return false,
            OwnerLifecycleResponseEntry::Ready { .. } => return false,
        };
        *entry = OwnerLifecycleResponseEntry::Ready {
            key,
            envelope: Box::new(envelope.clone()),
            expires_at,
        };
        drop(entries);
        let _ = notify.send(Some(envelope));
        true
    }

    fn lock_entries(&self) -> MutexGuard<'_, VecDeque<OwnerLifecycleResponseEntry>> {
        match self.entries.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

impl OwnerLifecycleResponseLeader {
    pub(crate) fn publish(mut self, envelope: OwnerControlEnvelope) {
        let _ = self
            .cache
            .complete_key(self.key, &self.reservation, envelope);
        self.published = true;
    }
}

impl Drop for OwnerLifecycleResponseLeader {
    fn drop(&mut self) {
        if !self.published {
            let _ = self
                .cache
                .complete_key(self.key, &self.reservation, self.fallback.clone());
        }
    }
}

impl OwnerLifecycleResponseFollower {
    pub(crate) async fn wait(mut self) -> OwnerControlEnvelope {
        loop {
            if let Some(envelope) = self.rx.borrow().clone() {
                return envelope;
            }
            if self.rx.changed().await.is_err() {
                return self.fallback;
            }
        }
    }
}

fn enforce_lifecycle_cache_bound(
    entries: &mut VecDeque<OwnerLifecycleResponseEntry>,
    requester: EndpointId,
) -> bool {
    while entries
        .iter()
        .filter(|entry| entry.key().requester == requester)
        .count()
        >= MAX_CACHED_LIFECYCLE_RESPONSES_PER_REQUESTER
    {
        let Some(ready_index) = entries.iter().position(|entry| {
            entry.key().requester == requester
                && matches!(entry, OwnerLifecycleResponseEntry::Ready { .. })
        }) else {
            return false;
        };
        entries.remove(ready_index);
    }
    while entries.len() >= MAX_CACHED_LIFECYCLE_RESPONSES {
        let Some(ready_index) = entries
            .iter()
            .position(|entry| matches!(entry, OwnerLifecycleResponseEntry::Ready { .. }))
        else {
            return false;
        };
        entries.remove(ready_index);
    }
    true
}

#[cfg(test)]
fn lifecycle_unavailable_envelope(request_id: u64) -> OwnerControlEnvelope {
    crate::proto::node::OwnerControlEnvelope {
        r#gen: crate::protocol::NODE_PROTOCOL_GENERATION,
        handshake: None,
        request: None,
        response: None,
        error: Some(crate::proto::node::OwnerControlError {
            code: crate::proto::node::OwnerControlErrorCode::ControlUnavailable as i32,
            message: "owner-control lifecycle request did not complete".to_string(),
            request_id: Some(request_id),
            current_revision: None,
        }),
    }
}

#[cfg(test)]
mod tests;
