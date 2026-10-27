//! Bounded signature verification, independent of the membership lock.

use super::{decode_cache_affinity, decode_node_record, log_malformed_record};
use crate::mesh::Node;
use crate::proto::node::{NodeRecord, SignedCacheAffinity, SignedNodeRecord};
use crate::protocol::InboundGossip;
use iroh::EndpointId;
use mesh_llm_membership::signed_record::{
    CACHE_AFFINITY_RECORD, NODE_RECORD, RecordHeader, VerifiedRecord,
};
use mesh_llm_routing::cache_inventory::CacheAffinityAdvertisement;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

const WINDOW: Duration = Duration::from_secs(10);
const MAX_RECORDS: usize = 4096;
const MAX_BYTES: usize = 16 * 1024 * 1024;
const MAX_SENDERS: usize = 2048;

struct Usage {
    since: Instant,
    records: usize,
    bytes: usize,
}

/// Limits both sustained work from one authenticated sender and aggregate
/// blocking-pool concurrency. Full accounting tables fail closed; churn cannot
/// evict another sender's still-active quota and reset its allowance.
pub(crate) struct VerificationGate {
    slots: Arc<Semaphore>,
    senders: Mutex<HashMap<EndpointId, Usage>>,
}

impl Default for VerificationGate {
    fn default() -> Self {
        Self {
            slots: Arc::new(Semaphore::new(2)),
            senders: Mutex::new(HashMap::new()),
        }
    }
}

impl VerificationGate {
    pub(super) fn reserve(
        &self,
        sender: EndpointId,
        records: usize,
        bytes: usize,
        now: Instant,
    ) -> Option<OwnedSemaphorePermit> {
        if records > MAX_RECORDS || bytes > MAX_BYTES {
            return None;
        }
        let permit = self.slots.clone().try_acquire_owned().ok()?;
        let mut senders = self.senders.lock().unwrap_or_else(|e| e.into_inner());
        senders.retain(|_, usage| now.saturating_duration_since(usage.since) < WINDOW);
        if !senders.contains_key(&sender) && senders.len() >= MAX_SENDERS {
            return None;
        }
        let usage = senders.entry(sender).or_insert(Usage {
            since: now,
            records: 0,
            bytes: 0,
        });
        if records > MAX_RECORDS - usage.records || bytes > MAX_BYTES - usage.bytes {
            return None;
        }
        usage.records += records;
        usage.bytes += bytes;
        Some(permit)
    }
}

pub(in crate::mesh) struct PreparedNode {
    pub record: VerifiedRecord,
    pub body: Box<NodeRecord>,
    pub hop: Option<crate::proto::node::HopObservation>,
}

#[derive(Default)]
pub(in crate::mesh) struct PreparedRecords {
    pub nodes: Vec<PreparedNode>,
    pub caches: Vec<(VerifiedRecord, CacheAffinityAdvertisement)>,
    // Presence only controls compatibility-mode downgrade cleanup. It never
    // authorizes a record or supplies peer data before signature verification.
    pub sender_node_record_present: bool,
    pub sender_cache_record_present: bool,
}

impl Node {
    pub(in crate::mesh) async fn prepare_gossip_records(
        &self,
        remote: EndpointId,
        inbound: &mut InboundGossip,
    ) -> Option<PreparedRecords> {
        let records = inbound.signed_records.len() + inbound.signed_cache_affinity.len();
        if records == 0 {
            return Some(PreparedRecords::default());
        }
        let bytes = inbound
            .signed_records
            .iter()
            .map(|r| r.signed.len())
            .chain(inbound.signed_cache_affinity.iter().map(|r| r.signed.len()))
            .fold(0usize, usize::saturating_add);
        let Some(permit) = self
            .gossip_verification
            .reserve(remote, records, bytes, Instant::now())
        else {
            tracing::debug!(peer = %remote.fmt_short(), records, bytes, "gossip signature work budget exhausted");
            return None;
        };
        let nodes = std::mem::take(&mut inbound.signed_records);
        let caches = std::mem::take(&mut inbound.signed_cache_affinity);
        // The permit lives in the blocking job: cancelling the async caller
        // cannot admit replacement work while this job is still running.
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            verify_records(remote, nodes, caches)
        })
        .await
        .map_err(|error| {
            tracing::warn!(%error, "gossip signature verification task failed");
        })
        .ok()
    }
}

fn verify_records(
    remote: EndpointId,
    nodes: Vec<SignedNodeRecord>,
    caches: Vec<SignedCacheAffinity>,
) -> PreparedRecords {
    let mut prepared = PreparedRecords::default();
    for wire in nodes {
        prepared.sender_node_record_present |= RecordHeader::parse(&NODE_RECORD, &wire.signed)
            .is_ok_and(|header| header.endpoint_id == remote);
        let result = VerifiedRecord::verify(NODE_RECORD, &wire.signed, &wire.signature)
            .and_then(|record| decode_node_record(record.body()).map(|body| (record, body)));
        match result {
            Ok((record, body)) => prepared.nodes.push(PreparedNode {
                record,
                body,
                hop: wire.hop,
            }),
            Err(error) => log_malformed_record("node record", remote, error),
        }
    }
    for wire in caches {
        prepared.sender_cache_record_present |=
            RecordHeader::parse(&CACHE_AFFINITY_RECORD, &wire.signed)
                .is_ok_and(|header| header.endpoint_id == remote);
        let result = VerifiedRecord::verify(CACHE_AFFINITY_RECORD, &wire.signed, &wire.signature)
            .and_then(|record| decode_cache_affinity(record.body()).map(|body| (record, body)));
        match result {
            Ok(record) => prepared.caches.push(record),
            Err(error) => log_malformed_record("cache affinity", remote, error),
        }
    }
    prepared
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sender(seed: u8) -> EndpointId {
        iroh::SecretKey::from_bytes(&[seed; 32]).public()
    }

    #[test]
    fn quotas_are_per_sender_and_reset_only_after_the_window() {
        let gate = VerificationGate::default();
        let now = Instant::now();
        drop(gate.reserve(sender(1), MAX_RECORDS, 1, now).unwrap());
        assert!(gate.reserve(sender(1), 1, 1, now + WINDOW / 2).is_none());
        drop(gate.reserve(sender(2), 1, MAX_BYTES, now).unwrap());
        assert!(gate.reserve(sender(2), 0, 1, now).is_none());
        assert!(gate.reserve(sender(1), 1, 1, now + WINDOW).is_some());
    }

    #[test]
    fn concurrent_jobs_keep_their_slots_until_the_worker_releases_them() {
        let gate = VerificationGate::default();
        let now = Instant::now();
        let first = gate.reserve(sender(1), 1, 1, now).unwrap();
        let second = gate.reserve(sender(2), 1, 1, now).unwrap();
        assert!(gate.reserve(sender(3), 1, 1, now).is_none());
        drop(first);
        assert!(gate.reserve(sender(3), MAX_RECORDS, 1, now).is_some());
        drop(second);
    }

    #[test]
    fn oversized_batches_do_not_consume_a_slot_or_reset_existing_usage() {
        let gate = VerificationGate::default();
        let now = Instant::now();
        assert!(gate.reserve(sender(1), MAX_RECORDS + 1, 0, now).is_none());
        assert!(gate.reserve(sender(1), 0, MAX_BYTES + 1, now).is_none());
        drop(
            gate.reserve(sender(1), MAX_RECORDS, MAX_BYTES, now)
                .unwrap(),
        );
        assert!(gate.reserve(sender(1), 1, 1, now).is_none());
    }

    #[test]
    fn sender_churn_cannot_evict_an_active_quota() {
        let gate = VerificationGate::default();
        let now = Instant::now();
        for index in 0..MAX_SENDERS {
            let mut seed = [0; 32];
            seed[..8].copy_from_slice(&(index as u64).to_be_bytes());
            let id = iroh::SecretKey::from_bytes(&seed).public();
            drop(gate.reserve(id, 1, 1, now).unwrap());
        }
        assert!(gate.reserve(sender(255), 1, 1, now).is_none());
        assert!(gate.reserve(sender(255), 1, 1, now + WINDOW).is_some());
    }
}
