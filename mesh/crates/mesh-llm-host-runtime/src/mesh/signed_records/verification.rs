//! Bounded signature verification, independent of the membership lock.

use super::verification_cache::VerificationCache;
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
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::Instant;

const WINDOW: Duration = Duration::from_secs(10);
const MAX_RECORDS: usize = crate::protocol::MAX_GOSSIP_RECORDS;
const MAX_BYTES: usize = 16 * 1024 * 1024;
const MAX_SENDERS: usize = 2048;
const SLOT_WAIT: Duration = Duration::from_millis(250);

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
    verified: VerificationCache,
}

impl Default for VerificationGate {
    fn default() -> Self {
        Self {
            slots: Arc::new(Semaphore::new(2)),
            senders: Mutex::new(HashMap::new()),
            verified: VerificationCache::default(),
        }
    }
}

impl VerificationGate {
    /// Wait briefly for competing exchanges, then charge only admitted work.
    /// Cancellation or timeout while queued must not consume a sender's quota.
    pub(super) async fn reserve(
        &self,
        sender: EndpointId,
        records: usize,
        bytes: usize,
    ) -> Option<OwnedSemaphorePermit> {
        if records > MAX_RECORDS || bytes > MAX_BYTES {
            return None;
        }
        let permit = tokio::time::timeout(SLOT_WAIT, self.slots.clone().acquire_owned())
            .await
            .ok()?
            .ok()?;
        let now = Instant::now();
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
            .reserve(remote, records, bytes)
            .await
        else {
            tracing::debug!(peer = %remote.fmt_short(), records, bytes, "gossip signature work budget exhausted");
            return None;
        };
        let nodes = std::mem::take(&mut inbound.signed_records);
        let caches = std::mem::take(&mut inbound.signed_cache_affinity);
        let gate = self.gossip_verification.clone();
        // The permit lives in the blocking job: cancelling the async caller
        // cannot admit replacement work while this job is still running.
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            verify_records(&gate.verified, remote, nodes, caches)
        })
        .await
        .map_err(|error| {
            tracing::warn!(%error, "gossip signature verification task failed");
        })
        .ok()
    }
}

fn verify_records(
    verified: &VerificationCache,
    remote: EndpointId,
    nodes: Vec<SignedNodeRecord>,
    caches: Vec<SignedCacheAffinity>,
) -> PreparedRecords {
    let mut prepared = PreparedRecords::default();
    for wire in nodes {
        prepared.sender_node_record_present |= RecordHeader::parse(&NODE_RECORD, &wire.signed)
            .is_ok_and(|header| header.endpoint_id == remote);
        let result = verified
            .verify(NODE_RECORD, &wire.signed, &wire.signature)
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
        let result = verified
            .verify(CACHE_AFFINITY_RECORD, &wire.signed, &wire.signature)
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
    fn repeated_batches_reuse_both_signatures_but_keep_current_hop_data() {
        use prost::Message;
        let cache = VerificationCache::default();
        let key = ed25519_dalek::SigningKey::from_bytes(&[1; 32]);
        let now = mesh_llm_membership::requirements::current_time_unix_ms();
        let node = VerifiedRecord::sign(
            NODE_RECORD,
            &key,
            1,
            now,
            &NodeRecord::default().encode_to_vec(),
        );
        let affinity = crate::proto::node::CacheAffinityAdvertisement {
            salt: vec![3; 32],
            epoch: 1,
            generated_at_unix_ms: now,
            ttl_ms: 120_000,
            entries: Vec::new(),
        };
        let affinity = VerifiedRecord::sign(
            CACHE_AFFINITY_RECORD,
            &key,
            1,
            now,
            &affinity.encode_to_vec(),
        );
        for latency in [10, 20] {
            let hop = crate::proto::node::HopObservation {
                latency_ms: Some(latency),
                ..Default::default()
            };
            let prepared = verify_records(
                &cache,
                sender(1),
                vec![SignedNodeRecord {
                    signed: node.signed_bytes().to_vec(),
                    signature: node.signature().to_vec(),
                    hop: Some(hop.clone()),
                }],
                vec![SignedCacheAffinity {
                    signed: affinity.signed_bytes().to_vec(),
                    signature: affinity.signature().to_vec(),
                }],
            );
            assert_eq!(prepared.nodes.len(), 1);
            assert_eq!(prepared.nodes[0].record, node);
            assert_eq!(prepared.nodes[0].hop, Some(hop));
            assert_eq!(prepared.caches.len(), 1);
            assert_eq!(prepared.caches[0].0, affinity);
            assert!(prepared.sender_node_record_present && prepared.sender_cache_record_present);
        }
        assert_eq!(
            cache
                .verifications
                .load(std::sync::atomic::Ordering::Relaxed),
            2
        );
    }

    #[test]
    fn cached_signature_does_not_bypass_body_validation() {
        let cache = VerificationCache::default();
        let record = VerifiedRecord::sign(
            NODE_RECORD,
            &ed25519_dalek::SigningKey::from_bytes(&[1; 32]),
            1,
            1_000_000,
            &[0xff],
        );
        for _ in 0..2 {
            let prepared = verify_records(
                &cache,
                sender(1),
                vec![SignedNodeRecord {
                    signed: record.signed_bytes().to_vec(),
                    signature: record.signature().to_vec(),
                    hop: None,
                }],
                Vec::new(),
            );
            assert!(prepared.nodes.is_empty());
        }
        assert_eq!(
            cache
                .verifications
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
    }

    #[tokio::test(start_paused = true)]
    async fn quotas_are_per_sender_and_reset_only_after_the_window() {
        let gate = VerificationGate::default();
        drop(gate.reserve(sender(1), MAX_RECORDS, 1).await.unwrap());
        tokio::time::advance(WINDOW / 2).await;
        assert!(gate.reserve(sender(1), 1, 1).await.is_none());
        drop(gate.reserve(sender(2), 1, MAX_BYTES).await.unwrap());
        assert!(gate.reserve(sender(2), 0, 1).await.is_none());
        tokio::time::advance(WINDOW / 2).await;
        assert!(gate.reserve(sender(1), 1, 1).await.is_some());
    }

    #[tokio::test(start_paused = true)]
    async fn concurrent_jobs_wait_for_a_worker_to_release_its_slot() {
        let gate = VerificationGate::default();
        let first = gate.reserve(sender(1), 1, 1).await.unwrap();
        let second = gate.reserve(sender(2), 1, 1).await.unwrap();
        let pending = gate.reserve(sender(3), MAX_RECORDS, MAX_BYTES);
        tokio::pin!(pending);
        assert!(futures_util::poll!(&mut pending).is_pending());
        tokio::time::advance(SLOT_WAIT / 2).await;
        drop(first);
        assert!(pending.await.is_some());
        drop(second);
        assert!(gate.reserve(sender(3), 1, 0).await.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn slot_timeout_does_not_consume_quota() {
        let gate = VerificationGate::default();
        let workers = gate.slots.clone().acquire_many_owned(2).await.unwrap();
        let started = Instant::now();
        assert!(
            gate.reserve(sender(1), MAX_RECORDS, MAX_BYTES)
                .await
                .is_none()
        );
        assert!(started.elapsed() >= SLOT_WAIT);
        drop(workers);
        assert!(
            gate.reserve(sender(1), MAX_RECORDS, MAX_BYTES)
                .await
                .is_some()
        );
    }

    #[tokio::test(start_paused = true)]
    async fn cancelled_slot_wait_does_not_consume_quota_or_leak_a_permit() {
        let gate = VerificationGate::default();
        let workers = gate.slots.clone().acquire_many_owned(2).await.unwrap();
        let mut pending = Box::pin(gate.reserve(sender(1), MAX_RECORDS, MAX_BYTES));
        assert!(futures_util::poll!(&mut pending).is_pending());
        drop(pending);
        drop(workers);
        drop(
            gate.reserve(sender(1), MAX_RECORDS, MAX_BYTES)
                .await
                .unwrap(),
        );
        assert_eq!(gate.slots.available_permits(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn quota_window_is_evaluated_after_the_slot_wait() {
        let gate = VerificationGate::default();
        drop(
            gate.reserve(sender(1), MAX_RECORDS, MAX_BYTES)
                .await
                .unwrap(),
        );
        tokio::time::advance(WINDOW - SLOT_WAIT / 2).await;
        let workers = gate.slots.clone().acquire_many_owned(2).await.unwrap();
        let pending = gate.reserve(sender(1), MAX_RECORDS, MAX_BYTES);
        tokio::pin!(pending);
        assert!(futures_util::poll!(&mut pending).is_pending());
        tokio::time::advance(SLOT_WAIT / 2).await;
        drop(workers);
        assert!(pending.await.is_some());
    }

    #[tokio::test]
    async fn contended_verification_preserves_the_signed_payload() {
        let node = Node::new_for_tests(crate::mesh::NodeRole::Worker)
            .await
            .unwrap();
        let record = VerifiedRecord::sign(
            NODE_RECORD,
            &ed25519_dalek::SigningKey::from_bytes(&[1; 32]),
            1,
            mesh_llm_membership::requirements::current_time_unix_ms(),
            &prost::Message::encode_to_vec(&NodeRecord::default()),
        );
        let mut inbound = InboundGossip {
            announcements: Vec::new(),
            signed_records: vec![SignedNodeRecord {
                signed: record.signed_bytes().to_vec(),
                signature: record.signature().to_vec(),
                hop: None,
            }],
            signed_cache_affinity: Vec::new(),
        };
        let workers = node
            .gossip_verification
            .slots
            .clone()
            .acquire_many_owned(2)
            .await
            .unwrap();
        let pending = node.prepare_gossip_records(sender(2), &mut inbound);
        tokio::pin!(pending);
        assert!(futures_util::poll!(&mut pending).is_pending());
        drop(workers);
        let prepared = pending
            .await
            .expect("brief contention must not drop records");
        assert_eq!(prepared.nodes.len(), 1);
        assert_eq!(
            prepared.nodes[0].record.signed_bytes(),
            record.signed_bytes()
        );
        node.endpoint.close().await;
    }

    #[tokio::test(start_paused = true)]
    async fn oversized_batches_do_not_consume_a_slot_or_reset_existing_usage() {
        let gate = VerificationGate::default();
        assert!(gate.reserve(sender(1), MAX_RECORDS + 1, 0).await.is_none());
        assert!(gate.reserve(sender(1), 0, MAX_BYTES + 1).await.is_none());
        drop(
            gate.reserve(sender(1), MAX_RECORDS, MAX_BYTES)
                .await
                .unwrap(),
        );
        assert!(gate.reserve(sender(1), 1, 1).await.is_none());
    }

    #[tokio::test(start_paused = true)]
    async fn sender_churn_cannot_evict_an_active_quota() {
        let gate = VerificationGate::default();
        for index in 0..MAX_SENDERS {
            let mut seed = [0; 32];
            seed[..8].copy_from_slice(&(index as u64).to_be_bytes());
            let id = iroh::SecretKey::from_bytes(&seed).public();
            drop(gate.reserve(id, 1, 1).await.unwrap());
        }
        assert!(gate.reserve(sender(255), 1, 1).await.is_none());
        tokio::time::advance(WINDOW).await;
        assert!(gate.reserve(sender(255), 1, 1).await.is_some());
    }
}
