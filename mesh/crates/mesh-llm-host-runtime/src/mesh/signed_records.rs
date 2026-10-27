//! Signed records in gossip: signing this node's records, admitting the
//! records peers relay, and merging them with unsigned announcements.
//!
//! Two kinds are gossiped mesh-wide: the node record (the node's
//! self-description) and its cache-affinity evidence. Both are relayed
//! byte-for-byte. A signed genesis policy can require origin-signed relayed
//! records. Existing meshes retain the legacy compatibility path until their
//! operator explicitly creates a strict mesh.

use super::{Node, PeerAnnouncement};
use crate::mesh::requirements::current_time_unix_ms;
use crate::proto::node::{
    CacheAffinityAdvertisement as WireCacheAffinity, NodeRecord, SignedCacheAffinity,
    SignedNodeRecord,
};
use crate::protocol::node_record::{
    hop_observation_from_ann, node_record_from_wire_ann, node_record_to_local,
};
use crate::protocol::{
    InboundGossip, OutboundGossip, local_ann_to_proto_ann, proto_cache_affinity_to_local,
    sanitize_cache_affinity_for_ann,
};
use iroh::{EndpointAddr, EndpointId};
use mesh_llm_membership::signed_record::{
    HeldRecords, RecordAdmission, RecordError, RecordSkip, VerifiedRecord,
};
use mesh_llm_routing::cache_inventory::CacheAffinityAdvertisement;
use prost::Message;
use std::collections::HashMap;

type Announcement = (EndpointAddr, PeerAnnouncement);

mod verification;
pub(crate) use verification::VerificationGate;

impl Node {
    pub(super) async fn collect_outbound_gossip(&self) -> OutboundGossip {
        let mut announcements = self.collect_announcements().await;
        let (signed_records, signed_cache_affinity) =
            self.collect_signed_records(&announcements).await;
        if self.requires_signed_gossip().await {
            // The direct entry retains merged demand, admission proof and
            // plugin keys. Third-party descriptions travel only as signed bytes.
            announcements.retain(|ann| ann.addr.id == self.id());
        }
        OutboundGossip {
            announcements,
            signed_records,
            signed_cache_affinity,
        }
    }

    /// This node's own records plus every held record for a peer we are
    /// rebroadcasting. Each relayed node record carries our latency view of
    /// that peer as unsigned hop data.
    async fn collect_signed_records(
        &self,
        announcements: &[PeerAnnouncement],
    ) -> (Vec<SignedNodeRecord>, Vec<SignedCacheAffinity>) {
        let local_id = self.endpoint.id();
        let now = current_time_unix_ms();
        let local_wire = announcements
            .iter()
            .find(|ann| ann.addr.id == local_id)
            .map(|ann| (local_ann_to_proto_ann(ann), &ann.addr));
        let node_body = local_wire
            .as_ref()
            .map(|(wire, addr)| node_record_from_wire_ann(wire, addr).encode_to_vec());
        let cache_body = local_wire
            .as_ref()
            .and_then(|(wire, _)| wire.cache_affinity.as_ref())
            .map(Message::encode_to_vec);
        let signing_key =
            ed25519_dalek::SigningKey::from_bytes(&self.endpoint_secret_key.to_bytes());
        let mut state = self.state.lock().await;
        let mut node_records = Vec::with_capacity(announcements.len());
        let mut cache_records = Vec::with_capacity(announcements.len());
        if let Some(body) = node_body {
            let own = state.node_records.refresh_local(&signing_key, &body, now);
            node_records.push(node_record_wire(own, None));
        }
        if let Some(body) = cache_body {
            let own = state
                .cache_affinity_records
                .refresh_local(&signing_key, &body, now);
            cache_records.push(cache_affinity_wire(own));
        }
        for ann in announcements.iter().filter(|ann| ann.addr.id != local_id) {
            if let Some(record) = state.node_records.fresh(&ann.addr.id, now) {
                node_records.push(node_record_wire(
                    record,
                    Some(hop_observation_from_ann(ann)),
                ));
            }
            if let Some(record) = state.cache_affinity_records.fresh(&ann.addr.id, now) {
                cache_records.push(cache_affinity_wire(record));
            }
        }
        (node_records, cache_records)
    }

    pub(super) async fn requires_signed_gossip(&self) -> bool {
        self.requirement_mesh_state.lock().await.as_ref().map_or(
            self.local_mesh_requirements.require_signed_gossip,
            |state| state.policy.requirements.require_signed_gossip,
        )
    }

    /// Verify off the membership lock and commit against the current sequence.
    /// Strict meshes accept unsigned data only from the authenticated origin.
    pub(super) async fn resolve_inbound_gossip(
        &self,
        remote: EndpointId,
        mut inbound: InboundGossip,
    ) -> Vec<Announcement> {
        let strict = self.requires_signed_gossip().await;
        let record_count = inbound.signed_records.len();
        let cache_count = inbound.signed_cache_affinity.len();
        let prepared = self.prepare_gossip_records(remote, &mut inbound).await;
        let verification_completed = prepared.is_some();
        let prepared = prepared.unwrap_or_default();
        let local_id = self.endpoint.id();
        let (mut node_records, cache_affinity) = {
            let mut state = self.state.lock().await;
            let now = current_time_unix_ms();
            let node_records = admit_node_records(
                &mut state.node_records,
                local_id,
                remote,
                prepared.nodes,
                now,
            );
            let cache_affinity = admit_cache_affinity(
                &mut state.cache_affinity_records,
                local_id,
                remote,
                prepared.caches,
                now,
            );
            // A sender that no longer signs its own records (for example, it
            // was downgraded) is described only by its unsigned entry, so
            // stop relaying the records we hold for it.
            if !strict && verification_completed && !prepared.sender_node_record_present {
                state.node_records.remove(&remote);
            }
            if !strict && verification_completed && !prepared.sender_cache_record_present {
                state.cache_affinity_records.remove(&remote);
            }
            (node_records, cache_affinity)
        };
        let unsigned_received = inbound.announcements.len();
        let mut announcements = merge_unsigned(
            remote,
            strict,
            inbound.announcements,
            &mut node_records.replacements,
        );
        let unsigned_applied = announcements.len();
        let relayed_records_applied = node_records.replacements.len();
        announcements.extend(node_records.replacements.into_values());
        apply_signed_cache_affinity(&mut announcements, &cache_affinity.advertisements);
        tracing::debug!(
            sender = %remote.fmt_short(),
            records = record_count,
            relayed_records_applied,
            cache_affinity_records = cache_count,
            cache_affinity_applied = cache_affinity.advertisements.len(),
            unsigned_received,
            unsigned_applied,
            "gossip: resolved signed records"
        );
        announcements
    }
}

fn node_record_wire(
    record: &VerifiedRecord,
    hop: Option<crate::proto::node::HopObservation>,
) -> SignedNodeRecord {
    SignedNodeRecord {
        signed: record.signed_bytes().to_vec(),
        signature: record.signature().to_vec(),
        hop,
    }
}

fn cache_affinity_wire(record: &VerifiedRecord) -> SignedCacheAffinity {
    SignedCacheAffinity {
        signed: record.signed_bytes().to_vec(),
        signature: record.signature().to_vec(),
    }
}

/// Keeps each unsigned entry unless an accepted record replaces it. A
/// replaced entry still lends its cache affinity to the record-derived
/// announcement, for senders that do not sign cache affinity.
fn merge_unsigned(
    remote: EndpointId,
    strict: bool,
    unsigned: Vec<Announcement>,
    replacements: &mut HashMap<EndpointId, Announcement>,
) -> Vec<Announcement> {
    let mut kept = Vec::with_capacity(unsigned.len());
    for (addr, ann) in unsigned {
        if strict && addr.id != remote {
            continue;
        }
        match replacements.get_mut(&addr.id) {
            Some((_, from_record)) if addr.id != remote => {
                from_record.cache_affinity = ann.cache_affinity;
            }
            _ => kept.push((addr, ann)),
        }
    }
    kept
}

/// Signed cache affinity replaces whatever unsigned copy an announcement
/// carries, then gets the same routable-model filter as unsigned evidence.
fn apply_signed_cache_affinity(
    announcements: &mut [Announcement],
    signed: &HashMap<EndpointId, CacheAffinityAdvertisement>,
) {
    for (addr, ann) in announcements {
        if let Some(advertisement) = signed.get(&addr.id) {
            ann.cache_affinity = Some(advertisement.clone());
            ann.cache_affinity = sanitize_cache_affinity_for_ann(ann);
        }
    }
}

struct AdmittedNodeRecords {
    /// Announcements rebuilt from accepted records about nodes other than
    /// the sender. Each replaces the frame's unsigned entry for that node.
    replacements: HashMap<EndpointId, Announcement>,
}

fn admit_node_records(
    held: &mut HeldRecords,
    local_id: EndpointId,
    remote: EndpointId,
    records: Vec<verification::PreparedNode>,
    now: u64,
) -> AdmittedNodeRecords {
    let mut admitted = AdmittedNodeRecords {
        replacements: HashMap::new(),
    };
    for wire in records {
        let admission = held.admit_verified(local_id, remote, wire.record, wire.body, now);
        match admission {
            RecordAdmission::Accepted { record, body } => {
                let id = record.endpoint_id();
                if id != remote
                    && let Some(entry) =
                        relayed_record_announcement(remote, id, &body, wire.hop.as_ref())
                {
                    admitted.replacements.insert(id, entry);
                }
            }
            RecordAdmission::Skipped {
                endpoint_id,
                reason,
            } => {
                log_skipped_record("node record", remote, endpoint_id, reason);
            }
        }
    }
    admitted
}

struct AdmittedCacheAffinity {
    advertisements: HashMap<EndpointId, CacheAffinityAdvertisement>,
}

fn admit_cache_affinity(
    held: &mut HeldRecords,
    local_id: EndpointId,
    remote: EndpointId,
    records: Vec<(VerifiedRecord, CacheAffinityAdvertisement)>,
    now: u64,
) -> AdmittedCacheAffinity {
    let mut admitted = AdmittedCacheAffinity {
        advertisements: HashMap::new(),
    };
    for (record, body) in records {
        let admission = held.admit_verified(local_id, remote, record, body, now);
        match admission {
            RecordAdmission::Accepted { record, body } => {
                let id = record.endpoint_id();
                admitted.advertisements.insert(id, body);
            }
            RecordAdmission::Skipped {
                endpoint_id,
                reason,
            } => {
                log_skipped_record("cache affinity", remote, endpoint_id, reason);
            }
        }
    }
    admitted
}

fn decode_node_record(body: &[u8]) -> Result<Box<NodeRecord>, RecordError> {
    let record = NodeRecord::decode(body).map_err(|_| RecordError::Decode)?;
    mesh_llm_protocol::protocol::validate_node_record(&record).map_err(|_| RecordError::Decode)?;
    Ok(Box::new(record))
}

/// Applies the same bounds and freshness checks as unsigned cache affinity.
fn decode_cache_affinity(body: &[u8]) -> Result<CacheAffinityAdvertisement, RecordError> {
    let advertisement = WireCacheAffinity::decode(body).map_err(|_| RecordError::Decode)?;
    proto_cache_affinity_to_local(&advertisement).ok_or(RecordError::Decode)
}

fn relayed_record_announcement(
    remote: EndpointId,
    id: EndpointId,
    body: &NodeRecord,
    hop: Option<&crate::proto::node::HopObservation>,
) -> Option<Announcement> {
    let entry = node_record_to_local(id, body, hop);
    if entry.is_none() {
        tracing::debug!(
            peer = %id.fmt_short(),
            relay = %remote.fmt_short(),
            "gossip: signed node record could not be converted"
        );
    }
    entry
}

fn log_malformed_record(kind: &'static str, remote: EndpointId, error: RecordError) {
    tracing::debug!(
        relay = %remote.fmt_short(),
        kind,
        %error,
        "gossip: ignoring malformed signed record"
    );
}

fn log_skipped_record(
    kind: &'static str,
    remote: EndpointId,
    peer: EndpointId,
    reason: RecordSkip,
) {
    match reason {
        RecordSkip::OwnRecord | RecordSkip::Superseded => {}
        RecordSkip::Expired | RecordSkip::Invalid(RecordError::Decode) => tracing::debug!(
            peer = %peer.fmt_short(),
            relay = %remote.fmt_short(),
            kind,
            ?reason,
            "gossip: ignoring stale or unusable signed record"
        ),
        RecordSkip::Invalid(error) => tracing::warn!(
            peer = %peer.fmt_short(),
            relay = %remote.fmt_short(),
            kind,
            %error,
            "gossip: rejecting invalid signed record"
        ),
    }
}
