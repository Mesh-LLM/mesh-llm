//! Signed records in gossip: signing this node's records, admitting the
//! records peers relay, and merging them with unsigned announcements.
//!
//! Two kinds are gossiped mesh-wide: the node record (the node's
//! self-description) and its cache-affinity evidence. Both are relayed
//! byte-for-byte. Unsigned announcements are still sent and applied so nodes
//! that predate signed records keep interoperating; a usable signed record
//! only takes precedence over the unsigned data sent alongside it.

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

impl Node {
    pub(super) async fn collect_outbound_gossip(&self) -> OutboundGossip {
        let announcements = self.collect_announcements().await;
        let (signed_records, signed_cache_affinity) =
            self.collect_signed_records(&announcements).await;
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

    /// Turns a decoded frame into the announcements gossip applies.
    ///
    /// A node record that verifies and is current replaces the sender's
    /// unsigned entry for the same node, and signed cache affinity replaces
    /// the unsigned copy. Every other unsigned entry still applies, including
    /// entries from nodes that predate signed records and the unsigned twin
    /// of a record that was skipped. The sender's own unsigned entry is
    /// always kept: the connection authenticates it and it carries
    /// direct-only data (demand, admission proof) that records leave out.
    pub(super) async fn resolve_inbound_gossip(
        &self,
        remote: EndpointId,
        inbound: InboundGossip,
    ) -> Vec<Announcement> {
        let local_id = self.endpoint.id();
        let now = current_time_unix_ms();
        let (mut node_records, cache_affinity) = {
            let mut state = self.state.lock().await;
            let node_records = admit_node_records(
                &mut state.node_records,
                local_id,
                remote,
                &inbound.signed_records,
                now,
            );
            let cache_affinity = admit_cache_affinity(
                &mut state.cache_affinity_records,
                local_id,
                remote,
                &inbound.signed_cache_affinity,
                now,
            );
            // A sender that no longer signs its own records (for example, it
            // was downgraded) is described only by its unsigned entry, so
            // stop relaying the records we hold for it.
            if !node_records.sender_signed {
                state.node_records.remove(&remote);
            }
            if !cache_affinity.sender_signed {
                state.cache_affinity_records.remove(&remote);
            }
            (node_records, cache_affinity)
        };
        let unsigned_received = inbound.announcements.len();
        let mut announcements = merge_unsigned(
            remote,
            inbound.announcements,
            &mut node_records.replacements,
        );
        let unsigned_applied = announcements.len();
        let relayed_records_applied = node_records.replacements.len();
        announcements.extend(node_records.replacements.into_values());
        apply_signed_cache_affinity(&mut announcements, &cache_affinity.advertisements);
        tracing::debug!(
            sender = %remote.fmt_short(),
            records = inbound.signed_records.len(),
            relayed_records_applied,
            cache_affinity_records = inbound.signed_cache_affinity.len(),
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
    unsigned: Vec<Announcement>,
    replacements: &mut HashMap<EndpointId, Announcement>,
) -> Vec<Announcement> {
    let mut kept = Vec::with_capacity(unsigned.len());
    for (addr, ann) in unsigned {
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
    /// Whether the frame carried a record naming the sender itself.
    sender_signed: bool,
    /// Announcements rebuilt from accepted records about nodes other than
    /// the sender. Each replaces the frame's unsigned entry for that node.
    replacements: HashMap<EndpointId, Announcement>,
}

fn admit_node_records(
    held: &mut HeldRecords,
    local_id: EndpointId,
    remote: EndpointId,
    records: &[SignedNodeRecord],
    now: u64,
) -> AdmittedNodeRecords {
    let mut admitted = AdmittedNodeRecords {
        sender_signed: false,
        replacements: HashMap::new(),
    };
    for wire in records {
        let admission = held.admit(
            local_id,
            remote,
            &wire.signed,
            &wire.signature,
            now,
            decode_node_record,
        );
        match admission {
            Ok(RecordAdmission::Accepted { record, body }) => {
                let id = record.endpoint_id();
                if id == remote {
                    admitted.sender_signed = true;
                } else if let Some(entry) = relayed_record_announcement(remote, id, &body, wire) {
                    admitted.replacements.insert(id, entry);
                }
            }
            Ok(RecordAdmission::Skipped {
                endpoint_id,
                reason,
            }) => {
                admitted.sender_signed |= endpoint_id == remote;
                log_skipped_record("node record", remote, endpoint_id, reason);
            }
            Err(error) => log_malformed_record("node record", remote, error),
        }
    }
    admitted
}

struct AdmittedCacheAffinity {
    sender_signed: bool,
    advertisements: HashMap<EndpointId, CacheAffinityAdvertisement>,
}

fn admit_cache_affinity(
    held: &mut HeldRecords,
    local_id: EndpointId,
    remote: EndpointId,
    records: &[SignedCacheAffinity],
    now: u64,
) -> AdmittedCacheAffinity {
    let mut admitted = AdmittedCacheAffinity {
        sender_signed: false,
        advertisements: HashMap::new(),
    };
    for wire in records {
        let admission = held.admit(
            local_id,
            remote,
            &wire.signed,
            &wire.signature,
            now,
            decode_cache_affinity,
        );
        match admission {
            Ok(RecordAdmission::Accepted { record, body }) => {
                let id = record.endpoint_id();
                admitted.sender_signed |= id == remote;
                admitted.advertisements.insert(id, body);
            }
            Ok(RecordAdmission::Skipped {
                endpoint_id,
                reason,
            }) => {
                admitted.sender_signed |= endpoint_id == remote;
                log_skipped_record("cache affinity", remote, endpoint_id, reason);
            }
            Err(error) => log_malformed_record("cache affinity", remote, error),
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
    wire: &SignedNodeRecord,
) -> Option<Announcement> {
    let entry = node_record_to_local(id, body, wire.hop.as_ref());
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
