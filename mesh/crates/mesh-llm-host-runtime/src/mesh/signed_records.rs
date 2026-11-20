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
    MAX_NODE_RECORD_BODY_BYTES, hop_observation_from_ann, node_record_from_wire_ann,
    node_record_to_local,
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

pub(super) mod verification;
mod verification_cache;
pub(crate) use verification::VerificationGate;

impl Node {
    pub(super) async fn collect_outbound_gossip(&self) -> OutboundGossip {
        let filtering = self
            .plugin_manager()
            .await
            .is_some_and(|manager| manager.peer_evidence().has_verifiers());
        let mut pins = filtering.then(HashMap::new);
        let mut announcements = Vec::new();
        for ann in self.collect_announcements().await {
            if ann.addr.id == self.id() {
                announcements.push(ann);
            } else if let Some((eligible, record)) = self
                .evidence_announcement_with_record(ann.addr.id, &ann, true)
                .await
            {
                if let Some(pins) = &mut pins {
                    pins.insert(ann.addr.id, record);
                }
                announcements.push(eligible);
            }
        }
        let requires_signed_gossip = self.requires_signed_gossip().await;
        let (signed_records, signed_cache_affinity) = self
            .collect_signed_records(&mut announcements, pins.as_ref())
            .await;
        if requires_signed_gossip {
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

    /// Gossip describing only this node: its own announcement and signed
    /// records, with no rebroadcast of other peers.
    pub(super) async fn collect_own_outbound_gossip(&self) -> OutboundGossip {
        let local = self.snapshot_local_announcement_data().await;
        let mut announcements = vec![self.build_local_announcement(local)];
        let (signed_records, signed_cache_affinity) =
            self.collect_signed_records(&mut announcements, None).await;
        OutboundGossip {
            announcements,
            signed_records,
            signed_cache_affinity,
        }
    }

    /// This node's own records plus every held record for a peer we are
    /// rebroadcasting. Each relayed node record carries our latency view of
    /// that peer as unsigned hop data.
    pub(super) async fn collect_signed_records(
        &self,
        announcements: &mut Vec<PeerAnnouncement>,
        pins: Option<&HashMap<EndpointId, Option<Vec<u8>>>>,
    ) -> (Vec<SignedNodeRecord>, Vec<SignedCacheAffinity>) {
        let local_id = self.endpoint.id();
        let now = current_time_unix_ms();
        let local_wire = announcements
            .iter()
            .find(|ann| ann.addr.id == local_id)
            .map(|ann| (local_ann_to_proto_ann(ann), &ann.addr));
        let manager = self.plugin_manager().await;
        let cache_body = local_wire
            .as_ref()
            .and_then(|(wire, _)| wire.cache_affinity.as_ref())
            .map(Message::encode_to_vec);
        let signing_key =
            ed25519_dalek::SigningKey::from_bytes(&self.endpoint_secret_key.to_bytes());
        let mesh = self.mesh_id().await;
        let mut state = self.state.lock().await;
        let evidence = match &manager {
            Some(manager) => manager.peer_evidence().publications(),
            None => Default::default(),
        };
        let node_body = local_wire.as_ref().and_then(|(wire, addr)| {
            let mut record = node_record_from_wire_ann(wire, addr);
            record.peer_evidence = evidence;
            if record.encoded_len() > MAX_NODE_RECORD_BODY_BYTES {
                tracing::warn!("local node record exceeds the signed record size limit; reduce advertised metadata");
                return None;
            }
            Some(record.encode_to_vec())
        });
        if let Some(pins) = pins {
            announcements.retain(|ann| {
                ann.addr.id == local_id
                    || pins.get(&ann.addr.id).is_some_and(|expected| {
                        let current = state
                            .node_records
                            .fresh(&ann.addr.id, current_time_unix_ms());
                        let Some(manager) = &manager else {
                            return false;
                        };
                        let registry = manager.peer_evidence();
                        let provider = super::peer_evidence::announcement_is_provider(ann);
                        match expected {
                            Some(bytes) => current
                                .filter(|record| record.signed_bytes() == bytes)
                                .and_then(|held| NodeRecord::decode(held.body()).ok())
                                .is_some_and(|record| {
                                    (!provider && !super::peer_evidence::is_provider(&record))
                                        || (mesh
                                            .as_deref()
                                            .is_none_or(|mesh| mesh == record.mesh_id)
                                            && registry.provider_cached(
                                                &record.mesh_id,
                                                &ann.addr.id.to_string(),
                                                &record.peer_evidence,
                                            ))
                                }),
                            None => {
                                current.is_none()
                                    && (!provider
                                        || (state.has_fresh_direct_admission(&ann.addr.id)
                                            && registry.provider_cached(
                                                mesh.as_deref().unwrap_or_default(),
                                                &ann.addr.id.to_string(),
                                                &Default::default(),
                                            )))
                            }
                        }
                    })
            });
        }
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

    pub(super) async fn signed_cache_affinity(
        &self,
        peer: EndpointId,
    ) -> Option<CacheAffinityAdvertisement> {
        let state = self.state.lock().await;
        let held = state
            .cache_affinity_records
            .fresh(&peer, current_time_unix_ms())?;
        decode_cache_affinity(held.body()).ok()
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
pub(super) fn decode_cache_affinity(
    body: &[u8],
) -> Result<CacheAffinityAdvertisement, RecordError> {
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
