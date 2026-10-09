//! Conversion between the signed `NodeRecord` and the gossip announcement
//! model the peer table consumes.
//!
//! A record holds only what a node asserts about itself. Converting it back
//! reuses `proto_ann_to_local` so every remote-bytes bound that applies to
//! legacy announcements applies to signed records too.

use super::proto_ann_to_local;
use crate::mesh::PeerAnnouncement;
use crate::proto::node as wire;
use iroh::{EndpointAddr, EndpointId, RelayUrl, TransportAddr};

/// Projects this node's outbound wire announcement onto the fields it signs.
/// Hop measurements, derived fields, deprecated duplicates, short-lived data
/// and per-connection proofs are left out.
pub(crate) fn node_record_from_wire_ann(
    ann: &wire::PeerAnnouncement,
    addr: &EndpointAddr,
) -> wire::NodeRecord {
    wire::NodeRecord {
        role: ann.role,
        http_port: ann.http_port,
        version: ann.version.clone().unwrap_or_default(),
        addrs: addr
            .addrs
            .iter()
            .filter_map(transport_addr_to_wire)
            .collect(),
        hardware: ann.hardware.clone(),
        vram_bytes: ann.vram_bytes,
        mesh_id: ann.mesh_id.clone().unwrap_or_default(),
        mesh_policy_hash: ann.mesh_policy_hash.clone().unwrap_or_default(),
        serving_models: ann.serving_models.clone(),
        hosted_models: ann.hosted_models.clone(),
        served_models: ann.served_model_descriptors.clone(),
        served_runtime: ann.served_model_runtime.clone(),
        requested_models: ann.requested_models.clone(),
        explicit_model_interests: ann.explicit_model_interests.clone(),
        catalog_models: ann.catalog_models.clone(),
        model_source: ann.model_source.clone().unwrap_or_default(),
        experts_summary: ann.experts_summary.clone(),
        first_joined_mesh_ts: ann.first_joined_mesh_ts,
        subprotocols: ann.subprotocols.clone(),
        admission_state: ann.inference_admission_state.unwrap_or_default(),
        throughput: ann.advertised_model_throughput.clone(),
        lightning_offers: ann.lightning_offers.clone(),
        owner_attestation: ann.owner_attestation.clone(),
        genesis_policy: ann.genesis_policy.clone(),
        release_attestation: ann.release_attestation.clone(),
    }
}

/// Rebuilds the announcement for a verified record. The id comes from the
/// signed header, so the record cannot describe any other node. `hop` is the
/// sending relay's unsigned latency observation.
pub(crate) fn node_record_to_local(
    endpoint_id: EndpointId,
    record: &wire::NodeRecord,
    hop: Option<&wire::HopObservation>,
) -> Option<(EndpointAddr, PeerAnnouncement)> {
    let hop = hop.cloned().unwrap_or_default();
    let ann = wire::PeerAnnouncement {
        endpoint_id: endpoint_id.as_bytes().to_vec(),
        role: record.role,
        http_port: record.http_port,
        version: non_empty(&record.version),
        serving_models: record.serving_models.clone(),
        requested_models: record.requested_models.clone(),
        experts_summary: record.experts_summary.clone(),
        catalog_models: record.catalog_models.clone(),
        vram_bytes: record.vram_bytes,
        model_source: non_empty(&record.model_source),
        primary_serving: record.serving_models.first().cloned(),
        mesh_id: non_empty(&record.mesh_id),
        hosted_models: record.hosted_models.clone(),
        hosted_models_known: Some(true),
        served_model_descriptors: record.served_models.clone(),
        served_model_identities: record
            .served_models
            .iter()
            .filter_map(|descriptor| descriptor.identity.clone())
            .collect(),
        served_model_runtime: record.served_runtime.clone(),
        owner_attestation: record.owner_attestation.clone(),
        hardware: record.hardware.clone(),
        first_joined_mesh_ts: record.first_joined_mesh_ts,
        explicit_model_interests: record.explicit_model_interests.clone(),
        subprotocols: record.subprotocols.clone(),
        latency_ms: hop.latency_ms,
        latency_source: hop.latency_source,
        latency_age_ms: hop.latency_age_ms,
        latency_observer_id: (!hop.latency_observer_id.is_empty())
            .then_some(hop.latency_observer_id),
        advertised_model_throughput: record.throughput.clone(),
        mesh_policy_hash: non_empty(&record.mesh_policy_hash),
        genesis_policy: record.genesis_policy.clone(),
        release_attestation: record.release_attestation.clone(),
        inference_admission_state: (record.admission_state != 0).then_some(record.admission_state),
        lightning_offers: record.lightning_offers.clone(),
        ..Default::default()
    };
    let (_, mut local) = proto_ann_to_local(&ann)?;
    let addr = EndpointAddr {
        id: endpoint_id,
        addrs: record
            .addrs
            .iter()
            .filter_map(transport_addr_from_wire)
            .collect(),
    };
    local.addr = addr.clone();
    Some((addr, local))
}

/// The unsigned latency a relay attaches when forwarding a record, taken
/// from the announcement it rebuilt for the same peer.
pub(crate) fn hop_observation_from_ann(ann: &PeerAnnouncement) -> wire::HopObservation {
    wire::HopObservation {
        latency_ms: ann.latency_ms,
        latency_source: ann.latency_source.map(|source| source as i32),
        latency_age_ms: ann
            .latency_age_ms
            .map(|age| u32::try_from(age).unwrap_or(u32::MAX)),
        latency_observer_id: ann
            .latency_observer_id
            .map(|id| id.as_bytes().to_vec())
            .unwrap_or_default(),
    }
}

fn non_empty(value: &str) -> Option<String> {
    (!value.is_empty()).then(|| value.to_string())
}

fn transport_addr_to_wire(addr: &TransportAddr) -> Option<wire::TransportAddr> {
    let addr = match addr {
        TransportAddr::Relay(url) => wire::transport_addr::Addr::RelayUrl(url.to_string()),
        TransportAddr::Ip(socket) => wire::transport_addr::Addr::Ip(socket.to_string()),
        // Custom transports are not advertised over gossip.
        _ => return None,
    };
    Some(wire::TransportAddr { addr: Some(addr) })
}

fn transport_addr_from_wire(addr: &wire::TransportAddr) -> Option<TransportAddr> {
    match addr.addr.as_ref()? {
        wire::transport_addr::Addr::RelayUrl(url) => {
            url.parse::<RelayUrl>().ok().map(TransportAddr::Relay)
        }
        wire::transport_addr::Addr::Ip(socket) => socket.parse().ok().map(TransportAddr::Ip),
    }
}

#[cfg(test)]
mod tests;
