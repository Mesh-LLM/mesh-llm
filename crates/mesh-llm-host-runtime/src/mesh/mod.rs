//! Mesh membership via iroh QUIC connections.
//!
//! Mesh control traffic uses QUIC ALPN `mesh-llm/1` and multiplexes bi-streams
//! by first byte. Latency-sensitive and path-maintenance flows keep dedicated
//! stream bytes. Skippy activation transport remains on the latency-sensitive
//! `skippy-stage/2` ALPN.

pub use mesh_llm_types::mesh::{
    DEMAND_TTL_SECS, MAX_SPLIT_RTT_MS, ModelDemand, ModelRuntimeDescriptor, ModelSourceKind,
    ServedModelDescriptor, ServedModelIdentity, ServedModelMetadata,
    infer_available_model_descriptors, infer_local_served_model_descriptor,
    infer_served_model_descriptors, merge_demand,
};

use anyhow::{Context, Result};
use base64::Engine;
use iroh::endpoint::Connection;
use iroh::{Endpoint, EndpointAddr, EndpointId, SecretKey, TransportAddr};
use mesh_llm_events::OutputEvent;
use prost::Message;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, HashSet, VecDeque};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;

use tokio::sync::{Mutex, watch};

use self::requirements::{
    DirectPeerProofStatus, MeshRequirementDecision, MeshRequirementPolicySummary,
    MeshRequirementRejectReason, MeshRequirementRejectionEvent, MeshRequirementRejectionSource,
    evaluate_direct_peer_admission, peer_release_attestation_status,
};
use crate::crypto::{
    DEFAULT_NODE_CERT_LIFETIME_SECS, OwnershipStatus, OwnershipSummary, SignedNodeOwnership,
    TrustPolicy, TrustStore, default_node_ownership_path, save_node_ownership, sign_node_ownership,
    verify_control_plane_target_node, verify_node_ownership,
};
use crate::protocol::*;

use self::artifact_transfer_io::{
    PartialArtifactGuard, append_artifact_transfer_body, select_partial_artifact,
    write_artifact_transfer_chunk,
};

#[cfg(test)]
use self::artifact_transfer_io::read_artifact_transfer_chunk;

use skippy_protocol::proto::stage as skippy_stage_proto;

const PRETTY_LOCAL_REQUEST_WINDOW_SECS: u64 = 24 * 60 * 60;
const EPHEMERAL_QUIC_PORT: u16 = 0;
const SIGNED_BOOTSTRAP_TOKEN_LIFETIME_MS: u64 = 24 * 60 * 60 * 1000;
const RECENT_MESH_REJECTION_LIMIT: usize = 16;

pub(crate) fn emit_mesh_info(message: String) {
    let _ = mesh_llm_events::emit_event(OutputEvent::Info {
        message,
        context: None,
    });
}

pub(crate) fn emit_mesh_warning(message: String) {
    let _ = mesh_llm_events::emit_event(OutputEvent::Warning {
        message,
        context: None,
    });
}

pub(crate) fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub(crate) fn current_time_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub(crate) fn elapsed_ms_u64(duration: std::time::Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

mod artifact_transfer_io;
mod connection_reservation;
mod connections;
mod direct_path;
mod gossip;
mod heartbeat;
mod identity_persistence;
mod lan_bootstrap;
mod model_identity;
mod node;
mod node_identity;
mod node_requirements;
mod owner_control;
mod owner_control_response;
mod peer_state;
mod plugin_config;
mod plugin_mesh;
mod plugin_streams;
pub mod requirements;
mod stage_artifacts;
mod stage_proto;
mod stage_transport;
mod stage_transport_bridge;
mod stun;

use connection_reservation::*;
use connections::*;
use model_identity::*;
use node_identity::*;
pub(crate) use node_requirements::*;
use owner_control::*;
use peer_state::*;
#[allow(unused_imports)]
use plugin_mesh::*;
#[expect(
    unused_imports,
    reason = "sibling mesh modules import this parent prelude with `super::*`"
)]
use stage_artifacts::*;
use stage_transport::*;
use stun::*;

pub use connections::{QuicBindSelection, RelayConfig, RelayPolicy, detect_primary_lan_ipv4};
pub use gossip::backfill_legacy_descriptors;
#[expect(
    unused_imports,
    reason = "public compatibility re-export for existing mesh identity callers"
)]
pub use identity_persistence::{
    clear_public_identity, default_node_key_path, generate_mesh_id, load_last_mesh_id,
    load_node_key_from_path, mark_was_public, save_last_mesh_id, save_node_key_to_path,
    was_previously_public,
};
pub(crate) use node::{LocalRequestMetricsSampler, PeerDownReport, peer_down_endpoint_id};
#[expect(
    unused_imports,
    reason = "public compatibility re-export for existing mesh node callers"
)]
pub use node::{
    LocalRequestMetricsSnapshot, Node, RouteEntry, RoutingTable, detect_vram_bytes_capped,
};
pub(crate) use peer_state::{
    ControlListenerLifecycle, DEAD_PEER_TTL, MeshState, PEER_DOWN_REPORTER_COOLDOWN_SECS,
    PEER_STALE_SECS, public_model_id_from_identity, resolve_peer_leaving,
};
#[expect(
    unused_imports,
    reason = "public compatibility re-export for existing mesh state callers"
)]
pub use peer_state::{
    DisplayLatency, DisplayLatencySource, MeshCatalogEntry, NodeRole, OwnerRuntimeConfig,
    PeerAnnouncement, PeerInfo, PropagatedLatencyObservation,
};
pub(crate) use stage_transport::{
    ARTIFACT_TRANSFER_BUFFER_BYTES, ARTIFACT_TRANSFER_INVALID_OFFSET_ERROR,
    ARTIFACT_TRANSFER_OPEN_TIMEOUT, ARTIFACT_TRANSFER_READ_IDLE_TIMEOUT, ConnectionCaptureEvent,
    HttpCaptureEvent, MeshBiStream, PeerLifecycleCaptureEvent, SelectedPathObservation,
    StageTopologyState,
};
#[expect(
    unused_imports,
    reason = "public compatibility re-export for existing split-stage routing callers"
)]
pub use stage_transport::{InflightRequestGuard, SplitStagePathRejection, SplitStagePathSnapshot};
pub use stage_transport::{
    StageAssignment, StageEndpoint, StageRuntimeStatus, StageTopologyInstance, TunnelChannels,
};
pub(crate) use stage_transport_bridge::{StageTransportBridge, StageTransportBridgeLabel};

#[allow(unused_imports)]
use gossip::{apply_transitive_ann, peer_meaningfully_changed};
pub(crate) use heartbeat::resolve_peer_down;
#[expect(
    unused_imports,
    reason = "sibling mesh modules import this parent prelude with `super::*`"
)]
use heartbeat::{HeartbeatFailurePolicy, heartbeat_failure_policy_for_peer};
use heartbeat::{PeerDownReportDisposition, peer_down_report_disposition};
use stage_proto::*;

#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
mod public_identity_tests;
