//! Mesh peer membership: peer identity, admission state, and discovery helpers.
//!
//! This crate owns the peer-membership foundations of Mesh — peer role, node
//! and mesh identity persistence, discovery address classification,
//! cache-affinity gossip, the dependency-neutral release-attestation and
//! throughput-hint contracts, the mesh genesis/requirements admission policy,
//! peer-announcement/latency state, peer health and connection admission
//! lifecycle — independent of
//! inference/split policy (which stays in Skippy) and of the serving transport
//! and owner/plugin surfaces (which stay in `mesh-llm-host-runtime` for their
//! own later extractions).
//!
//! Extracted so far: `types` (`NodeRole`), `address`, `identity_persistence`,
//! `lan_bootstrap`, `cache_affinity_gossip`, the pure leaves of
//! `model_identity`, `weights_digest` (with an injected cache directory),
//! `release_attestation`, `advertised_throughput`, `selected_path`,
//! `requirements`, `peer_state`, `peer_health`, `adopted_membership`,
//! `state`, and `connection_reservation`.
//!
//! `peer_state` carries the neutral `PeerInfo` state and its non-routing
//! accessors. The serving-routing projections over `PeerInfo`
//! (`routable_models`, `routes_model`, …) stay in `mesh-llm-host-runtime` as
//! free functions because they call
//! `public_model_id_from_identity`/`canonical_demand_model_ref`, which depend
//! on `skippy_model_ref` and the host `models` catalog — a Skippy/host
//! boundary (see the remaining-boundary list in this crate's `README.md`).

pub mod address;
#[cfg(feature = "host-io")]
pub mod adopted_membership;
pub mod advertised_throughput;
#[cfg(feature = "host-io")]
pub mod announcements;
pub mod cache_affinity_gossip;
#[cfg(feature = "host-io")]
pub mod connection_reservation;
#[cfg(feature = "host-io")]
pub mod identity_persistence;
pub mod lan_bootstrap;
pub mod model_identity;
pub mod peer_health;
#[cfg(feature = "host-io")]
pub mod peer_state;
pub mod release_attestation;
pub mod requirements;
pub mod selected_path;
#[cfg(feature = "host-io")]
pub mod state;
#[cfg(feature = "host-io")]
pub mod transitions;
pub mod types;
pub mod weights_digest;

pub use address::{is_global_ipv4_candidate, is_public_ipv4_candidate};
pub use cache_affinity_gossip::{
    advertised_state_changed, local_advertisement, merge_advertisement,
};
#[cfg(feature = "host-io")]
pub use identity_persistence::{
    adopted_mesh_membership_path, clear_public_identity, default_node_key_path, generate_mesh_id,
    identity_home_dir, identity_state_dir, load_last_mesh_id, load_node_key_from_path,
    load_or_create_key, mark_was_public, mesh_genesis_policy_path, save_last_mesh_id,
    save_node_key_to_path, was_previously_public,
};
pub use lan_bootstrap::{is_private_lan_ipv4, lan_ipv4_candidates};
pub use model_identity::{
    format_hf_canonical_ref, identity_hash_for, local_gguf_identity_from_source,
    parse_hf_ref_parts, parse_hf_resolve_url_parts, unknown_identity,
};
pub use types::NodeRole;
pub use weights_digest::{file_fingerprint, weights_digest_for_file_in};
