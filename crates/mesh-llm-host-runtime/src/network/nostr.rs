//! Publish and discover mesh-llm meshes via Nostr relays.
//!
//! A running mesh publishes a replaceable event (kind 31990, d-tag "mesh-llm")
//! containing bootstrap metadata, a join token, served models, VRAM, node count, etc.
//! Other nodes can discover available meshes and auto-join.

mod auto;
mod contracts;
mod discovery;
mod keys;
mod model_packs;
mod publish;

pub use auto::{AutoDecision, is_auto_eligible, score_mesh, smart_auto};
pub use contracts::{DEFAULT_RELAYS, DiscoveredMesh, MeshListing};
pub use discovery::{MeshFilter, discover};
pub use keys::{load_or_create_keys, rotate_keys};
pub use model_packs::{auto_model_pack, default_models_for_vram};
pub use publish::{
    PublishLoopConfig, PublishStateUpdate, Publisher, publish_loop, publish_watchdog,
};
