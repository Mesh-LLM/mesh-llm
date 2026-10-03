//! Core peer-membership state types.
//!
//! Moved from host-runtime `mesh/peer_state.rs`. The client previously
//! duplicated `NodeRole` in `mesh-client/src/mesh/types.rs`; both now import it
//! from here.

use serde::{Deserialize, Serialize};

/// Role a node plays in the mesh.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum NodeRole {
    /// Provides staged GPU compute for a specific model.
    #[default]
    Worker,
    /// Runs the local serving runtime for a specific model and provides the HTTP API.
    Host { http_port: u16 },
    /// Lite client — no compute, accesses the API via tunnel.
    Client,
}
