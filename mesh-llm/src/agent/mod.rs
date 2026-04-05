//! ACP agent pooling — share AI agents across the mesh.
//!
//! An **agent provider** spawns an ACP-speaking subprocess (e.g. `goose acp`)
//! and accepts tunneled sessions from mesh peers over QUIC.
//!
//! An **agent client** runs a local ACP HTTP server and routes sessions to
//! available providers discovered through mesh gossip.

pub mod client;
pub mod provider;
pub mod relay;
