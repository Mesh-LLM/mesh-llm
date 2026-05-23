//! Embed mesh-llm in a Rust application.
//!
//! This module is the **public, stable** Rust API for running a mesh-llm node
//! in-process. Use it when you want to ship mesh-llm as part of your own
//! binary instead of shelling out to the `mesh-llm` CLI.
//!
//! Everything else in this crate is `pub(crate)` or `pub` only because Rust
//! doesn't have a smaller visibility — those modules are internal and may
//! change in any release. Build against `mesh_llm::embed::*` and you get a
//! curated, documented surface.
//!
//! # Quickstart
//!
//! ```no_run
//! use std::collections::HashMap;
//! use mesh_llm::embed::{NodeBuilder, NodeRole, QuicBindSelection};
//!
//! # async fn run() -> anyhow::Result<()> {
//! let mut relay_auths = HashMap::new();
//! relay_auths.insert(
//!     "https://gated.example/".to_string(),
//!     "<nip98-bearer-or-static-token>".to_string(),
//! );
//!
//! let handle = NodeBuilder::new()
//!     .role(NodeRole::Client)
//!     .relays(["https://gated.example/"])
//!     .relay_auths(relay_auths)
//!     .bind(QuicBindSelection::default())
//!     .max_vram_gb(0.0)
//!     .enumerate_host(true)
//!     .start()
//!     .await?;
//!
//! handle.start_accepting();
//! handle.set_display_name("my-app".to_string()).await;
//!
//! let invite = handle.invite_token();
//! println!("share this with peers: {invite}");
//!
//! // ... later, when your application is done with the mesh:
//! drop(handle);
//! # Ok(())
//! # }
//! ```
//!
//! # Scope
//!
//! `NodeBuilder` + `NodeHandle` give you the in-process equivalent of the
//! `mesh-llm` CLI's mesh layer: peer membership, gossip, invite tokens,
//! relay registration (including [`--relay-auth`][relay-auth] for gated
//! iroh relays), and model advertisement.
//!
//! It does **not** start the HTTP proxy, the management console, the TUI,
//! local inference, or the auto-discovery / auto-mode loops. Those are
//! orchestrated by the CLI's `runtime::run` and would need their own
//! embed surface; we'll add one if there is demand.
//!
//! [relay-auth]: https://github.com/Mesh-LLM/mesh-llm/pull/641

use crate::mesh;
use anyhow::Result;
use std::collections::HashMap;

pub use crate::mesh::{NodeRole, QuicBindSelection, RelayConfig};

/// Builder for an in-process mesh-llm [`NodeHandle`].
///
/// All fields are optional and have CLI-equivalent defaults:
/// - `role`: [`NodeRole::Client`]
/// - `relays`: empty (mesh-llm's bundled default iroh relays are used)
/// - `relay_auths`: empty (no gated relays)
/// - `bind`: ephemeral port, OS-chosen bind IP
/// - `max_vram_gb`: `None` (use all available VRAM)
/// - `enumerate_host`: `true` (publish hardware survey to gossip)
#[derive(Debug, Default)]
pub struct NodeBuilder {
    role: NodeRole,
    relays: Vec<String>,
    relay_auths: HashMap<String, String>,
    bind: QuicBindSelection,
    max_vram_gb: Option<f64>,
    enumerate_host: bool,
}

impl NodeBuilder {
    /// Start with all defaults. Equivalent to `NodeBuilder::default()` plus
    /// `enumerate_host = true`, which matches the CLI's default behaviour.
    pub fn new() -> Self {
        Self {
            enumerate_host: true,
            ..Default::default()
        }
    }

    /// Set the role this node plays in the mesh.
    pub fn role(mut self, role: NodeRole) -> Self {
        self.role = role;
        self
    }

    /// Set the iroh relay URLs this node will register with. Pass an empty
    /// iterator (or skip the call) to use the bundled default relays.
    pub fn relays<I, S>(mut self, relays: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.relays = relays.into_iter().map(Into::into).collect();
        self
    }

    /// Set the full per-relay auth-token map. Keys must exactly match the
    /// relay URLs in [`Self::relays`] (or the bundled defaults, if you
    /// didn't override). Relays not present in this map register
    /// unauthenticated, which is correct for public relays.
    ///
    /// Tokens are forwarded verbatim to `iroh::RelayConfig::with_auth_token`
    /// and sent as `Authorization: Bearer <token>` on the relay WebSocket
    /// upgrade. For relays running `AccessConfig::Restricted` with NIP-98
    /// admission this should be a base64-encoded signed kind:27235 event;
    /// for relays with a custom static token, just the static token.
    pub fn relay_auths(mut self, relay_auths: HashMap<String, String>) -> Self {
        self.relay_auths = relay_auths;
        self
    }

    /// Add or replace a single relay-auth entry.
    pub fn relay_auth(mut self, relay_url: impl Into<String>, token: impl Into<String>) -> Self {
        self.relay_auths.insert(relay_url.into(), token.into());
        self
    }

    /// Set the QUIC bind selection (IP and/or port).
    pub fn bind(mut self, bind: QuicBindSelection) -> Self {
        self.bind = bind;
        self
    }

    /// Set the VRAM cap in gigabytes. `None` means "use whatever the hardware
    /// reports". Pass `Some(0.0)` for a client-only node that should not
    /// advertise any VRAM.
    pub fn max_vram_gb(mut self, max_vram_gb: Option<f64>) -> Self {
        self.max_vram_gb = max_vram_gb;
        self
    }

    /// Whether to publish a hardware survey (GPU name, VRAM, hostname, etc.)
    /// to gossip. Defaults to `true`. Set to `false` to keep the node
    /// hardware profile private.
    pub fn enumerate_host(mut self, enumerate_host: bool) -> Self {
        self.enumerate_host = enumerate_host;
        self
    }

    /// Bring the node online. Returns a [`NodeHandle`] once the iroh
    /// endpoint has been bound and (best-effort) the home relay has been
    /// reached.
    pub async fn start(self) -> Result<NodeHandle> {
        let relay = RelayConfig {
            urls: &self.relays,
            auths: &self.relay_auths,
        };
        let (node, _channels) = mesh::Node::start(
            self.role,
            relay,
            self.bind,
            self.max_vram_gb,
            self.enumerate_host,
            // OwnerRuntimeConfig is for owner-control endpoints, which the
            // embed surface doesn't expose yet. None ⇒ no control listener.
            None,
            // Embedders don't have a `mesh-llm` config file; pass None to
            // skip config-driven model defaults.
            None,
        )
        .await?;
        Ok(NodeHandle { inner: node })
    }
}

/// A running in-process mesh-llm node.
///
/// Drop the handle to stop accepting new mesh traffic (in-flight tasks may
/// continue briefly). A graceful `shutdown()` API may be added later.
#[derive(Clone)]
pub struct NodeHandle {
    inner: mesh::Node,
}

impl NodeHandle {
    /// Start accepting incoming mesh connections.
    ///
    /// The bind happens in [`NodeBuilder::start`], but the accept loop waits
    /// until you call this. That lets you finish wiring listeners (display
    /// name, models) before the node is reachable.
    pub fn start_accepting(&self) {
        self.inner.start_accepting();
    }

    /// The stable endpoint ID of this node (derived from its iroh secret key).
    pub fn id(&self) -> String {
        format!("{:?}", self.inner.id())
    }

    /// An invite token that lets another node join this one with
    /// [`NodeHandle::join`].
    pub fn invite_token(&self) -> String {
        self.inner.invite_token()
    }

    /// Join an existing mesh via an invite token produced by another node's
    /// [`Self::invite_token`].
    pub async fn join(&self, invite_token: &str) -> Result<()> {
        self.inner.join(invite_token).await
    }

    /// Set a human-readable display name advertised to peers.
    pub async fn set_display_name(&self, name: String) {
        self.inner.set_display_name(name).await;
    }

    /// Replace the set of models this node advertises as available.
    pub async fn set_models(&self, models: Vec<String>) {
        self.inner.set_models(models).await;
    }

    /// Get the current set of advertised models.
    pub async fn models(&self) -> Vec<String> {
        self.inner.models().await
    }

    /// Read the current role.
    pub async fn role(&self) -> NodeRole {
        self.inner.role().await
    }

    /// A simplified view of currently-known peers. Each entry is one peer's
    /// `(endpoint_id, role, models)`. This is intentionally narrow — if you
    /// need the full peer record, build it from gossip yourself or open an
    /// issue describing your use case.
    pub async fn peers(&self) -> Vec<EmbedPeer> {
        self.inner
            .peers()
            .await
            .into_iter()
            .map(|peer| EmbedPeer {
                id: format!("{:?}", peer.id),
                role: peer.role,
                models: peer.models,
                rtt_ms: peer.rtt_ms,
            })
            .collect()
    }
}

/// Public peer summary exposed by [`NodeHandle::peers`]. Stable subset of
/// the internal `PeerInfo`.
#[derive(Debug, Clone)]
pub struct EmbedPeer {
    /// Hex-formatted endpoint ID, suitable for logging and display.
    pub id: String,
    /// Role this peer advertises in gossip.
    pub role: NodeRole,
    /// Models the peer advertises as available.
    pub models: Vec<String>,
    /// Last observed round-trip-time in milliseconds, if known.
    pub rtt_ms: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_defaults_match_cli_intent() {
        let b = NodeBuilder::new();
        assert!(b.enumerate_host, "enumerate_host defaults to true");
        assert_eq!(b.role, NodeRole::Worker);
        assert!(b.relays.is_empty());
        assert!(b.relay_auths.is_empty());
        assert_eq!(b.max_vram_gb, None);
    }

    #[test]
    fn builder_threads_relay_auths_through() {
        let b = NodeBuilder::new()
            .relays(["https://gated.example/"])
            .relay_auth("https://gated.example/", "bearer-token");
        assert_eq!(b.relays, vec!["https://gated.example/".to_string()]);
        assert_eq!(
            b.relay_auths
                .get("https://gated.example/")
                .map(String::as_str),
            Some("bearer-token"),
        );
    }

    #[test]
    fn builder_relay_auths_replaces_full_map() {
        let mut map = HashMap::new();
        map.insert("https://a/".to_string(), "ta".to_string());
        map.insert("https://b/".to_string(), "tb".to_string());
        let b = NodeBuilder::new()
            .relay_auth("https://stale/", "stale") // should be replaced
            .relay_auths(map);
        assert_eq!(b.relay_auths.len(), 2);
        assert!(!b.relay_auths.contains_key("https://stale/"));
        assert_eq!(
            b.relay_auths.get("https://a/").map(String::as_str),
            Some("ta")
        );
    }
}
