//! Embed API surface tests.
//!
//! These tests pin the `mesh_llm::embed::*` API as seen by an external
//! consumer. Regressions in method signatures, default values, or the
//! visibility of re-exported types will fail at compile time here.
//!
//! Live-network behaviour (actual relay registration) is covered separately
//! by `embed_gated_relay.rs` (gated behind `#[ignore]`).

use std::collections::HashMap;

use mesh_llm::embed::{NodeBuilder, NodeRole, QuicBindSelection};

#[test]
fn node_builder_is_chainable_and_owns_its_state() {
    let mut auths = HashMap::new();
    auths.insert("https://gated.example/".to_string(), "tok".to_string());

    // The whole point: an embedder can configure a node end-to-end with
    // chained calls and no awaits. The builder is fully synchronous up to
    // `.start()`.
    let _builder = NodeBuilder::new()
        .role(NodeRole::Client)
        .relays(["https://gated.example/", "https://public.iroh/"])
        .relay_auths(auths)
        .relay_auth("https://another-gated.example/", "tok2")
        .bind(QuicBindSelection {
            ip: None,
            port: None,
        })
        .max_vram_gb(Some(0.0))
        .enumerate_host(false);
}

#[test]
fn node_role_default_is_worker() {
    assert_eq!(NodeRole::default(), NodeRole::Worker);
}

#[test]
fn node_role_serializes_for_external_consumers() {
    // The embed surface re-exports NodeRole. External consumers commonly
    // round-trip it through serde for config files; pin that behaviour.
    let role = NodeRole::Host { http_port: 9337 };
    let json = serde_json::to_string(&role).expect("NodeRole should serialize");
    assert!(
        json.contains("Host"),
        "NodeRole::Host should serialize with a `Host` variant tag: {json}"
    );
    let back: NodeRole = serde_json::from_str(&json).expect("NodeRole should round-trip");
    assert_eq!(back, role);
}

#[test]
fn quic_bind_selection_is_default_constructible() {
    let q = QuicBindSelection::default();
    assert!(q.ip.is_none());
    assert!(q.port.is_none());
}
