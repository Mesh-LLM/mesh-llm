//! `GET /api/plugin-keys`: the public keys plugins on this node announce, and
//! the ones this node's directly-connected peers announce, each verified
//! against that peer's own node key (`mesh::plugin_keys`). Keyed by plugin
//! name, hex-encoded; nodes by their full 64-hex endpoint id. Public facts:
//! nothing here is secret.
//!
//! `withdrawn_plugin_keys` and `withdrawn_peers` list the keys a node has
//! signed a withdrawal of (plugin name to the withdrawn key); a peer stays
//! there after it leaves, so a plugin forgets a withdrawn key at its next
//! poll even if that peer is no longer listed under `peers`.

use std::collections::BTreeMap;

use serde::Serialize;
use tokio::net::TcpStream;

use super::super::{MeshApi, http::respond_json};
use crate::mesh::plugin_keys::{BoundPluginKey, Withdrawal};

pub(super) const ROUTE: &str = "/api/plugin-keys";

#[derive(Debug, Serialize, PartialEq, Eq)]
pub(super) struct PluginKeysResponse {
    pub(super) node_id: String,
    pub(super) plugin_keys: BTreeMap<String, String>,
    pub(super) peers: BTreeMap<String, BTreeMap<String, String>>,
    pub(super) withdrawn_plugin_keys: BTreeMap<String, String>,
    pub(super) withdrawn_peers: BTreeMap<String, BTreeMap<String, String>>,
}

fn withdrawn_by_plugin(withdrawals: &[Withdrawal]) -> BTreeMap<String, String> {
    withdrawals
        .iter()
        .map(|w| (w.plugin.clone(), hex::encode(w.key_id)))
        .collect()
}

fn by_plugin(keys: &[BoundPluginKey]) -> BTreeMap<String, String> {
    keys.iter()
        .map(|key| (key.plugin.clone(), hex::encode(key.public_key)))
        .collect()
}

pub(super) fn response(
    node_id: &iroh::EndpointId,
    own: &[BoundPluginKey],
    peers: &std::collections::HashMap<iroh::EndpointId, Vec<BoundPluginKey>>,
    own_withdrawals: &[Withdrawal],
    peer_withdrawals: &std::collections::HashMap<iroh::EndpointId, Vec<Withdrawal>>,
) -> PluginKeysResponse {
    PluginKeysResponse {
        node_id: hex::encode(node_id.as_bytes()),
        plugin_keys: by_plugin(own),
        peers: peers
            .iter()
            .map(|(peer, keys)| (hex::encode(peer.as_bytes()), by_plugin(keys)))
            .collect(),
        withdrawn_plugin_keys: withdrawn_by_plugin(own_withdrawals),
        withdrawn_peers: peer_withdrawals
            .iter()
            .map(|(peer, withdrawals)| {
                (
                    hex::encode(peer.as_bytes()),
                    withdrawn_by_plugin(withdrawals),
                )
            })
            .collect(),
    }
}

pub(super) async fn handle(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let node = state.inner.lock().await.node.clone();
    let body = response(
        &node.endpoint.id(),
        &node.plugin_keys.own(),
        &node.plugin_keys.peers(),
        &node.plugin_keys.own_withdrawals(),
        &node.plugin_keys.peer_withdrawals(),
    );
    respond_json(stream, 200, &body).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::plugin_keys::{bind, withdraw};

    #[test]
    fn lists_own_and_peer_keys_by_full_id() {
        let node = iroh::SecretKey::generate();
        let peer = iroh::SecretKey::generate();
        let own = vec![bind(&node, "capsules", [3; 32])];
        let peers = std::collections::HashMap::from([(
            peer.public(),
            vec![bind(&peer, "capsules", [4; 32])],
        )]);
        let body = response(&node.public(), &own, &peers, &[], &Default::default());
        assert_eq!(body.node_id, hex::encode(node.public().as_bytes()));
        assert_eq!(body.plugin_keys["capsules"], hex::encode([3u8; 32]));
        assert_eq!(
            body.peers[&hex::encode(peer.public().as_bytes())]["capsules"],
            hex::encode([4u8; 32])
        );
    }

    #[test]
    fn lists_own_and_departed_peer_withdrawals_by_full_id() {
        let node = iroh::SecretKey::generate();
        let peer = iroh::SecretKey::generate();
        let own = [withdraw(&node, "capsules", [3; 32])];
        let peers = std::collections::HashMap::from([(
            peer.public(),
            vec![withdraw(&peer, "capsules", [4; 32])],
        )]);
        let body = response(&node.public(), &[], &Default::default(), &own, &peers);
        assert!(body.peers.is_empty(), "a departed peer lists no key");
        assert_eq!(
            body.withdrawn_plugin_keys["capsules"],
            hex::encode([3u8; 32])
        );
        assert_eq!(
            body.withdrawn_peers[&hex::encode(peer.public().as_bytes())]["capsules"],
            hex::encode([4u8; 32])
        );
    }
}
