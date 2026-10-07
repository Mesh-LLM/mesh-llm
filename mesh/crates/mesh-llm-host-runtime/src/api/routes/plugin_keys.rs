//! `GET /api/plugin-keys`: the public keys plugins on this node announce, and
//! the ones this node's directly-connected peers announce, each verified
//! against that peer's own node key (`mesh::plugin_keys`). Keyed by plugin
//! name, hex-encoded; nodes by their full 64-hex endpoint id. Public facts:
//! nothing here is secret.

use std::collections::BTreeMap;

use serde::Serialize;
use tokio::net::TcpStream;

use super::super::{MeshApi, http::respond_json};
use crate::mesh::plugin_keys::BoundPluginKey;

pub(super) const ROUTE: &str = "/api/plugin-keys";

#[derive(Debug, Serialize, PartialEq, Eq)]
pub(super) struct PluginKeysResponse {
    pub(super) node_id: String,
    pub(super) plugin_keys: BTreeMap<String, String>,
    pub(super) peers: BTreeMap<String, BTreeMap<String, String>>,
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
) -> PluginKeysResponse {
    PluginKeysResponse {
        node_id: hex::encode(node_id.as_bytes()),
        plugin_keys: by_plugin(own),
        peers: peers
            .iter()
            .map(|(peer, keys)| (hex::encode(peer.as_bytes()), by_plugin(keys)))
            .collect(),
    }
}

pub(super) async fn handle(stream: &mut TcpStream, state: &MeshApi) -> anyhow::Result<()> {
    let node = state.inner.lock().await.node.clone();
    let body = response(&node.endpoint.id(), &node.plugin_keys.own(), &node.plugin_keys.peers());
    respond_json(stream, 200, &body).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::plugin_keys::bind;

    #[test]
    fn lists_own_and_peer_keys_by_full_id() {
        let node = iroh::SecretKey::generate();
        let peer = iroh::SecretKey::generate();
        let own = vec![bind(&node, "capsules", [3; 32])];
        let peers = std::collections::HashMap::from([(
            peer.public(),
            vec![bind(&peer, "capsules", [4; 32])],
        )]);
        let body = response(&node.public(), &own, &peers);
        assert_eq!(body.node_id, hex::encode(node.public().as_bytes()));
        assert_eq!(body.plugin_keys["capsules"], hex::encode([3u8; 32]));
        assert_eq!(
            body.peers[&hex::encode(peer.public().as_bytes())]["capsules"],
            hex::encode([4u8; 32])
        );
    }
}
