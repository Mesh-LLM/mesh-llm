//! Public keys plugins on this node sign with, bound to this node by its own
//! key and announced in gossip (`PluginKey` in `node.proto`), so a peer's
//! plugin can check what a plugin here signed without keys configured by hand.
//!
//! A plugin opts in with a `PluginKeyRequest`; nothing is announced for a
//! plugin that did not ask, and a plugin can only set or withdraw its own key.
//! A key received from a peer is kept only when its binding verifies against
//! that peer's own node key, and only from the directly-connected sender's own
//! entry: plugin keys are not relayed transitively, like `claimed_log_head`.
//! A peer's keys are dropped when it leaves this node's peer list.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, PoisonError};

use iroh::{EndpointId, SecretKey};

/// Domain-separation tag of the binding's signed bytes; see `PluginKey`.
pub(crate) const SIG_DOMAIN: &[u8] = b"mesh-llm-plugin-key-v1:";
pub(crate) const SIGNATURE_ALGORITHM: &str = "ed25519";
/// At most this many plugin keys per node, sent or kept.
pub(crate) const MAX_PLUGIN_KEYS: usize = 16;
/// The longest plugin name a key may carry, in bytes.
pub(crate) const MAX_PLUGIN_NAME_BYTES: usize = 128;

/// One plugin's public key, with this node's (or the announcing peer's)
/// signature binding it to that node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BoundPluginKey {
    pub(crate) plugin: String,
    pub(crate) public_key: [u8; 32],
    pub(crate) binding_signature: [u8; 64],
}

fn write_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(bytes);
}

/// The bytes a node signs to bind `public_key` to itself for `plugin`.
pub(crate) fn sig_input(node: &EndpointId, plugin: &str, public_key: &[u8; 32]) -> Vec<u8> {
    let mut out = SIG_DOMAIN.to_vec();
    write_bytes(&mut out, node.as_bytes());
    write_bytes(&mut out, plugin.as_bytes());
    write_bytes(&mut out, public_key);
    write_bytes(&mut out, SIGNATURE_ALGORITHM.as_bytes());
    out
}

/// A plugin name a key may be announced under: the host's own name for the
/// plugin connection, which is never the plugin's claim.
pub(crate) fn valid_plugin_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_PLUGIN_NAME_BYTES
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

/// Whether `public_key` is a usable Ed25519 public key.
pub(crate) fn valid_public_key(public_key: &[u8; 32]) -> bool {
    ed25519_dalek::VerifyingKey::from_bytes(public_key).is_ok()
}

/// Bind `public_key` to the node `secret` belongs to, for `plugin`.
pub(crate) fn bind(secret: &SecretKey, plugin: &str, public_key: [u8; 32]) -> BoundPluginKey {
    let signing_key = ed25519_dalek::SigningKey::from_bytes(&secret.to_bytes());
    let signature = ed25519_dalek::Signer::sign(
        &signing_key,
        &sig_input(&secret.public(), plugin, &public_key),
    );
    BoundPluginKey {
        plugin: plugin.to_string(),
        public_key,
        binding_signature: signature.to_bytes(),
    }
}

/// Whether `key`'s binding is `node`'s own signature.
pub(crate) fn verify(node: &EndpointId, key: &BoundPluginKey) -> bool {
    let Ok(verifying_key) = ed25519_dalek::VerifyingKey::from_bytes(node.as_bytes()) else {
        return false;
    };
    let signature = ed25519_dalek::Signature::from_bytes(&key.binding_signature);
    verifying_key
        .verify_strict(&sig_input(node, &key.plugin, &key.public_key), &signature)
        .is_ok()
}

pub(crate) fn to_proto(keys: &[BoundPluginKey]) -> Vec<crate::proto::node::PluginKey> {
    keys.iter()
        .take(MAX_PLUGIN_KEYS)
        .map(|key| crate::proto::node::PluginKey {
            plugin: key.plugin.clone(),
            public_key: key.public_key.to_vec(),
            binding_signature: key.binding_signature.to_vec(),
            signature_algorithm: SIGNATURE_ALGORITHM.to_string(),
        })
        .collect()
}

/// The keys in `keys` whose binding is `node`'s own signature, at most
/// [`MAX_PLUGIN_KEYS`], one per plugin (the first). Anything malformed or
/// unverified is dropped, never kept as unverified.
pub(crate) fn verified_from_proto(
    node: &EndpointId,
    keys: &[crate::proto::node::PluginKey],
) -> Vec<BoundPluginKey> {
    let mut out: Vec<BoundPluginKey> = Vec::new();
    for key in keys.iter().take(MAX_PLUGIN_KEYS) {
        if key.signature_algorithm != SIGNATURE_ALGORITHM || !valid_plugin_name(&key.plugin) {
            continue;
        }
        let (Ok(public_key), Ok(binding_signature)) = (
            <[u8; 32]>::try_from(key.public_key.as_slice()),
            <[u8; 64]>::try_from(key.binding_signature.as_slice()),
        ) else {
            continue;
        };
        let bound = BoundPluginKey {
            plugin: key.plugin.clone(),
            public_key,
            binding_signature,
        };
        if verify(node, &bound) && !out.iter().any(|kept| kept.plugin == bound.plugin) {
            out.push(bound);
        }
    }
    out
}

#[derive(Default)]
struct State {
    own: BTreeMap<String, BoundPluginKey>,
    peers: HashMap<EndpointId, Vec<BoundPluginKey>>,
}

/// This node's announced plugin keys and the verified ones its peers announced.
#[derive(Clone, Default)]
pub(crate) struct PluginKeys {
    inner: Arc<Mutex<State>>,
}

impl PluginKeys {
    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Announce `key` for its plugin, replacing that plugin's earlier key.
    /// Refused when this would exceed [`MAX_PLUGIN_KEYS`].
    pub(crate) fn set_own(&self, key: BoundPluginKey) -> Result<(), String> {
        let mut state = self.state();
        if !state.own.contains_key(&key.plugin) && state.own.len() >= MAX_PLUGIN_KEYS {
            return Err(format!(
                "this node already announces {MAX_PLUGIN_KEYS} plugin keys"
            ));
        }
        state.own.insert(key.plugin.clone(), key);
        Ok(())
    }

    /// Stop announcing `plugin`'s key.
    pub(crate) fn remove_own(&self, plugin: &str) {
        self.state().own.remove(plugin);
    }

    pub(crate) fn own(&self) -> Vec<BoundPluginKey> {
        self.state().own.values().cloned().collect()
    }

    /// What `peer` announces now: replaces what it announced before; an empty
    /// list forgets it.
    pub(crate) fn set_peer(&self, peer: EndpointId, keys: Vec<BoundPluginKey>) {
        let mut state = self.state();
        if keys.is_empty() {
            state.peers.remove(&peer);
        } else {
            state.peers.insert(peer, keys);
        }
    }

    pub(crate) fn peers(&self) -> HashMap<EndpointId, Vec<BoundPluginKey>> {
        self.state().peers.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plugin_public_key() -> [u8; 32] {
        ed25519_dalek::SigningKey::from_bytes(&[7; 32])
            .verifying_key()
            .to_bytes()
    }

    #[test]
    fn a_binding_verifies_against_its_own_node_only() {
        let node = SecretKey::generate();
        let other = SecretKey::generate();
        let key = bind(&node, "capsules", plugin_public_key());
        assert!(verify(&node.public(), &key));
        assert!(
            !verify(&other.public(), &key),
            "another node's key never verifies it"
        );
    }

    #[test]
    fn a_changed_plugin_name_key_or_signature_does_not_verify() {
        let node = SecretKey::generate();
        let key = bind(&node, "capsules", plugin_public_key());
        let mut renamed = key.clone();
        renamed.plugin = "other".to_string();
        assert!(!verify(&node.public(), &renamed));
        let mut swapped = key.clone();
        swapped.public_key[0] ^= 1;
        assert!(!verify(&node.public(), &swapped));
        let mut forged = key;
        forged.binding_signature[0] ^= 1;
        assert!(!verify(&node.public(), &forged));
    }

    #[test]
    fn proto_round_trip_keeps_verified_keys_and_drops_the_rest() {
        let node = SecretKey::generate();
        let good = bind(&node, "capsules", plugin_public_key());
        let mut wire = to_proto(std::slice::from_ref(&good));
        let mut bad_sig = wire[0].clone();
        bad_sig.plugin = "second".to_string();
        let mut bad_len = wire[0].clone();
        bad_len.plugin = "third".to_string();
        bad_len.public_key.pop();
        let mut bad_alg = wire[0].clone();
        bad_alg.signature_algorithm = "rsa".to_string();
        let mut bad_name = wire[0].clone();
        bad_name.plugin = "has space".to_string();
        let duplicate = wire[0].clone();
        wire.extend([bad_sig, bad_len, bad_alg, bad_name, duplicate]);
        assert_eq!(verified_from_proto(&node.public(), &wire), vec![good]);
    }

    #[test]
    fn at_most_sixteen_keys_are_read() {
        let node = SecretKey::generate();
        let keys: Vec<BoundPluginKey> = (0..20)
            .map(|i| bind(&node, &format!("plugin-{i}"), plugin_public_key()))
            .collect();
        let wire: Vec<_> = keys
            .iter()
            .map(|key| to_proto(std::slice::from_ref(key)).remove(0))
            .collect();
        assert_eq!(
            verified_from_proto(&node.public(), &wire).len(),
            MAX_PLUGIN_KEYS
        );
    }

    #[test]
    fn own_keys_replace_per_plugin_and_are_capped() {
        let node = SecretKey::generate();
        let keys = PluginKeys::default();
        keys.set_own(bind(&node, "capsules", plugin_public_key()))
            .unwrap();
        let replacement = bind(&node, "capsules", [9; 32]);
        keys.set_own(replacement.clone()).unwrap();
        assert_eq!(keys.own(), vec![replacement]);
        for i in 1..MAX_PLUGIN_KEYS {
            keys.set_own(bind(&node, &format!("p{i}"), plugin_public_key()))
                .unwrap();
        }
        assert!(
            keys.set_own(bind(&node, "one-too-many", plugin_public_key()))
                .is_err()
        );
        keys.remove_own("capsules");
        assert_eq!(keys.own().len(), MAX_PLUGIN_KEYS - 1);
    }

    #[test]
    fn a_peer_announcing_no_keys_is_forgotten() {
        let node = SecretKey::generate();
        let keys = PluginKeys::default();
        keys.set_peer(
            node.public(),
            vec![bind(&node, "capsules", plugin_public_key())],
        );
        assert_eq!(keys.peers().len(), 1);
        keys.set_peer(node.public(), Vec::new());
        assert!(keys.peers().is_empty());
    }

    #[test]
    fn plugin_names_are_bounded() {
        assert!(valid_plugin_name("capsules"));
        assert!(valid_plugin_name("example-plugin_name.v2"));
        assert!(!valid_plugin_name(""));
        assert!(!valid_plugin_name("a b"));
        assert!(!valid_plugin_name(&"x".repeat(MAX_PLUGIN_NAME_BYTES + 1)));
    }
}
