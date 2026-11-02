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
//!
//! A plugin that withdraws its key also makes this node sign a
//! [`Withdrawal`] of it (`PluginKeyWithdrawal`), announced on the same entry.
//! A receiver keeps a peer's verified withdrawals after the peer leaves, up to
//! [`MAX_RETAINED_WITHDRAWAL_PEERS`] peers, so a plugin forgets the key at its
//! next poll even when the host that withdrew it is offline.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, PoisonError};

use iroh::{EndpointId, SecretKey};

/// Domain-separation tag of the binding's signed bytes; see `PluginKey`.
pub(crate) const SIG_DOMAIN: &[u8] = b"mesh-llm-plugin-key-v1:";
pub(crate) const SIGNATURE_ALGORITHM: &str = "ed25519";
/// At most this many plugin keys per node, sent or kept.
pub(crate) const MAX_PLUGIN_KEYS: usize = 16;
/// Domain-separation tag of a withdrawal's signed bytes; see `PluginKeyWithdrawal`.
pub(crate) const WITHDRAWAL_SIG_DOMAIN: &[u8] = b"mesh-llm-plugin-key-withdrawal-v1:";
/// How many departed peers' withdrawals a node keeps; the oldest are dropped
/// first, so the memory a peer can cost is bounded.
pub(crate) const MAX_RETAINED_WITHDRAWAL_PEERS: usize = 256;
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

/// A node's signed statement that it no longer vouches for `key_id`, the
/// public key it announced for `plugin`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Withdrawal {
    pub(crate) plugin: String,
    pub(crate) key_id: [u8; 32],
    pub(crate) withdrawal_signature: [u8; 64],
}

/// The keys and withdrawals one node announces.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Announced {
    pub(crate) keys: Vec<BoundPluginKey>,
    pub(crate) withdrawals: Vec<Withdrawal>,
}

impl From<Vec<BoundPluginKey>> for Announced {
    fn from(keys: Vec<BoundPluginKey>) -> Self {
        Self {
            keys,
            withdrawals: Vec::new(),
        }
    }
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

/// The bytes a node signs to withdraw `key_id` for `plugin`.
pub(crate) fn withdrawal_sig_input(node: &EndpointId, plugin: &str, key_id: &[u8; 32]) -> Vec<u8> {
    let mut out = WITHDRAWAL_SIG_DOMAIN.to_vec();
    write_bytes(&mut out, node.as_bytes());
    write_bytes(&mut out, plugin.as_bytes());
    write_bytes(&mut out, key_id);
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

/// Withdraw `key_id` for `plugin`, signed by the node `secret` belongs to.
pub(crate) fn withdraw(secret: &SecretKey, plugin: &str, key_id: [u8; 32]) -> Withdrawal {
    let signing_key = ed25519_dalek::SigningKey::from_bytes(&secret.to_bytes());
    let signature = ed25519_dalek::Signer::sign(
        &signing_key,
        &withdrawal_sig_input(&secret.public(), plugin, &key_id),
    );
    Withdrawal {
        plugin: plugin.to_string(),
        key_id,
        withdrawal_signature: signature.to_bytes(),
    }
}

/// Whether `withdrawal` is `node`'s own signature.
pub(crate) fn verify_withdrawal(node: &EndpointId, withdrawal: &Withdrawal) -> bool {
    let Ok(verifying_key) = ed25519_dalek::VerifyingKey::from_bytes(node.as_bytes()) else {
        return false;
    };
    let signature = ed25519_dalek::Signature::from_bytes(&withdrawal.withdrawal_signature);
    verifying_key
        .verify_strict(
            &withdrawal_sig_input(node, &withdrawal.plugin, &withdrawal.key_id),
            &signature,
        )
        .is_ok()
}

pub(crate) fn withdrawals_to_proto(
    withdrawals: &[Withdrawal],
) -> Vec<crate::proto::node::PluginKeyWithdrawal> {
    withdrawals
        .iter()
        .take(MAX_PLUGIN_KEYS)
        .map(|withdrawal| crate::proto::node::PluginKeyWithdrawal {
            plugin: withdrawal.plugin.clone(),
            key_id: withdrawal.key_id.to_vec(),
            withdrawal_signature: withdrawal.withdrawal_signature.to_vec(),
            signature_algorithm: SIGNATURE_ALGORITHM.to_string(),
        })
        .collect()
}

/// The withdrawals in `withdrawals` that are `node`'s own signature, at most
/// [`MAX_PLUGIN_KEYS`], one per plugin (the first). Anything malformed or
/// unverified is dropped.
pub(crate) fn verified_withdrawals_from_proto(
    node: &EndpointId,
    withdrawals: &[crate::proto::node::PluginKeyWithdrawal],
) -> Vec<Withdrawal> {
    let mut out: Vec<Withdrawal> = Vec::new();
    for wire in withdrawals.iter().take(MAX_PLUGIN_KEYS) {
        if wire.signature_algorithm != SIGNATURE_ALGORITHM || !valid_plugin_name(&wire.plugin) {
            continue;
        }
        let (Ok(key_id), Ok(withdrawal_signature)) = (
            <[u8; 32]>::try_from(wire.key_id.as_slice()),
            <[u8; 64]>::try_from(wire.withdrawal_signature.as_slice()),
        ) else {
            continue;
        };
        let withdrawal = Withdrawal {
            plugin: wire.plugin.clone(),
            key_id,
            withdrawal_signature,
        };
        if verify_withdrawal(node, &withdrawal)
            && !out.iter().any(|kept| kept.plugin == withdrawal.plugin)
        {
            out.push(withdrawal);
        }
    }
    out
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
    own_withdrawn: BTreeMap<String, Withdrawal>,
    peers: HashMap<EndpointId, Vec<BoundPluginKey>>,
    /// A peer's last verified withdrawals, with when they arrived (a counter,
    /// for dropping the oldest); kept after the peer leaves.
    peer_withdrawn: HashMap<EndpointId, (u64, Vec<Withdrawal>)>,
    withdrawn_clock: u64,
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
    /// Announcing the very key it withdrew reinstates it (the withdrawal is
    /// dropped); a different key leaves the withdrawal of the old one in
    /// place. Refused when this would exceed [`MAX_PLUGIN_KEYS`].
    pub(crate) fn set_own(&self, key: BoundPluginKey) -> Result<(), String> {
        let mut state = self.state();
        if !state.own.contains_key(&key.plugin) && state.own.len() >= MAX_PLUGIN_KEYS {
            return Err(format!(
                "this node already announces {MAX_PLUGIN_KEYS} plugin keys"
            ));
        }
        if state
            .own_withdrawn
            .get(&key.plugin)
            .is_some_and(|w| w.key_id == key.public_key)
        {
            state.own_withdrawn.remove(&key.plugin);
        }
        state.own.insert(key.plugin.clone(), key);
        Ok(())
    }

    /// Stop announcing `plugin`'s key, without announcing a withdrawal.
    pub(crate) fn remove_own(&self, plugin: &str) {
        self.state().own.remove(plugin);
    }

    /// Stop announcing `plugin`'s key and announce `sign(key)`'s withdrawal of
    /// it in its place, replacing any earlier withdrawal for the plugin. Does
    /// nothing when no key is announced for `plugin`.
    pub(crate) fn withdraw_own(
        &self,
        plugin: &str,
        sign: impl FnOnce(&BoundPluginKey) -> Withdrawal,
    ) {
        let mut state = self.state();
        if let Some(key) = state.own.remove(plugin) {
            let withdrawal = sign(&key);
            // At most MAX_PLUGIN_KEYS are announced, so make room by dropping
            // another plugin's withdrawal, the first by name.
            if !state.own_withdrawn.contains_key(plugin)
                && state.own_withdrawn.len() >= MAX_PLUGIN_KEYS
                && let Some(first) = state.own_withdrawn.keys().next().cloned()
            {
                state.own_withdrawn.remove(&first);
            }
            state.own_withdrawn.insert(plugin.to_string(), withdrawal);
        }
    }

    pub(crate) fn own(&self) -> Vec<BoundPluginKey> {
        self.state().own.values().cloned().collect()
    }

    /// Everything this node announces now, taken under one lock.
    pub(crate) fn own_announced(&self) -> Announced {
        let state = self.state();
        Announced {
            keys: state.own.values().cloned().collect(),
            withdrawals: state.own_withdrawn.values().cloned().collect(),
        }
    }

    pub(crate) fn own_withdrawals(&self) -> Vec<Withdrawal> {
        self.state().own_withdrawn.values().cloned().collect()
    }

    /// What `peer` announces now: replaces the keys it announced before;
    /// nothing forgets them. Its earlier withdrawals are kept, whatever it
    /// announces now (a host that restarted or rotated has lost them but its
    /// old key is no less withdrawn), unless it now announces the same key
    /// again or withdraws another key for that plugin. A key a kept or new
    /// withdrawal covers is not listed.
    pub(crate) fn set_peer(&self, peer: EndpointId, announced: impl Into<Announced>) {
        let Announced {
            mut keys,
            withdrawals,
        } = announced.into();
        let mut state = self.state();
        let mut merged = withdrawals;
        for kept in state
            .peer_withdrawn
            .remove(&peer)
            .map(|(_, earlier)| earlier)
            .unwrap_or_default()
        {
            let superseded = merged.iter().any(|w| w.plugin == kept.plugin);
            let reinstated = keys
                .iter()
                .any(|key| key.plugin == kept.plugin && key.public_key == kept.key_id);
            if !superseded && !reinstated {
                merged.push(kept);
            }
        }
        merged.truncate(MAX_PLUGIN_KEYS);
        keys.retain(|key| {
            !merged
                .iter()
                .any(|w| w.plugin == key.plugin && w.key_id == key.public_key)
        });
        if keys.is_empty() {
            state.peers.remove(&peer);
        } else {
            state.peers.insert(peer, keys);
        }
        if merged.is_empty() {
            return;
        }
        state.withdrawn_clock += 1;
        let stamp = state.withdrawn_clock;
        state.peer_withdrawn.insert(peer, (stamp, merged));
        if state.peer_withdrawn.len() > MAX_RETAINED_WITHDRAWAL_PEERS
            && let Some(oldest) = state
                .peer_withdrawn
                .iter()
                .min_by_key(|(_, (stamp, _))| *stamp)
                .map(|(peer, _)| *peer)
        {
            state.peer_withdrawn.remove(&oldest);
        }
    }

    /// `peer` left: stop listing its keys, but keep its withdrawals so a
    /// plugin still learns of them.
    pub(crate) fn forget_peer(&self, peer: EndpointId) {
        self.state().peers.remove(&peer);
    }

    /// `peer` was rejected or removed: forget its keys and its withdrawals.
    pub(crate) fn purge_peer(&self, peer: EndpointId) {
        let mut state = self.state();
        state.peers.remove(&peer);
        state.peer_withdrawn.remove(&peer);
    }

    pub(crate) fn peers(&self) -> HashMap<EndpointId, Vec<BoundPluginKey>> {
        self.state().peers.clone()
    }

    pub(crate) fn peer_withdrawals(&self) -> HashMap<EndpointId, Vec<Withdrawal>> {
        self.state()
            .peer_withdrawn
            .iter()
            .map(|(peer, (_, withdrawals))| (*peer, withdrawals.clone()))
            .collect()
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
        keys.set_peer(node.public(), Vec::<BoundPluginKey>::new());
        assert!(keys.peers().is_empty());
    }

    #[test]
    fn a_withdrawal_verifies_against_its_own_node_only() {
        let node = SecretKey::generate();
        let other = SecretKey::generate();
        let withdrawal = withdraw(&node, "capsules", plugin_public_key());
        assert!(verify_withdrawal(&node.public(), &withdrawal));
        assert!(!verify_withdrawal(&other.public(), &withdrawal));
        let mut renamed = withdrawal.clone();
        renamed.plugin = "other".to_string();
        assert!(!verify_withdrawal(&node.public(), &renamed));
        let mut swapped = withdrawal.clone();
        swapped.key_id[0] ^= 1;
        assert!(!verify_withdrawal(&node.public(), &swapped));
        // A key's binding is not a withdrawal of it, and the reverse.
        let bound = bind(&node, "capsules", plugin_public_key());
        assert_ne!(bound.binding_signature, withdrawal.withdrawal_signature);
    }

    #[test]
    fn withdrawals_round_trip_and_unverified_ones_are_dropped() {
        let node = SecretKey::generate();
        let good = withdraw(&node, "capsules", plugin_public_key());
        let mut wire = withdrawals_to_proto(std::slice::from_ref(&good));
        let mut bad_sig = wire[0].clone();
        bad_sig.plugin = "second".to_string();
        let mut bad_len = wire[0].clone();
        bad_len.plugin = "third".to_string();
        bad_len.key_id.pop();
        let mut bad_alg = wire[0].clone();
        bad_alg.signature_algorithm = "rsa".to_string();
        let duplicate = wire[0].clone();
        wire.extend([bad_sig, bad_len, bad_alg, duplicate]);
        assert_eq!(
            verified_withdrawals_from_proto(&node.public(), &wire),
            vec![good]
        );
    }

    #[test]
    fn withdrawing_own_key_swaps_it_for_a_withdrawal_until_it_is_announced_again() {
        let node = SecretKey::generate();
        let keys = PluginKeys::default();
        keys.withdraw_own("capsules", |_| unreachable!("nothing to withdraw"));
        assert!(keys.own_withdrawals().is_empty());
        keys.set_own(bind(&node, "capsules", plugin_public_key()))
            .unwrap();
        keys.withdraw_own("capsules", |key| {
            withdraw(&node, &key.plugin, key.public_key)
        });
        assert!(keys.own().is_empty());
        assert_eq!(
            keys.own_withdrawals(),
            vec![withdraw(&node, "capsules", plugin_public_key())]
        );
        keys.set_own(bind(&node, "capsules", plugin_public_key()))
            .unwrap();
        assert!(
            keys.own_withdrawals().is_empty(),
            "the same key is reinstated"
        );
    }

    #[test]
    fn a_peers_withdrawn_key_is_not_listed_and_its_withdrawal_outlives_the_peer() {
        let peer = SecretKey::generate();
        let keys = PluginKeys::default();
        let key = bind(&peer, "capsules", plugin_public_key());
        let withdrawal = withdraw(&peer, "capsules", plugin_public_key());
        keys.set_peer(
            peer.public(),
            Announced {
                keys: vec![key.clone()],
                withdrawals: vec![withdrawal.clone()],
            },
        );
        assert!(keys.peers().is_empty(), "a withdrawn key is not listed");
        keys.forget_peer(peer.public());
        assert_eq!(
            keys.peer_withdrawals()[&peer.public()],
            vec![withdrawal.clone()]
        );
        // It reconnects announcing nothing (a restart): the withdrawal stays.
        keys.set_peer(peer.public(), Vec::<BoundPluginKey>::new());
        assert_eq!(
            keys.peer_withdrawals()[&peer.public()],
            vec![withdrawal.clone()]
        );
        assert!(keys.peers().is_empty());
        // It announces the same key again: reinstated, the withdrawal is gone.
        keys.set_peer(peer.public(), vec![key.clone()]);
        assert_eq!(keys.peers()[&peer.public()], vec![key]);
        assert!(keys.peer_withdrawals().is_empty());
        keys.set_peer(
            peer.public(),
            Announced {
                keys: Vec::new(),
                withdrawals: vec![withdrawal],
            },
        );
        keys.purge_peer(peer.public());
        assert!(keys.peer_withdrawals().is_empty());
    }

    #[test]
    fn a_rotation_keeps_the_old_keys_withdrawal_and_a_later_one_supersedes_it() {
        let peer = SecretKey::generate();
        let keys = PluginKeys::default();
        let w1 = withdraw(&peer, "capsules", [1; 32]);
        keys.set_peer(
            peer.public(),
            Announced {
                keys: Vec::new(),
                withdrawals: vec![w1.clone()],
            },
        );
        // The host rotated to K2 and no longer sends the withdrawal of K1.
        let k2 = bind(&peer, "capsules", plugin_public_key());
        keys.set_peer(peer.public(), vec![k2.clone()]);
        assert_eq!(keys.peers()[&peer.public()], vec![k2]);
        assert_eq!(keys.peer_withdrawals()[&peer.public()], vec![w1]);
        let w2 = withdraw(&peer, "capsules", plugin_public_key());
        keys.set_peer(
            peer.public(),
            Announced {
                keys: Vec::new(),
                withdrawals: vec![w2.clone()],
            },
        );
        assert_eq!(keys.peer_withdrawals()[&peer.public()], vec![w2]);
        assert!(keys.peers().is_empty());
    }

    #[test]
    fn own_withdrawal_survives_a_rotation_and_is_capped() {
        let node = SecretKey::generate();
        let keys = PluginKeys::default();
        keys.set_own(bind(&node, "capsules", [1; 32])).unwrap();
        keys.withdraw_own("capsules", |k| withdraw(&node, &k.plugin, k.public_key));
        keys.set_own(bind(&node, "capsules", plugin_public_key()))
            .unwrap();
        assert_eq!(
            keys.own_withdrawals(),
            vec![withdraw(&node, "capsules", [1; 32])]
        );
        // Announcing the withdrawn key itself reinstates it.
        keys.set_own(bind(&node, "capsules", [1; 32])).unwrap();
        assert!(keys.own_withdrawals().is_empty());
        for i in 0..=MAX_PLUGIN_KEYS {
            let plugin = format!("p{i:02}");
            keys.set_own(bind(&node, &plugin, plugin_public_key())).ok();
            keys.withdraw_own(&plugin, |k| withdraw(&node, &k.plugin, k.public_key));
            keys.remove_own(&plugin);
        }
        assert!(keys.own_withdrawals().len() <= MAX_PLUGIN_KEYS);
    }

    #[test]
    fn departed_peers_withdrawals_are_bounded_and_the_oldest_go_first() {
        let keys = PluginKeys::default();
        let peers: Vec<SecretKey> = (0..=MAX_RETAINED_WITHDRAWAL_PEERS)
            .map(|_| SecretKey::generate())
            .collect();
        for peer in &peers {
            keys.set_peer(
                peer.public(),
                Announced {
                    keys: Vec::new(),
                    withdrawals: vec![withdraw(peer, "capsules", plugin_public_key())],
                },
            );
        }
        let kept = keys.peer_withdrawals();
        assert_eq!(kept.len(), MAX_RETAINED_WITHDRAWAL_PEERS);
        assert!(
            !kept.contains_key(&peers[0].public()),
            "the oldest is dropped"
        );
        assert!(kept.contains_key(&peers[MAX_RETAINED_WITHDRAWAL_PEERS].public()));
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
