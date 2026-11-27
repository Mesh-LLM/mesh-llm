use crate::crypto::owner_id_from_verifying_key;
use crate::proto::node::{MeshPolicyPayload, SignedMeshPolicy};
use prost::Message;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

pub(crate) mod applicator;
pub(crate) mod config_overlay;

#[derive(Debug, Default)]
pub(crate) struct PolicyStore {
    current: RwLock<Option<StoredPolicy>>,
    /// Per-owner monotonic generation counters for stale-rejection.
    /// Each owner's generation is tracked independently so that a low
    /// generation from owner B is never rejected because owner A's
    /// accepted generation happens to be higher.
    owner_gens: RwLock<HashMap<String, u32>>,
}

#[derive(Debug, Clone)]
pub(crate) struct StoredPolicy {
    pub signed: SignedMeshPolicy,
    pub payload: MeshPolicyPayload,
    pub received_at: std::time::Instant,
}

#[derive(Debug)]
pub(crate) enum PolicyUpdateResult {
    Accepted { policy_gen: u32 },
    Stale { current_gen: u32 },
    Invalid(String),
}

impl PolicyStore {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Update the store with an incoming signed policy.
    ///
    /// `authorized_owner_ids` is the set of owner IDs that are permitted to
    /// inject policy into this node.  Pass the node's own verified owner ID
    /// (and any additionally trusted owners).  An empty slice rejects every
    /// policy because no owner is authorised.
    ///
    /// `node_mesh_id` is this node's current mesh ID (empty string or `None`
    /// means no check).  When the payload carries a non-empty `mesh_id`, it
    /// must match this value or the policy is rejected.
    pub async fn update(
        &self,
        incoming: SignedMeshPolicy,
        authorized_owner_ids: &[String],
        node_mesh_id: Option<&str>,
    ) -> PolicyUpdateResult {
        let payload = match MeshPolicyPayload::decode(incoming.policy_payload_bytes.as_slice()) {
            Ok(p) => p,
            Err(e) => return PolicyUpdateResult::Invalid(format!("decode error: {e}")),
        };
        // Verify owner public key and signature before trusting any payload field.
        let vk = {
            let key_hex = &payload.owner_sign_public_key;
            let key_bytes_vec = match hex::decode(key_hex) {
                Ok(b) => b,
                Err(_) => {
                    return PolicyUpdateResult::Invalid(
                        "owner_sign_public_key is not valid hex".into(),
                    )
                }
            };
            let key_arr: [u8; 32] = match key_bytes_vec.try_into() {
                Ok(a) => a,
                Err(_) => {
                    return PolicyUpdateResult::Invalid(
                        "owner_sign_public_key must be 32 bytes".into(),
                    )
                }
            };
            match ed25519_dalek::VerifyingKey::from_bytes(&key_arr) {
                Ok(k) => k,
                Err(_) => {
                    return PolicyUpdateResult::Invalid(
                        "owner_sign_public_key is not a valid Ed25519 key".into(),
                    )
                }
            }
        };
        // owner_id must be sha256(public_key) — prevents impersonation.
        let derived_owner_id = owner_id_from_verifying_key(&vk);
        if payload.owner_id != derived_owner_id {
            return PolicyUpdateResult::Invalid(
                "owner_id does not match sha256(owner_sign_public_key)".into(),
            );
        }
        // Reject policies from owners not on this node's trust list.
        // This prevents any valid Ed25519 key-pair from injecting policy.
        if !authorized_owner_ids.iter().any(|id| id == &payload.owner_id) {
            return PolicyUpdateResult::Invalid(
                "owner_id is not authorised to control this node".into(),
            );
        }
        // Reject policies that target a different mesh (P2c).
        // A non-empty `mesh_id` in the payload is an explicit scope guard.
        if !payload.mesh_id.is_empty() {
            if let Some(nid) = node_mesh_id {
                if !nid.is_empty() && payload.mesh_id != nid {
                    return PolicyUpdateResult::Invalid(format!(
                        "policy mesh_id {:?} does not match this node's mesh_id {:?}",
                        payload.mesh_id, nid,
                    ));
                }
            }
        }
        // Signature must cover the raw policy_payload_bytes.
        let sig_arr: [u8; 64] = match incoming.signature.as_slice().try_into() {
            Ok(a) => a,
            Err(_) => {
                return PolicyUpdateResult::Invalid("signature must be 64 bytes".into())
            }
        };
        let sig = ed25519_dalek::Signature::from_bytes(&sig_arr);
        if vk
            .verify_strict(&incoming.policy_payload_bytes, &sig)
            .is_err()
        {
            return PolicyUpdateResult::Invalid("policy signature verification failed".into());
        }
        // Per-owner stale check: compare against this owner's last accepted
        // generation only, not against whatever other owner's gen happens to
        // sit in `current`.
        let mut owner_gens_guard = self.owner_gens.write().await;
        let owner_current_gen = owner_gens_guard
            .get(&payload.owner_id)
            .copied()
            .unwrap_or(0);
        if payload.policy_gen <= owner_current_gen && owner_current_gen > 0 {
            return PolicyUpdateResult::Stale {
                current_gen: owner_current_gen,
            };
        }
        let policy_gen = payload.policy_gen;
        // Update the per-owner generation tracking. Acquire the `current`
        // write lock BEFORE dropping `owner_gens_guard` so that two concurrent
        // updates for the same owner cannot publish out of order: the one that
        // wins the `owner_gens` insert holds both locks until it commits.
        owner_gens_guard.insert(payload.owner_id.clone(), policy_gen);
        let mut guard = self.current.write().await;
        drop(owner_gens_guard);
        // P2b tiebreaking: when multiple authorized owners can submit policies,
        // avoid a "last write wins" race when their generations are equal.
        // Use lexicographic ordering of owner_id as a stable tiebreaker so all
        // nodes converge to the same current policy under concurrent updates.
        let should_replace = match guard.as_ref() {
            None => true,
            Some(existing) => {
                let new_gen = payload.policy_gen;
                let old_gen = existing.payload.policy_gen;
                if new_gen != old_gen {
                    new_gen > old_gen
                } else {
                    // Same generation: lexicographically smaller owner_id wins
                    // (deterministic tiebreaker — both nodes converge to the same winner).
                    payload.owner_id < existing.payload.owner_id
                }
            }
        };
        if should_replace {
            *guard = Some(StoredPolicy {
                signed: incoming,
                payload,
                received_at: std::time::Instant::now(),
            });
        }
        PolicyUpdateResult::Accepted { policy_gen }
    }

    pub async fn current(&self) -> Option<StoredPolicy> {
        self.current.read().await.clone()
    }

    /// Synchronous snapshot of the current policy for use in non-async contexts
    /// such as building outbound gossip. Returns `None` if no policy is stored
    /// or if the lock is unavailable.
    pub fn current_sync(&self) -> Option<StoredPolicy> {
        self.current.try_read().ok().and_then(|g| g.clone())
    }

    pub async fn clear(&self) {
        *self.current.write().await = None;
    }

    pub async fn persist(&self, data_dir: &std::path::Path) -> anyhow::Result<()> {
        if let Some(stored) = self.current().await {
            let bytes = prost::Message::encode_to_vec(&stored.signed);
            let path = data_dir.join("mesh_policy.bin");
            let tmp = data_dir.join("mesh_policy.bin.tmp");
            tokio::fs::write(&tmp, &bytes).await?;
            tokio::fs::rename(&tmp, &path).await?;
            // Persist owner_gens so per-owner stale-rejection survives a
            // restart (finding #5).
            let gens_snapshot = self.owner_gens.read().await.clone();
            let gens_bytes = serde_json::to_vec(&gens_snapshot)?;
            let gens_path = data_dir.join("mesh_policy_gens.json");
            let gens_tmp = data_dir.join("mesh_policy_gens.json.tmp");
            tokio::fs::write(&gens_tmp, &gens_bytes).await?;
            tokio::fs::rename(&gens_tmp, &gens_path).await?;
        }
        Ok(())
    }

    /// Delete the persisted policy file (and its owner-gens sidecar) from
    /// disk. Called by the DELETE handler so a cleared policy is not
    /// re-applied on restart.
    pub async fn delete_persisted(&self, data_dir: &std::path::Path) -> anyhow::Result<()> {
        for name in &["mesh_policy.bin", "mesh_policy_gens.json"] {
            let path = data_dir.join(name);
            match tokio::fs::remove_file(&path).await {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
        Ok(())
    }

    pub async fn load_from_disk(
        data_dir: &std::path::Path,
    ) -> anyhow::Result<Option<crate::proto::node::SignedMeshPolicy>> {
        let path = data_dir.join("mesh_policy.bin");
        match tokio::fs::read(&path).await {
            Ok(bytes) => {
                let signed = prost::Message::decode(bytes.as_slice())?;
                // Restore owner_gens sidecar if present so previously-rejected
                // generations remain rejected after a restart.
                // (This instance is not `self` — the caller wires it into the
                // store via `apply_if_newer`, which calls `update()`; the
                // per-owner gen will be (re-)recorded there.  The sidecar is
                // read by the store's own `restore_owner_gens` helper so the
                // store can pre-populate `owner_gens` before accepting gossip.)
                Ok(Some(signed))
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Restore persisted per-owner generation counters from disk so stale
    /// policies are still rejected after a restart.
    pub async fn restore_owner_gens(&self, data_dir: &std::path::Path) {
        let path = data_dir.join("mesh_policy_gens.json");
        match tokio::fs::read(&path).await {
            Ok(bytes) => {
                if let Ok(gens) =
                    serde_json::from_slice::<HashMap<String, u32>>(&bytes)
                {
                    let mut guard = self.owner_gens.write().await;
                    for (owner, r#gen) in gens {
                        guard.entry(owner).or_insert(r#gen);
                    }
                }
            }
            // Sidecar absent (old installation) — harmless, start fresh.
            Err(_) => {}
        }
    }
}
