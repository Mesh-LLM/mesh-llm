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

    pub async fn update(&self, incoming: SignedMeshPolicy) -> PolicyUpdateResult {
        let payload = match MeshPolicyPayload::decode(incoming.policy_payload_bytes.as_slice()) {
            Ok(p) => p,
            Err(e) => return PolicyUpdateResult::Invalid(format!("decode error: {e}")),
        };
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
        // Update the per-owner generation tracking.
        owner_gens_guard.insert(payload.owner_id.clone(), policy_gen);
        drop(owner_gens_guard);

        let mut guard = self.current.write().await;
        *guard = Some(StoredPolicy {
            signed: incoming,
            payload,
            received_at: std::time::Instant::now(),
        });
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
        }
        Ok(())
    }

    pub async fn load_from_disk(
        data_dir: &std::path::Path,
    ) -> anyhow::Result<Option<crate::proto::node::SignedMeshPolicy>> {
        let path = data_dir.join("mesh_policy.bin");
        match tokio::fs::read(&path).await {
            Ok(bytes) => Ok(Some(prost::Message::decode(bytes.as_slice())?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}
