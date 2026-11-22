use crate::proto::node::{MeshPolicyPayload, SignedMeshPolicy};
use prost::Message;
use std::sync::Arc;
use tokio::sync::RwLock;

pub(crate) mod applicator;
pub(crate) mod config_overlay;

#[derive(Debug, Default)]
pub(crate) struct PolicyStore {
    current: RwLock<Option<StoredPolicy>>,
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
        let mut guard = self.current.write().await;
        if let Some(current) = guard.as_ref() {
            if payload.policy_gen <= current.payload.policy_gen {
                return PolicyUpdateResult::Stale {
                    current_gen: current.payload.policy_gen,
                };
            }
        }
        let policy_gen = payload.policy_gen;
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
