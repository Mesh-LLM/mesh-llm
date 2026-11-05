//! Renewal and persistence of the ownership certificate for one node identity.

use super::Node;
use crate::crypto::{SignedNodeOwnership, default_node_ownership_path, save_node_ownership};
use std::path::{Path, PathBuf};

/// Resolve the same storage scope used when the node first signs its certificate.
pub(super) fn owner_attestation_path(identity_directory: Option<&Path>) -> anyhow::Result<PathBuf> {
    match identity_directory {
        Some(directory) => Ok(directory.join("node-ownership.json")),
        None => Ok(default_node_ownership_path()?),
    }
}

impl Node {
    /// Retry failed writes on every heartbeat, within this identity's storage.
    /// A failed disk write must not discard the renewed in-memory certificate or
    /// fall back to another identity's storage.
    async fn save_renewed_owner_attestation(&self) {
        let Some(path) = &self.owner_attestation_path else {
            return;
        };
        let mut unsaved = self.unsaved_owner_attestation.lock().await;
        let Some(renewed) = unsaved.as_ref() else {
            return;
        };
        match save_node_ownership(path, renewed) {
            Ok(()) => *unsaved = None,
            Err(error) => tracing::warn!(
                "could not save the renewed ownership certificate, retrying on the next heartbeat: {error:#}"
            ),
        }
    }

    /// The ownership certificate is signed at startup for 7 days. Re-signing it
    /// inside its renewal window keeps peers that enforce an owner policy from
    /// dropping a node that runs longer than that.
    pub(in crate::mesh) async fn renew_owner_attestation_on_heartbeat(&self, now_unix_ms: u64) {
        match self.renew_owner_attestation_if_due(now_unix_ms).await {
            Ok(Some(renewed)) => {
                tracing::info!(
                    expires_at_unix_ms = renewed.claim.expires_at_unix_ms,
                    "renewed this node's ownership certificate"
                );
                if self.owner_attestation_path.is_some() {
                    *self.unsaved_owner_attestation.lock().await = Some(renewed);
                }
            }
            Ok(None) => {}
            Err(error) => tracing::warn!(
                "could not renew this node's ownership certificate, retrying on the next heartbeat: {error:#}"
            ),
        }
        self.save_renewed_owner_attestation().await;
    }

    /// Swaps in a renewed ownership certificate, and the summary `/api/status`
    /// reports, once the current one is inside its renewal window. Gossip reads
    /// the certificate on every exchange, so peers get the new one next.
    pub(crate) async fn renew_owner_attestation_if_due(
        &self,
        now_unix_ms: u64,
    ) -> anyhow::Result<Option<SignedNodeOwnership>> {
        let Some(owner_keypair) = self.owner_keypair.as_ref() else {
            return Ok(None);
        };
        let Some(current) = self.owner_attestation.lock().await.clone() else {
            return Ok(None);
        };
        let Some(renewed) = crate::mesh::renewed_owner_attestation(
            owner_keypair,
            self.id(),
            &current,
            now_unix_ms,
        )?
        else {
            return Ok(None);
        };
        let summary = crate::mesh::verify_node_ownership(
            Some(&renewed),
            self.id().as_bytes(),
            &*self.trust_store.lock().await,
            crate::mesh::TrustPolicy::Off,
            now_unix_ms,
        );
        *self.owner_attestation.lock().await = Some(renewed.clone());
        *self.owner_summary.lock().await = summary;
        Ok(Some(renewed))
    }
}

#[cfg(test)]
mod tests;

// HOME can redirect the default certificate safely in a Unix child process.
// Windows resolves its home through the known-folder API instead.
#[cfg(all(test, unix))]
mod persistence_tests;
