//! Mesh memberships through the embedded handle: serving the node's models
//! into further meshes while it runs, without loading them again.

use super::EmbeddedServeHandle;
use crate::api::RuntimeControlRequest;
use crate::runtime::{MeshMembershipConfig, MeshMembershipStatus};
use anyhow::{Context, Result};

impl EmbeddedServeHandle {
    /// Serves this node's models into a further mesh, with its own identity,
    /// owner allowlist, peers and local API port, replacing a membership of
    /// the same name, which stops first: if the new one cannot start, neither
    /// runs. Returns once the membership accepts connections; its join tokens
    /// are dialled in the background.
    pub async fn add_membership(
        &self,
        config: MeshMembershipConfig,
    ) -> Result<MeshMembershipStatus> {
        self.control(|resp| RuntimeControlRequest::AddMembership { config, resp })
            .await?
    }

    /// Leaves a membership's mesh. Returns whether there was one by that name.
    pub async fn remove_membership(&self, name: impl Into<String>) -> Result<bool> {
        let name = name.into();
        self.control(|resp| RuntimeControlRequest::RemoveMembership { name, resp })
            .await
    }

    pub async fn memberships(&self) -> Result<Vec<MeshMembershipStatus>> {
        self.control(|resp| RuntimeControlRequest::Memberships { resp })
            .await
    }

    async fn control<T>(
        &self,
        request: impl FnOnce(tokio::sync::oneshot::Sender<T>) -> RuntimeControlRequest,
    ) -> Result<T> {
        let control_tx = self
            .control_tx
            .as_ref()
            .context("embedded mesh runtime control channel is unavailable")?;
        let (resp, rx) = tokio::sync::oneshot::channel();
        control_tx
            .send(request(resp))
            .map_err(|_| anyhow::anyhow!("embedded mesh runtime control channel is closed"))?;
        rx.await
            .context("embedded mesh runtime control response dropped")
    }
}
