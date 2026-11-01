//! Model discovery uses the runtime channel, independently of HTTP listeners.

use super::EmbeddedServeHandle;
use crate::models::discovery::{self, MeshModel, MeshModelPage, MeshModelQuery};
use anyhow::{Context, Result};

impl EmbeddedServeHandle {
    /// Query advertisements in the node's primary mesh directly in process.
    pub async fn mesh_models(&self, query: MeshModelQuery) -> Result<MeshModelPage> {
        discovery::query_models(self.discover_models().await?, query)
    }

    /// Read one exact public serving ID, returning an error when it is absent.
    pub async fn mesh_model(&self, id: &str) -> Result<MeshModel> {
        discovery::lookup(self.discover_models().await?, id)
    }

    async fn discover_models(&self) -> Result<Vec<MeshModel>> {
        let control = self
            .control_tx
            .as_ref()
            .context("embedded mesh runtime control channel is unavailable")?;
        let (resp, response) = tokio::sync::oneshot::channel();
        control
            .send(crate::api::RuntimeControlRequest::DiscoverModels { resp })
            .map_err(|_| anyhow::anyhow!("embedded mesh runtime control channel is closed"))?;
        response
            .await
            .context("embedded mesh model discovery response dropped")?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn native_discovery_queries_runtime_channel_without_http() {
        let (control, mut requests) = tokio::sync::mpsc::unbounded_channel();
        let (shutdown, _) = tokio::sync::watch::channel(false);
        let handle = EmbeddedServeHandle {
            api_base_url: "http://127.0.0.1:1/v1".into(),
            console_url: "http://127.0.0.1:1".into(),
            invite_token: None,
            control_tx: Some(control),
            task: None,
            shutdown,
            native_node: None,
        };
        let runtime = tokio::spawn(async move {
            match requests.recv().await.expect("discovery request") {
                crate::api::RuntimeControlRequest::DiscoverModels { resp } => {
                    let _ = resp.send(Ok(Vec::new()));
                }
                _ => panic!("expected model discovery"),
            }
        });
        assert!(
            handle
                .mesh_models(MeshModelQuery::default())
                .await
                .unwrap()
                .data
                .is_empty()
        );
        runtime.await.unwrap();
    }
}
