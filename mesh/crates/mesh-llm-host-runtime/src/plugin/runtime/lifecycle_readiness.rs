//! Lifecycle authority belongs to an authenticated, initialized runtime generation.

use anyhow::{Context, Result, bail};
use rmcp::model::ServerConfig;
use tokio::sync::mpsc;

use super::{ExternalPlugin, PendingResponses, proto};

pub(super) enum LifecycleOperation<'a> {
    Callback(&'a proto::OpenAiExchangeHookManifest),
    Body(&'a str, &'a proto::OpenAiExchangeHookManifest),
}

impl ExternalPlugin {
    pub(super) async fn publish_initialized_state(
        &self,
        generation: u64,
        init: &proto::InitializeResponse,
        server_info: ServerConfig,
    ) -> Result<()> {
        let mut runtime = self.runtime.lock().await;
        let runtime = runtime
            .as_mut()
            .filter(|runtime| runtime.generation == generation)
            .context("plugin generation changed during initialization")?;
        let declaration = init
            .manifest
            .as_ref()
            .and_then(|manifest| manifest.openai_exchange_hook.clone());
        super::socket_auth::validate_lifecycle_declaration(
            declaration.is_some(),
            runtime.authenticated_peer,
        )?;
        *self.server_info.lock().await = Some(server_info);
        *self.manifest.lock().await = init.manifest.clone();
        runtime.initialized_lifecycle = declaration;
        Ok(())
    }

    pub(super) async fn lifecycle_runtime_handles(
        &self,
        operation: LifecycleOperation<'_>,
    ) -> Result<(mpsc::Sender<proto::Envelope>, PendingResponses)> {
        let runtime = self.runtime.lock().await;
        let runtime = runtime
            .as_ref()
            .context("plugin lifecycle runtime unavailable")?;
        if !runtime.authenticated_peer {
            bail!("plugin lifecycle runtime is not authenticated");
        }
        let declaration = runtime
            .initialized_lifecycle
            .as_ref()
            .context("plugin lifecycle generation has not completed initialization")?;
        if let LifecycleOperation::Body(_, expected) = &operation
            && declaration.as_ref() != *expected
        {
            bail!("plugin lifecycle declaration changed before body negotiation");
        }
        let allowed = match operation {
            LifecycleOperation::Callback(expected) => declaration.as_ref() == expected,
            LifecycleOperation::Body("openai_exchange_original" | "openai_exchange_request", _) => {
                declaration.request_body
            }
            LifecycleOperation::Body("openai_exchange_effective", _) => {
                declaration.effective_request_body
            }
            LifecycleOperation::Body("openai_exchange_response", _) => declaration.response_body,
            LifecycleOperation::Body(_, _) => false,
        };
        if !allowed {
            bail!("plugin lifecycle operation is not declared by the initialized generation");
        }
        // These channels belong only to this verified generation. A reconnect
        // cannot redirect the operation through a replacement's channels.
        Ok((runtime.outbound_tx.clone(), runtime.pending.clone()))
    }
}

pub(super) fn lifecycle_stream_kind(request: &proto::OpenStreamRequest) -> Option<String> {
    let metadata: serde_json::Value =
        serde_json::from_str(request.metadata_json.as_deref()?).ok()?;
    metadata["kind"]
        .as_str()
        .filter(|kind| kind.starts_with("openai_exchange_"))
        .map(str::to_owned)
}

#[cfg(test)]
#[path = "lifecycle_readiness_tests.rs"]
mod tests;
