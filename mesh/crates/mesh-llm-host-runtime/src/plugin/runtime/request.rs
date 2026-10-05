//! Request replay policy at the transport boundary, before any restart can
//! turn a lost payment reply into a fresh process's `not_open` response.

use super::{ExternalPlugin, plugin_error, proto};
use crate::plugin::ToolCallResult;
use crate::plugin::operations::TransportReplay;
use anyhow::{Result, bail};

impl ExternalPlugin {
    pub(crate) async fn call_tool_with_replay(
        &self,
        tool_name: &str,
        arguments_json: &str,
        timeout: Option<std::time::Duration>,
        replay: TransportReplay,
    ) -> Result<ToolCallResult> {
        let response = self
            .invoke_service_with_replay(
                proto::ServiceKind::Operation,
                tool_name,
                arguments_json,
                timeout,
                replay,
            )
            .await?;
        Ok(ToolCallResult {
            content_json: response.output_json,
            is_error: response.is_error,
        })
    }

    pub(crate) async fn invoke_service(
        &self,
        kind: proto::ServiceKind,
        service_name: &str,
        input_json: &str,
        timeout: Option<std::time::Duration>,
    ) -> Result<proto::InvokeServiceResponse> {
        self.invoke_service_with_replay(
            kind,
            service_name,
            input_json,
            timeout,
            TransportReplay::Allow,
        )
        .await
    }

    async fn invoke_service_with_replay(
        &self,
        kind: proto::ServiceKind,
        service_name: &str,
        input_json: &str,
        timeout: Option<std::time::Duration>,
        replay: TransportReplay,
    ) -> Result<proto::InvokeServiceResponse> {
        let response = self
            .request_with_replay(
                proto::envelope::Payload::InvokeServiceRequest(proto::InvokeServiceRequest {
                    kind: kind as i32,
                    service_name: service_name.to_string(),
                    input_json: input_json.to_string(),
                }),
                timeout,
                replay,
            )
            .await?;
        match response.payload {
            Some(proto::envelope::Payload::InvokeServiceResponse(resp)) => Ok(resp),
            Some(proto::envelope::Payload::ErrorResponse(err)) => {
                Err(plugin_error(&self.spec.name, "invoke_service", &err))
            }
            _ => bail!(
                "Plugin '{}' returned an unexpected payload for 'invoke_service'",
                self.spec.name
            ),
        }
    }

    pub(super) async fn request_with_timeout(
        &self,
        payload: proto::envelope::Payload,
        timeout: Option<std::time::Duration>,
    ) -> Result<proto::Envelope> {
        self.request_with_replay(payload, timeout, TransportReplay::Allow)
            .await
    }

    async fn request_with_replay(
        &self,
        payload: proto::envelope::Payload,
        timeout: Option<std::time::Duration>,
        replay: TransportReplay,
    ) -> Result<proto::Envelope> {
        for attempt in 0..2 {
            self.ensure_running().await?;
            let (generation, outbound_tx, pending) = self.runtime_handles().await?;
            match self
                .request_once(generation, outbound_tx, pending, payload.clone(), timeout)
                .await
            {
                Ok(response) => return Ok(response),
                Err(err) if attempt == 0 && replay == TransportReplay::Allow => {
                    tracing::debug!(
                        plugin = %self.spec.name,
                        error = %err,
                        "Retrying plugin request after restart"
                    );
                }
                Err(err) => return Err(err),
            }
        }
        bail!("Plugin '{}' request failed after restart", self.spec.name)
    }
}

#[cfg(test)]
mod tests;
