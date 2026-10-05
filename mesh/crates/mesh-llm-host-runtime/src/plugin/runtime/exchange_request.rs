//! Single-send lifecycle calls: observer deadlines never restart the process.

use super::{ExternalPlugin, PROTOCOL_VERSION, PendingResponses, proto};
use anyhow::{Context, Result, bail};
use std::sync::atomic::Ordering;
use tokio::sync::oneshot;
use tokio::time::Instant;

struct PendingExchangeRequest {
    pending: PendingResponses,
    request_id: u64,
}

impl Drop for PendingExchangeRequest {
    fn drop(&mut self) {
        if let Ok(mut pending) = self.pending.try_lock() {
            pending.remove(&self.request_id);
            return;
        }
        let pending = self.pending.clone();
        let request_id = self.request_id;
        tokio::spawn(async move {
            pending.lock().await.remove(&request_id);
        });
    }
}

impl ExternalPlugin {
    pub(crate) async fn invoke_exchange_service(
        &self,
        service_name: &str,
        input_json: &str,
        deadline: Instant,
    ) -> Result<proto::InvokeServiceResponse> {
        let response = tokio::time::timeout_at(
            deadline,
            self.request_exchange_once(proto::InvokeServiceRequest {
                kind: proto::ServiceKind::OpenaiExchange as i32,
                service_name: service_name.to_owned(),
                input_json: input_json.to_owned(),
            }),
        )
        .await
        .context("Plugin exchange hook deadline expired")??;
        match response.payload {
            Some(proto::envelope::Payload::InvokeServiceResponse(response)) => Ok(response),
            Some(proto::envelope::Payload::ErrorResponse(error)) => Err(super::plugin_error(
                &self.spec.name,
                "openai_exchange",
                &error,
            )),
            _ => bail!(
                "Plugin '{}' returned an unexpected exchange reply",
                self.spec.name
            ),
        }
    }

    async fn request_exchange_once(
        &self,
        request: proto::InvokeServiceRequest,
    ) -> Result<proto::Envelope> {
        // Preflight already requires a live manifest. Do not start or retry a
        // process here: the exchange grant's deadline belongs to this call.
        let (_, outbound, pending) = self.runtime_handles().await?;
        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel();
        pending.lock().await.insert(request_id, tx);
        let _cleanup = PendingExchangeRequest {
            pending,
            request_id,
        };
        outbound
            .send(proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: self.spec.name.clone(),
                request_id,
                payload: Some(proto::envelope::Payload::InvokeServiceRequest(request)),
            })
            .await
            .context("Plugin exchange request channel closed")?;
        rx.await
            .context("Plugin exchange response channel closed")?
    }
}
