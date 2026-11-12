//! Single-send lifecycle calls: observer deadlines never restart the process.
//! Only a private hook call held past its limit does.

use super::lifecycle_readiness::{LifecycleOperation, lifecycle_stream_kind};
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
        declaration: &proto::OpenAiExchangeHookManifest,
        input_json: &str,
        deadline: Instant,
    ) -> Result<proto::InvokeServiceResponse> {
        tokio::time::timeout_at(deadline, self.exchange_call(declaration, input_json, None))
            .await
            .context("Plugin exchange hook deadline expired")?
    }

    /// A private hook call: waits past the exchange's deadline, until the
    /// plugin replies, its connection ends or `hold_until`. A hook still
    /// working then is restarted, which ends its work, so a hung hook cannot
    /// hold a private binding (and narrow the plugin's nested calls) forever.
    pub(crate) async fn invoke_private_exchange_service(
        &self,
        declaration: &proto::OpenAiExchangeHookManifest,
        input_json: &str,
        hold_until: Instant,
    ) -> Result<proto::InvokeServiceResponse> {
        self.exchange_call(declaration, input_json, Some(hold_until))
            .await
    }

    async fn exchange_call(
        &self,
        declaration: &proto::OpenAiExchangeHookManifest,
        input_json: &str,
        hold_until: Option<Instant>,
    ) -> Result<proto::InvokeServiceResponse> {
        let response = self
            .request_lifecycle_once(
                proto::envelope::Payload::InvokeServiceRequest(proto::InvokeServiceRequest {
                    invocation_id: None,
                    kind: proto::ServiceKind::OpenaiExchange as i32,
                    service_name: declaration.handler.clone(),
                    input_json: input_json.to_owned(),
                }),
                LifecycleOperation::Callback(declaration),
                hold_until,
            )
            .await?;
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

    pub(crate) async fn open_stream(
        &self,
        request: proto::OpenStreamRequest,
    ) -> Result<proto::OpenStreamResponse> {
        if lifecycle_stream_kind(&request).is_some() {
            bail!("lifecycle body streams require the negotiated declaration");
        }
        // Preserve ordinary plugin stream startup/retry behavior.
        let response = self
            .request(proto::envelope::Payload::OpenStreamRequest(request))
            .await?;
        Self::stream_response(&self.spec.name, response)
    }

    pub(crate) async fn open_lifecycle_stream(
        &self,
        request: proto::OpenStreamRequest,
        declaration: &proto::OpenAiExchangeHookManifest,
    ) -> Result<proto::OpenStreamResponse> {
        let kind = lifecycle_stream_kind(&request).context("lifecycle stream kind missing")?;
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(super::REQUEST_TIMEOUT_SECS),
            self.request_lifecycle_once(
                proto::envelope::Payload::OpenStreamRequest(request),
                LifecycleOperation::Body(&kind, declaration),
                None,
            ),
        )
        .await
        .context("Plugin lifecycle stream deadline expired")??;
        Self::stream_response(&self.spec.name, response)
    }

    fn stream_response(name: &str, response: proto::Envelope) -> Result<proto::OpenStreamResponse> {
        match response.payload {
            Some(proto::envelope::Payload::OpenStreamResponse(response)) => Ok(response),
            Some(proto::envelope::Payload::ErrorResponse(error)) => {
                Err(super::plugin_error(name, "open_stream", &error))
            }
            _ => bail!("Plugin '{}' returned an unexpected stream reply", name),
        }
    }

    async fn request_lifecycle_once(
        &self,
        payload: proto::envelope::Payload,
        operation: LifecycleOperation<'_>,
        restart_at: Option<Instant>,
    ) -> Result<proto::Envelope> {
        // Preflight already requires a live manifest. Do not start or retry a
        // process here: the exchange grant's deadline belongs to this call.
        let (generation, outbound, pending) = self.lifecycle_runtime_handles(operation).await?;
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
                payload: Some(payload),
            })
            .await
            .context("Plugin exchange request channel closed")?;
        let reply = async {
            rx.await
                .context("Plugin exchange response channel closed")?
        };
        let Some(restart_at) = restart_at else {
            return reply.await;
        };
        match tokio::time::timeout_at(restart_at, reply).await {
            Ok(reply) => reply,
            Err(_) => {
                let reason = format!(
                    "Plugin '{}' exchange hook held a private request too long",
                    self.spec.name
                );
                self.handle_runtime_failure(Some(generation), reason.clone())
                    .await;
                bail!(reason)
            }
        }
    }
}
