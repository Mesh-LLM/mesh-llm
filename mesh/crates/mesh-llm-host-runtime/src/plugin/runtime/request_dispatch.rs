//! Queue admission and cancellation-safe correlation for control requests.
use super::{ExternalPlugin, PROTOCOL_VERSION, PendingResponses, proto};
use anyhow::{Context, Result, bail};
use std::sync::atomic::Ordering;
use tokio::sync::{mpsc, oneshot};

struct PendingRequest {
    pending: PendingResponses,
    request_id: u64,
}

impl Drop for PendingRequest {
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
    pub(super) async fn request_once(
        &self,
        generation: u64,
        outbound_tx: mpsc::Sender<proto::Envelope>,
        pending: PendingResponses,
        payload: proto::envelope::Payload,
        timeout: Option<std::time::Duration>,
    ) -> Result<proto::Envelope> {
        let deadline = timeout.map(|duration| tokio::time::Instant::now() + duration);
        let call = async {
            // Reserve before allocating correlation state. Cancellation while
            // waiting for capacity cannot leave an orphan pending response.
            let permit = outbound_tx
                .reserve()
                .await
                .context("Plugin request queue closed before admission")?;
            let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
            let (tx, rx) = oneshot::channel();
            pending.lock().await.insert(request_id, tx);
            let _cleanup = PendingRequest {
                pending,
                request_id,
            };
            permit.send(proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: self.spec.name.clone(),
                request_id,
                payload: Some(payload),
            });
            rx.await.context("Plugin dropped the response channel")?
        };
        let response = match deadline {
            Some(deadline) => match tokio::time::timeout_at(deadline, call).await {
                Ok(response) => response,
                Err(_) => {
                    self.handle_runtime_failure(
                        Some(generation),
                        format!("Plugin '{}' timed out", self.spec.name),
                    )
                    .await;
                    bail!("Plugin '{}' timed out", self.spec.name);
                }
            },
            None => call.await,
        };
        if let Err(error) = &response {
            self.handle_runtime_failure(Some(generation), error.to_string())
                .await;
        }
        response
    }
}
