//! Request replay policy at the transport boundary, before any restart can
//! turn a lost payment reply into a fresh process's `not_open` response.

use super::{ExternalPlugin, proto};
use anyhow::{Result, bail};

fn permits_transport_replay(payload: &proto::envelope::Payload) -> bool {
    // The operation name is the wallet.v1 wire contract, independent of the
    // plugin name or its current (possibly cleared) capability advertisement.
    !matches!(payload, proto::envelope::Payload::InvokeServiceRequest(request)
        if request.kind == proto::ServiceKind::Operation as i32
            && request.service_name == "wallet_pay")
}

impl ExternalPlugin {
    pub(super) async fn request_with_timeout(
        &self,
        payload: proto::envelope::Payload,
        timeout: Option<std::time::Duration>,
    ) -> Result<proto::Envelope> {
        let replay_allowed = permits_transport_replay(&payload);
        for attempt in 0..2 {
            self.ensure_running().await?;
            let (generation, outbound_tx, pending) = self.runtime_handles().await?;
            match self
                .request_once(generation, outbound_tx, pending, payload.clone(), timeout)
                .await
            {
                Ok(response) => return Ok(response),
                Err(err) if attempt == 0 && replay_allowed => {
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
