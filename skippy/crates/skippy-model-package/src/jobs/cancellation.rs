//! Acknowledged cancellation requests are distinct from observed terminal state.
use super::HfJobsClient;
use anyhow::Result;
use serde::Serialize;
use std::time::Instant;

/// Public CLI receipt. `canceled` retains the historical field with truthful false
/// until a separate owner actually observes terminal cancellation. No status is inferred.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CancellationReceipt {
    namespace: String,
    job_id: String,
    canceled: bool,
    cancel_requested: bool,
    cancel_accepted: bool,
    cancel_confirmed: bool,
}
impl HfJobsClient {
    /// Send the explicit cancellation request, recording API acknowledgment only.
    /// Existing `cancel`/`cancel_until` signatures remain unchanged.
    pub async fn cancel_receipt(
        &self,
        namespace: &str,
        job_id: &str,
    ) -> Result<CancellationReceipt> {
        self.cancel_until(
            namespace,
            job_id,
            Instant::now() + self.limits.request_timeout,
        )
        .await?;
        Ok(CancellationReceipt {
            namespace: namespace.into(),
            job_id: job_id.into(),
            canceled: false,
            cancel_requested: true,
            cancel_accepted: true,
            cancel_confirmed: false,
        })
    }
}
