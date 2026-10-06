//! Immutable conversion evidence; failed evidence never becomes successful conversion admission.
use super::super::receipts::{self, Locator, LogAllowance, Observed};
use super::*;
use crate::{
    jobs::{MonitorEnd, MonitorLimits, MonitorReceipt},
    snapshot_promotion::{
        policy::ArtifactIdentity,
        regular_publication::{Plan, Publisher},
    },
};
#[derive(Serialize)]
pub struct ConversionEvidence {
    pub native_receipt: Value,
    pub conversion_admitted: bool,
}
#[derive(Serialize)]
pub struct CollectedConversion {
    pub job_id: String,
    pub monitor: MonitorReceipt,
    pub locator: Locator,
    pub evidence: ConversionEvidence,
    pub image_observed: bool,
    pub cost_observed: bool,
}
/// At most28800 polls at10s covers the original72h budget without weakening certification defaults.
pub fn monitor_limits() -> MonitorLimits {
    MonitorLimits {
        poll_interval: std::time::Duration::from_secs(10),
        ..MonitorLimits::default()
    }
}
fn observe(
    native: Value,
    locator: &Locator,
    submitted: &SubmittedConversionDelivery,
) -> Result<ConversionEvidence> {
    let d = &submitted.native.declaration;
    if native["schema_version"] != 1
        || native["workflow"] != "generic-conversion"
        || native["request_sha256"] != locator.receipt_request_sha256
        || native["transport_input_sha256"] != d.transport_input_sha256
        || !matches!(
            native["status"].as_str(),
            Some("CONVERSION_COMPLETED" | "FAILED")
        )
        || !matches!(
            submitted.expected_status.as_str(),
            "DRY_RUN_COMPLETED" | "LOCAL_ARTIFACT_READY" | "PUBLISHED"
        )
    {
        bail!("generic immutable evidence correlation/workflow refused");
    }
    let operator = &native["operator"];
    let conversion = &operator["conversion_receipt"];
    let pin = |v: &Value| {
        v.as_str().is_some_and(|s| {
            s.len() == 64
                && s.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        })
    };
    let admitted = pin(&native["operator_request_sha256"])
        && pin(&operator["conversion_request_sha256"])
        && locator.delivery_complete
        && native["status"] == "CONVERSION_COMPLETED"
        && native["error"].is_null()
        && operator["status"] == "OPERATOR_COMPLETED"
        && operator["error"].is_null()
        && operator["request_sha256"] == native["operator_request_sha256"]
        && operator["conversion_request_sha256"] == conversion["request_sha256"]
        && conversion["status"] == submitted.expected_status
        && conversion["error"].is_null()
        && (submitted.expected_status != "PUBLISHED"
            || conversion["publication_completed"] == true);
    Ok(ConversionEvidence {
        native_receipt: native,
        conversion_admitted: admitted,
    })
}
impl HfJobsClient {
    /// One absolute deadline covers Jobs observation and immutable receipt bytes.
    /// A terminal failure may yield correlated observations, with conversion_admitted=false.
    pub async fn collect_conversion_until<C: Future<Output = ()>>(
        &self,
        namespace: &str,
        submitted: &SubmittedConversionDelivery,
        publisher: &Publisher,
        deadline: Instant,
        cancellation: C,
    ) -> Result<CollectedConversion> {
        let mut cancel = Box::pin(cancellation);
        let (monitor, locator, native) = collect_native_until(
            self,
            namespace,
            &submitted.native.job_id,
            &submitted.native.declaration,
            publisher,
            deadline,
            cancel.as_mut(),
        )
        .await?;
        let mut evidence = observe(native, &locator, submitted)?;
        evidence.conversion_admitted &= monitor.end == MonitorEnd::Completed;
        if Instant::now() >= deadline || cancel.as_mut().now_or_never().is_some() {
            bail!("generic delivery final boundary refused");
        }
        Ok(CollectedConversion {
            job_id: submitted.native.job_id.clone(),
            monitor,
            locator,
            evidence,
            image_observed: false,
            cost_observed: false,
        })
    }
}
#[cfg(test)]
#[path = "collection/tests.rs"]
mod tests;

/// Shared transport observation; workflow admission remains in each owning classifier.
pub(in crate::jobs::delivery) async fn collect_native_until<C: Future<Output = ()>>(
    client: &HfJobsClient,
    namespace: &str,
    job_id: &str,
    declaration: &DeliveryDeclaration,
    publisher: &Publisher,
    deadline: Instant,
    cancellation: C,
) -> Result<(MonitorReceipt, Locator, Value)> {
    if !declaration.submitted {
        bail!("generic delivery not submitted");
    }
    let limits = monitor_limits();
    let mut cancel = Box::pin(cancellation);
    let mut observed = Observed::default();
    let monitor = client
        .monitor_until(
            namespace,
            job_id,
            deadline,
            cancel.as_mut(),
            limits,
            |row| {
                for line in row.lines() {
                    observed.line(line)?;
                }
                Ok(())
            },
        )
        .await;
    if !matches!(
        monitor.end,
        MonitorEnd::Completed | MonitorEnd::TerminalFailure
    ) || observed.ambiguous
    {
        bail!("generic delivery requires unambiguous terminal monitor");
    }
    if observed.locator.is_none() {
        receipts::final_logs(
            client,
            namespace,
            job_id,
            deadline,
            cancel.as_mut(),
            LogAllowance {
                lines: limits.max_log_lines.saturating_sub(monitor.log_lines),
                bytes: limits.max_log_bytes.saturating_sub(monitor.log_bytes),
            },
            &mut observed,
        )
        .await?;
    }
    if observed.ambiguous {
        bail!("generic delivery contradictory locators");
    }
    let locator = observed
        .locator
        .ok_or_else(|| anyhow::anyhow!("generic delivery locator absent"))?;
    receipts::correlated(&locator, declaration)?;
    let plan = Plan {
        repo: locator.repo.clone(),
        parent_commit: locator.parent_commit.clone(),
        paths: vec![locator.path_in_repo.clone()],
    };
    let identity = ArtifactIdentity {
        sha256: locator.artifact_sha256.clone(),
        byte_size: locator.byte_size,
    };
    let bytes = publisher
        .retrieve_json_until(
            &plan,
            &locator.path_in_repo,
            &locator.commit_oid,
            &identity,
            deadline,
            cancel.as_mut(),
        )
        .await?;

    let native: Value = serde_json::from_slice(&bytes)
        .map_err(|_| anyhow::anyhow!("native receipt JSON refused"))?;
    if Instant::now() >= deadline || cancel.as_mut().now_or_never().is_some() {
        bail!("native receipt terminal boundary refused");
    }
    Ok((monitor, locator, native))
}
