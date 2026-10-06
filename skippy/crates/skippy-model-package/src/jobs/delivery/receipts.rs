//! Consumes a native locator only after the existing correlated Jobs monitor and immutable byte retrieval.
use super::{DeliveryDeclaration, SubmittedCertificationDelivery};
use crate::{
    jobs::{HfJobsClient, MonitorEnd, MonitorLimits, MonitorReceipt},
    snapshot_promotion::{
        policy::ArtifactIdentity,
        regular_publication::{Plan, Publisher},
    },
};
use anyhow::{Result, bail};
use futures::{
    Future, FutureExt as _, StreamExt as _,
    future::{Either, select},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Instant;
#[derive(Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Locator {
    pub schema_version: u32,
    pub transport_input_sha256: String,
    pub receipt_request_sha256: String,
    pub repo: String,
    pub parent_commit: String,
    pub commit_oid: String,
    pub path_in_repo: String,
    pub artifact_sha256: String,
    pub byte_size: u64,
    pub delivery_complete: bool,
}
#[derive(Default)]
struct Observed {
    locator: Option<Locator>,
    ambiguous: bool,
}
impl Observed {
    fn line(&mut self, line: &str) -> Result<()> {
        let Some(body) = line.strip_prefix("MESH_NATIVE_DELIVERY ") else {
            return Ok(());
        };
        if body.len() > 4096 {
            bail!("native delivery locator byte bound refused");
        }
        let parsed: Locator = serde_json::from_str(body)
            .map_err(|_| anyhow::anyhow!("native delivery locator malformed"))?;
        if self.locator.as_ref().is_some_and(|prior| prior != &parsed) {
            self.ambiguous = true;
        } else {
            self.locator = Some(parsed);
        }
        Ok(())
    }
}
fn hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn correlated(locator: &Locator, declaration: &DeliveryDeclaration) -> Result<()> {
    if locator.schema_version != 1
        || locator.transport_input_sha256 != declaration.transport_input_sha256
        || locator.repo != declaration.evidence_repo
        || locator.parent_commit != declaration.evidence_parent_commit
        || locator.path_in_repo != declaration.evidence_path
        || !hex(&locator.receipt_request_sha256, 64)
        || !hex(&locator.commit_oid, 40)
        || !hex(&locator.artifact_sha256, 64)
        || !(1..=1024 * 1024).contains(&locator.byte_size)
    {
        bail!("native delivery locator source/destination/correlation refused");
    }
    Ok(())
}
fn native_receipt(
    value: &Value,
    locator: &Locator,
    declaration: &DeliveryDeclaration,
) -> Result<()> {
    if value["schema_version"] != 1
        || value["request_sha256"] != locator.receipt_request_sha256
        || value["transport_input_sha256"] != declaration.transport_input_sha256
        || value["status"] != "CERTIFIED"
        || !value["error"].is_null()
        || value["bootstrap"]["status"] != "BOOTSTRAP_COMPLETED"
        || value["acquisition"]["certification"]["status"] != "PASS"
        || value["acquisition"]["certification"]["source_unchanged"] != true
    {
        bail!("immutable native certification receipt refused");
    }
    Ok(())
}
#[derive(Serialize)]
pub struct VerifiedCertificationDelivery {
    pub receipt_log_lines: usize,
    pub receipt_log_bytes: usize,
    pub job_id: String,
    pub monitor: MonitorReceipt,
    pub locator: Locator,
    pub native_receipt: Value,
    pub image_observed: bool,
    pub cost_observed: bool,
}
impl HfJobsClient {
    /// One caller budget covers Jobs polling/log reconnects and immutable evidence retrieval.
    /// The supplied regular Publisher owns explicit credential/origin policy; no ambient auth.
    pub async fn collect_certification_until<C: Future<Output = ()>>(
        &self,
        namespace: &str,
        submitted: &SubmittedCertificationDelivery,
        publisher: &Publisher,
        deadline: Instant,
        cancellation: C,
        limits: MonitorLimits,
    ) -> Result<VerifiedCertificationDelivery> {
        if !submitted.declaration.submitted {
            bail!("native certification was not submitted");
        }
        let mut cancel = Box::pin(cancellation);
        let mut observed = Observed::default();
        let monitor = self
            .monitor_until(
                namespace,
                &submitted.job_id,
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
        let (receipt_log_lines, receipt_log_bytes) =
            if monitor.end == MonitorEnd::Completed && observed.locator.is_none() {
                final_logs(
                    self,
                    namespace,
                    &submitted.job_id,
                    deadline,
                    cancel.as_mut(),
                    LogAllowance {
                        lines: limits.max_log_lines.saturating_sub(monitor.log_lines),
                        bytes: limits.max_log_bytes.saturating_sub(monitor.log_bytes),
                    },
                    &mut observed,
                )
                .await?
            } else {
                (0, 0)
            };
        if monitor.end != MonitorEnd::Completed || observed.ambiguous {
            bail!(
                "native delivery requires correlated completed Jobs monitor and one unambiguous locator"
            );
        }
        let locator = observed
            .locator
            .ok_or_else(|| anyhow::anyhow!("native delivery locator absent"))?;
        correlated(&locator, &submitted.declaration)?;
        if !locator.delivery_complete {
            bail!("native delivery export is not terminal certification");
        }
        if Instant::now() >= deadline || cancel.as_mut().now_or_never().is_some() {
            bail!("native delivery collection cancelled or expired");
        }
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
            .map_err(|_| anyhow::anyhow!("immutable native receipt JSON refused"))?;
        native_receipt(&native, &locator, &submitted.declaration)?;
        if Instant::now() >= deadline || cancel.as_mut().now_or_never().is_some() {
            bail!("native delivery final collection boundary refused");
        }
        Ok(VerifiedCertificationDelivery {
            receipt_log_lines,
            receipt_log_bytes,
            job_id: submitted.job_id.clone(),
            monitor,
            locator,
            native_receipt: native,
            image_observed: false,
            cost_observed: false,
        })
    }
}

async fn wait<T, W: Future<Output = Result<T>>, C: Future<Output = ()>>(
    work: W,
    cancel: std::pin::Pin<&mut C>,
    deadline: Instant,
) -> Result<T> {
    if Instant::now() >= deadline {
        bail!("native delivery deadline expired");
    }
    match select(
        cancel,
        tokio::time::timeout_at(tokio::time::Instant::from_std(deadline), work).boxed_local(),
    )
    .await
    {
        Either::Left(_) => bail!("native delivery collection cancelled"),
        Either::Right((Err(_), _)) => bail!("native delivery collection deadline expired"),
        Either::Right((Ok(result), retained)) => {
            if retained.now_or_never().is_some() || Instant::now() >= deadline {
                bail!("native delivery terminal boundary refused");
            }
            result
        }
    }
}
struct LogAllowance {
    lines: usize,
    bytes: usize,
}
async fn final_logs<C: Future<Output = ()>>(
    client: &HfJobsClient,
    namespace: &str,
    id: &str,
    deadline: Instant,
    mut cancel: std::pin::Pin<&mut C>,
    allowance: LogAllowance,
    observed: &mut Observed,
) -> Result<(usize, usize)> {
    let stream = wait(
        client.stream_logs_until(namespace, id, deadline),
        cancel.as_mut(),
        deadline,
    )
    .await?;
    let mut stream = Box::pin(stream);
    let mut lines = 0usize;
    let mut bytes = 0usize;
    loop {
        let row = wait(async { Ok(stream.next().await) }, cancel.as_mut(), deadline).await?;
        let Some(row) = row else {
            break;
        };
        let row = row?;
        lines = lines
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("native delivery log count overflow"))?;
        bytes = bytes
            .checked_add(row.len())
            .ok_or_else(|| anyhow::anyhow!("native delivery log byte overflow"))?;
        if lines > allowance.lines || bytes > allowance.bytes {
            bail!("native delivery aggregate log allowance exhausted");
        }
        for line in row.lines() {
            observed.line(line)?;
        }
    }
    Ok((lines, bytes))
}
#[cfg(test)]
#[path = "receipts/tests.rs"]
mod tests;

/// Retrieve request-correlated immutable native evidence even when the native job failed.
/// Returned FAILED bytes are evidence, never certification or remote-delivery acceptance.
pub async fn retrieve_native_job_receipt_until<C: Future<Output = ()>>(
    publisher: &Publisher,
    locator: &Locator,
    declaration: &DeliveryDeclaration,
    deadline: Instant,
    cancellation: C,
) -> Result<Value> {
    correlated(locator, declaration)?;
    let mut cancellation = Box::pin(cancellation);
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
            cancellation.as_mut(),
        )
        .await?;
    let native: Value = serde_json::from_slice(&bytes)
        .map_err(|_| anyhow::anyhow!("immutable native receipt JSON refused"))?;
    if native["schema_version"] != 1
        || native["request_sha256"] != locator.receipt_request_sha256
        || native["transport_input_sha256"] != declaration.transport_input_sha256
        || !matches!(
            native["status"].as_str(),
            Some("CERTIFIED" | "COMPOSED" | "FAILED")
        )
    {
        bail!("immutable partial native receipt correlation refused");
    }
    if Instant::now() >= deadline || cancellation.as_mut().now_or_never().is_some() {
        bail!("partial native receipt final collection boundary refused");
    }
    Ok(native)
}
