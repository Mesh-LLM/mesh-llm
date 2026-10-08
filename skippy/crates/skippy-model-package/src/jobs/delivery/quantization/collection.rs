//! Collect correlated quant observations; remote Jobs completion is not native artifact admission.
use super::super::{
    generic::collection::{
        NativeCollectionPolicy, collect_native_with_policy_until, quant_monitor_limits,
    },
    receipts::Locator,
};
use super::*;
use crate::{
    jobs::{MonitorEnd, MonitorReceipt},
    snapshot_promotion::regular_publication::Publisher,
};
#[derive(Serialize)]
pub struct CollectedQuantization {
    pub job_id: String,
    pub monitor: MonitorReceipt,
    pub locator: Locator,
    pub native_receipt: Value,
    pub quantization_admitted: bool,
    pub package_admitted: bool,
    pub image_observed: bool,
    pub cost_observed: bool,
}
fn observed(
    native: &Value,
    locator: &Locator,
    submitted: &SubmittedConversionDelivery,
) -> Result<bool> {
    let workflow = match submitted.expected_status.as_str() {
        "QUANTIZATION_PUBLISHED" => "quantization",
        "QUANTIZATION_PACKAGED" => "quantization-and-package",
        _ => bail!("quant expected workflow status refused"),
    };
    if native["schema_version"] != 1
        || native["workflow"] != workflow
        || native["request_sha256"] != locator.receipt_request_sha256
        || native["transport_input_sha256"] != submitted.native.declaration.transport_input_sha256
        || !matches!(
            native["status"].as_str(),
            Some("QUANTIZATION_PUBLISHED" | "QUANTIZATION_PACKAGED" | "FAILED")
        )
    {
        bail!("quant immutable request/transport/workflow correlation refused");
    }
    let op = &native["operator"];
    let pin = |v: &Value, n: usize| {
        v.as_str().is_some_and(|s| {
            s.len() == n
                && s.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        })
    };
    Ok(locator.delivery_complete
        && native["status"] == submitted.expected_status
        && native["error"].is_null()
        && pin(&native["operator_request_sha256"], 64)
        && op["request_sha256"] == native["operator_request_sha256"]
        && op["status"] == submitted.expected_status
        && op["error"].is_null()
        && op["completed_job"] == true
        && op["full_roster_verified"] == true
        && pin(&op["final_commit"], 40)
        && op["verify_job"]["completed"] == true
        && (workflow == "quantization"
            || op["package"]["completed"] == true && pin(&op["package"]["final_commit"], 40)))
}
impl HfJobsClient {
    pub async fn collect_quantization_until<C: Future<Output = ()>>(
        &self,
        namespace: &str,
        submitted: &SubmittedConversionDelivery,
        publisher: &Publisher,
        deadline: Instant,
        cancellation: C,
    ) -> Result<CollectedQuantization> {
        let mut cancel = Box::pin(cancellation);
        let policy = NativeCollectionPolicy {
            deadline,
            limits: quant_monitor_limits(submitted.native.declaration.timeout_seconds)?,
        };
        let (monitor, locator, native) = collect_native_with_policy_until(
            self,
            namespace,
            &submitted.native.job_id,
            &submitted.native.declaration,
            publisher,
            policy,
            cancel.as_mut(),
        )
        .await?;
        let admitted =
            observed(&native, &locator, submitted)? && monitor.end == MonitorEnd::Completed;
        if Instant::now() >= deadline || cancel.as_mut().now_or_never().is_some() {
            bail!("quant collection final boundary refused");
        }
        Ok(CollectedQuantization {
            job_id: submitted.native.job_id.clone(),
            monitor,
            locator,
            native_receipt: native,
            quantization_admitted: admitted,
            package_admitted: admitted && submitted.expected_status == "QUANTIZATION_PACKAGED",
            image_observed: false,
            cost_observed: false,
        })
    }
}

#[cfg(test)]
#[path = "collection/tests.rs"]
mod tests;
