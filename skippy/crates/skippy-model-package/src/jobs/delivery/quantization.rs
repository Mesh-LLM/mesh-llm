//! Pinned supplied-tool quant Jobs delivery through the existing submission and evidence owners.
use super::generic::{PreparedConversionDelivery, SubmittedConversionDelivery};
use super::*;
#[path = "quantization/collection.rs"]
mod collection;
#[path = "quantization/admission.rs"]
mod quant_admission;
pub use collection::CollectedQuantization;
#[cfg(test)]
#[path = "quantization/tests.rs"]
pub(super) mod tests;
const ROUTE: &str = "quant-job-worker";
pub fn prepare(
    bytes: &[u8],
    mounts: &[ModelMount],
    plan: &CpuJobPlan,
) -> Result<PreparedConversionDelivery> {
    if bytes.len() > 65536 {
        bail!("inline quant request byte bound");
    }
    prepare_inner(bytes, mounts, plan)
}
pub fn prepare_mounted(
    bytes: &[u8],
    mounts: &[ModelMount],
    plan: &CpuJobPlan,
    locator: &request_transport::MountedRequest,
) -> Result<PreparedConversionDelivery> {
    locator.admit(bytes, mounts)?;
    let mut prepared = prepare_inner(bytes, mounts, plan)?;
    let encoded = serde_json::to_string(locator)?;
    if encoded.len() > 8192 {
        bail!("mounted quant locator byte bound");
    }
    prepared.native.declaration.transport_input_sha256 = admission::digest(bytes);
    prepared.native.spec.arguments[3] = "--mounted-input-environment".into();
    prepared.native.spec.arguments[4] = request_transport::LOCATOR_ENVIRONMENT.into();
    prepared.native.spec.secrets.remove(INPUT_KEY);
    prepared
        .native
        .spec
        .environment
        .insert(request_transport::LOCATOR_ENVIRONMENT.into(), encoded);
    Ok(prepared)
}
fn prepare_inner(
    bytes: &[u8],
    mounts: &[ModelMount],
    plan: &CpuJobPlan,
) -> Result<PreparedConversionDelivery> {
    let request = quant_admission::request(bytes, mounts, plan)?;
    let worker_input = serde_json::to_string(&request)?;
    let authority = &request["authority"];
    let image = admission::text(authority, "image")?.to_owned();
    let expected_status = if request["workflow"] == "quantization" {
        "QUANTIZATION_PUBLISHED"
    } else {
        "QUANTIZATION_PACKAGED"
    }
    .into();
    let declaration = DeliveryDeclaration {
        schema_version: 1,
        transport_input_sha256: admission::digest(worker_input.as_bytes()),
        runner_sha256: admission::text(&request["runner"], "sha256")?.into(),
        mesh_commit: admission::text(authority, "mesh_commit")?.into(),
        image: image.clone(),
        flavor: plan.flavor.clone(),
        timeout_seconds: plan.timeout_seconds,
        cpu_plan_receipt_sha256: admission::digest(&serde_json::to_vec(plan)?),
        declared_estimate_usd: plan.max_cost_usd,
        evidence_repo: admission::text(&request["receipt_export"], "repo")?.into(),
        evidence_parent_commit: admission::text(&request["receipt_export"], "parent_commit")?
            .into(),
        evidence_path: admission::text(&request["receipt_export"], "path_in_repo")?.into(),
        image_observed: false,
        submitted: false,
        native_certification_completed: false,
    };
    let spec = JobSpec {
        docker_image: image,
        command: vec![admission::text(&request["runner"], "path")?.into()],
        arguments: [
            "automation",
            "hf-certify",
            ROUTE,
            "--input-environment",
            INPUT_KEY,
            "--output-directory",
            OUTPUT,
        ]
        .map(str::to_owned)
        .into(),
        environment: HashMap::new(),
        secrets: HashMap::from([(INPUT_KEY.into(), worker_input)]),
        flavor: plan.flavor.clone(),
        timeout_seconds: plan.timeout_seconds,
        volumes: mounts
            .iter()
            .map(|m| JobVolume {
                volume_type: "model".into(),
                source: m.repo.clone(),
                mount_path: m.mount_path.clone(),
                read_only: Some(true),
                revision: Some(m.revision.clone()),
            })
            .collect(),
    };
    Ok(PreparedConversionDelivery {
        native: PreparedCertificationDelivery { spec, declaration },
        expected_status,
    })
}
