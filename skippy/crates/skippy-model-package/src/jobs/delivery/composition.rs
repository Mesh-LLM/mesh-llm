//! Default MTP composition Jobs over the existing bounded submission and immutable evidence owners.
use super::generic::{PreparedConversionDelivery, SubmittedConversionDelivery};
use super::receipts::Locator;
use super::*;
use crate::{
    jobs::{MonitorEnd, MonitorReceipt},
    snapshot_promotion::regular_publication::Publisher,
};
use serde_json::json;
pub fn prepare(
    bytes: &[u8],
    mounts: &[ModelMount],
    plan: &CpuJobPlan,
) -> Result<PreparedConversionDelivery> {
    if bytes.len() > 65536 {
        bail!("inline composition request byte bound");
    }
    prepare_inner(bytes, mounts, plan)
}
pub fn prepare_mounted(
    bytes: &[u8],
    mounts: &[ModelMount],
    plan: &CpuJobPlan,
    locator: &super::request_transport::MountedRequest,
) -> Result<PreparedConversionDelivery> {
    locator.admit(bytes, mounts)?;
    let mut prepared = prepare_inner(bytes, mounts, plan)?;
    let encoded = serde_json::to_string(locator)?;
    if encoded.len() > 8192 {
        bail!("mounted composition locator byte bound");
    }
    prepared.native.declaration.transport_input_sha256 = admission::digest(bytes);
    prepared.native.spec.arguments[3] = "--mounted-input-environment".into();
    prepared.native.spec.arguments[4] = super::request_transport::LOCATOR_ENVIRONMENT.into();
    prepared.native.spec.secrets.remove(INPUT_KEY);
    prepared.native.spec.environment.insert(
        super::request_transport::LOCATOR_ENVIRONMENT.into(),
        encoded,
    );
    Ok(prepared)
}
fn prepare_inner(
    bytes: &[u8],
    mounts: &[ModelMount],
    plan: &CpuJobPlan,
) -> Result<PreparedConversionDelivery> {
    let request = request(bytes, mounts, plan)?;
    let worker_input = serde_json::to_string(&request)?;
    let bootstrap = &request["operator"]["bootstrap"];
    let image = admission::text(bootstrap, "image")?.to_owned();
    let declaration = DeliveryDeclaration {
        schema_version: 1,
        transport_input_sha256: admission::digest(worker_input.as_bytes()),
        runner_sha256: admission::text(&request["runner"], "sha256")?.into(),
        mesh_commit: admission::text(bootstrap, "mesh_commit")?.into(),
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
    let expected_status = if request["operator"]["dry_run"] == true {
        "DRY_RUN_NOT_EXECUTED"
    } else {
        "COMPOSED_PUBLISHED"
    }
    .into();
    let spec = JobSpec {
        docker_image: image,
        command: vec![admission::text(&request["runner"], "path")?.into()],
        arguments: [
            "automation",
            "hf-certify",
            "composition-job-worker",
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
fn request(bytes: &[u8], mounts: &[ModelMount], plan: &CpuJobPlan) -> Result<Value> {
    if bytes.is_empty() || bytes.len() > super::request_transport::MAX_REQUEST_BYTES {
        bail!("composition Jobs input byte bound");
    }
    let v: Value = serde_json::from_slice(bytes)?;
    let keys = [
        "schema_version",
        "workflow",
        "timeout_secs",
        "runner",
        "operator",
        "receipt_export",
    ];
    if v.as_object()
        .is_none_or(|o| o.len() != keys.len() || keys.iter().any(|k| !o.contains_key(*k)))
        || v["schema_version"] != 1
        || v["workflow"] != "default-mtp-composition"
    {
        bail!("composition Jobs closed workflow");
    }
    admission::mounts_admitted(mounts)?;
    let op = &v["operator"];
    admission::resource(
        &json!({"bootstrap":op["bootstrap"],"timeout_secs":v["timeout_secs"]}),
        plan,
        259200,
    )?;
    admission::artifact(&v["runner"], mounts, false)?;
    let runner = Path::new(admission::text(&v["runner"], "path")?);
    if runner.starts_with("/work")
        || runner.starts_with("/models")
        || op["schema_version"] != 1
        || op["overall_seconds"].as_u64() != Some(plan.timeout_seconds)
        || !op["credential_file"].is_null()
        || op["confirm_publication"] != true && op["dry_run"] != true
    {
        bail!("composition Jobs explicit budget/authority/image runner refused");
    }
    for key in [
        "staging_helper",
        "repository_helper",
        "repository_helper_source",
        "publisher_helper",
        "publisher_source",
        "tokenizer_profile",
    ] {
        admission::artifact(&op[key], mounts, false)?;
    }
    let parts = op["target_parts"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("ordered target parts"))?;
    if !(2..=128).contains(&parts.len()) {
        bail!("composition complete target roster");
    }
    for (i, p) in parts.iter().enumerate() {
        admission::artifact(p, mounts, true)?;
        if parts[..i].iter().any(|q| q["path"] == p["path"]) {
            bail!("composition target duplicate");
        }
    }
    for key in ["checkpoint", "tokenizer_source"] {
        let source = &op[key];
        let revision = admission::text(source, "revision")?;
        let repo = admission::text(source, "repo")?;
        let files = source["files"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("composition immutable source files"))?;
        if revision.len() != 40
            || !revision
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            || repo.split('/').count() != 2
            || files.is_empty()
            || files.len() > 1024
            || files.iter().any(|(n, h)| {
                n.is_empty()
                    || n.contains('/')
                    || n.contains('\\')
                    || h.as_str().is_none_or(|s| {
                        s.len() != 64
                            || !s
                                .bytes()
                                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
                    })
            })
        {
            bail!("composition immutable checkpoint/tokenizer pins refused");
        }
    }
    let sidecars = op["sidecars"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("composition sidecar roster"))?;
    if sidecars.len() > 31 {
        bail!("composition sidecar bound");
    }
    for s in sidecars {
        admission::artifact(&s["artifact"], mounts, false)?;
    }
    admission::export(&v["receipt_export"], mounts, plan.timeout_seconds)?;
    Ok(v)
}
#[derive(Serialize)]
pub struct CollectedComposition {
    pub job_id: String,
    pub monitor: MonitorReceipt,
    pub locator: Locator,
    pub native_receipt: Value,
    pub composition_admitted: bool,
    pub image_observed: bool,
    pub cost_observed: bool,
}
fn observe(
    native: &Value,
    locator: &Locator,
    submitted: &SubmittedConversionDelivery,
) -> Result<bool> {
    if native["schema_version"] != 1
        || native["workflow"] != "default-mtp-composition"
        || native["request_sha256"] != locator.receipt_request_sha256
        || native["transport_input_sha256"] != submitted.native.declaration.transport_input_sha256
        || !matches!(
            native["status"].as_str(),
            Some("COMPOSITION_COMPLETED" | "FAILED")
        )
        || !matches!(
            submitted.expected_status.as_str(),
            "COMPOSED_PUBLISHED" | "DRY_RUN_NOT_EXECUTED"
        )
    {
        bail!("composition immutable correlation refused");
    }
    let op = &native["operator"];
    Ok(locator.delivery_complete
        && native["status"] == "COMPOSITION_COMPLETED"
        && native["error"].is_null()
        && op["request_sha256"] == native["operator_request_sha256"]
        && op["request_sha256"].as_str().is_some_and(|s| {
            s.len() == 64
                && s.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        })
        && op["status"] == submitted.expected_status
        && op["error"].is_null()
        && (submitted.expected_status == "DRY_RUN_NOT_EXECUTED"
            || (op["ordered_publication"]["status"] == "PUBLISHED"
                && op["ordered_publication"]["final_receipt"]["publication"]["completed"] == true)))
}
impl HfJobsClient {
    pub async fn collect_composition_until<C: Future<Output = ()>>(
        &self,
        namespace: &str,
        submitted: &SubmittedConversionDelivery,
        publisher: &Publisher,
        deadline: Instant,
        cancellation: C,
    ) -> Result<CollectedComposition> {
        let mut cancel = Box::pin(cancellation);
        let (monitor, locator, native) = generic::collection::collect_native_until(
            self,
            namespace,
            &submitted.native.job_id,
            &submitted.native.declaration,
            publisher,
            deadline,
            cancel.as_mut(),
        )
        .await?;
        let admitted =
            observe(&native, &locator, submitted)? && monitor.end == MonitorEnd::Completed;
        if Instant::now() >= deadline || cancel.as_mut().now_or_never().is_some() {
            bail!("composition collection final boundary");
        }
        Ok(CollectedComposition {
            job_id: submitted.native.job_id.clone(),
            monitor,
            locator,
            native_receipt: native,
            composition_admitted: admitted,
            image_observed: false,
            cost_observed: false,
        })
    }
}
#[cfg(test)]
#[path = "composition/tests.rs"]
mod tests;
