//! Closed native certification Jobs delivery. A supplied image is declared, never observed here.
use super::{CpuJobPlan, HfJobsClient, JobSpec, JobStage, JobVolume, estimate_cost_usd};
use anyhow::{Result, bail};
use futures::{
    Future, FutureExt as _,
    future::{Either, select},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::{
    collections::HashMap,
    path::{Component, Path},
    time::Instant,
};
#[path = "delivery/admission.rs"]
mod admission;
#[path = "delivery/composition.rs"]
pub mod composition;
#[path = "delivery/generic.rs"]
pub mod generic;
#[cfg(unix)]
#[path = "delivery/generic_cli.rs"]
pub mod generic_cli;
#[path = "delivery/receipts.rs"]
pub mod receipts;
#[path = "delivery/request_transport.rs"]
pub mod request_transport;
#[cfg(test)]
#[path = "delivery/tests.rs"]
mod tests;
const INPUT_KEY: &str = "MESH_HF_JOB_INPUT";
const OUTPUT: &str = "/work/native-job";
#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelMount {
    pub repo: String,
    pub revision: String,
    pub mount_path: String,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeliveryDeclaration {
    pub schema_version: u32,
    pub transport_input_sha256: String,
    pub runner_sha256: String,
    pub mesh_commit: String,
    pub image: String,
    pub flavor: String,
    pub timeout_seconds: u64,
    pub cpu_plan_receipt_sha256: String,
    pub declared_estimate_usd: f64,
    pub evidence_repo: String,
    pub evidence_parent_commit: String,
    pub evidence_path: String,
    pub image_observed: bool,
    pub submitted: bool,
    pub native_certification_completed: bool,
}
/// No Debug or input accessor: the worker request may contain a signed projector URL.
pub struct PreparedCertificationDelivery {
    spec: JobSpec,
    declaration: DeliveryDeclaration,
}
impl PreparedCertificationDelivery {
    /// The fixed explicit secret is materialized as an owned private file by the native worker.
    pub fn with_publication_credential(mut self, token: String) -> Result<Self> {
        let _ = crate::snapshot_promotion::regular_publication::Secret::new(token.clone())?;
        self.spec
            .secrets
            .insert("MESH_HF_PUBLICATION_TOKEN".into(), token);
        Ok(self)
    }
    pub fn declaration(&self) -> &DeliveryDeclaration {
        &self.declaration
    }
    /// Transport admission is intentionally distinct from the worker's full native admission.
    /// Readonly immutable mounts supply GGUFs; the declared image must already contain
    /// the pinned Linux runner/tools and writable /work parent. This function builds no image.
    pub fn prepare(bytes: &[u8], mounts: &[ModelMount], plan: &CpuJobPlan) -> Result<Self> {
        let request = admission::request(bytes, mounts, plan)?;
        let worker_input = serde_json::to_string(&request)?;
        let bootstrap = &request["bootstrap"];
        let image = admission::text(bootstrap, "image")?.to_owned();
        let runner = admission::text(&request["runner"], "path")?.to_owned();
        let plan_sha = admission::digest(&serde_json::to_vec(plan)?);
        let declaration = DeliveryDeclaration {
            schema_version: 1,
            transport_input_sha256: admission::digest(worker_input.as_bytes()),
            runner_sha256: admission::text(&request["runner"], "sha256")?.into(),
            mesh_commit: admission::text(bootstrap, "mesh_commit")?.into(),
            image: image.clone(),
            flavor: plan.flavor.clone(),
            timeout_seconds: plan.timeout_seconds,
            cpu_plan_receipt_sha256: plan_sha,
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
            command: vec![runner],
            arguments: [
                "automation",
                "hf-certify",
                "job-worker",
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
        Ok(Self { spec, declaration })
    }
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SubmittedCertificationDelivery {
    pub declaration: DeliveryDeclaration,
    pub job_id: String,
    pub stage: JobStage,
}
impl HfJobsClient {
    /// Uses the existing bounded submit engine. A response only acknowledges delivery;
    /// neither COMPLETED Jobs state nor POST success is a native certification receipt.
    /// Cancelling an in-flight POST leaves remote acceptance uncertain and sends no cancel.
    pub async fn submit_certification_until<C: Future<Output = ()>>(
        &self,
        namespace: &str,
        prepared: PreparedCertificationDelivery,
        deadline: Instant,
        cancellation: C,
    ) -> Result<SubmittedCertificationDelivery> {
        if !prepared
            .spec
            .secrets
            .contains_key("MESH_HF_PUBLICATION_TOKEN")
        {
            bail!("explicit native receipt publication credential required before submission");
        }
        let cancel = Box::pin(cancellation);
        let mut cancel = cancel;
        if cancel.as_mut().now_or_never().is_some() || Instant::now() >= deadline {
            bail!("native certification submission cancelled or expired before request");
        }
        let result = select(
            cancel,
            Box::pin(self.submit_until(namespace, &prepared.spec, deadline)),
        )
        .await;
        let (job, mut retained) = match result {
            Either::Left(_) => {
                bail!("native certification submission cancelled; remote acceptance unknown")
            }
            Either::Right((job, retained)) => (job?, retained),
        };
        if retained.as_mut().now_or_never().is_some() || Instant::now() >= deadline {
            bail!(
                "native certification submission terminal boundary refused; remote acceptance unknown"
            );
        }
        if job.id.is_empty()
            || job.id.len() > 128
            || !job
                .id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
        {
            bail!("native certification submission returned invalid job identity");
        }
        let mut declaration = prepared.declaration;
        declaration.submitted = true;
        Ok(SubmittedCertificationDelivery {
            declaration,
            job_id: job.id,
            stage: job.status.stage,
        })
    }
}
