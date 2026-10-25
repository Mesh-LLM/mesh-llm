use super::admission::AdmittedProducer;
use super::coverage::{self, Case};
use super::error::{Checked, Rejected};
use super::identity::{ArtifactDigest, RunId, SourceSha};
use super::input::{Artifact, ArtifactList, ProducerRun, parse};
use super::rows::{Catalog, ProducerWorkflow, ProductRow};
use serde::Serialize;
use std::collections::BTreeSet;
use std::num::NonZeroU64;

#[derive(Debug, Serialize)]
pub(super) struct Selection {
    validation: &'static str,
    payload_verification: &'static str,
    repository: String,
    producer_run_id: RunId,
    producer_run_attempt: NonZeroU64,
    producer_workflow: ProducerWorkflow,
    source_sha: SourceSha,
    model_manifest: &'static str,
    model_cadence: &'static str,
    pub(super) products: Vec<SelectedProduct>,
}

#[derive(Debug, Serialize)]
pub(super) struct SelectedProduct {
    pub(super) row: ProductRow,
    platform: String,
    architecture: String,
    backend: String,
    target: String,
    pub(super) artifact_id: NonZeroU64,
    pub(super) artifact_name: String,
    pub(super) artifact_digest: ArtifactDigest,
    pub(super) cases: Vec<Case>,
}

pub(super) fn select(
    producer: &AdmittedProducer,
    catalog: &Catalog,
    artifact_bytes: &[u8],
) -> Checked<Selection> {
    let inventory: ArtifactList = parse(artifact_bytes)?;
    if inventory.total_count != inventory.artifacts.len() {
        return Err(Rejected::IncompleteArtifacts);
    }
    let mut ids = BTreeSet::new();
    for artifact in &inventory.artifacts {
        if !ids.insert(artifact.id) {
            return Err(Rejected::DuplicateArtifactId);
        }
    }
    let products = producer
        .rows()
        .iter()
        .map(|requested| {
            let row = catalog.row(*requested)?;
            let name = format!(
                "ci-product-{}-{}-{}",
                row.platform, row.architecture, row.backend
            );
            let artifact = exact_artifact(&inventory.artifacts, &name)?;
            let digest = bind_artifact(artifact, producer.run())?;
            Ok(SelectedProduct {
                row: *requested,
                platform: row.platform.clone(),
                architecture: row.architecture.clone(),
                backend: row.backend.clone(),
                target: row.target.clone(),
                artifact_id: artifact.id,
                artifact_name: name,
                artifact_digest: digest,
                cases: coverage::cases(),
            })
        })
        .collect::<Checked<Vec<_>>>()?;
    Ok(Selection {
        validation: "offline_metadata_selection",
        payload_verification: "required_before_execution",
        repository: producer.run().repository.full_name.clone(),
        producer_run_id: producer.run().id,
        producer_run_attempt: producer.run().run_attempt,
        producer_workflow: producer.workflow(),
        source_sha: producer.run().head_sha.clone(),
        model_manifest: "ci/model-artifacts/manifests/product-smoke.json",
        model_cadence: "main",
        products,
    })
}

fn exact_artifact<'a>(artifacts: &'a [Artifact], name: &str) -> Checked<&'a Artifact> {
    let mut matching = artifacts.iter().filter(|artifact| artifact.name == name);
    let artifact = matching.next().ok_or(Rejected::MissingArtifact)?;
    if matching.next().is_some() {
        return Err(Rejected::AmbiguousArtifact);
    }
    Ok(artifact)
}

fn bind_artifact(artifact: &Artifact, producer: &ProducerRun) -> Checked<ArtifactDigest> {
    if artifact.expired {
        return Err(Rejected::ExpiredArtifact);
    }
    let origin = artifact
        .workflow_run
        .as_ref()
        .ok_or(Rejected::ArtifactIdentity)?;
    if origin.id != producer.id
        || origin.repository_id != producer.repository.id
        || origin.head_repository_id != producer.head_repository.id
        || origin.head_branch != producer.head_branch
        || origin.head_sha != producer.head_sha
    {
        return Err(Rejected::ArtifactIdentity);
    }
    artifact
        .digest
        .as_deref()
        .and_then(ArtifactDigest::parse)
        .ok_or(Rejected::ArtifactDigest)
}
