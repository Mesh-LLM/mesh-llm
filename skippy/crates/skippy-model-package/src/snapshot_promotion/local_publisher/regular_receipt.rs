//! One pinned native-job JSON receipt, published through the existing regular-file owner.
use super::{SignalLatch, contract::Artifact, output, regular_input};
use crate::snapshot_promotion::{
    policy::ArtifactIdentity,
    regular_publication::{LocalFile, Plan, Publisher, Receipt as Publication, Secret},
};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: u32,
    repo: String,
    parent_commit: String,
    artifact: Artifact,
    receipt_request_sha256: String,
    credential_file: PathBuf,
    execution_timeout_ms: u64,
}
#[derive(Serialize)]
struct Receipt<'a> {
    schema_version: u32,
    request_sha256: &'a str,
    status: &'a str,
    receipt_request_sha256: &'a str,
    artifact_sha256: &'a str,
    publication: Option<&'a Publication>,
    input_custody_verified: bool,
    error: Option<&'a str>,
}
fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn pin(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn validate(input: &Input) -> Result<Plan> {
    if input.schema_version != 1
        || !(1..=86_400_000).contains(&input.execution_timeout_ms)
        || input.artifact.byte_size == 0
        || input.artifact.byte_size > 1024 * 1024
        || !pin(&input.artifact.sha256, 64)
        || !pin(&input.receipt_request_sha256, 64)
        || !input.artifact.path_in_repo.ends_with("/native-job.json")
    {
        bail!("regular native-job receipt schema/path/pins refused");
    }
    let plan = Plan {
        repo: input.repo.clone(),
        parent_commit: input.parent_commit.clone(),
        paths: vec![input.artifact.path_in_repo.clone()],
    };
    plan.validate()?;
    Ok(plan)
}
fn artifact(input: &Input, until: Instant) -> Result<LocalFile> {
    if Instant::now() >= until {
        bail!("receipt deadline expired before local admission");
    }
    let mut file = regular_input::open(&input.artifact.path, 1024 * 1024, false)?;
    let bytes = regular_input::read(&mut file, 1024 * 1024)?;
    if bytes.len() as u64 != input.artifact.byte_size || digest(&bytes) != input.artifact.sha256 {
        bail!("receipt exact-byte pin refused");
    }
    let receipt: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|_| anyhow::anyhow!("native-job JSON refused"))?;
    if receipt["schema_version"] != 1
        || receipt["request_sha256"].as_str() != Some(input.receipt_request_sha256.as_str())
        || !matches!(
            receipt["status"].as_str(),
            Some("CERTIFIED" | "COMPOSED" | "FAILED")
        )
    {
        bail!("native-job typed receipt correlation refused");
    }
    Ok(LocalFile {
        path_in_repo: input.artifact.path_in_repo.clone(),
        file,
        identity: ArtifactIdentity {
            sha256: input.artifact.sha256.clone(),
            byte_size: input.artifact.byte_size,
        },
    })
}
pub(super) fn execute(input_path: &Path, output_path: &Path) -> Result<bool> {
    let mut source = regular_input::open(input_path, 512 * 1024, false)?;
    let bytes = regular_input::read(&mut source, 512 * 1024)?;
    let raw_hash = digest(&bytes);
    let input: Input = serde_json::from_slice(&bytes)
        .map_err(|_| anyhow::anyhow!("regular receipt input refused"))?;
    let request_hash = digest(&serde_json::to_vec(&input)?);
    let plan = validate(&input)?;
    let until = Instant::now()
        .checked_add(Duration::from_millis(input.execution_timeout_ms))
        .ok_or_else(|| anyhow::anyhow!("regular receipt deadline overflow"))?;
    let file = artifact(&input, until)?;
    let mut credential = regular_input::open(&input.credential_file, 8192, true)?;
    let mut token = String::from_utf8(regular_input::read(&mut credential, 8192)?)?;
    if token.ends_with('\n') {
        token.pop();
    }
    let publisher = Publisher::new(Secret::new(token)?)?;
    let output = output::Output::fresh(output_path)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let latch = SignalLatch::install()?;
        use tokio::signal::unix::{SignalKind, signal};
        let mut term = signal(SignalKind::terminate())?;
        let mut interrupt = signal(SignalKind::interrupt())?;
        let cancellation = async {
            tokio::select! {_=term.recv()=>(),_=interrupt.recv()=>()}
        };
        let mut observer = |publication: &Publication| {
            output.write(
                "progress.json",
                &Receipt {
                    schema_version: 1,
                    request_sha256: &request_hash,
                    status: "IN_PROGRESS",
                    receipt_request_sha256: &input.receipt_request_sha256,
                    artifact_sha256: &input.artifact.sha256,
                    publication: Some(publication),
                    input_custody_verified: false,
                    error: None,
                },
                false,
            )
        };
        let mut publication = publisher
            .publish_observed_until(&plan, vec![file], until, cancellation, &mut observer)
            .await;
        let custody =
            regular_input::read(&mut source, 512 * 1024).is_ok_and(|b| digest(&b) == raw_hash);
        let complete =
            publication.completed && custody && !latch.cancelled() && Instant::now() < until;
        if !complete {
            publication.completed = false;
            publication.error = Some("regular receipt helper terminal admission refused".into());
        }
        output.write(
            "publication.json",
            &Receipt {
                schema_version: 1,
                request_sha256: &request_hash,
                status: if complete {
                    "PUBLISHED_REGULAR_RECEIPT"
                } else {
                    "FAILED"
                },
                receipt_request_sha256: &input.receipt_request_sha256,
                artifact_sha256: &input.artifact.sha256,
                publication: Some(&publication),
                input_custody_verified: custody,
                error: if complete {
                    None
                } else {
                    Some("regular_receipt_operation_incomplete")
                },
            },
            true,
        )?;
        Ok(complete)
    })
}
#[cfg(test)]
#[path = "regular_receipt/tests.rs"]
mod tests;
