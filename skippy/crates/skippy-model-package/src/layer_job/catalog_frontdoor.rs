//! Native catalog read/project/parent-bound dataset publication.
use crate::snapshot_promotion::{
    local_publisher::{SignalLatch, regular_input},
    package_upload,
};
use anyhow::{Result, bail};
use clap::Args;
use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
#[derive(Args, serde::Serialize)]
pub(super) struct Options {
    #[arg(long)]
    pub manifest: PathBuf,
    #[arg(long)]
    pub source_repo: String,
    #[arg(long)]
    pub source_revision: String,
    #[arg(long)]
    pub source_file: String,
    #[arg(long)]
    pub target_repo: String,
    #[arg(long)]
    pub credential_file: PathBuf,
    #[arg(long)]
    pub output_directory: PathBuf,
    #[arg(long)]
    pub create_pr: bool,
    #[arg(long)]
    pub confirm: bool,
    #[arg(long, default_value_t = 3600)]
    pub timeout_seconds: u64,
}
pub(super) fn run(options: Options, latch: &SignalLatch) -> Result<()> {
    if !options.confirm || !(1..=86400).contains(&options.timeout_seconds) {
        bail!("explicit catalog publication confirmation/time required");
    }
    let until = Instant::now() + Duration::from_secs(options.timeout_seconds);
    let bytes = regular_input::read(
        &mut regular_input::open(&options.manifest, super::MANIFEST_LIMIT as u64, false)?,
        super::MANIFEST_LIMIT,
    )?;
    let (manifest, _) = super::project(
        &bytes,
        &options.source_repo,
        &options.source_revision,
        options.create_pr,
    )?;
    let input = package_upload::catalog::CatalogInput {
        source_repo: &options.source_repo,
        source_revision: &options.source_revision,
        source_file: &options.source_file,
        target_repo: &options.target_repo,
        model_id: &manifest.model_id,
        layer_count: u64::from(manifest.layer_count),
    };
    let token = super::upload_frontdoor::token(&options.credential_file)?;
    let publisher = package_upload::Publisher::new(token)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let prepared = runtime.block_on(async {
        tokio::select! {
            result=publisher.prepare_catalog_until(&input,until)=>result,
            ()=super::cli::cancelled(latch)=>Err(anyhow::anyhow!("catalog read cancelled")),
        }
    })?;
    super::cli::terminal(latch, until)?;
    if regular_input::read(
        &mut regular_input::open(&options.manifest, super::MANIFEST_LIMIT as u64, false)?,
        super::MANIFEST_LIMIT,
    )? != bytes
    {
        bail!("catalog manifest changed");
    }
    let root = super::cli::output(&options.output_directory)?;
    super::cli::fresh(&root, "entry.json", &prepared.bytes)?;
    let file = regular_input::open(&root.join("entry.json"), 1024 * 1024, false)?;
    let identity = crate::snapshot_promotion::policy::ArtifactIdentity {
        byte_size: prepared.bytes.len() as u64,
        sha256: Sha256::digest(&prepared.bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
    };
    let plan = package_upload::Plan {
        repo: "meshllm/catalog".into(),
        kind: package_upload::RepositoryKind::Dataset,
        revision: "main".into(),
        path: prepared.entry_path.clone(),
        create_pr: options.create_pr,
        maximum_attempts: 8,
        expected_parent: Some(prepared.observed_parent.clone()),
    };
    let request_sha256: String = Sha256::digest(serde_json::to_vec(&options)?)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let mut receipt = runtime.block_on(publisher.upload_until(
        &plan,
        package_upload::Artifact {
            file,
            identity,
            unlink_path: None,
        },
        until,
        super::cli::cancelled(latch),
        &mut |_| Ok(()),
    ));
    if let Err(error) = super::cli::terminal(latch, until) {
        receipt.completed = false;
        receipt.error.get_or_insert(error.to_string());
    }
    let complete = receipt.completed && receipt.error.is_none();
    super::cli::fresh(
        &root,
        "catalog.json",
        &serde_json::to_vec(
            &serde_json::json!({"schema_version":1,"request_sha256":request_sha256,"status":if complete{"CATALOG_PUBLISHED"}else{"FAILED"},"observed_parent":prepared.observed_parent,"entry_missing":prepared.missing_entry,"publication":receipt}),
        )?,
    )?;
    if !complete {
        bail!("catalog publication incomplete; inspect partial receipt");
    }
    Ok(())
}
