//! Full package card preparation; publication uses the same owned artifact uploader separately.
use crate::snapshot_promotion::local_publisher::{SignalLatch, regular_input};
use anyhow::{Result, bail};
use clap::Args;
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
    pub experimental: bool,
    #[arg(long, default_value = "text-generation")]
    pub pipeline_tag: String,
    #[arg(long, default_value = "main")]
    pub mesh_llm_ref: String,
}
pub(super) fn run(options: Options, latch: &SignalLatch) -> Result<()> {
    let until = Instant::now() + Duration::from_secs(300);
    let bytes = regular_input::read(
        &mut regular_input::open(&options.manifest, super::MANIFEST_LIMIT as u64, false)?,
        super::MANIFEST_LIMIT,
    )?;
    let (manifest, projection) = super::project(
        &bytes,
        &options.source_repo,
        &options.source_revision,
        options.experimental,
    )?;
    let client = super::SourceClient::new(Some(super::upload_frontdoor::token(
        &options.credential_file,
    )?))?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let license=runtime.block_on(async{tokio::select!{
        value=client.license_until(&options.source_repo,&options.source_revision,until)=>Ok(value),
        ()=super::cli::cancelled(latch)=>Err(anyhow::anyhow!("model card cancelled")),
    }})?;
    super::cli::terminal(latch, until)?;
    let pipeline = options.pipeline_tag.trim();
    let pipeline = if pipeline.is_empty() {
        "text-generation"
    } else {
        pipeline
    };
    let card = super::card::render_complete(
        &manifest,
        &projection,
        super::card::CardContext {
            target: &options.target_repo,
            pipeline,
            source_file: &options.source_file,
            mesh_ref: &options.mesh_llm_ref,
            license: &license,
        },
    )?;
    if regular_input::read(
        &mut regular_input::open(&options.manifest, super::MANIFEST_LIMIT as u64, false)?,
        super::MANIFEST_LIMIT,
    )? != bytes
    {
        bail!("model card manifest changed");
    }
    let root = super::cli::output(&options.output_directory)?;
    super::cli::fresh(&root, "README.md", card.as_bytes())?;
    super::cli::terminal(latch, until)?;
    super::cli::fresh(&root, "license.json", &serde_json::to_vec(&license)?)?;
    Ok(())
}
