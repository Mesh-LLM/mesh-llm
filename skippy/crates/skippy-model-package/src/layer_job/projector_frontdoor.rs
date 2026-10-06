use crate::snapshot_promotion::{lfs_transfer, local_publisher::SignalLatch};
use anyhow::{Result, bail};
use clap::Args;
use std::{
    io::Write as _,
    path::PathBuf,
    time::{Duration, Instant},
};
#[derive(Args)]
pub(super) struct Options {
    #[arg(long)]
    pub repo: String,
    #[arg(long)]
    pub revision: String,
    #[arg(long)]
    pub file: String,
    #[arg(long)]
    pub credential_file: PathBuf,
    #[arg(long)]
    pub output_directory: PathBuf,
    #[arg(long, default_value_t = 3600)]
    pub timeout_seconds: u64,
}
pub(super) fn run(options: Options, latch: &SignalLatch) -> Result<()> {
    if !(1..=86400).contains(&options.timeout_seconds) {
        bail!("projector time bound refused");
    }
    super::repo(&options.repo)?;
    super::revision(&options.revision)?;
    super::projector::selection(&options.file)?;
    let until = Instant::now() + Duration::from_secs(options.timeout_seconds);
    let token = super::upload_frontdoor::token(&options.credential_file)?;
    let source = super::SourceClient::new(Some(token.clone()))?;
    let client = lfs_transfer::Client::new(token)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let pin=runtime.block_on(async{tokio::select!{
        result=source.projector_until(&options.repo,&options.revision,&options.file,until)=>result,
        ()=super::cli::cancelled(latch)=>Err(anyhow::anyhow!("projector metadata cancelled")),
    }})?;
    super::cli::terminal(latch, until)?;
    let root = super::cli::output(&options.output_directory)?;
    let mut staged = tempfile::NamedTempFile::new_in(&root)?;
    let observed=runtime.block_on(async{tokio::select!{
        result=client.acquire_repository_until(lfs_transfer::RepositoryRead{repo:&pin.repo,dataset:false,commit:&pin.revision,path:&pin.path,size:pin.byte_size,expected_sha256:pin.expected_sha256.as_deref()},staged.as_file_mut(),until)=>result,
        ()=super::cli::cancelled(latch)=>Err(anyhow::anyhow!("projector transfer cancelled")),
    }})?;
    super::cli::terminal(latch, until)?;
    let path = root.join("projector.gguf");
    staged.persist_noclobber(&path)?;
    super::cli::terminal(latch, until)?;
    super::cli::fresh(
        &root,
        "projector.json",
        &serde_json::to_vec(
            &serde_json::json!({"schema_version":1,"status":"ACQUIRED","source":pin,"observed_identity":observed,"digest_scope":if pin.expected_sha256.is_some(){"verified_declared_LFS_SHA256"}else{"observed_SHA256_at_immutable_source_path_and_size"}}),
        )?,
    )?;
    if let Err(error) = super::cli::terminal(latch, until) {
        std::fs::remove_file(root.join("projector.json"))?;
        return Err(error);
    }
    writeln!(mesh_llm_events::machine_out(), "{}", path.display())?;
    Ok(())
}
