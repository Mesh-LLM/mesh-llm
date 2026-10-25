//! Existing immutable byte transport exposed read-only for quant window resume.
use crate::snapshot_promotion::{
    local_publisher::{SignalLatch, regular_input},
    package_upload::{Plan, Publisher, RepositoryKind},
};
use anyhow::{Result, bail};
use clap::Args;
use serde::Serialize;
use serde_json::json;
use std::{
    io::{Seek as _, SeekFrom, Write as _},
    path::PathBuf,
    time::{Duration, Instant},
};
#[derive(Args, Serialize)]
pub(super) struct Options {
    #[arg(long)]
    repo: String,
    #[arg(long)]
    commit: String,
    #[arg(long)]
    artifact: PathBuf,
    #[arg(long)]
    relative_path: String,
    #[arg(long)]
    credential_file: PathBuf,
    #[arg(long)]
    output_directory: PathBuf,
    #[arg(long, default_value_t = 3600)]
    timeout_seconds: u64,
}
pub(super) fn run(options: Options, latch: &SignalLatch) -> Result<()> {
    if !(1..=86400).contains(&options.timeout_seconds) {
        bail!("immutable verification time bound");
    }
    let until = Instant::now() + Duration::from_secs(options.timeout_seconds);
    let check = || super::cli::terminal(latch, until);
    let mut file = regular_input::open(&options.artifact, 1024_u64.pow(4), false)?;
    let artifact = super::upload_frontdoor::hash(&mut file, until, latch)?;
    let plan = Plan {
        repo: options.repo.clone(),
        kind: RepositoryKind::Model,
        revision: options.commit.clone(),
        path: options.relative_path.clone(),
        create_pr: false,
        maximum_attempts: 1,
        expected_parent: None,
    };
    plan.validate()?;
    if options.output_directory.starts_with(&options.artifact)
        || options.artifact.starts_with(&options.output_directory)
        || options
            .credential_file
            .starts_with(&options.output_directory)
    {
        bail!("verification output overlaps immutable inputs");
    }
    let token = super::upload_frontdoor::token(&options.credential_file)?;
    let request_sha256 = {
        use sha2::{Digest as _, Sha256};
        Sha256::digest(serde_json::to_vec(&options)?)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    };
    let root = super::cli::output(&options.output_directory)?;
    let publisher = Publisher::new(token)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let verified = runtime.block_on(publisher.verify_artifact_until(
        &plan,
        artifact,
        until,
        super::cli::cancelled(latch),
    ));
    let complete = verified.completed && verified.error.is_none();
    let mut value = json!({"schema_version":1,"request_sha256":request_sha256,"status":if complete {"IMMUTABLE_VERIFIED"} else {"FAILED"},"verification":verified});
    if check().is_err() {
        value["status"] = json!("FAILED");
        value["terminal_refused"] = json!(true);
    }
    let temporary = tempfile::NamedTempFile::new_in(&root)?;
    let mut held = temporary.persist_noclobber(root.join("verification.json"))?;
    let write = |file: &mut std::fs::File, value: &serde_json::Value| -> Result<()> {
        let bytes = serde_json::to_vec_pretty(value)?;
        if bytes.len() > 1048576 {
            bail!("verification evidence bound");
        }
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&bytes)?;
        file.set_len(bytes.len() as u64)?;
        file.sync_all()?;
        Ok(())
    };
    write(&mut held, &value)?;
    if check().is_err() {
        value["status"] = json!("FAILED");
        value["terminal_refused"] = json!(true);
        write(&mut held, &value)?;
        bail!("immutable verification terminal refused; observations retained");
    }
    if !complete {
        bail!("immutable verification refused; observations retained");
    }
    Ok(())
}
