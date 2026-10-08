//! Read-only whole immutable quant commit acquisition for native verification/package follow-on.
use crate::snapshot_promotion::{
    local_publisher::{SignalLatch, regular_input},
    package_upload::{CommitRequest, Publisher},
};
use anyhow::{Result, bail};
use clap::Args;
use serde_json::json;
use std::{
    io::{Seek as _, SeekFrom, Write as _},
    path::PathBuf,
    time::{Duration, Instant},
};
#[derive(Args)]
pub(super) struct Options {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    credential_file: PathBuf,
    #[arg(long)]
    output_directory: PathBuf,
    #[arg(long, default_value_t = 3600)]
    timeout_seconds: u64,
}
pub(super) fn run(options: Options, latch: &SignalLatch) -> Result<()> {
    use sha2::{Digest as _, Sha256};
    if !(1..=86400).contains(&options.timeout_seconds) {
        bail!("quant commit time bound");
    }
    let until = Instant::now() + Duration::from_secs(options.timeout_seconds);
    let bytes = regular_input::read(
        &mut regular_input::open(&options.input, 8 * 1048576, false)?,
        8 * 1048576,
    )?;
    let input: CommitRequest = serde_json::from_slice(&bytes)?;
    input.validate()?;
    if options.input.starts_with(&options.output_directory)
        || options
            .credential_file
            .starts_with(&options.output_directory)
    {
        bail!("quant commit output overlaps immutable request/credential");
    }
    let request_sha256: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let token = super::upload_frontdoor::token(&options.credential_file)?;
    let root = super::cli::output(&options.output_directory)?;
    let publisher = Publisher::new(token)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let receipt = runtime.block_on(publisher.acquire_quant_commit_until(
        &input,
        &root.join("artifacts"),
        until,
        super::cli::cancelled(latch),
    ));
    let complete = receipt.completed && receipt.error.is_none();
    let mut value = json!({"schema_version":1,"request_sha256":request_sha256,"status":if complete {"QUANT_COMMIT_BYTES_VERIFIED"}else{"FAILED"},"tensor_model_qualified":false,"verification":receipt});
    if super::cli::terminal(latch, until).is_err() {
        value["status"] = json!("FAILED");
        value["terminal_refused"] = json!(true);
    }
    let temporary = tempfile::NamedTempFile::new_in(&root)?;
    let mut held = temporary.persist_noclobber(root.join("commit.json"))?;
    let write = |f: &mut std::fs::File, v: &serde_json::Value| -> Result<()> {
        let bytes = serde_json::to_vec_pretty(v)?;
        if bytes.len() > 8 * 1048576 {
            bail!("quant commit receipt bound");
        }
        f.seek(SeekFrom::Start(0))?;
        f.write_all(&bytes)?;
        f.set_len(bytes.len() as u64)?;
        f.sync_all()?;
        Ok(())
    };
    write(&mut held, &value)?;
    if super::cli::terminal(latch, until).is_err() {
        value["status"] = json!("FAILED");
        value["terminal_refused"] = json!(true);
        write(&mut held, &value)?;
        bail!("quant commit terminal refused; partial acquisition retained");
    }
    if !complete {
        bail!("quant commit bytes incomplete; inspect observations");
    }
    Ok(())
}
