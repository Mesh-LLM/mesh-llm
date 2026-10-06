//! Explicit one-artifact upload frontdoor; source pins are observed FD bytes, not model certification.
use crate::snapshot_promotion::{
    local_publisher::{SignalLatch, regular_input},
    package_upload,
};
use anyhow::{Result, bail};
use clap::Args;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    io::{Read, Seek},
    path::PathBuf,
    time::{Duration, Instant},
};
#[derive(Args, Serialize)]
pub(super) struct Options {
    #[arg(long)]
    pub repo: String,
    #[arg(long, default_value = "main")]
    pub revision: String,
    #[arg(long)]
    pub artifact: PathBuf,
    #[arg(long)]
    pub relative_path: String,
    #[arg(long)]
    pub credential_file: PathBuf,
    #[arg(long)]
    pub output_directory: PathBuf,
    #[arg(long, default_value_t = 8)]
    pub maximum_attempts: u8,
    #[arg(long, default_value_t = 3600)]
    pub timeout_seconds: u64,
    #[arg(long)]
    pub dataset: bool,
    #[arg(long)]
    pub create_pr: bool,
    #[arg(long)]
    pub unlink_after_success: bool,
    #[arg(long)]
    pub admit_only: bool,
    #[arg(long)]
    pub confirm: bool,
}
fn hash(
    file: &mut std::fs::File,
    until: Instant,
    latch: &SignalLatch,
) -> Result<package_upload::Artifact> {
    let size = file.metadata()?.len();
    if size == 0 || size > 1024_u64 * 1024 * 1024 * 1024 {
        bail!("artifact size refused");
    }
    file.rewind()?;
    let mut digest = Sha256::new();
    let mut seen = 0_u64;
    let mut bytes = [0; 65536];
    loop {
        super::cli::terminal(latch, until)?;
        let n = file.read(&mut bytes)?;
        if n == 0 {
            break;
        }
        seen = seen
            .checked_add(n as u64)
            .ok_or_else(|| anyhow::anyhow!("artifact overflow"))?;
        if seen > size {
            bail!("artifact grew");
        }
        digest.update(&bytes[..n]);
    }
    if seen != size {
        bail!("artifact changed size");
    }
    super::cli::terminal(latch, until)?;
    file.rewind()?;
    Ok(package_upload::Artifact {
        file: file.try_clone()?,
        identity: crate::snapshot_promotion::policy::ArtifactIdentity {
            byte_size: size,
            sha256: digest
                .finalize()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect(),
        },
        unlink_path: None,
    })
}
pub(super) fn token(path: &std::path::Path) -> Result<String> {
    let mut file = regular_input::open(path, 4096, true)?;
    let bytes = regular_input::read(&mut file, 4096)?;
    let token = String::from_utf8(bytes)?;
    if token.is_empty() || token.bytes().any(|b| !b.is_ascii_graphic()) {
        bail!("private credential refused");
    }
    Ok(token)
}
pub(super) fn run(options: Options, latch: &SignalLatch) -> Result<()> {
    if !(1..=86400).contains(&options.timeout_seconds) || (!options.admit_only && !options.confirm)
    {
        bail!("explicit bounded upload confirmation required");
    }
    let until = Instant::now()
        .checked_add(Duration::from_secs(options.timeout_seconds))
        .ok_or_else(|| anyhow::anyhow!("upload deadline overflow"))?;
    let request = serde_json::to_vec(&options)?;
    let request_sha256: String = Sha256::digest(&request)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let plan = package_upload::Plan {
        repo: options.repo.clone(),
        kind: if options.dataset {
            package_upload::RepositoryKind::Dataset
        } else {
            package_upload::RepositoryKind::Model
        },
        revision: options.revision.clone(),
        path: options.relative_path.clone(),
        create_pr: options.create_pr,
        maximum_attempts: options.maximum_attempts,
        expected_parent: None,
    };
    plan.validate()?;
    super::cli::terminal(latch, until)?;
    let mut file = regular_input::open(&options.artifact, 1024_u64 * 1024 * 1024 * 1024, false)?;
    let mut artifact = hash(&mut file, until, latch)?;
    if options.unlink_after_success {
        artifact.unlink_path = Some(options.artifact.clone());
    }
    let publisher = package_upload::Publisher::new(token(&options.credential_file)?)?;
    let root = super::cli::output(&options.output_directory)?;
    if options.admit_only {
        super::cli::terminal(latch, until)?;
        super::cli::fresh(
            &root,
            "upload.json",
            &serde_json::to_vec(
                &serde_json::json!({"schema_version":1,"request_sha256":request_sha256,"status":"ADMITTED","source_identity_origin":"observed_local_fd","identity":artifact.identity,"publication_performed":false}),
            )?,
        )?;
        return Ok(());
    }
    let mut progress = |receipt: &package_upload::Receipt| -> Result<()> {
        super::cli::terminal(latch, until)?;
        let value = serde_json::json!({"schema_version":1,"request_sha256":request_sha256,"status":"INCOMPLETE","publication":receipt});
        use std::io::Write;
        let mut staged = tempfile::NamedTempFile::new_in(&root)?;
        staged.write_all(&serde_json::to_vec(&value)?)?;
        staged.flush()?;
        staged.as_file().sync_all()?;
        staged.persist(root.join("progress.json"))?;
        Ok(())
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let mut receipt = runtime.block_on(publisher.upload_until(
        &plan,
        artifact,
        until,
        super::cli::cancelled(latch),
        &mut progress,
    ));
    if let Err(error) = super::cli::terminal(latch, until) {
        receipt.completed = false;
        receipt.error.get_or_insert(error.to_string());
    }
    let completed = receipt.completed && receipt.error.is_none();
    super::cli::fresh(
        &root,
        "upload.json",
        &serde_json::to_vec(
            &serde_json::json!({"schema_version":1,"request_sha256":request_sha256,"status":if completed{"PUBLISHED"}else{"FAILED"},"source_identity_origin":"observed_local_fd","publication":receipt}),
        )?,
    )?;
    if !completed {
        bail!("package artifact publication incomplete; inspect correlated receipt");
    }
    Ok(())
}

#[derive(Args, Serialize)]
pub(super) struct RepoOptions {
    #[arg(long)]
    pub repo: String,
    #[arg(long)]
    pub credential_file: PathBuf,
    #[arg(long)]
    pub output_directory: PathBuf,
    #[arg(long, default_value_t = 300)]
    pub timeout_seconds: u64,
    #[arg(long)]
    pub confirm: bool,
}
pub(super) fn ensure_repo(options: RepoOptions, latch: &SignalLatch) -> Result<()> {
    if !options.confirm || !(1..=1200).contains(&options.timeout_seconds) {
        bail!("explicit bounded repository confirmation required");
    }
    package_upload::Plan {
        repo: options.repo.clone(),
        kind: package_upload::RepositoryKind::Model,
        revision: "main".into(),
        path: "model-package.json".into(),
        create_pr: false,
        maximum_attempts: 1,
        expected_parent: None,
    }
    .validate()?;
    let until = Instant::now() + Duration::from_secs(options.timeout_seconds);
    let request_sha256: String = Sha256::digest(serde_json::to_vec(&options)?)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    super::cli::terminal(latch, until)?;
    let publisher = package_upload::Publisher::new(token(&options.credential_file)?)?;
    let root = super::cli::output(&options.output_directory)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let mut receipt = runtime.block_on(publisher.ensure_model_repo_until(
        &options.repo,
        true,
        until,
        super::cli::cancelled(latch),
    ));
    if let Err(error) = super::cli::terminal(latch, until) {
        receipt.completed = false;
        receipt.error.get_or_insert(error.to_string());
    }
    let completed = receipt.completed && receipt.error.is_none();
    super::cli::fresh(
        &root,
        "repository.json",
        &serde_json::to_vec(
            &serde_json::json!({"schema_version":1,"request_sha256":request_sha256,"status":if completed{"REPOSITORY_READY"}else{"FAILED"},"repository":receipt}),
        )?,
    )?;
    if !completed {
        bail!("repository provisioning incomplete; inspect correlated receipt");
    }
    Ok(())
}
