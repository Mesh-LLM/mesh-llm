use super::{
    policy::{self, SnapshotTransport},
    transport::HubTransport,
};
use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Parser)]
#[command(
    name = "promote-layer-package-snapshot",
    about = "Preview or atomically publish a staged layer-package snapshot"
)]
struct Cli {
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand)]
enum Action {
    Prepare {
        #[arg(long)]
        repo: String,
        #[arg(long)]
        source_revision: String,
        /// Non-secret nonce used in the staging branch name.
        #[arg(long)]
        token: String,
        /// Create the staging branch. Default only previews the plan.
        #[arg(long)]
        confirm: bool,
    },
    Promote {
        #[arg(long)]
        repo: String,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        staging_revision: String,
        #[arg(long)]
        parent_commit: String,
        /// Commit the complete snapshot to main. Default only previews the plan.
        #[arg(long)]
        confirm: bool,
    },
}

fn manifest_bytes(path: &Path) -> Result<Vec<u8>> {
    const LIMIT: u64 = 64 * 1024 * 1024;
    let metadata = std::fs::metadata(path).context("inspect promotion manifest")?;
    if !metadata.is_file() || metadata.len() > LIMIT {
        bail!("promotion manifest must be a bounded regular file");
    }
    let mut bytes = Vec::new();
    File::open(path)?.take(LIMIT + 1).read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())? > LIMIT {
        bail!("promotion manifest exceeds input limit");
    }
    Ok(bytes)
}

pub fn run(output: &mut dyn std::io::Write, diagnostics: &mut dyn std::io::Write) -> Result<()> {
    match Cli::parse().action {
        Action::Prepare {
            repo,
            source_revision,
            token,
            confirm,
        } => {
            let mut transport = HubTransport::new(repo)?;
            let plan = if confirm {
                policy::prepare_with(&mut transport, &source_revision, &token)?
            } else {
                let parent = transport.main_revision()?;
                policy::prepare(&source_revision, &token, &parent)?
            };
            if confirm {
                // Two lines are consumed by the existing embedded job caller.
                writeln!(
                    output,
                    "{}\n{}",
                    plan.staging_revision(),
                    plan.parent_commit()
                )?;
            } else {
                writeln!(output, "{}", serde_json::to_string_pretty(&plan)?)?;
            }
        }
        Action::Promote {
            repo,
            manifest,
            staging_revision,
            parent_commit,
            confirm,
        } => {
            let bytes = manifest_bytes(&manifest)?;
            let plan = policy::promote(&bytes, &staging_revision, &parent_commit)?;
            if confirm {
                let mut transport = HubTransport::new(repo)?;
                if let Some(warning) = policy::promote_with(&mut transport, &plan)? {
                    let _ = writeln!(
                        diagnostics,
                        "WARNING: promotion succeeded, staging cleanup failed: {warning}"
                    );
                }
            } else {
                writeln!(output, "{}", serde_json::to_string_pretty(&plan)?)?;
            }
        }
    }
    output.flush()?;
    diagnostics.flush()?;
    Ok(())
}
