use crate::snapshot_promotion::local_publisher::{SignalLatch, regular_input};
use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Parser)]
#[command(
    name = "model-package-layer-job",
    about = "Native source/projector/card ownership and explicitly confirmed repository, artifact and dataset-catalog operations; no runtime qualification"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Projector(super::projector_frontdoor::Options),
    FormatBytes {
        #[arg(long)]
        bytes: u64,
    },
    WorkspaceEstimate {
        #[arg(long)]
        bytes: u64,
    },
    GenerationDefaults {
        #[arg(long)]
        file: PathBuf,
    },
    PrepareCard(super::card_frontdoor::Options),
    UpdateCatalog(super::catalog_frontdoor::Options),
    Upload(super::upload_frontdoor::Options),
    VerifyUpload(super::verification_frontdoor::Options),
    VerifyQuantCommit(super::commit_frontdoor::Options),
    EnsureRepo(super::upload_frontdoor::RepoOptions),
    Source {
        #[arg(long)]
        repo: String,
        #[arg(long, default_value = "main")]
        revision: String,
        #[arg(long)]
        credential_file: Option<PathBuf>,
        #[arg(long)]
        output_directory: PathBuf,
        #[arg(long, default_value_t = 300)]
        timeout_seconds: u64,
    },
    Project {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        source_repo: String,
        #[arg(long)]
        source_revision: String,
        #[arg(long)]
        source_admission_file: PathBuf,
        #[arg(long)]
        target_repo: String,
        #[arg(long)]
        output_directory: PathBuf,
        #[arg(long)]
        experimental: bool,
        #[arg(long, default_value = "text-generation")]
        pipeline_tag: String,
    },
}
fn read(path: &Path, cap: usize, credential: bool) -> Result<Vec<u8>> {
    regular_input::read(&mut regular_input::open(path, cap as u64, credential)?, cap)
}
pub(super) fn output(path: &Path) -> Result<PathBuf> {
    if !path.is_absolute() {
        bail!("fresh absolute output required");
    }
    match std::fs::symlink_metadata(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        _ => bail!("fresh output refused"),
    }
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("output parent absent"))?
        .canonicalize()?;
    let root = parent.join(
        path.file_name()
            .ok_or_else(|| anyhow::anyhow!("output leaf absent"))?,
    );
    std::fs::create_dir(&root)?;
    Ok(root)
}
pub(super) fn fresh(root: &Path, name: &str, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut staged = tempfile::NamedTempFile::new_in(root)?;
    staged.write_all(bytes)?;
    staged.flush()?;
    staged.as_file().sync_all()?;
    staged.persist_noclobber(root.join(name))?;
    Ok(())
}
pub fn run(output: &mut dyn std::io::Write) -> Result<()> {
    #[cfg(not(unix))]
    {
        bail!("layer job input custody currently requires Unix");
    }
    #[cfg(unix)]
    {
        run_unix(Cli::parse(), output)
    }
}
#[cfg(unix)]
fn run_unix(cli: Cli, writer: &mut dyn std::io::Write) -> Result<()> {
    let latch = SignalLatch::install()?;
    match cli.command {
        Command::Projector(options) => super::projector_frontdoor::run(options, &latch, writer)?,
        Command::FormatBytes { bytes } => writeln!(writer, "{}", super::workspace::format(bytes))?,
        Command::WorkspaceEstimate { bytes } => {
            writeln!(writer, "{}", super::workspace::estimate(bytes)?)?
        }
        Command::GenerationDefaults { file } => {
            writeln!(writer, "{}", super::workspace::generation(&file)?)?
        }
        Command::PrepareCard(options) => super::card_frontdoor::run(options, &latch)?,
        Command::UpdateCatalog(options) => super::catalog_frontdoor::run(options, &latch)?,
        Command::Upload(options) => super::upload_frontdoor::run(options, &latch)?,
        Command::VerifyUpload(options) => super::verification_frontdoor::run(options, &latch)?,
        Command::VerifyQuantCommit(options) => super::commit_frontdoor::run(options, &latch)?,
        Command::EnsureRepo(options) => super::upload_frontdoor::ensure_repo(options, &latch)?,
        Command::Source {
            repo,
            revision,
            credential_file,
            output_directory,
            timeout_seconds,
        } => {
            if !(1..=1200).contains(&timeout_seconds) {
                bail!("source time bound refused");
            }
            let deadline = Instant::now() + Duration::from_secs(timeout_seconds);
            let token = credential_file
                .map(|p| {
                    read(&p, 4096, true).and_then(|b| String::from_utf8(b).map_err(Into::into))
                })
                .transpose()?;
            let client = super::SourceClient::new(token)?;
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            let (source, files) = runtime.block_on(async {
                tokio::select! {
                    result=client.admit_until(&repo,&revision,deadline)=>result,
                    ()=cancelled(&latch)=>Err(anyhow::anyhow!("source admission cancelled")),
                }
            })?;
            terminal(&latch, deadline)?;
            let root = output(&output_directory)?;
            for (name, bytes) in files {
                terminal(&latch, deadline)?;
                fresh(&root, &name, &bytes)?;
            }
            terminal(&latch, deadline)?;
            fresh(&root, "source.json", &serde_json::to_vec(&source)?)?;
            terminal(&latch, deadline)?;
            writeln!(writer, "{}", source.revision)?;
        }
        Command::Project {
            manifest,
            source_repo,
            source_revision,
            source_admission_file,
            target_repo,
            output_directory,
            experimental,
            pipeline_tag,
        } => {
            let deadline = Instant::now() + Duration::from_secs(30);
            let source_bytes = read(&source_admission_file, 65536, false)?;
            let source: super::Source = serde_json::from_slice(&source_bytes)?;
            if source.repo != source_repo || source.revision != source_revision {
                bail!("admitted source receipt mismatch");
            }
            let bytes = read(&manifest, super::MANIFEST_LIMIT, false)?;
            let (package, projection) =
                super::project(&bytes, &source_repo, &source_revision, experimental)?;
            let card = super::card::render(
                &package,
                &projection,
                &target_repo,
                &pipeline_tag,
                source.license.as_deref(),
            )?;
            terminal(&latch, deadline)?;
            if read(&manifest, super::MANIFEST_LIMIT, false)? != bytes
                || read(&source_admission_file, 65536, false)? != source_bytes
            {
                bail!("local manifest/source receipt changed");
            }
            let root = output(&output_directory)?;
            fresh(&root, "README.preview.md", card.as_bytes())?;
            terminal(&latch, deadline)?;
            fresh(&root, "projection.json", &serde_json::to_vec(&projection)?)?;
            terminal(&latch, deadline)?;
            writeln!(
                writer,
                "{}\n{}\n{}",
                projection.source_identity, projection.layer_count, projection.total_bytes
            )?;
        }
    }
    writer.flush()?;
    Ok(())
}
#[cfg(unix)]
pub(super) fn terminal(latch: &SignalLatch, deadline: Instant) -> Result<()> {
    if latch.cancelled() {
        bail!("layer job cancelled");
    }
    super::source::guard(deadline)
}
#[cfg(unix)]
pub(super) async fn cancelled(latch: &SignalLatch) {
    loop {
        if latch.cancelled() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

#[cfg(test)]
mod output_tests {
    use super::*;
    struct Refusal(bool);
    impl std::io::Write for Refusal {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if self.0 {
                Err(std::io::ErrorKind::BrokenPipe.into())
            } else {
                Ok(bytes.len())
            }
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Err(std::io::ErrorKind::BrokenPipe.into())
        }
    }
    #[test]
    fn supplied_plain_output_has_one_newline_and_propagates_writer_refusal() {
        let cli = || Cli {
            command: Command::FormatBytes { bytes: 1024 },
        };
        let mut output = Vec::new();
        run_unix(cli(), &mut output).unwrap();
        assert_eq!(output, b"1.0 KiB\n");
        for reject_write in [true, false] {
            assert!(run_unix(cli(), &mut Refusal(reject_write)).is_err());
        }
    }
}
