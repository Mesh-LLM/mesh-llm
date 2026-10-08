//! Supplied existing-repo Unix publisher helper. Local admission is not remote publication.
mod contract;
mod output;
pub(crate) mod regular_input;
#[cfg(unix)]
mod regular_receipt;
use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use contract::{Input, Receipt};
use sha2::{Digest, Sha256};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};
#[derive(Parser)]
#[command(
    name = "model-package-publish",
    about = "Admit or publish pinned local model artifacts; publication requires explicit credentials and external authorization"
)]
struct Cli {
    #[command(subcommand)]
    operation: Operation,
    #[arg(long, global = true, help = "Required pinned input document")]
    input: Option<PathBuf>,
    #[arg(long, global = true, help = "Required fresh output directory")]
    output_directory: Option<PathBuf>,
}
#[derive(Clone, Copy, Subcommand)]
enum Operation {
    Admit,
    Publish,
    #[cfg(unix)]
    PublishRegularReceipt,
}
pub fn run(help: &mut dyn std::io::Write) -> Result<bool> {
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(error)
            if error.kind() == clap::error::ErrorKind::DisplayHelp
                || error.kind() == clap::error::ErrorKind::DisplayVersion =>
        {
            write!(help, "{error}")?;
            help.flush()?;
            return Ok(true);
        }
        Err(_) => bail!("publisher arguments refused"),
    };
    execute(cli)
        .map_err(|_| anyhow::anyhow!("publisher local admission or receipt publication refused"))
}
fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn execute(cli: Cli) -> Result<bool> {
    let input_path = cli
        .input
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("publisher input absent"))?;
    let output_directory = cli
        .output_directory
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("publisher output directory absent"))?;
    #[cfg(unix)]
    if matches!(cli.operation, Operation::PublishRegularReceipt) {
        return regular_receipt::execute(input_path, output_directory);
    }
    let mut input_file = regular_input::open(input_path, 512 * 1024, false)?;
    let bytes = regular_input::read(&mut input_file, 512 * 1024)?;
    let raw_hash = digest(&bytes);
    let input: Input = serde_json::from_slice(&bytes)?;
    validate(&input, cli.operation)?;
    let hash = digest(&serde_json::to_vec(&input)?);
    let until = Instant::now()
        .checked_add(Duration::from_millis(input.execution_timeout_ms))
        .ok_or_else(|| anyhow::anyhow!("publisher deadline overflow"))?;
    let output = output::Output::fresh(output_directory)?;
    let admitted = artifacts(&input, until);
    let mut finalization = Finalization {
        output: &output,
        hash: &hash,
        input_file: &mut input_file,
        raw_hash: &raw_hash,
        until,
    };
    let mut success = false;
    if let Ok(mut artifacts) = admitted
        && super::model_publication::Publisher::admit_until(&mut artifacts, until).is_ok()
    {
        match cli.operation {
            Operation::Admit => success = Instant::now() < until,
            #[cfg(unix)]
            Operation::PublishRegularReceipt => bail!("regular receipt dispatch refused"),
            Operation::Publish => {
                if let Ok(token) = credential(&input) {
                    return publish(artifacts, token, &mut finalization);
                }
            }
        }
    }
    finalization.finish(cli.operation, success, None, || false)
}
struct Finalization<'a> {
    output: &'a output::Output,
    hash: &'a str,
    input_file: &'a mut std::fs::File,
    raw_hash: &'a str,
    until: Instant,
}
impl Finalization<'_> {
    fn finish(
        &mut self,
        operation: Operation,
        mut success: bool,
        mut publication: Option<super::model_publication::Receipt>,
        cancelled: impl FnOnce() -> bool,
    ) -> Result<bool> {
        let custody = regular_input::read(self.input_file, 512 * 1024)
            .is_ok_and(|bytes| digest(&bytes) == self.raw_hash);
        success &= custody && !cancelled() && Instant::now() < self.until;
        if !success && let Some(observed) = publication.as_mut() {
            observed.completed = false;
            observed.error = Some("helper terminal admission failed".into());
        }
        self.output.write(
            "publication.json",
            &Receipt {
                schema_version: 1,
                request_sha256: self.hash,
                status: if success {
                    match operation {
                        Operation::Admit => "ADMITTED",
                        Operation::Publish => "PUBLISHED",
                        #[cfg(unix)]
                        Operation::PublishRegularReceipt => "PUBLISHED_REGULAR_RECEIPT",
                    }
                } else {
                    "FAILED"
                },
                publication: publication.as_ref(),
                source_custody_verified: custody,
                error: if success {
                    None
                } else {
                    Some("publisher_operation_incomplete")
                },
            },
            true,
        )?;
        Ok(success)
    }
}

fn validate(input: &Input, operation: Operation) -> Result<()> {
    if input.schema_version != 1
        || !(1..=86_400_000).contains(&input.execution_timeout_ms)
        || input.shards.is_empty()
        || input.shards.len() > 128
        || input.sidecars.len() > 32
    {
        bail!("publisher input schema or bounds refused");
    }
    match operation {
        Operation::Admit if input.credential_file.is_some() => {
            bail!("admit must not acquire credentials")
        }
        Operation::Publish if input.credential_file.is_none() => {
            bail!("publish private credential file required")
        }
        _ => Ok(()),
    }
}
fn artifacts(input: &Input, until: Instant) -> Result<super::model_publication::Input> {
    let mut shards = Vec::new();
    for artifact in &input.shards {
        if artifact.byte_size == 0 || artifact.byte_size > 1024u64.pow(4) {
            bail!("shard size refused");
        }
        let mut file = regular_input::open(&artifact.path, artifact.byte_size, false)?;
        regular_input::hash(&mut file, &artifact.sha256, artifact.byte_size, until)?;
        shards.push(super::model_publication::Shard {
            path_in_repo: artifact.path_in_repo.clone(),
            object: super::lfs_transfer::Object {
                file,
                oid: artifact.sha256.clone(),
                size: artifact.byte_size,
            },
        });
    }
    let mut sidecars = Vec::new();
    let mut total = 0u64;
    for artifact in &input.sidecars {
        total = total
            .checked_add(artifact.byte_size)
            .ok_or_else(|| anyhow::anyhow!("sidecar size overflow"))?;
        if artifact.byte_size > 1024 * 1024 || total > 8 * 1024 * 1024 {
            bail!("sidecar size refused");
        }
        let mut file = regular_input::open(&artifact.path, artifact.byte_size, false)?;
        regular_input::hash(&mut file, &artifact.sha256, artifact.byte_size, until)?;
        sidecars.push(super::regular_publication::LocalFile {
            path_in_repo: artifact.path_in_repo.clone(),
            file,
            identity: super::policy::ArtifactIdentity {
                sha256: artifact.sha256.clone(),
                byte_size: artifact.byte_size,
            },
        });
    }
    Ok(super::model_publication::Input {
        repo: input.repo.clone(),
        parent_commit: input.parent_commit.clone(),
        shards,
        sidecars,
    })
}
fn credential(input: &Input) -> Result<String> {
    let path = input
        .credential_file
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("credential file absent"))?;
    let mut file = regular_input::open(path, 8192, true)?;
    let bytes = regular_input::read(&mut file, 8192)?;
    let mut token = String::from_utf8(bytes)?;
    if token.ends_with('\n') {
        token.pop();
    }
    if token.is_empty()
        || token
            .bytes()
            .any(|b| b.is_ascii_control() || b.is_ascii_whitespace())
    {
        bail!("credential contents refused");
    }
    Ok(token)
}
fn publish(
    input: super::model_publication::Input,
    token: String,
    finalization: &mut Finalization<'_>,
) -> Result<bool> {
    let _ = skippy_model_hf::configure_hf_tls_provider();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        #[cfg(unix)]
        {
            use tokio::signal::unix::{SignalKind, signal};
            let latch = match SignalLatch::install() {
                Ok(latch) => latch,
                Err(_) => return finalization.finish(Operation::Publish, false, None, || true),
            };
            let signals = signal(SignalKind::terminate()).and_then(|terminate| {
                signal(SignalKind::interrupt()).map(|interrupt| (terminate, interrupt))
            });
            let (mut terminate, mut interrupt) = match signals {
                Ok(signals) => signals,
                Err(_) => return finalization.finish(Operation::Publish, false, None, || true),
            };
            let publisher = match super::model_publication::Publisher::new(token) {
                Ok(publisher) => publisher,
                Err(_) => return finalization.finish(Operation::Publish, false, None, || false),
            };
            let cancellation = async {
                tokio::select! {_=terminate.recv()=>(),_=interrupt.recv()=>()}
            };
            let mut observer = |observed: &super::model_publication::Receipt| {
                finalization
                    .output
                    .write(
                        "progress.json",
                        &Receipt {
                            schema_version: 1,
                            request_sha256: finalization.hash,
                            status: "IN_PROGRESS",
                            publication: Some(observed),
                            source_custody_verified: false,
                            error: None,
                        },
                        false,
                    )
                    .map_err(|_| anyhow::anyhow!("publisher progress persistence failed"))
            };
            let receipt = publisher
                .publish_observed_until(input, finalization.until, cancellation, &mut observer)
                .await;
            let success = receipt.completed;
            finalization.finish(Operation::Publish, success, Some(receipt), || {
                latch.cancelled()
            })
        }
        #[cfg(not(unix))]
        {
            let _ = (input, token);
            finalization.finish(Operation::Publish, false, None, || true)
        }
    })
}

// The callback observes OS delivery even while the Tokio driver cannot poll.
#[cfg(unix)]
pub(crate) struct SignalLatch {
    flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
    registrations: Vec<signal_hook_registry::SigId>,
}
#[cfg(unix)]
impl SignalLatch {
    pub(crate) fn install() -> Result<Self> {
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        let mut latch = Self {
            flag: Arc::new(AtomicBool::new(false)),
            registrations: Vec::new(),
        };
        for signal in [libc::SIGTERM, libc::SIGINT] {
            let flag = latch.flag.clone();
            // SAFETY: the signal callback performs only a lock-free atomic store;
            // allocation and registration occur outside signal context.
            let registration = unsafe {
                signal_hook_registry::register(signal, move || flag.store(true, Ordering::SeqCst))
            }?;
            latch.registrations.push(registration);
        }
        Ok(latch)
    }
    pub(crate) fn cancelled(&self) -> bool {
        self.flag.load(std::sync::atomic::Ordering::SeqCst)
    }
}
#[cfg(unix)]
impl Drop for SignalLatch {
    fn drop(&mut self) {
        // Remove only owned callbacks. The registry/Tokio process handlers remain installed.
        for registration in self.registrations.drain(..) {
            signal_hook_registry::unregister(registration);
        }
    }
}

// Serializes tests that raise process-directed signals with tests whose run path
// listens for SIGTERM/SIGINT, so one test cannot cancel another.
#[cfg(all(test, unix))]
pub(crate) static PROCESS_SIGNAL_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(all(test, unix))]
#[path = "local_publisher/tests.rs"]
mod tests;
