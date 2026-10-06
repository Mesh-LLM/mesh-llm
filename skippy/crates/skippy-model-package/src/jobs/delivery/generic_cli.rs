//! Explicit native Jobs delivery: declarations, acknowledged submission, immutable collection.
use super::generic::{PreparedConversionDelivery, SubmittedConversionDelivery};
use super::{CpuJobPlan, HfJobsClient, ModelMount};
use crate::{
    jobs::TransportLimits,
    snapshot_promotion::{
        local_publisher::{SignalLatch, regular_input},
        regular_publication::{Publisher, Secret},
    },
};
use anyhow::{Result, bail};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    io::Write as _,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Parser)]
#[command(
    name = "model-package-generic-jobs",
    about = "Prepare, explicitly submit, or collect pinned native generic conversion Jobs; supplied image/tool declarations are not hosted qualification"
)]
struct Cli {
    #[command(subcommand)]
    operation: Operation,
    #[arg(long, global = true)]
    input: Option<PathBuf>,
    #[arg(long, global = true)]
    output_directory: Option<PathBuf>,
    #[arg(long, global = true)]
    credential_file: Option<PathBuf>,
    #[arg(long, global = true)]
    submitted_file: Option<PathBuf>,
    #[arg(long, global = true, default_value_t = 300)]
    timeout_seconds: u64,
    #[arg(long, global = true)]
    confirm_submission: bool,
}
#[derive(Clone, Copy, Subcommand)]
enum Operation {
    Prepare,
    Submit,
    Collect,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: u32,
    namespace: String,
    worker_input: Value,
    mounts: Vec<ModelMount>,
    cpu_plan: CpuJobPlan,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Acknowledgment {
    schema_version: u32,
    namespace: String,
    input_sha256: String,
    submitted: SubmittedConversionDelivery,
}
fn read(path: &Path, max: usize) -> Result<Vec<u8>> {
    let mut file = regular_input::open(path, max as u64, false)?;
    regular_input::read(&mut file, max)
}
fn prepare(input: &Input) -> Result<PreparedConversionDelivery> {
    if input.schema_version != 1
        || input.namespace.is_empty()
        || input.namespace.len() > 128
        || !input
            .namespace
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
    {
        bail!("generic Jobs facade schema/namespace refused");
    }
    PreparedConversionDelivery::prepare(
        &serde_json::to_vec(&input.worker_input)?,
        &input.mounts,
        &input.cpu_plan,
    )
}
fn hash(input: &Input) -> Result<String> {
    Ok(super::admission::digest(&serde_json::to_vec(input)?))
}
fn correlated(input: &Input, ack: &Acknowledgment) -> Result<()> {
    let expected = prepare(input)?;
    let mut declaration = serde_json::to_value(expected.declaration())?;
    declaration["submitted"] = json!(true);
    if ack.schema_version != 1
        || ack.namespace != input.namespace
        || ack.input_sha256 != hash(input)?
        || serde_json::to_value(&ack.submitted.native.declaration)? != declaration
        || ack.submitted.expected_status != expected.expected_status()
        || ack.submitted.native.job_id.is_empty()
        || ack.submitted.native.job_id.len() > 128
        || !ack
            .submitted
            .native
            .job_id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_-".contains(&b))
    {
        bail!("generic Jobs acknowledgment request/declaration/job correlation refused");
    }
    Ok(())
}
fn token(path: &Path) -> Result<String> {
    let mut file = regular_input::open(path, 8192, true)?;
    let bytes = regular_input::read(&mut file, 8192)?;
    let value =
        String::from_utf8(bytes).map_err(|_| anyhow::anyhow!("credential Unicode refused"))?;
    // Existing token policies reject empty/control/whitespace bytes. No trim or environment fallback.
    let _ = Secret::new(value.clone())?;
    Ok(value)
}
fn root(path: &Path) -> Result<PathBuf> {
    if !path.is_absolute() {
        bail!("generic Jobs facade output must be absolute");
    }
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("output parent"))?
        .canonicalize()?;
    let root = parent.join(
        path.file_name()
            .ok_or_else(|| anyhow::anyhow!("output leaf"))?,
    );
    match std::fs::symlink_metadata(&root) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        _ => bail!("generic Jobs facade output must be fresh"),
    }
    std::fs::create_dir(&root)?;
    Ok(root)
}
fn write(root: &Path, name: &str, value: &impl Serialize) -> Result<()> {
    let bytes = serde_json::to_vec_pretty(value)?;
    if bytes.len() > 16 * 1048576 {
        bail!("generic Jobs facade receipt byte bound");
    }
    let mut file = tempfile::NamedTempFile::new_in(root)?;
    file.write_all(&bytes)?;
    file.as_file().sync_all()?;
    file.persist_noclobber(root.join(name))?;
    Ok(())
}
async fn cancellation(latch: &SignalLatch) {
    while !latch.cancelled() {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
fn final_result(
    root: &Path,
    mut value: Value,
    decision: (bool, &str, &str),
    until: Instant,
    cancelled: impl Fn() -> bool,
) -> Result<bool> {
    let (candidate, success, refusal) = decision;
    let admitted = candidate && Instant::now() < until && !cancelled();
    value["status"] = json!(if admitted { success } else { refusal });
    value["operation_completed"] = json!(admitted);
    if !admitted {
        value["conversion_admitted"] = json!(false);
    }
    write(root, "result.json", &value)?;
    let complete = admitted && Instant::now() < until && !cancelled();
    if admitted && !complete {
        value["status"] = json!(refusal);
        value["operation_completed"] = json!(false);
        value["conversion_admitted"] = json!(false);
        value["terminal_refused"] = json!(true);
        let bytes = serde_json::to_vec_pretty(&value)?;
        let mut file = tempfile::NamedTempFile::new_in(root)?;
        file.write_all(&bytes)?;
        file.as_file().sync_all()?;
        if !std::fs::symlink_metadata(root.join("result.json"))?.is_file() {
            bail!("owned facade result changed before terminal downgrade");
        }
        file.persist(root.join("result.json"))?;
    }
    Ok(complete)
}
async fn submit(
    input: &Input,
    token: String,
    until: Instant,
    latch: &SignalLatch,
    root: &Path,
) -> Result<bool> {
    let prepared = prepare(input)?.with_publication_credential(token.clone())?;
    let client =
        HfJobsClient::new_admitted("https://huggingface.co", token, TransportLimits::default())?;
    write(
        root,
        "submission-attempt.json",
        &json!({"schema_version":1,"namespace":input.namespace,"input_sha256":hash(input)?,"submission_attempted":true,"remote_acceptance":"unconfirmed","conversion_admitted":false}),
    )?;
    let result = client
        .submit_conversion_until(&input.namespace, prepared, until, cancellation(latch))
        .await;
    match result {
        Ok(submitted) => {
            let ack = Acknowledgment {
                schema_version: 1,
                namespace: input.namespace.clone(),
                input_sha256: hash(input)?,
                submitted,
            };
            correlated(input, &ack)?;
            // Preserve an acknowledged job even if terminal local admission fails; never claim native success.
            write(root, "submitted.json", &ack)?;
            final_result(
                root,
                json!({"schema_version":1,"job_id":ack.submitted.native.job_id,"submission_acknowledged":true,"conversion_admitted":false,"remote_cancel_requested":false}),
                (true, "SUBMITTED", "FAILED"),
                until,
                || latch.cancelled(),
            )
        }
        Err(_) => {
            write(
                root,
                "result.json",
                &json!({"schema_version":1,"status":"FAILED","submission_attempted":true,"remote_acceptance":"unconfirmed","conversion_admitted":false,"error":"submission did not produce an admitted acknowledgment; inspect authorized Jobs status before any retry"}),
            )?;
            Ok(false)
        }
    }
}
async fn collect(
    input: &Input,
    ack: &Acknowledgment,
    token: String,
    until: Instant,
    latch: &SignalLatch,
    root: &Path,
) -> Result<bool> {
    correlated(input, ack)?;
    let client = HfJobsClient::new_admitted(
        "https://huggingface.co",
        token.clone(),
        TransportLimits::default(),
    )?;
    let publisher = Publisher::new(Secret::new(token)?)?;
    let result = client
        .collect_conversion_until(
            &input.namespace,
            &ack.submitted,
            &publisher,
            until,
            cancellation(latch),
        )
        .await;
    match result {
        Ok(mut observed) => {
            let candidate = observed.evidence.conversion_admitted;
            observed.evidence.conversion_admitted = false;
            write(
                root,
                "collected.json",
                &json!({"schema_version":1,"candidate_conversion_admitted":candidate,"final_admission":false,"observation":observed}),
            )?;
            final_result(
                root,
                json!({"schema_version":1,"input_sha256":hash(input)?,"job_id":ack.submitted.native.job_id,"conversion_admitted":candidate,"native_certified":false,"remote_cancel_requested":false}),
                (candidate, "CONVERSION_ADMITTED", "OBSERVATIONS_ONLY"),
                until,
                || latch.cancelled(),
            )
        }
        Err(_) => {
            write(
                root,
                "result.json",
                &json!({"schema_version":1,"status":"FAILED","input_sha256":hash(input)?,"job_id":ack.submitted.native.job_id,"conversion_admitted":false,"error":"monitor or immutable receipt correlation incomplete","remote_cancel_requested":false}),
            )?;
            Ok(false)
        }
    }
}
pub fn run() -> Result<bool> {
    run_args(std::env::args_os())
}
fn run_args(args: impl IntoIterator<Item = std::ffi::OsString>) -> Result<bool> {
    let cli = match Cli::try_parse_from(args) {
        Ok(c) => c,
        Err(e)
            if e.kind() == clap::error::ErrorKind::DisplayHelp
                || e.kind() == clap::error::ErrorKind::DisplayVersion =>
        {
            e.print()?;
            return Ok(true);
        }
        Err(_) => bail!("generic Jobs closed CLI refused"),
    };
    if !(5..=259200).contains(&cli.timeout_seconds) {
        bail!("generic Jobs local transport/monitor budget refused");
    }
    let until = Instant::now() + Duration::from_secs(cli.timeout_seconds);
    let input_path = cli
        .input
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("required generic Jobs input absent"))?;
    let output_path = cli
        .output_directory
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("required generic Jobs output absent"))?;
    let input: Input = serde_json::from_slice(&read(input_path, 1048576)?)
        .map_err(|_| anyhow::anyhow!("generic Jobs typed facade input refused"))?;
    let prepared = prepare(&input)?;
    match cli.operation {
        Operation::Prepare => {
            if cli.credential_file.is_some()
                || cli.submitted_file.is_some()
                || cli.confirm_submission
            {
                bail!("prepare cannot consume credentials or submission authority");
            }
            let root = root(output_path)?;
            write(&root, "declaration.json", prepared.declaration())?;
            write(
                &root,
                "result.json",
                &json!({"schema_version":1,"status":"PREPARED","input_sha256":hash(&input)?,"submitted":false,"conversion_admitted":false,"image_observed":false,"cost_observed":false}),
            )?;
            Ok(Instant::now() < until)
        }
        Operation::Submit | Operation::Collect => {
            if matches!(cli.operation, Operation::Submit)
                && (!cli.confirm_submission || cli.submitted_file.is_some())
            {
                bail!("submit requires explicit confirmation and no prior submission file");
            }
            if matches!(cli.operation, Operation::Collect) && cli.confirm_submission {
                bail!("collect cannot submit");
            }
            let ack = if matches!(cli.operation, Operation::Collect) {
                let path = cli
                    .submitted_file
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("collect requires submitted file"))?;
                let value: Acknowledgment = serde_json::from_slice(&read(path, 1048576)?)?;
                correlated(&input, &value)?;
                Some(value)
            } else {
                None
            };
            let token =
                token(cli.credential_file.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("explicit private credential file required")
                })?)?;
            let latch = SignalLatch::install()?;
            let root = root(output_path)?;
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            runtime.block_on(async {
                if let Some(ack) = ack {
                    collect(&input, &ack, token, until, &latch, &root).await
                } else {
                    submit(&input, token, until, &latch, &root).await
                }
            })
        }
    }
}
#[cfg(test)]
#[path = "generic_cli/tests.rs"]
mod tests;
