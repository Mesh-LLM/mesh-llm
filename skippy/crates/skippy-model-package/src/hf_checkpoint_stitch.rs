//! Immutable local MTP checkpoint staging; conversion and publication are separate owners.
#[path = "hf_checkpoint_stitch/contract.rs"]
mod contract;
#[path = "hf_checkpoint_stitch/stitch.rs"]
mod stitch;
#[cfg(all(test, unix))]
#[path = "hf_checkpoint_stitch/tests.rs"]
mod tests;
use crate::{
    competitive_acquisition::{
        contract::{check, digest},
        publish,
    },
    snapshot_promotion::local_publisher::regular_input as read,
};
use anyhow::{Result, bail};
pub use contract::{Request, Source};
use serde_json::{Value, json};
use std::{
    path::Path,
    time::{Duration, Instant},
};
pub fn run(args: &[String]) -> Result<()> {
    if args.len() != 2 || args[0] != "--input" {
        bail!("usage: model-package-mtp-checkpoint --input ABS");
    }
    let mut fd = read::open(Path::new(&args[1]), 1024 * 1024, false)?;
    let bytes = read::read(&mut fd, 1024 * 1024)?;
    let input: Request =
        serde_json::from_value(crate::competitive_acquisition::json::unique(&bytes)?)?;
    input.validate()?;
    #[cfg(unix)]
    let signal = crate::snapshot_promotion::local_publisher::SignalLatch::install()?;
    let deadline = Instant::now()
        .checked_add(Duration::from_secs(input.timeout_seconds))
        .ok_or_else(|| anyhow::anyhow!("deadline overflow"))?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let mut receipt = json!({"schema_version":1,"status":"FAILED","error":null,"request_sha256":digest(&serde_json::to_vec(&input)?),"request_transport_sha256":digest(&bytes),"output_owned":false,"checkpoint_source":null,"tokenizer_source":null,"checkpoint_directory":null,"checkpoint_files":[],"tokenizer_profile":null,"lineage":null,"loaded_model_qualification":false});
    let result = runtime.block_on(async {
        let work = stitch::execute(&input, deadline, &mut receipt, None);
        let stop = crate::competitive_acquisition::cancelled();
        let timer = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline));
        futures::pin_mut!(work, stop, timer);
        match futures::future::select(work, futures::future::select(stop, timer)).await {
            futures::future::Either::Left((r, remaining)) => {
                use futures::FutureExt as _;
                r?;
                if remaining.now_or_never().is_some() {
                    bail!("terminal deadline/cancellation");
                }
                Ok(())
            }
            futures::future::Either::Right(_) => {
                Err(anyhow::anyhow!("checkpoint deadline/cancellation"))
            }
        }
    });
    let result = result.and_then(|()| {
        if read::read(&mut fd, 1024 * 1024)? != bytes {
            bail!("request custody changed");
        }
        #[cfg(unix)]
        if signal.cancelled() {
            bail!("terminal cancellation refused");
        }
        check(deadline)
    });
    receipt["error"] = result
        .as_ref()
        .err()
        .map_or(Value::Null, |e| json!(e.to_string()));
    receipt["status"] = json!(if result.is_ok() {
        "STAGED_NOT_CONVERTED"
    } else {
        "FAILED"
    });
    if receipt["output_owned"] == true {
        publish(
            &input.output_directory.join("checkpoint-stitch.json"),
            &serde_json::to_vec_pretty(&receipt)?,
        )?;
    }
    result
}
