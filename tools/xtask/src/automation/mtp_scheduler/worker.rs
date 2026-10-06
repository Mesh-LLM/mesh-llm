use super::Input;
use crate::{
    command::DynResult,
    process::{Cancellation, ProcessSpec, Value},
};
use std::{
    path::Path,
    time::{Duration, Instant},
};
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let [path, label, address] = args else {
        return Err("MTP worker requires input/old-or-new/address".into());
    };
    if !["old", "new"].contains(&label.as_str()) {
        return Err("invalid MTP arm".into());
    }
    let mut input: Input = serde_json::from_slice(&super::read(Path::new(path))?)?;
    super::profile(&input)?;
    // Worker accepts only the parent's private serialized input and fixed arm output leaves.
    let socket: std::net::SocketAddr = address.parse()?;
    if !socket.ip().is_loopback() || socket.port() == 0 {
        return Err("MTP worker requires local explicit port".into());
    }
    admit_existing(&mut input, Path::new(path), label, address)?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let deadline = Instant::now() + super::execution::arm_budget(&input)?;
    let mut publication = super::final_publication::ReceiptFile::new(
        &input.output_dir.join(format!("{label}-result.json")),
    )?;
    let (mut receipt, result) = measure(
        &input,
        label,
        socket,
        &interrupt.cancellation(),
        &mut publication,
    );
    publication.finish(&mut receipt, result, interrupt, deadline, &mut |_| Ok(()))
}
fn measure(
    input: &Input,
    label: &str,
    address: std::net::SocketAddr,
    cancellation: &Cancellation,
    publication: &mut super::final_publication::ReceiptFile,
) -> (serde_json::Value, DynResult<()>) {
    let mut result = serde_json::json!({"status":"starting","concurrency_sweep":[]});
    let measured = (|| -> DynResult<()> {
        publication.write(&result)?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(ready(
            address,
            Duration::from_secs(input.startup_seconds),
            cancellation,
        ))?;
        result["status"] = "measuring".into();
        for (index, &concurrency) in input.concurrency.iter().enumerate() {
            if cancellation.is_cancelled() {
                return Err("MTP measurement cancelled".into());
            }
            let requests = input.requests.max(concurrency);
            let spec = ProcessSpec {
                executable: input.client_bin.clone(),
                arguments: vec![
                    "--addr".into(),
                    address.to_string().into(),
                    "--requests".into(),
                    requests.to_string().into(),
                    "--concurrency".into(),
                    concurrency.to_string().into(),
                    "--activation-width".into(),
                    input.activation_width.to_string().into(),
                ]
                .into_iter()
                .map(Value::Public)
                .collect(),
                cwd: input.output_dir.clone(),
                environment: super::environment(input),
            };
            let raw = crate::process::supervise_raw(
                &spec,
                &super::limits(Duration::from_secs(input.client_seconds)),
                cancellation,
                crate::process::RawCaptureOptions {
                    stdout: std::num::NonZeroUsize::new(16 * 1024 * 1024),
                    stderr: std::num::NonZeroUsize::new(65536),
                },
            )?;
            crate::command::write_json_file(
                &input
                    .output_dir
                    .join(format!("{label}-client-{index}-{concurrency}-receipt.json")),
                &serde_json::json!({"process":format!("{:?}",raw.process)}),
            )?;
            let stdout = raw
                .stdout
                .as_ref()
                .ok_or("missing client stdout")?
                .as_bytes();
            let stderr = raw
                .stderr
                .as_ref()
                .ok_or("missing client stderr")?
                .as_bytes();
            if !raw.process.success()
                || stdout.len() as u64 != raw.process.stdout.bytes_seen
                || stderr.len() as u64 != raw.process.stderr.bytes_seen
                || !raw.process.stdout.line_capture_complete
                || !raw.process.stderr.line_capture_complete
            {
                return Err("MTP client failed or capture incomplete".into());
            }
            let report: serde_json::Value = serde_json::from_slice(stdout)?;
            let metrics = super::metrics::summarize(&report, requests, concurrency)?;
            result["concurrency_sweep"].as_array_mut().ok_or("sweep state")?.push(serde_json::json!({"concurrency":concurrency,"metrics":metrics,"per_request":report["per_request"]}));
            publication.write(&result)?;
        }
        if cancellation.is_cancelled() {
            return Err("MTP measurement cancelled at terminal boundary".into());
        }
        Ok(())
    })();
    (result, measured)
}

async fn cancelled(cancellation: &Cancellation) {
    while !cancellation.is_cancelled() {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}
async fn ready(
    address: std::net::SocketAddr,
    budget: Duration,
    cancellation: &Cancellation,
) -> DynResult<()> {
    use tokio::io::AsyncReadExt;
    let deadline = Instant::now() + budget;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err("MTP stage readiness deadline exceeded".into());
        }
        let probe = async {
            let mut stream = tokio::net::TcpStream::connect(address).await?;
            let mut magic = [0; 4];
            stream.read_exact(&mut magic).await?;
            Ok::<_, std::io::Error>(magic)
        };
        tokio::select! {
            ()=cancelled(cancellation)=>return Err("MTP readiness cancelled".into()),
            response=tokio::time::timeout(remaining.min(Duration::from_secs(1)),probe)=>{
                if let Ok(Ok(magic))=response{if magic==0x5352_4459_i32.to_le_bytes(){return Ok(());}return Err("MTP stage ready magic mismatch".into());}
            }
        }
        tokio::select! {()=cancelled(cancellation)=>return Err("MTP readiness cancelled".into()),()=tokio::time::sleep(remaining.min(Duration::from_millis(100)))=>{}}
    }
}
fn admit_existing(input: &mut Input, path: &Path, label: &str, address: &str) -> DynResult<()> {
    let root = input.output_dir.canonicalize()?;
    if !root.is_dir()
        || root != input.output_dir
        || super::regular(path, 16 * 1024 * 1024)? != root.join("input.json")
    {
        return Err("worker requires canonical owned serialized input root".into());
    }
    for file in [
        &mut input.old_bin,
        &mut input.new_bin,
        &mut input.client_bin,
    ] {
        if super::regular(file, 256 * 1024 * 1024)? != *file {
            return Err("worker binary must be canonical regular file".into());
        }
    }
    for directory in [
        &input.package,
        &input.native_build,
        &root.join("private-home"),
    ] {
        if !directory.is_dir() || directory.canonicalize()? != *directory {
            return Err("worker directory must be existing canonical directory".into());
        }
    }
    let stage: serde_json::Value =
        serde_json::from_slice(&super::read(&root.join(format!("{label}-stage.json")))?)?;
    if stage != super::config(input, address) {
        return Err("worker config differs from bounded input/address".into());
    }
    if std::fs::symlink_metadata(root.join(format!("{label}-result.json"))).is_ok() {
        return Err("worker refuses existing result output".into());
    }
    Ok(())
}
