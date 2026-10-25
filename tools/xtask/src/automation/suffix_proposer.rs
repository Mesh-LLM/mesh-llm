#[path = "suffix_proposer/contract.rs"]
mod contract;
#[path = "suffix_proposer/evidence.rs"]
mod evidence;
#[path = "suffix_proposer/execution.rs"]
mod execution;
#[path = "suffix_proposer/sample.rs"]
mod sample;
#[path = "suffix_proposer/summary.rs"]
mod summary;
use crate::{automation::command_interrupt::Interrupt, command::DynResult};
use serde_json::json;
use std::{
    path::Path,
    time::{Duration, Instant},
};
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        println!(
            "automation suffix-proposer --input ABS_JSON --output-directory FRESH_DIRECTORY; already-serving endpoints, no runtime custody or loaded-byte attestation"
        );
        return Ok(());
    }
    let [a, path, b, output] = args else {
        return Err("suffix-proposer closed flags".into());
    };
    if a != "--input" || b != "--output-directory" {
        return Err("suffix-proposer closed flags".into());
    }
    let bytes = evidence::read(Path::new(path), 1048576)?;
    let input: contract::Input = serde_json::from_slice(&bytes)?;
    input.validate()?;
    let workloads = contract::workloads(&input)?;
    let before = evidence::digest(&bytes);
    let workload_sha = evidence::digest(&serde_json::to_vec(&workloads)?);
    let requested = std::path::absolute(output)?;
    std::fs::create_dir_all(requested.parent().ok_or("output parent")?)?;
    std::fs::create_dir(&requested)?;
    let root = requested.canonicalize()?;
    let interrupt = Interrupt::install()?;
    let c = interrupt.cancellation();
    let deadline = Instant::now() + Duration::from_millis(input.execution_timeout_ms);
    let manifest = json!({"schema_version":1,"input":input,"request_sha256":before,"workloads_sha256":workload_sha,"workloads":workloads.iter().map(|w|&w.name).collect::<Vec<_>>(),"identity_scope":"external endpoints; model listing and operator declarations are not loaded-byte/binary/split attestation","shuffle_algorithm":"seeded splitmix64 Fisher-Yates; not Python RNG byte equivalence","host":{"os":std::env::consts::OS,"arch":std::env::consts::ARCH}});
    evidence::publish(&root.join("manifest.json"), &manifest)?;
    let mut report = json!({"schema_version":1,"status":"FAILED","request_sha256":before,"samples":[],"completed_warmups":0,"identity_before":null,"identity_after":null,"source_unchanged":false,"summary":null,"error":null});
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(execution::execute(
        &input,
        &workloads,
        &root,
        deadline,
        &c,
        &mut report,
    ));
    let unchanged = evidence::read(Path::new(path), 1048576)
        .is_ok_and(|v| evidence::digest(&v) == before)
        && contract::workloads(&input).is_ok_and(|v| {
            evidence::digest(&serde_json::to_vec(&v).unwrap_or_default()) == workload_sha
        });
    report["source_unchanged"] = json!(unchanged);
    let signal = interrupt.finish();
    if let Err(error) = result {
        report["error"] = json!(error.to_string());
    } else if let Err(error) = signal {
        report["error"] = json!(error.to_string());
    } else if !unchanged || c.is_cancelled() || Instant::now() >= deadline {
        report["error"] = json!("suffix source changed or cancelled/deadline");
    } else {
        report["status"] = json!("PASS");
    }
    evidence::publish(&root.join("report.json"), &report)?;
    if report["status"] == "PASS" {
        Ok(())
    } else {
        Err("suffix failed; partial artifacts retained".into())
    }
}
#[cfg(test)]
#[path = "suffix_proposer/tests.rs"]
mod tests;
