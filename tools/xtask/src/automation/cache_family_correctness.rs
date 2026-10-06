//! Actual cache correctness producer stage; serving/baseline producers remain separate.
#[path = "cache_family_correctness/admission.rs"]
mod admission;
#[path = "cache_family_correctness/catalog.rs"]
mod catalog;
#[path = "cache_family_correctness/report.rs"]
mod report;
#[cfg(test)]
#[path = "cache_family_correctness/tests.rs"]
mod tests;
use crate::{
    command::DynResult,
    process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness,
        Value as Argument,
    },
};
use catalog::{Input, Topology};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
const USAGE: &str = "cargo xtool automation cache-family-correctness [admit-worker] --input ABSOLUTE_JSON --output ABSOLUTE_FRESH_DIRECTORY_OR_WORKER_FILE";
fn paths(args: &[String]) -> DynResult<(PathBuf, PathBuf)> {
    if args.len() != 4 {
        return Err(USAGE.into());
    }
    let mut values = BTreeMap::new();
    for pair in args.as_chunks::<2>().0 {
        if !["--input", "--output"].contains(&pair[0].as_str())
            || values
                .insert(pair[0].as_str(), PathBuf::from(&pair[1]))
                .is_some()
        {
            return Err(USAGE.into());
        }
    }
    let input = values.remove("--input").ok_or(USAGE)?;
    let output = values.remove("--output").ok_or(USAGE)?;
    if !input.is_absolute() || !output.is_absolute() {
        return Err("cache stage paths must be absolute".into());
    }
    Ok((input, output))
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        println!("{USAGE}");
        return Ok(());
    }
    if let [verb, rest @ ..] = args
        && verb == "admit-worker"
    {
        let (input, output) = paths(rest)?;
        return admission::run(&input, &output);
    }
    let (path, output) = paths(args)?;
    let bytes = crate::automation::waiting_prefix::adaptive_identity::bounded(&path, 1024 * 1024)?;
    let input: Input = serde_json::from_slice(&bytes)?;
    input.validate()?;
    let parent = output.parent().ok_or("cache output parent")?;
    if !std::fs::symlink_metadata(parent)?.is_dir() {
        return Err("cache output parent must be regular".into());
    }
    std::fs::create_dir(&output)?; // Refuses existing files, directories and dangling links.
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let result = execute(&input, &output, &cancellation);
    let finish = interrupt.finish();
    let value = result?;
    finish?;
    println!("{}", serde_json::to_string(&value)?);
    if value["status"] != "completed" {
        return Err("cache correctness stage incomplete/failed; partial evidence retained".into());
    }
    Ok(())
}
fn publish(path: &Path, value: &Value) -> DynResult<()> {
    crate::automation::waiting_prefix::adaptive_identity::fresh(
        path,
        &serde_json::to_vec_pretty(value)?,
    )
}
fn complete(r: &process::ProcessReport) -> bool {
    r.success()
        && !r.cleanup.forced
        && r.failure.is_none()
        && r.cleanup.complete
        && r.cleanup.failure.is_none()
        && !r.cleanup.graceful_signal_failed
        && [&r.stdout, &r.stderr]
            .iter()
            .all(|s| s.line_capture_complete && !s.truncated && s.oversized_lines == 0)
}
struct Execution<'a> {
    deadline: Instant,
    cancellation: &'a Cancellation,
}
fn child(
    spec: &ProcessSpec,
    path: &Path,
    execution: &Execution<'_>,
    cap: u64,
) -> DynResult<process::ProcessReport> {
    let budget = execution
        .deadline
        .saturating_duration_since(Instant::now())
        .checked_sub(Duration::from_secs(3))
        .filter(|d| !d.is_zero())
        .ok_or("cache stage budget exhausted before child")?
        .min(Duration::from_secs(cap));
    Ok(process::supervise(
        spec,
        &Limits {
            execution: budget,
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 1024 * 1024,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        execution.cancellation,
        OutputFiles {
            stdout: Some(path.with_extension("stdout.log")),
            stderr: Some(path.with_extension("stderr.log")),
        },
    )?)
}
fn admit(
    input: &Input,
    directory: &Path,
    name: &str,
    execution: &Execution<'_>,
) -> DynResult<admission::Receipt> {
    let request = directory.join(format!("{name}-request.json"));
    let output = directory.join(format!("{name}-receipt.json"));
    let bytes = serde_json::to_vec(input)?;
    crate::automation::waiting_prefix::adaptive_identity::fresh(&request, &bytes)?;
    let args = vec![
        "automation".into(),
        "cache-family-correctness".into(),
        "admit-worker".into(),
        "--input".into(),
        request.clone().into_os_string(),
        "--output".into(),
        output.clone().into_os_string(),
    ];
    let process = child(
        &ProcessSpec {
            executable: std::env::current_exe()?,
            arguments: args.into_iter().map(Argument::Public).collect(),
            cwd: directory.into(),
            environment: BTreeMap::new(),
        },
        &directory.join(name),
        execution,
        input.cell_seconds,
    )?;
    publish(
        &directory.join(format!("{name}-process.json")),
        &json!({"outcome":format!("{:?}",process.outcome),"exit_code":process.status.and_then(|s|s.code()),"cleanup_complete":process.cleanup.complete,"forced":process.cleanup.forced,"failure_present":process.failure.is_some(),"cleanup_failure_present":process.cleanup.failure.is_some(),"graceful_signal_failed":process.cleanup.graceful_signal_failed,"stdout_suppressed_lines":process.stdout.suppressed_lines,"stderr_suppressed_lines":process.stderr.suppressed_lines,"stdout_bytes_seen":process.stdout.bytes_seen,"stderr_bytes_seen":process.stderr.bytes_seen}),
    )?;
    if !complete(&process) {
        return Err("cache identity worker did not complete cleanly".into());
    }
    let receipt: admission::Receipt = serde_json::from_slice(
        &crate::automation::waiting_prefix::adaptive_identity::bounded(&output, 1024 * 1024)?,
    )?;
    if receipt.request_sha256 != admission::hash(&bytes) {
        return Err("cache admission receipt request mismatch".into());
    }
    Ok(receipt)
}
fn arguments(
    receipt: &admission::Receipt,
    topology: Topology,
    path: &Path,
) -> DynResult<Vec<Argument>> {
    let input = &receipt.admitted;
    let (start, end, index) = topology.range(receipt.layers)?;
    let mut args = vec![
        "state-handoff".into(),
        "--model".into(),
        input.model.clone().into_os_string(),
        "--model-id".into(),
        input.model_id.clone().into(),
        "--stage-server-bin".into(),
        input.stage_server.clone().into_os_string(),
        "--layer-end".into(),
        receipt.layers.to_string().into(),
        "--ctx-size".into(),
        input.ctx_size.to_string().into(),
        format!("--n-gpu-layers={}", input.n_gpu_layers).into(),
        "--activation-width".into(),
        receipt.activation_width.to_string().into(),
        "--stage-load-mode".into(),
        "runtime-slice".into(),
        "--state-layer-start".into(),
        start.to_string().into(),
        "--state-layer-end".into(),
        end.to_string().into(),
        "--state-stage-index".into(),
        index.to_string().into(),
        "--state-payload-kind".into(),
        catalog::family(&input.case_key)?.1.into(),
        "--prefix-token-count".into(),
        input.prefix_tokens.to_string().into(),
        "--cache-hit-repeats".into(),
        input.cache_hit_repeats.to_string().into(),
        "--runtime-lane-count".into(),
        input.runtime_lane_count.to_string().into(),
        "--source-bind-addr".into(),
        format!("127.0.0.1:{}", input.source_port).into(),
        "--restore-bind-addr".into(),
        format!("127.0.0.1:{}", input.restore_port).into(),
        "--report-out".into(),
        path.to_path_buf().into_os_string(),
    ];

    if input.borrow_resident_hits {
        args.push("--borrow-resident-hits".into());
    }
    if input.cache_decoded_result_hits {
        args.push("--cache-decoded-result-hits".into());
    }
    let mut arguments: Vec<_> = args.into_iter().map(Argument::Public).collect();
    if let Some(prompt) = &input.prompt {
        arguments.push(Argument::Public("--prompt".into()));
        arguments.push(Argument::Secret(prompt.clone().into()));
    }
    Ok(arguments)
}
fn one(
    input: &Input,
    topology: Topology,
    directory: &Path,
    execution: &Execution<'_>,
) -> DynResult<Value> {
    let receipt = admit(input, directory, "before", execution)?;
    let output = directory.join("state-handoff.json");
    let arguments = arguments(&receipt, topology, &output)?;
    let mut environment: BTreeMap<_, _> = input
        .settings
        .iter()
        .map(|(k, v)| (k.clone().into(), Argument::Public(v.clone().into())))
        .collect();
    environment.insert(
        "LLAMA_STAGE_BUILD_DIR".into(),
        Argument::Public(receipt.admitted.native_build.clone().into_os_string()),
    );
    let started = Instant::now();
    let process = child(
        &ProcessSpec {
            executable: receipt.admitted.correctness.clone(),
            arguments,
            cwd: directory.into(),
            environment,
        },
        &directory.join("correctness"),
        execution,
        input.cell_seconds,
    )?;
    let observed = json!({"outcome":format!("{:?}",process.outcome),"exit_code":process.status.and_then(|s|s.code()),"cleanup_complete":process.cleanup.complete,"cleanup_forced":process.cleanup.forced,"failure_present":process.failure.is_some(),"cleanup_failure_present":process.cleanup.failure.is_some(),"graceful_signal_failed":process.cleanup.graceful_signal_failed,"stdout_suppressed_lines":process.stdout.suppressed_lines,"stderr_suppressed_lines":process.stderr.suppressed_lines,"stdout_bytes_seen":process.stdout.bytes_seen,"stderr_bytes_seen":process.stderr.bytes_seen,"runner_elapsed_ms":started.elapsed().as_secs_f64()*1000.0});
    publish(&directory.join("correctness-process.json"), &observed)?;
    if !complete(&process) {
        return Ok(json!({"status":"failed-process","lifecycle":observed}));
    }
    let value: Value = serde_json::from_slice(
        &crate::automation::waiting_prefix::adaptive_identity::bounded(&output, 1024 * 1024)?,
    )?;
    report::accept(&value, &receipt, topology)?;
    let after = admit(&receipt.admitted, directory, "after", execution)?;
    if serde_json::to_value(&after.admitted)? != serde_json::to_value(&receipt.admitted)?
        || after.layers != receipt.layers
        || after.activation_width != receipt.activation_width
    {
        return Err("cache artifact/config changed during trial".into());
    }
    Ok(
        json!({"status":"pass","lifecycle":observed,"skippy":value,"admitted":receipt.admitted,"identity_scope":"provided_byte_pins_not_source_build_or_loaded_runtime_attestation"}),
    )
}
fn execute(input: &Input, output: &Path, cancellation: &Cancellation) -> DynResult<Value> {
    let execution = Execution {
        deadline: Instant::now() + Duration::from_secs(input.execution_seconds),
        cancellation,
    };
    publish(&output.join("input.json"), &serde_json::to_value(input)?)?;
    let mut rows = Vec::new();
    let mut failed = false;
    for (index, topology) in input.topologies.iter().enumerate() {
        if cancellation.is_cancelled() || Instant::now() >= execution.deadline {
            failed = true;
            break;
        }
        let directory = output.join(format!("topology-{index:02}"));
        std::fs::create_dir(&directory)?;
        let result = one(input, *topology, &directory, &execution).unwrap_or_else(
            |_| json!({"status":"refused","reason":"identity_report_or_deadline_admission_failed"}),
        );
        failed |= result["status"] != "pass";
        rows.push(json!({"case_key":input.case_key,"family":catalog::family(&input.case_key)?.0,"payload":catalog::family(&input.case_key)?.1,"topology":topology,"evidence":result}));
        publish(
            &directory.join("trial.json"),
            rows.last().ok_or("trial row")?,
        )?;
    }
    let value = json!({"schema_version":1,"scope":"cache_correctness_producer_stage_not_complete_family_benchmark","status":if failed||cancellation.is_cancelled()||Instant::now()>=execution.deadline{"failed-or-incomplete"}else{"completed"},"planned_topologies":input.topologies.len(),"completed_topologies":rows.len(),"effective_settings":input.settings,"native_build":input.native_build,"rows":rows});
    publish(&output.join("cache-correctness-stage.json"), &value)?;
    Ok(value)
}
