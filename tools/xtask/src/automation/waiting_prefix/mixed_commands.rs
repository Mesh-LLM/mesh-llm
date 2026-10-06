//! Native mixed workload planning and explicitly declared-phase reports.
//! Runtime qualification belongs to the future retained parent bridge, not a supplied log file.
use super::{
    adaptive_identity as identity,
    mixed_counters::Projection,
    mixed_summary, mixed_worker,
    mixed_workload::{self, Manifest, PromptRecord, Role, Shape},
    options,
};
use crate::command::DynResult;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::path::Path;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Plan {
    schema_version: u64,
    arm: identity::Input,
    shape: Shape,
    split: bool,
    manifest: Option<Manifest>,
    request_timeout_secs: f64,
    timeout_secs: u64,
    readiness_timeout_secs: u64,
    suppressed_token_ids: Vec<u32>,
}
fn input<T: serde::de::DeserializeOwned>(path: &str) -> DynResult<T> {
    Ok(serde_json::from_slice(&identity::bounded(
        Path::new(path),
        16 * 1024 * 1024,
    )?)?)
}
fn output_args(args: &[String]) -> DynResult<std::collections::BTreeMap<&str, &str>> {
    options(args, &["--input", "--output"], &["--input", "--output"])
}
pub(super) fn plan(args: &[String]) -> DynResult<()> {
    let opts = output_args(args)?;
    let input: Plan = input(opts["--input"])?;
    if input.schema_version != 1 {
        return Err("mixed plan schema invalid".into());
    }
    input.arm.validate()?;
    input.shape.validate()?;
    if let Some(m) = &input.manifest {
        m.validate(&input.shape)?;
    }
    let config_dir = Path::new(opts["--output"])
        .parent()
        .ok_or("mixed config parent absent")?;
    let count = if input.split { 2 } else { 1 };
    let mut configs = Vec::new();
    let mut arguments = Vec::new();
    for index in 0..count {
        configs.push(mixed_workload::config(
            &input.arm,
            &input.shape,
            index,
            input.split,
        )?);
        arguments.push(
            mixed_workload::arguments(
                &input.arm,
                &input.shape,
                &config_dir.join(format!("stage-{index}.json")),
                index,
                input.split,
            )?
            .into_iter()
            .map(|s| {
                s.into_string()
                    .map_err(|_| "mixed argument path must be UTF-8")
            })
            .collect::<Result<Vec<_>, _>>()?,
        );
    }
    let mut workers = Vec::new();
    for round in 1..=input.shape.rounds {
        let mut worker = mixed_worker::Input {
            schema_version: 1,
            round,
            version: input.arm.version,
            base_url: format!("http://127.0.0.1:{}/v1", input.arm.openai_port),
            model: input.arm.model_id.clone(),
            request_timeout_secs: input.request_timeout_secs,
            timeout_secs: input.timeout_secs,
            readiness_timeout_secs: input.readiness_timeout_secs,
            warmup: PromptRecord {
                family: "synthetic-warmup".into(),
                prompt: mixed_workload::stable_prompt(16, -1, Role::Prefill)?,
                provenance: Map::new(),
            },
            requests: mixed_workload::requests(&input.shape, round, input.manifest.as_ref())?,
            suppressed_token_ids: input.suppressed_token_ids.clone(),
            manifest_metadata: input
                .manifest
                .as_ref()
                .map_or_else(Map::new, |m| m.metadata.clone()),
            provenance: json!({"declared_arm":input.arm,"shape":input.shape,"split":input.split})
                .as_object()
                .unwrap()
                .clone(),
            workload_sha256: String::new(),
        };
        worker.workload_sha256 = worker.workload_sha()?;
        worker.validate()?;
        workers.push(worker);
    }
    identity::fresh(
        Path::new(opts["--output"]),
        &serde_json::to_vec_pretty(
            &json!({"schema_version":1,"identity_status":"caller-declared-unverified","configs":configs,"arguments":arguments,"workers":workers,"native_profile":"standalone-static-skippy-server"}),
        )?,
    )
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CallerCell {
    worker: Value,
    warmup_lines: Vec<String>,
    measured_lines: Vec<String>,
    capture_complete: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Report {
    schema_version: u64,
    rounds: u64,
    n_batch: u32,
    prefills: usize,
    cells: Vec<CallerCell>,
}
pub(super) fn report(args: &[String]) -> DynResult<()> {
    let opts = options(
        args,
        &["--input", "--output", "--report"],
        &["--input", "--output"],
    )?;
    let input: Report = input(opts["--input"])?;
    if input.schema_version != 1
        || !(1..=128).contains(&input.rounds)
        || input.cells.len() != input.rounds as usize * 2
        || input.n_batch == 0
        || input.prefills == 0
    {
        return Err("mixed report declared roster invalid".into());
    }
    let mut cells = Vec::new();
    for cell in &input.cells {
        let mut projection = Projection::default();
        for line in &cell.warmup_lines {
            projection.observe(line.as_bytes());
        }
        let boundary = projection.warmup_boundary()?;
        for line in &cell.measured_lines {
            projection.observe(line.as_bytes());
        }
        let count = cell.worker["requests"]
            .as_array()
            .ok_or("mixed report worker roster absent")?
            .len();
        let mut counters = projection.measured(
            &boundary,
            count,
            cell.worker["input_sha256"]
                .as_str()
                .ok_or("worker identity absent")?,
            cell.capture_complete,
        )?;
        counters.phase_provenance = "caller-declared-separated-lines-not-owner-runtime-proof";
        let mut value = cell.worker.clone();
        let mut summary = mixed_summary::requests(&value)?;
        mixed_summary::counters(
            &mut summary,
            &value,
            &counters,
            input.n_batch,
            input.prefills,
        )?;
        value["summary"] = summary;
        cells.push(value);
    }
    let mut comparison = mixed_summary::compare(&cells, input.rounds)?;
    comparison["qualification_scope"] =
        json!("caller-declared-phase-report-no-owned-runtime-admission");
    comparison["cells"] = json!(cells);
    identity::fresh(
        Path::new(opts["--output"]),
        &serde_json::to_vec_pretty(&comparison)?,
    )?;
    if let Some(path) = opts.get("--report") {
        identity::fresh(
            Path::new(path),
            mixed_summary::render(&comparison).as_bytes(),
        )?;
    }
    Ok(())
}
