//! Full pinned old/new comparison, retaining every cell and failure.
use super::{
    acceptance, aggregation, metrics_client, metrics_correlation, metrics_summary, native_identity,
    options, publish, report, rounds, server_cell, synthetic_prompts, telemetry, workload_plan,
};
use crate::{
    command::DynResult,
    process::{Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Binary {
    path: PathBuf,
    sha256: String,
    supplied_commit: String,
}
impl Binary {
    fn verify(&mut self) -> DynResult<()> {
        if !self.path.is_absolute()
            || self.supplied_commit.len() != 40
            || !self
                .supplied_commit
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("A/B binary requires an absolute path and full supplied commit".into());
        }
        self.path = self.path.canonicalize()?;
        let actual =
            crate::product::digest::file_sha256(&self.path).map_err(|error| error.error)?;
        if actual != self.sha256 {
            return Err("A/B binary differs from its admitted SHA-256".into());
        }
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema_version: u64,
    old: Binary,
    new: Binary,
    model_id: String,
    model_path: PathBuf,
    model_sha256: String,
    catalog: PathBuf,
    profile: String,
    contract: Option<PathBuf>,
    prompt_manifest: Option<PathBuf>,
    native_runtime_root: PathBuf,
    native_runtime_sha256: String,
    payload: String,
    n_gpu_layers: i32,
    request_timeout_secs: f64,
    startup_timeout_secs: u64,
    telemetry_timeout_secs: u64,
    cell_timeout_secs: u64,
    metrics_http: String,
    metrics_otlp_grpc: String,
    metrics_timeout_secs: u64,
}

struct Prepared {
    input: Input,
    plan: workload_plan::Plan,
    prompts: Vec<super::Prompt>,
    layer_end: u64,
}

fn prepare(mut input: Input) -> DynResult<Prepared> {
    if input.schema_version != 1
        || !input.model_path.is_absolute()
        || !input.native_runtime_root.is_absolute()
    {
        return Err("A/B input requires schema1 and absolute model/runtime paths".into());
    }
    input.old.verify()?;
    input.new.verify()?;
    input.model_path = input.model_path.canonicalize()?;
    input.native_runtime_root = input.native_runtime_root.canonicalize()?;
    if !input.native_runtime_root.is_dir() {
        return Err("A/B native-runtime root must be a directory".into());
    }
    native_identity::verify(&input.native_runtime_root, &input.native_runtime_sha256)?;
    let catalog = std::fs::read(&input.catalog)?;
    let contract = input.contract.as_ref().map(std::fs::read).transpose()?;
    let manifest = input
        .prompt_manifest
        .as_ref()
        .map(std::fs::read)
        .transpose()?;
    let plan = workload_plan::resolve(
        &catalog,
        &input.profile,
        &input.model_id,
        &input.model_sha256,
        contract.as_deref(),
        manifest.as_deref(),
    )?;
    let prompts = if let Some(bytes) = manifest {
        super::prompt_manifest(&bytes)?.prompts
    } else {
        synthetic_prompts::interleaved(
            plan.workload.families,
            plan.workload.requests_per_family,
            plan.workload.prefix_blocks,
        )?
    };
    crate::automation::replay_matrix::model_preflight::verify(
        &input.model_path,
        &input.model_sha256,
        plan.workload.ctx_size,
    )?;
    let dimensions =
        crate::automation::replay_matrix::model_preflight::dimensions::inspect(&input.model_path)?
            .ok_or("A/B runner requires full GGUF dimensions")?;
    let prepared = Prepared {
        input,
        plan,
        prompts,
        layer_end: dimensions.block_count,
    };
    let cells = rounds::schedule(prepared.plan.workload.rounds)?;
    let maximum = prepared
        .input
        .cell_timeout_secs
        .checked_add(200)
        .and_then(|seconds| seconds.checked_mul(cells.len() as u64))
        .ok_or("A/B deadline budget overflow")?;
    if maximum > 86400 {
        return Err("complete A/B comparison exceeds one-day process budget".into());
    }
    // Admit every shape before creating outputs or starting either binary.
    for (round, version) in cells {
        prepared.cell(round, version)?.validate()?;
    }
    Ok(prepared)
}

impl Prepared {
    fn collector(
        &self,
        round: u64,
        version: acceptance::Version,
    ) -> DynResult<metrics_client::Endpoint> {
        // Fresh entropy per prepared cell invocation; never share old/new run IDs.
        let version = match version {
            acceptance::Version::Old => "old",
            acceptance::Version::New => "new",
        };
        let mut random = [0_u8; 16];
        getrandom::fill(&mut random).map_err(|_| "collector run identity entropy unavailable")?;
        let nonce = hex::encode(random);
        Ok(metrics_client::Endpoint {
            http: self.input.metrics_http.clone(),
            otlp_grpc: self.input.metrics_otlp_grpc.clone(),
            run_id: format!("waiting-prefix-{round}-{version}-{nonce}"),
            timeout_secs: self.input.metrics_timeout_secs,
        })
    }
    fn binary(&self, version: acceptance::Version) -> &Binary {
        match version {
            acceptance::Version::Old => &self.input.old,
            acceptance::Version::New => &self.input.new,
        }
    }
    fn cell(&self, round: u64, version: acceptance::Version) -> DynResult<server_cell::Input> {
        let workload = &self.plan.workload;
        let binary = self.binary(version);
        Ok(serde_json::from_value(
            serde_json::json!({"schema_version":1,"binary":binary.path,
            "binary_sha256":binary.sha256,"native_runtime_root":self.input.native_runtime_root,
            "admission_concurrency":workload.admission_concurrency,"execution_timeout_secs":self.input.cell_timeout_secs,
            "stage":{"model_id":self.input.model_id,"model_path":self.input.model_path,
                "source_model_sha256":self.input.model_sha256,"layer_end":self.layer_end,"ctx_size":workload.ctx_size,
                "lane_count":workload.lanes,"n_gpu_layers":self.input.n_gpu_layers,"payload":self.input.payload,
                "cache_entries":workload.cache_entries},
            "worker":{"schema_version":1,"server_log":std::env::temp_dir().join("owned-pending.log"),
                "metrics":self.collector(round,version)?,"metrics_directory":std::env::temp_dir(),
                "cache_seed":self.plan.cache_seed,"startup_timeout_secs":self.input.startup_timeout_secs,
                "telemetry_timeout_secs":self.input.telemetry_timeout_secs,
                "phase":{"schema_version":1,"round":round,"version":version,"base_url":"http://127.0.0.1:1/v1",
                    "model":self.input.model_id,"output_tokens":workload.output_tokens,
                    "request_timeout_secs":self.input.request_timeout_secs,"stagger_ms":workload.stagger_ms,
                    "prompts":self.prompts}}}),
        )?)
    }
}

#[derive(Serialize)]
struct CellRecord {
    round: u64,
    version: acceptance::Version,
    directory: PathBuf,
    process_status: Option<i32>,
    process_clean: bool,
    evidence: Option<serde_json::Value>,
    evidence_bytes: u64,
    error: Option<String>,
}

#[derive(Serialize)]
struct Comparison<'a> {
    schema_version: u64,
    workload_plan: &'a workload_plan::Plan,
    old: &'a Binary,
    new: &'a Binary,
    provided_native_runtime_root: &'a Path,
    provided_native_runtime_sha256: &'a str,
    cells: Vec<CellRecord>,
    aggregate: Vec<acceptance::Aggregate>,
    collector_summary: Vec<metrics_summary::Summary>,
    acceptance: Option<acceptance::Acceptance>,
    error: Option<String>,
}

fn read_cell(path: &Path, budget: u64) -> DynResult<(serde_json::Value, u64)> {
    use std::io::Read;
    let limit = budget.min(64 * 1024 * 1024);
    let mut reader = std::fs::File::open(path)?;
    if !reader.metadata()?.is_file() {
        return Err("cell evidence must be a regular file".into());
    }
    let mut bytes = Vec::new();
    (&mut reader).take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err("cell evidence exceeds its individual or comparison byte budget".into());
    }
    Ok((serde_json::from_slice(&bytes)?, bytes.len() as u64))
}

fn owner_spec(input: &Path, directory: &Path) -> DynResult<ProcessSpec> {
    let mut environment = BTreeMap::new();
    for name in ["PATH", "SYSTEMROOT", "WINDIR"] {
        if let Some(value) = std::env::var_os(name) {
            environment.insert(name.into(), Value::Secret(value));
        }
    }
    Ok(ProcessSpec {
        executable: std::env::current_exe()?,
        arguments: vec![
            Value::Public("automation".into()),
            Value::Public("waiting-prefix".into()),
            Value::Public("server-cell".into()),
            Value::Public("--input".into()),
            Value::Public(input.as_os_str().into()),
            Value::Public("--output-directory".into()),
            Value::Public(directory.as_os_str().into()),
        ],
        cwd: directory
            .parent()
            .ok_or("missing comparison directory")?
            .into(),
        environment,
    })
}

fn execute(
    prepared: &Prepared,
    directory: &Path,
    round: u64,
    version: acceptance::Version,
    cancellation: &Cancellation,
    evidence_budget: u64,
) -> CellRecord {
    let mut record = CellRecord {
        round,
        version,
        directory: directory.to_path_buf(),
        process_status: None,
        process_clean: false,
        evidence: None,
        evidence_bytes: 0,
        error: None,
    };
    let result = (|| -> DynResult<()> {
        native_identity::verify(
            &prepared.input.native_runtime_root,
            &prepared.input.native_runtime_sha256,
        )?;
        let input = directory.with_extension("input.json");
        let cell = prepared.cell(round, version)?;
        let collector = serde_json::to_value(
            cell.worker
                .metrics
                .as_ref()
                .ok_or("missing admitted collector")?,
        )?;
        publish(&input, &serde_json::to_vec_pretty(&cell)?)?;
        let spec = owner_spec(&input, directory)?;
        let limits = Limits {
            execution: Duration::from_secs(prepared.input.cell_timeout_secs + 80),
            graceful_shutdown: Duration::from_secs(100),
            forced_shutdown: Duration::from_secs(20),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        };
        let report = crate::process::supervise(
            &spec,
            &limits,
            cancellation,
            OutputFiles {
                stdout: Some(directory.with_extension("owner.stdout.log")),
                stderr: Some(directory.with_extension("owner.stderr.log")),
            },
        )?;
        record.process_status = report.status.and_then(|status| status.code());
        record.process_clean = report.success();
        if directory.join("cell.json").exists() {
            let (evidence, bytes) = read_cell(&directory.join("cell.json"), evidence_budget)?;
            record.evidence = Some(evidence);
            record.evidence_bytes = bytes;
        }
        if !record.process_clean {
            return Err(format!("cell owner failed: {:?}", report.outcome).into());
        }
        let (lifecycle, _) = read_cell(&directory.join("lifecycle.json"), 1024 * 1024)?;
        if lifecycle["infrastructure_clean"] != true
            || lifecycle["worker_status"] != 0
            || !lifecycle["error"].is_null()
            || lifecycle["collector_run_id"] != collector["run_id"]
        {
            return Err("cell lifecycle did not prove clean measurement".into());
        }
        let evidence = record
            .evidence
            .as_ref()
            .ok_or("cell owner produced no measured evidence")?;
        if evidence["collector"]["endpoint"] != collector
            || evidence["collector"]["timings"].as_array().map(Vec::len)
                != usize::try_from(prepared.plan.requests_per_round).ok()
        {
            return Err(
                "cell collector identity or measured timing census differs from admission".into(),
            );
        }
        if evidence["round"] != round
            || evidence["version"] != serde_json::to_value(version)?
            || evidence["model"] != prepared.input.model_id
            || !evidence["error"].is_null()
        {
            return Err(
                "cell evidence identity or success differs from the scheduled workload".into(),
            );
        }
        Ok(())
    })();
    if let Err(error) = result {
        record.error = Some(error.to_string());
    }
    record
}

fn finish(comparison: &mut Comparison<'_>, prepared: &Prepared, directory: &Path) -> DynResult<()> {
    native_identity::verify(
        &prepared.input.native_runtime_root,
        &prepared.input.native_runtime_sha256,
    )?;
    let mut cells = Vec::<telemetry::Cell>::new();
    let mut old_timings = Vec::new();
    let mut new_timings = Vec::new();
    for record in &comparison.cells {
        if record.error.is_some() {
            return Err("A/B comparison contains failed cell ownership or measurement".into());
        }
        let raw = record
            .evidence
            .as_ref()
            .ok_or("missing A/B cell evidence")?;
        let cell: telemetry::Cell = serde_json::from_value(raw["summary"].clone())?;
        if cell.round != record.round || cell.version != record.version {
            return Err("measured summary identity differs from its scheduled cell".into());
        }
        let timings: Vec<metrics_correlation::Timing> =
            serde_json::from_value(raw["collector"]["timings"].clone())?;
        match record.version {
            acceptance::Version::Old => old_timings.push(timings),
            acceptance::Version::New => new_timings.push(timings),
        }
        cells.push(cell);
    }
    rounds::complete(
        &cells,
        prepared.plan.workload.rounds,
        prepared.plan.requests_per_round,
    )?;
    comparison.aggregate = aggregation::aggregate(aggregation::Input { cells })?;
    let contract = serde_json::from_value(prepared.plan.hardware_acceptance.clone())?;
    let acceptance = acceptance::evaluate(&comparison.aggregate, &contract)?;
    comparison.collector_summary = vec![
        metrics_summary::summarize(acceptance::Version::Old, &old_timings)?,
        metrics_summary::summarize(acceptance::Version::New, &new_timings)?,
    ];
    let mut markdown = report::render(&comparison.aggregate, &acceptance)?;
    markdown.push_str(&metrics_summary::render(&comparison.collector_summary)?);
    let passed = acceptance.passed;
    comparison.acceptance = Some(acceptance);
    publish(&directory.join("report.md"), markdown.as_bytes())?;
    if !passed {
        return Err("waiting-prefix comparison failed hardware acceptance".into());
    }
    Ok(())
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let opts = options(
        args,
        &["--input", "--output-directory"],
        &["--input", "--output-directory"],
    )?;
    let prepared = prepare(serde_json::from_slice(&std::fs::read(opts["--input"])?)?)?;
    let requested = std::path::absolute(opts["--output-directory"])?;
    let directory = requested
        .parent()
        .ok_or("comparison output requires a parent")?
        .canonicalize()?
        .join(
            requested
                .file_name()
                .ok_or("comparison output requires a fresh directory name")?,
        );
    if directory.starts_with(&prepared.input.native_runtime_root) {
        return Err("comparison output cannot be inside the pinned native artifact tree".into());
    }
    std::fs::create_dir(&directory)?;
    let directory = directory.canonicalize()?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let mut comparison = Comparison {
        schema_version: 1,
        workload_plan: &prepared.plan,
        old: &prepared.input.old,
        new: &prepared.input.new,
        provided_native_runtime_root: &prepared.input.native_runtime_root,
        provided_native_runtime_sha256: &prepared.input.native_runtime_sha256,
        cells: Vec::new(),
        aggregate: Vec::new(),
        collector_summary: Vec::new(),
        acceptance: None,
        error: None,
    };
    let mut retained_bytes = 0_u64;
    const COMPARISON_BYTES: u64 = 256 * 1024 * 1024;
    for (round, version) in rounds::schedule(prepared.plan.workload.rounds)? {
        if retained_bytes >= COMPARISON_BYTES {
            comparison.error = Some("A/B comparison retained-byte budget exhausted".into());
            break;
        }
        if cancellation.is_cancelled() {
            comparison.error = Some("A/B comparison interrupted before next launch".into());
            break;
        }
        let label = match version {
            acceptance::Version::Old => "old",
            acceptance::Version::New => "new",
        };
        let record = execute(
            &prepared,
            &directory.join(format!("round-{round}-{label}")),
            round,
            version,
            &cancellation,
            COMPARISON_BYTES - retained_bytes,
        );
        retained_bytes = retained_bytes
            .checked_add(record.evidence_bytes)
            .ok_or("comparison evidence byte count overflow")?;
        comparison.cells.push(record);
    }
    if comparison.error.is_none()
        && let Err(error) = finish(&mut comparison, &prepared, &directory)
    {
        comparison.error = Some(error.to_string());
    }
    if let Err(error) = interrupt.finish() {
        comparison.error = Some(match comparison.error.take() {
            Some(prior) => format!("{prior}; {error}"),
            None => error.to_string(),
        });
    }
    let mut bytes = serde_json::to_vec_pretty(&comparison)?;
    bytes.push(b'\n');
    publish(&directory.join("comparison.json"), &bytes)?;
    if comparison.error.is_some() {
        Err("A/B comparison failed; every attempted cell retained".into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
#[path = "round_runner_tests.rs"]
mod tests;
