//! Own one exact competitive cell. Matrix selection and reporting are separate phases.
use crate::{
    command::DynResult,
    process::{
        self, Value as Argument,
        retained::{Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState},
    },
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    config: PathBuf,
    config_sha256: String,
    cell: Value,
    model: super::competitive_launch::Artifact,
    backend: super::competitive_launch::Backend,
    manifest: Option<PathBuf>,
    benchy: Option<super::competitive_launch::Artifact>,
    output: PathBuf,
    timeout_seconds: u64,
    request_timeout_seconds: u64,
}
pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    use crate::repository::{check_args::Grammar, check_report::CheckReport};
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix competitive-run-cell --input PATH",
        values: &["--input"],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positionals").emit();
    }
    let started = Instant::now();
    let input: Input = serde_json::from_slice(&super::competitive_cell::read(
        Path::new(parsed.last("--input").ok_or("missing input")?),
        8 * 1024 * 1024,
    )?)?;
    execute(input, started)
}
fn execute(input: Input, started: Instant) -> DynResult<()> {
    let document = admit(&input)?;
    let deadline = started + Duration::from_secs(input.timeout_seconds);
    // Refuse prior evidence. Resume/quarantine must be an explicit matrix-owner operation.
    std::fs::create_dir(&input.output)?;
    let reservation = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    let port = reservation.local_addr()?.port();
    let mut prepared = super::competitive_launch::prepare(
        &document,
        &input.cell,
        &input.backend,
        &input.model,
        port,
        &input.output,
    )?;
    if matches!(input.cell["arm"].as_str(), Some("mesh" | "mesh-adaptive")) {
        let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
        let admission = crate::automation::skippy_cli_admission::prepare(
            &mut prepared.spec,
            &input.backend.executable.sha256,
            crate::automation::skippy_cli_admission::Role::Public,
            deadline,
            &interrupt.cancellation(),
            &input.output.join("cli-admission.json"),
        );
        let restored = interrupt.finish();
        admission?;
        restored?;
        prepared.command = std::iter::once(prepared.spec.executable.to_string_lossy().into_owned())
            .chain(prepared.spec.arguments.iter().map(|value| match value {
                Argument::Public(value) => value.to_string_lossy().into_owned(),
                Argument::Secret(_) => "<redacted>".into(),
            }))
            .collect();
    }
    version(&input, deadline)?;
    super::competitive_launch::file(&input.backend.executable)?;
    super::competitive_launch::file(&input.model)?;
    recheck_trees(&input.backend)?;
    let execution = deadline
        .saturating_duration_since(Instant::now())
        .checked_sub(Duration::from_secs(6))
        .ok_or("cell deadline exhausted before launch")?;
    if execution < Duration::from_secs(1) {
        return Err("insufficient cell execution budget".into());
    }
    let (server, worker) = launches(&input, &document, prepared, port, execution)?;
    let worker_output = input.output.join("worker");
    let mut owner = Owner {
        server: Some(server),
        worker: Some(worker),
        stop: false,
        policy: ExpectedExit::new(&[0], execution)?,
    };
    drop(reservation);
    let report = crate::automation::retained_session::run(&mut owner, &limits(execution))?;
    super::pass_lifecycle::retain(&report, &input.output)?;
    let receipt: super::pass_lifecycle::Receipt =
        serde_json::from_slice(&std::fs::read(input.output.join("lifecycle.json"))?)?;
    if !report.recovery_success()
        || !receipt.infrastructure_clean
        || receipt.worker_status != Some(0)
    {
        let members = report
            .members
            .iter()
            .map(|member| {
                format!(
                    "{}:{}:{:?}:{}",
                    String::from_utf8_lossy(member.member.name()),
                    member.disposition.label(),
                    member
                        .process
                        .status
                        .as_ref()
                        .and_then(std::process::ExitStatus::code),
                    member.process.cleanup.complete
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        return Err(format!("competitive cell failed; partial evidence retained without completion marker; outcome={:?}; members={members}", report.outcome).into());
    }
    if Instant::now() >= deadline {
        return Err("competitive deadline expired during cleanup; completion withheld".into());
    }
    complete(&input, &worker_output)
}
fn admit(input: &Input) -> DynResult<Value> {
    if !(10..=86400).contains(&input.timeout_seconds)
        || input.request_timeout_seconds == 0
        || input.request_timeout_seconds >= input.timeout_seconds
        || !input.output.is_absolute()
    {
        return Err(
            "competitive cell requires absolute output and bounded deadlines with cleanup reserve"
                .into(),
        );
    }

    let bytes = super::competitive_cell::read(&input.config, 8 * 1024 * 1024)?;
    if hex::encode(Sha256::digest(&bytes)) != input.config_sha256 {
        return Err("competitive config SHA mismatch".into());
    }
    let document: Value = serde_json::from_slice(&bytes)?;
    super::competitive_plan::admit_cell(&document, &bytes, &input.cell)?;
    let arm = input.cell["arm"].as_str().ok_or("arm")?;
    if ["vllm", "sglang"].contains(&arm) {
        if !cfg!(target_os = "linux") || input.cell["platform"] != "cuda" {
            return Err("optional backend requires Linux CUDA".into());
        }
        let model = document["models"]
            .as_array()
            .ok_or("models")?
            .iter()
            .find(|value| value["key"] == input.cell["model"])
            .ok_or("model")?;
        if model["comparison_support"][arm]["available"] == false {
            return Err(format!(
                "pinned model explicitly excludes {arm}: {}",
                model["comparison_support"][arm]["reason"]
            )
            .into());
        }
    }
    if input.cell["workload"] == "synthetic" {
        let benchy = input
            .benchy
            .as_ref()
            .ok_or("synthetic requires pinned benchy executable")?;
        super::competitive_launch::file(benchy)?;
        let tokenizer = input
            .backend
            .tokenizer
            .as_ref()
            .ok_or("synthetic requires pinned tokenizer")?;
        super::competitive_launch::tree(tokenizer)?;
        let model = document["models"]
            .as_array()
            .ok_or("models")?
            .iter()
            .find(|value| value["key"] == input.cell["model"])
            .ok_or("model")?;
        if model["tokenizer_sha256"].as_str() != Some(tokenizer.sha256.as_str()) {
            return Err("synthetic tokenizer differs from source pin".into());
        }
    } else if input.manifest.is_none() {
        return Err("trace requires manifest".into());
    }
    Ok(document)
}
fn launches(
    input: &Input,
    document: &Value,
    prepared: super::competitive_launch::Prepared,
    port: u16,
    execution: Duration,
) -> DynResult<(Launch, Launch)> {
    let worker_input = input.output.join("worker-input.json");
    let worker_output = input.output.join("worker");
    let mut worker = json!({"config":input.config,"config_sha256":input.config_sha256,"cell":input.cell,"base_url":format!("http://127.0.0.1:{port}/v1"),"served_model":prepared.served,"launch_provenance":prepared.provenance,"output":worker_output,"timeout_seconds":execution.as_secs(),"request_timeout_seconds":input.request_timeout_seconds.min(execution.as_secs())});
    let command = if input.cell["workload"] == "synthetic" {
        let benchy = input.benchy.as_ref().ok_or("benchy")?;
        worker["benchy"] = json!({"path":benchy.path,"sha256":benchy.sha256});
        worker["tokenizer"] = input
            .backend
            .tokenizer
            .as_ref()
            .ok_or("tokenizer")?
            .path
            .to_string_lossy()
            .into_owned()
            .into();
        "competitive-synthetic-cell"
    } else {
        worker["manifest"] = serde_json::to_value(input.manifest.as_ref().ok_or("manifest")?)?;
        "competitive-cell"
    };
    crate::command::write_json_file(&worker_input, &worker)?;
    crate::command::write_json_file(
        &input.output.join("launch.json"),
        &json!({"schema_version":1,"cell":input.cell,"config_sha256":input.config_sha256,"command":prepared.command,"launch_provenance":prepared.provenance,"source_pin_context":document["baseline"],"capacity_policy":{"mode":"declared-shared-context","comparison_kv_matched":input.backend.match_kv_capacity},"worker_input_sha256":crate::product::digest::file_sha256(&worker_input).map_err(|error|error.error)?}),
    )?;
    let server = Launch {
        member: MemberId::Seed,
        spec: prepared.spec,
        files: process::OutputFiles {
            stdout: Some(input.output.join("server.stdout.log")),
            stderr: Some(input.output.join("server.stderr.log")),
        },
        readiness_deadline: execution,
    };
    let worker = Launch {
        member: MemberId::WorkerOne,
        spec: process::ProcessSpec {
            executable: std::env::current_exe()?,
            arguments: ["automation", "replay-matrix", command, "--input"]
                .into_iter()
                .map(|value| Argument::Public(value.into()))
                .chain(std::iter::once(Argument::Public(
                    worker_input.into_os_string(),
                )))
                .collect(),
            cwd: input.backend.cwd.clone(),
            environment: super::competitive_launch::environment(None, false),
        },
        files: process::OutputFiles {
            stdout: Some(input.output.join("worker.stdout.log")),
            stderr: Some(input.output.join("worker.stderr.log")),
        },
        readiness_deadline: execution,
    };
    Ok((server, worker))
}
fn complete(input: &Input, worker_output: &Path) -> DynResult<()> {
    let summary_bytes =
        super::competitive_cell::read(&worker_output.join("worker-summary.json"), 8 * 1024 * 1024)?;
    let summary: Value = serde_json::from_slice(&summary_bytes)?;
    if summary["completed"] != true
        || summary["passed"] != true
        || summary["cell"] != input.cell
        || summary["config_sha256"] != input.config_sha256
    {
        return Err("worker summary not complete or source-correlated".into());
    }
    let completion = json!({"schema_version":2,"scope":"competitive_retained_cell","completed":true,"cell":input.cell,"config_sha256":input.config_sha256,"launch_sha256":crate::product::digest::file_sha256(&input.output.join("launch.json")).map_err(|error|error.error)?,"worker_summary_sha256":hex::encode(Sha256::digest(summary_bytes)),"lifecycle_sha256":crate::product::digest::file_sha256(&input.output.join("lifecycle.json")).map_err(|error|error.error)?});
    super::competitive_synthetic::write_new(&input.output.join("complete.json"), &completion)
}

fn version(input: &Input, deadline: Instant) -> DynResult<()> {
    let text = observe_version(
        &input.backend,
        input.cell["arm"].as_str().ok_or("arm")?,
        deadline,
    )?;
    if hex::encode(Sha256::digest(text.as_bytes())) != input.backend.version_sha256 {
        return Err("backend version output differs from prepared source context".into());
    }
    crate::command::write_json_file(
        &input.output.join("version.json"),
        &json!({"version":text,"sha256":input.backend.version_sha256}),
    )
}
pub(super) fn observe_version(
    backend: &super::competitive_launch::Backend,
    name: &str,
    deadline: Instant,
) -> DynResult<String> {
    let execution = deadline
        .saturating_duration_since(Instant::now())
        .checked_sub(Duration::from_secs(9))
        .ok_or("version probe lacks cleanup budget")?
        .min(Duration::from_secs(30));
    if execution.is_zero() {
        return Err("version budget exhausted".into());
    }
    let args = if name == "sglang" {
        vec![
            "-c".into(),
            "import importlib.metadata; print(importlib.metadata.version('sglang'))".into(),
        ]
    } else {
        vec!["--version".into()]
    };
    let spec = process::ProcessSpec {
        executable: backend.executable.path.clone(),
        arguments: args.into_iter().map(Argument::Public).collect(),
        cwd: backend.cwd.clone(),
        environment: super::competitive_launch::environment(backend.runtime.as_ref(), false),
    };
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let result = process::supervise_raw(
        &spec,
        &limits(execution),
        &interrupt.cancellation(),
        process::RawCaptureOptions {
            stdout: std::num::NonZeroUsize::new(65536),
            stderr: std::num::NonZeroUsize::new(65536),
        },
    );
    let finish = interrupt.finish();
    let report = result?;
    finish?;
    if !report.process.success()
        || !full_stream(&report.stdout, &report.process.stdout)
        || !full_stream(&report.stderr, &report.process.stderr)
    {
        return Err("version probe must exit cleanly with complete captured output".into());
    }
    let mut raw = report
        .stdout
        .as_ref()
        .ok_or("version stdout missing")?
        .as_bytes()
        .to_vec();
    raw.extend(
        report
            .stderr
            .as_ref()
            .ok_or("version stderr missing")?
            .as_bytes(),
    );
    let text = std::str::from_utf8(&raw)?.trim();
    if text.is_empty() {
        return Err("empty backend version".into());
    }
    Ok(text.to_owned())
}
pub(super) fn limits(execution: Duration) -> process::Limits {
    process::Limits {
        execution,
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 65536,
        readiness: process::Readiness::None,
        completion: process::Completion::Exit,
    }
}
struct Owner {
    server: Option<Launch>,
    worker: Option<Launch>,
    stop: bool,
    policy: ExpectedExit,
}
impl Coordinator for Owner {
    type Rejection = String;
    fn line(
        &mut self,
        _member: MemberId,
        _line: process::ObservedLine<'_>,
    ) -> process::ProbeDecision<String> {
        process::ProbeDecision::Pending
    }
    fn tick(&mut self, context: Context<'_>) -> Action<String> {
        if self.stop {
            return Action::Complete;
        }
        if let Some(server) = self.server.take() {
            return Action::Start(server);
        }
        if context.members.iter().any(|member| {
            member.member == MemberId::Seed && matches!(member.state, MemberState::Starting)
        }) {
            return Action::Admit(MemberId::Seed);
        }
        if let Some(worker) = self.worker.take() {
            return Action::StartExpected {
                launch: worker,
                policy: self.policy.clone(),
            };
        }
        if context.members.iter().any(|member| {
            member.member == MemberId::WorkerOne
                && matches!(member.state, MemberState::ExpectedExit { .. })
        }) {
            self.stop = true;
            return Action::Stop(MemberId::Seed);
        }
        Action::Pending
    }
}

fn recheck_trees(backend: &super::competitive_launch::Backend) -> DynResult<()> {
    for artifact in [
        &backend.runtime,
        &backend.tokenizer,
        &backend.comparison_model,
    ]
    .into_iter()
    .flatten()
    {
        super::competitive_launch::tree(artifact)?;
    }
    if let Some(config) = &backend.hf_config {
        super::competitive_launch::hf_config_directory(config)?;
    }
    Ok(())
}

fn full_stream(raw: &Option<process::RawBytes>, observed: &process::StreamReport) -> bool {
    raw.as_ref()
        .is_some_and(|raw| u64::try_from(raw.as_bytes().len()).ok() == Some(observed.bytes_seen))
        && observed.line_capture_complete
        && observed.oversized_lines == 0
}
