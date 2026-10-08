//! Retained ownership of a pinned stage server and its isolated measurement worker.
use super::{cell_worker, metrics_client, options, publish, stage_config, telemetry_sink};
use crate::process::retained::{
    Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState, Report,
};
use crate::process::{
    Completion, Limits, ObservedLine, OutputFiles, ProbeDecision, ProcessSpec, Readiness, Value,
};
use crate::{
    automation::{command_interrupt::Interrupt, private_state::PrivateState},
    command::DynResult,
};
use serde::{Deserialize, Serialize};
use std::{
    net::TcpListener,
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    schema_version: u64,
    binary: PathBuf,
    binary_sha256: String,
    native_runtime_root: PathBuf,
    stage: stage_config::Input,
    pub(super) worker: cell_worker::Input,
    admission_concurrency: u64,
    execution_timeout_secs: u64,
}

impl Input {
    pub(super) fn validate(&self) -> DynResult<()> {
        if self.schema_version != 1
            || !self.binary.is_absolute()
            || !self.native_runtime_root.is_absolute()
            || self.binary_sha256.len() != 64
            || !self
                .binary_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            || self.admission_concurrency > 10000
            || !(1..=86400).contains(&self.execution_timeout_secs)
        {
            return Err("invalid server-cell identity, concurrency or deadline".into());
        }
        let admitted = self.admission()?;
        self.stage.validate()?;
        let seed = self.worker.seed_phase()?;
        self.worker.validate(seed.as_ref())?;
        let create_budget = self
            .worker
            .metrics
            .as_ref()
            .map_or(0.0, |endpoint| endpoint.timeout_secs as f64);
        if self.worker.deadline_budget(seed.as_ref()) + create_budget
            >= self.execution_timeout_secs as f64
        {
            return Err(
                "server deadline must cover readiness, seed, measured requests and telemetry"
                    .into(),
            );
        }
        if seed
            .as_ref()
            .is_some_and(|phase| phase.prompts.len() as u64 > admitted)
        {
            return Err("server admission must cover every cache-seed request".into());
        }
        if self.stage.model_id != self.worker.phase.model {
            return Err("server stage and request model identities differ".into());
        }
        if self.worker.startup_timeout_secs >= self.execution_timeout_secs {
            return Err("server startup deadline must precede its execution deadline".into());
        }
        Ok(())
    }

    fn admission(&self) -> DynResult<u64> {
        let requests = u64::try_from(self.worker.phase.prompts.len())?;
        let admitted = if self.admission_concurrency == 0 {
            requests
        } else {
            self.admission_concurrency
        };
        if admitted < requests || admitted > u64::from(self.stage.lane_count) {
            return Err(
                "A/B admission must cover every measured request and fit the stage lanes".into(),
            );
        }
        Ok(admitted)
    }

    fn preflight(&mut self) -> DynResult<serde_json::Value> {
        self.validate()?;
        self.binary = self.binary.canonicalize()?;
        self.native_runtime_root = self.native_runtime_root.canonicalize()?;
        if !self.binary.is_file() || !self.native_runtime_root.is_dir() {
            return Err("server binary and native-runtime directory must exist".into());
        }
        let digest =
            crate::product::digest::file_sha256(&self.binary).map_err(|error| error.error)?;
        if digest != self.binary_sha256 {
            return Err("server binary SHA-256 mismatch".into());
        }
        self.stage.model_path = self.stage.model_path.canonicalize()?;
        let model = crate::automation::replay_matrix::model_preflight::verify(
            &self.stage.model_path,
            &self.stage.source_model_sha256,
            u64::from(self.stage.ctx_size),
        )?;
        let dimensions = crate::automation::replay_matrix::model_preflight::dimensions::inspect(
            &self.stage.model_path,
        )?
        .ok_or("server cell requires complete GGUF dimensions")?;
        if u64::from(self.stage.layer_end) != dimensions.block_count {
            return Err("single-stage layer range differs from the complete GGUF model".into());
        }
        Ok(serde_json::to_value(model)?)
    }
}

struct Owner {
    server: Option<Launch>,
    worker: Option<Launch>,
    stopping: bool,
    policy: ExpectedExit,
    telemetry: Option<telemetry_sink::Sink>,
}

impl Owner {
    fn bind_deadline(&mut self, remaining: Duration) -> DynResult<()> {
        if remaining.is_zero() {
            return Err("server cell deadline expired during collector creation".into());
        }
        for launch in [&mut self.server, &mut self.worker].into_iter().flatten() {
            launch.readiness_deadline = launch.readiness_deadline.min(remaining);
        }
        self.policy = ExpectedExit::new(&[0, 1], remaining)?;
        Ok(())
    }
}

impl Coordinator for Owner {
    type Rejection = String;
    fn line(&mut self, member: MemberId, line: ObservedLine<'_>) -> ProbeDecision<String> {
        if member == MemberId::Seed
            && line.stream == crate::process::Stream::Stderr
            && let Some(sink) = &mut self.telemetry
            && let Err(error) = sink.observe(line.bytes)
        {
            return ProbeDecision::Rejected(error);
        }
        ProbeDecision::Pending
    }
    fn tick(&mut self, context: Context<'_>) -> Action<String> {
        if let Some(sink) = &mut self.telemetry
            && let Err(error) = sink.tick()
        {
            return Action::Reject(error);
        }
        if self.stopping {
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
        if context.members.iter().any(|member| {
            member.member == MemberId::Seed && matches!(member.state, MemberState::Ready { .. })
        }) && let Some(worker) = self.worker.take()
        {
            return Action::StartExpected {
                launch: worker,
                policy: self.policy.clone(),
            };
        }
        if context.members.iter().any(|member| {
            member.member == MemberId::WorkerOne
                && matches!(member.state, MemberState::ExpectedExit { .. })
        }) {
            self.stopping = true;
            return Action::Stop(MemberId::Seed);
        }
        Action::Pending
    }
}

#[derive(Serialize)]
struct Receipt {
    schema_version: u64,
    binary_sha256: String,
    model_identity: serde_json::Value,
    worker_status: Option<i32>,
    infrastructure_clean: bool,
    error: Option<String>,
    collector_run_id: Option<String>,
}

impl Receipt {
    fn observe(&mut self, report: &Report<String>) {
        self.worker_status = report
            .members
            .iter()
            .find(|member| member.member == MemberId::WorkerOne)
            .and_then(|member| member.process.status.as_ref())
            .and_then(std::process::ExitStatus::code);
        self.infrastructure_clean = report.recovery_success() && report.members.len() == 2;
        if !self.infrastructure_clean || self.worker_status != Some(0) {
            self.error = Some(format!(
                "server cell failed: {:?}; worker status {:?}",
                report.outcome, self.worker_status
            ));
        }
    }
    fn fail(&mut self, error: impl std::fmt::Display) {
        self.error = Some(match self.error.take() {
            Some(prior) => format!("{prior}; {error}"),
            None => error.to_string(),
        });
    }
}

fn limits(execution: Duration) -> Limits {
    Limits {
        execution,
        graceful_shutdown: Duration::from_secs(30),
        forced_shutdown: Duration::from_secs(10),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}

fn launches(input: &Input, directory: &Path, state: &PrivateState, port: u16) -> DynResult<Owner> {
    let execution = Duration::from_secs(input.execution_timeout_secs);
    let mut environment = state.environment(&input.native_runtime_root);
    environment.insert("SKIPPY_TELEMETRY_STDERR".into(), Value::Public("1".into()));
    environment.insert(
        "SKIPPY_NATIVE_MTP_GREEDY_SAMPLING_FASTPATH".into(),
        Value::Public("1".into()),
    );
    let mut arguments = vec![
        "serve-openai".into(),
        "--config".into(),
        directory.join("stage.json").into_os_string(),
        "--bind-addr".into(),
        format!("127.0.0.1:{port}").into(),
        "--generation-concurrency".into(),
        input.admission()?.to_string().into(),
        "--telemetry-level".into(),
        "debug".into(),
    ];
    if let Some(metrics) = &input.worker.metrics {
        arguments.extend([
            "--metrics-otlp-grpc".into(),
            metrics.otlp_grpc.clone().into(),
        ]);
    }
    let server = Launch {
        member: MemberId::Seed,
        spec: ProcessSpec {
            executable: input.binary.clone(),
            arguments: arguments.into_iter().map(Value::Public).collect(),
            cwd: directory.to_path_buf(),
            environment,
        },
        files: OutputFiles {
            stdout: Some(directory.join("server.stdout.log")),
            stderr: Some(directory.join("server.stderr.log")),
        },
        readiness_deadline: execution,
    };
    let worker = Launch {
        member: MemberId::WorkerOne,
        spec: ProcessSpec {
            executable: std::env::current_exe()?,
            arguments: vec![
                Value::Public("automation".into()),
                Value::Public("waiting-prefix".into()),
                Value::Public("cell-worker".into()),
                Value::Public("--input".into()),
                Value::Public(directory.join("worker-input.json").into_os_string()),
                Value::Public("--output".into()),
                Value::Public(directory.join("cell.json").into_os_string()),
            ],
            cwd: directory.to_path_buf(),
            environment: state.environment(&input.native_runtime_root),
        },
        files: OutputFiles {
            stdout: Some(directory.join("worker.stdout.log")),
            stderr: Some(directory.join("worker.stderr.log")),
        },
        readiness_deadline: execution,
    };
    Ok(Owner {
        server: Some(server),
        worker: Some(worker),
        stopping: false,
        policy: ExpectedExit::new(&[0, 1], execution)?,
        telemetry: Some(telemetry_sink::Sink::create(&directory.join("server.log"))?),
    })
}

fn session(input: &Input, owner: &mut Owner, directory: &Path, receipt: &mut Receipt) {
    let interrupt = match Interrupt::install() {
        Ok(interrupt) => interrupt,
        Err(error) => {
            receipt.fail(error);
            return;
        }
    };
    let cancellation = interrupt.cancellation();
    let started = std::time::Instant::now();
    let result = (|| -> DynResult<()> {
        let server = owner
            .server
            .as_mut()
            .ok_or("waiting-prefix server absent")?;
        crate::automation::skippy_cli_admission::prepare(
            &mut server.spec,
            &input.binary_sha256,
            crate::automation::skippy_cli_admission::Role::Public,
            started + Duration::from_secs(input.execution_timeout_secs),
            &cancellation,
            &directory.join("cli-admission.json"),
        )?;
        if let Some(endpoint) = &input.worker.metrics {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()?;
            let mut config: serde_json::Value =
                serde_json::from_slice(&std::fs::read(directory.join("stage.json"))?)?;
            config["measurement_request_count"] =
                serde_json::json!(input.worker.phase.prompts.len());
            config["seed_request_count"] = serde_json::json!(
                input
                    .worker
                    .seed_phase()?
                    .map_or(0, |phase| phase.prompts.len())
            );
            runtime.block_on(metrics_client::create(endpoint, &config, &cancellation))?;
        }
        let remaining =
            Duration::from_secs(input.execution_timeout_secs).saturating_sub(started.elapsed());
        owner.bind_deadline(remaining)?;
        let report = crate::process::retained::run(owner, &limits(remaining), &cancellation)?;
        receipt.observe(&report);
        Ok(())
    })();
    if let Err(error) = result {
        receipt.fail(error);
    }
    if let Err(error) = interrupt.finish() {
        receipt.fail(error);
    }
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    let opts = options(
        args,
        &["--input", "--output-directory"],
        &["--input", "--output-directory"],
    )?;
    let mut input: Input = serde_json::from_slice(&std::fs::read(opts["--input"])?)?;
    let identity = input.preflight()?;
    let directory = std::path::absolute(opts["--output-directory"])?;
    std::fs::create_dir(&directory)?;
    let directory = directory.canonicalize()?;
    let reservation = TcpListener::bind(("127.0.0.1", 0))?;
    let port = reservation.local_addr()?.port();
    input.worker.phase.base_url = format!("http://127.0.0.1:{port}/v1");
    input.worker.server_log = directory.join("server.log");
    if input.worker.metrics.is_some() {
        input.worker.metrics_directory = Some(directory.clone());
    }
    let mut stage: serde_json::Value =
        serde_json::from_slice(&stage_config::prepare(&mut input.stage)?)?;
    if let Some(endpoint) = &input.worker.metrics {
        stage["run_id"] = serde_json::json!(endpoint.run_id);
    }
    publish(
        &directory.join("stage.json"),
        &serde_json::to_vec_pretty(&stage)?,
    )?;
    publish(
        &directory.join("worker-input.json"),
        &serde_json::to_vec_pretty(&input.worker)?,
    )?;
    let state = PrivateState::create(&std::env::temp_dir(), "waiting-prefix-server")?;
    state.prepare()?;
    let mut owner = launches(&input, &directory, &state, port)?;
    let mut receipt = Receipt {
        schema_version: 1,
        binary_sha256: input.binary_sha256.clone(),
        model_identity: identity,
        worker_status: None,
        infrastructure_clean: false,
        error: None,
        collector_run_id: input
            .worker
            .metrics
            .as_ref()
            .map(|endpoint| endpoint.run_id.clone()),
    };
    drop(reservation);
    session(&input, &mut owner, &directory, &mut receipt);
    if let Some(sink) = &mut owner.telemetry
        && let Err(error) = sink.finish()
    {
        receipt.fail(error);
    }
    if let Err(error) = state.retain_runtime_logs(&directory.join("native-runtime")) {
        receipt.fail(error);
    }
    if let Err(error) = state.finish(Ok::<(), String>(())) {
        receipt.fail(format!("private-state cleanup failed: {error:?}"));
    }
    let mut bytes = serde_json::to_vec_pretty(&receipt)?;
    bytes.push(b'\n');
    publish(&directory.join("lifecycle.json"), &bytes)?;
    if receipt.error.is_some() {
        Err("server cell failed; logs and lifecycle retained".into())
    } else {
        Ok(())
    }
}

#[cfg(test)]
#[path = "server_cell_tests.rs"]
mod tests;
