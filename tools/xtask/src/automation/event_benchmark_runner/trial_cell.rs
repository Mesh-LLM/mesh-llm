//! One owned host trial; command owner supplies its installed interruption scope.
use super::{
    evidence_io, lifecycle_budget, measurement_worker, paired_execution::Outcome, plan,
    trial_environment, trial_owner::Owner, trial_profile, trial_receipt, worker_frontends,
};
use crate::process::retained::{ExpectedExit, Launch, MemberId, Report};
use crate::process::{
    Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
};
use crate::{automation::private_state::PrivateState, command::DynResult};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    net::TcpListener,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

pub(super) struct Input<'a> {
    pub side: &'a plan::Side,
    pub entry: &'a plan::Entry,
    pub binary_sha256: &'a str,
    pub model: &'a Path,
    pub model_sha256: &'a str,
    pub native_runtime_root: &'a Path,
    pub directory: &'a Path,
    pub max_tokens: u64,
    pub readiness: Duration,
    pub request: Duration,
    pub poll: Duration,
    pub shutdown: Duration,
    pub remaining: Duration,
    /// Capture once for the paired matrix, before per-side overrides.
    pub inherited_profile: &'a BTreeMap<OsString, OsString>,
}

pub(super) struct Trial {
    pub outcome: Outcome,
    pub environment: BTreeMap<String, trial_environment::Entry>,
    pub worker_status: Option<i32>,
    pub cleanup_complete: bool,
    pub cleanup_forced: bool,
    pub capture_complete: bool,
    pub health_observation_error: Option<String>,
    pub inheritance: trial_profile::Inheritance,
}

impl Trial {
    pub(super) fn observe_final_health(&mut self, directory: &Path) {
        collect_health(self, directory);
    }

    pub(super) fn into_outcome(mut self) -> Outcome {
        self.outcome.health_observation_error = self.health_observation_error;
        self.outcome
    }
}

fn identity(path: &Path, expected: &str) -> DynResult<PathBuf> {
    if expected.len() != 64
        || !expected.bytes().all(|b| b.is_ascii_hexdigit())
        || !path.is_absolute()
        || !std::fs::symlink_metadata(path)?.file_type().is_file()
    {
        return Err("trial identity requires an absolute regular file and SHA256".into());
    }
    let canonical = path.canonicalize()?;
    if canonical != path {
        return Err("trial requires the admitted canonical path".into());
    }
    Ok(canonical)
}

fn raw(environment: &BTreeMap<OsString, Value>) -> BTreeMap<OsString, OsString> {
    environment
        .iter()
        .map(|(key, value)| {
            let raw = match value {
                Value::Public(raw) | Value::Secret(raw) => raw,
            };
            (key.clone(), raw.clone())
        })
        .collect()
}

fn limits(execution: Duration, shutdown: Duration) -> Limits {
    Limits {
        execution,
        graceful_shutdown: shutdown,
        forced_shutdown: shutdown,
        retained_bytes_per_stream: 4 * 1024 * 1024,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}

fn launches(
    input: &Input<'_>,
    state: &PrivateState,
    port: u16,
    execution: Duration,
) -> DynResult<(
    Owner,
    BTreeMap<String, trial_environment::Entry>,
    trial_profile::Inheritance,
)> {
    let (environment, inheritance) = trial_profile::apply(
        state.environment(input.native_runtime_root),
        input.inherited_profile,
        input.side.mode,
    );
    let snapshot = trial_environment::snapshot(&raw(&environment));
    let server = Launch {
        member: MemberId::Seed,
        spec: ProcessSpec {
            executable: input.side.binary.clone(),
            cwd: input.directory.into(),
            environment,
            arguments: [
                "serve".into(),
                "--local-model-only".into(),
                "--model".into(),
                input.model.as_os_str().to_owned(),
                "--port".into(),
                port.to_string().into(),
                "--speculative-strategy".into(),
                "disabled".into(),
                "--log-format".into(),
                "json".into(),
            ]
            .into_iter()
            .map(Value::Public)
            .collect(),
        },
        files: OutputFiles {
            stdout: Some(input.directory.join("server.stdout.log")),
            stderr: Some(input.directory.join("server.stderr.log")),
        },
        readiness_deadline: execution,
    };
    let worker = Launch {
        member: MemberId::WorkerOne,
        spec: ProcessSpec {
            executable: std::env::current_exe()?,
            cwd: input.directory.into(),
            environment: state.environment(input.native_runtime_root),
            arguments: [
                "automation".into(),
                "event-benchmark-run".into(),
                "measurement-worker".into(),
                "--input".into(),
                input.directory.join("worker-input.json").into_os_string(),
                "--output".into(),
                input.directory.join("worker.json").into_os_string(),
            ]
            .into_iter()
            .map(Value::Public)
            .collect(),
        },
        files: OutputFiles {
            stdout: Some(input.directory.join("worker.stdout.log")),
            stderr: Some(input.directory.join("worker.stderr.log")),
        },
        readiness_deadline: execution,
    };
    Ok((
        Owner {
            server: Some(server),
            worker: Some(worker),
            worker_policy: ExpectedExit::new(&[0, 1], execution)?,
            stopping: false,
            setup_ms: None,
            stop_started: None,
            shutdown_ms: None,
            api_readiness: Some((format!("http://127.0.0.1:{port}/v1"), port)),
            listener_ready: false,
            host_readiness_timeout: input.readiness,
            host_started: None,
        },
        snapshot,
        inheritance,
    ))
}

fn failed(trial: &mut Trial, error: impl std::fmt::Display) {
    trial.outcome.error = Some(match trial.outcome.error.take() {
        Some(prior) => format!("{prior}; {error}"),
        None => error.to_string(),
    });
}

fn observe(trial: &mut Trial, owner: &Owner, report: &Report<String>) {
    trial.outcome.launched = report.members.iter().any(|m| m.member == MemberId::Seed);
    trial.outcome.setup_ms = owner.setup_ms;
    trial.outcome.shutdown_ms = owner.shutdown_ms;
    trial.worker_status = report
        .members
        .iter()
        .find(|m| m.member == MemberId::WorkerOne)
        .and_then(|m| m.process.status.as_ref())
        .and_then(std::process::ExitStatus::code);
    trial.cleanup_complete = report.members.iter().all(|m| m.process.cleanup.complete);
    trial.cleanup_forced = report.members.iter().any(|m| m.process.cleanup.forced);
    trial.capture_complete = report
        .members
        .iter()
        .find(|m| m.member == MemberId::Seed)
        .is_some_and(|member| {
            [&member.process.stdout, &member.process.stderr]
                .iter()
                .all(|stream| !stream.truncated && stream.suppressed_lines == 0)
        });
    if !trial.capture_complete {
        trial.health_observation_error =
            Some("host capture incomplete or suppressed; final health cannot be qualified".into());
    }
    if !report.recovery_success() || report.members.len() != 2 {
        failed(
            trial,
            format!("retained trial failed: {:?}", report.outcome),
        );
    }
}

fn collect(trial: &mut Trial, input: &Input<'_>, request_sha256: &str) {
    match trial_receipt::worker(
        &input.directory.join("worker.json"),
        request_sha256,
        &input.entry.prompt_sha256(),
        trial.worker_status,
    ) {
        Ok(evidence) => {
            trial.outcome.readiness_ms = evidence.readiness_ms;
            trial.outcome.measurement = evidence.measurement;
            if let Some(error) = evidence.error {
                failed(trial, error);
            }
        }
        Err(error) => failed(trial, error),
    }
    trial.observe_final_health(input.directory);
}

fn collect_health(trial: &mut Trial, directory: &Path) {
    if trial.capture_complete {
        match trial_receipt::final_health(directory) {
            Ok(health) => trial.outcome.health = health,
            Err(error) => trial.health_observation_error = Some(error.to_string()),
        }
    }
}

fn prepare_owner(
    input: &Input<'_>,
    port: u16,
    execution: Duration,
) -> DynResult<(
    PrivateState,
    Owner,
    BTreeMap<String, trial_environment::Entry>,
    trial_profile::Inheritance,
)> {
    let state = PrivateState::create(&std::env::temp_dir(), "event-benchmark-trial")?;
    let prepared = (|| -> DynResult<_> {
        state.prepare()?;
        launches(input, &state, port, execution)
    })();
    match prepared {
        Ok((owner, environment, inheritance)) => Ok((state, owner, environment, inheritance)),
        Err(error) => Err(preparation_failure(state, error)),
    }
}

fn preparation_failure(
    state: PrivateState,
    error: crate::command::DynError,
) -> crate::command::DynError {
    match state.finish(Err::<(), _>(error.to_string())) {
        Err(cleanup) => {
            format!("trial preparation failed; private-state receipt: {cleanup:?}").into()
        }
        Ok(()) => "trial preparation failed without an error receipt".into(),
    }
}

pub(super) fn execute(input: &Input<'_>, cancellation: &Cancellation) -> DynResult<Trial> {
    let start = Instant::now();
    if cancellation.is_cancelled() {
        return Err("benchmark interrupted before trial".into());
    }
    if !input.directory.is_absolute() {
        return Err("trial evidence directory must be absolute".into());
    }
    identity(&input.side.binary, input.binary_sha256)?;
    identity(input.model, input.model_sha256)?;
    if !input.native_runtime_root.is_absolute() || !input.native_runtime_root.is_dir() {
        return Err("trial requires explicit native-runtime bundle directory".into());
    }
    let (execution, cleanup_budget) =
        lifecycle_budget::cell(input.readiness, input.request, input.shutdown)?;
    let remaining = input.remaining.saturating_sub(start.elapsed());
    if remaining <= execution + cleanup_budget {
        return Err("remaining matrix deadline cannot cover trial and owned-tree cleanup".into());
    }
    std::fs::create_dir(input.directory)?;
    let reservation = TcpListener::bind(("127.0.0.1", 0))?;
    let worker = measurement_worker::Input {
        schema_version: 1,
        port: reservation.local_addr()?.port(),
        prompt: input.entry.prompt(),
        prompt_sha256: input.entry.prompt_sha256(),
        max_tokens: input.max_tokens,
        readiness_timeout_ms: u64::try_from(input.readiness.as_millis())?,
        request_timeout_ms: u64::try_from(input.request.as_millis())?,
        readiness_poll_ms: u64::try_from(input.poll.as_millis())?,
    };
    worker.validate()?;
    let request_sha256 = worker_frontends::request_sha256(&worker)?;
    evidence_io::publish(
        &input.directory.join("worker-input.json"),
        &worker,
        evidence_io::INPUT_BYTES,
    )?;
    let (state, mut owner, environment, inheritance) =
        prepare_owner(input, worker.port, execution)?;
    let mut trial = Trial {
        outcome: Outcome::default(),
        environment,
        worker_status: None,
        cleanup_complete: false,
        cleanup_forced: false,
        capture_complete: false,
        health_observation_error: None,
        inheritance,
    };
    drop(reservation);
    let final_remaining = input.remaining.saturating_sub(start.elapsed());
    let result: DynResult<Report<String>> = if final_remaining <= execution + cleanup_budget {
        Err("trial preparation exhausted deadline reserved for owned cleanup".into())
    } else {
        crate::process::retained::run(&mut owner, &limits(execution, input.shutdown), cancellation)
            .map_err(Box::<dyn std::error::Error>::from)
    };
    match result {
        Ok(report) => observe(&mut trial, &owner, &report),
        Err(error) => failed(&mut trial, error),
    }
    collect(&mut trial, input, &request_sha256);
    if let Err(error) = state.retain_runtime_logs(&input.directory.join("native-runtime")) {
        failed(&mut trial, error);
    }
    if let Err(error) = state.finish(Ok::<(), String>(())) {
        failed(
            &mut trial,
            format!("private-state cleanup failed: {error:?}"),
        );
    }
    Ok(trial)
}

#[cfg(test)]
#[path = "trial_cell_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "trial_admission_tests.rs"]
mod admission_tests;
