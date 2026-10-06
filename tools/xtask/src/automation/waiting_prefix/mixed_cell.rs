//! Static native identity pre/postflight, actual retained mixed roles and typed phase evidence.
use super::{
    adaptive_cell as static_cell, adaptive_identity as identity,
    mixed_owner::Owner,
    mixed_phase::Observation,
    mixed_summary, mixed_worker,
    mixed_workload::{self, Role, Shape},
    options, publish,
};
use crate::process::retained::{ExpectedExit, Launch, MemberId};
use crate::{
    command::DynResult,
    process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Arg,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    net::TcpListener,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u64,
    pub arm: identity::Input,
    pub shape: Shape,
    pub split: bool,
    pub worker: mixed_worker::Input,
    pub timeout_secs: u64,
    pub startup_timeout_secs: u64,
}
impl Input {
    pub fn validate(&self) -> DynResult<()> {
        self.arm.validate()?;
        self.shape.validate()?;
        self.worker.validate()?;
        if self.schema_version != 1
            || !(10..=86400).contains(&self.timeout_secs)
            || self.startup_timeout_secs == 0
            || self.startup_timeout_secs >= self.timeout_secs
            || self.arm.round > self.shape.rounds
            || self.worker.round != self.arm.round
            || self.worker.version != self.arm.version
        {
            return Err("mixed cell arm/worker/round/deadline mismatch".into());
        }
        if self.worker.requests.len() != usize::try_from(self.shape.anchors + self.shape.prefills)?
        {
            return Err("mixed exact role roster mismatch".into());
        }
        for (index, row) in self.worker.requests.iter().enumerate() {
            let anchor = index < self.shape.anchors as usize;
            let role = if anchor { Role::Anchor } else { Role::Prefill };
            let tokens = if anchor {
                self.shape.anchor_output_tokens
            } else {
                self.shape.prefill_output_tokens
            };
            let delay = if anchor {
                0.0
            } else {
                self.shape.prefill_delay_ms
                    + (index - self.shape.anchors as usize) as f64 * self.shape.prefill_stagger_ms
            };
            if row.role != role || row.output_tokens != tokens || row.delay_ms != delay {
                return Err("mixed worker role timing/output differs from profile".into());
            }
        }
        Ok(())
    }
}
fn launch(
    input: &Input,
    directory: &Path,
    index: usize,
    mut environment: BTreeMap<std::ffi::OsString, Arg>,
    execution: Duration,
) -> DynResult<Launch> {
    if let Some(Arg::Public(root)) = environment.get(std::ffi::OsStr::new("MESH_LLM_RUNTIME_ROOT"))
    {
        let path = PathBuf::from(root).join(format!("stage-{index}"));
        environment.insert(
            "MESH_LLM_RUNTIME_ROOT".into(),
            Arg::Public(path.into_os_string()),
        );
    }
    Ok(Launch {
        member: if index == 1 {
            MemberId::Seed
        } else {
            MemberId::WorkerOne
        },
        spec: ProcessSpec {
            executable: input.arm.binary.clone(),
            arguments: mixed_workload::arguments(
                &input.arm,
                &input.shape,
                &directory.join(format!("stage-{index}.json")),
                index,
                input.split,
            )?
            .into_iter()
            .map(Arg::Public)
            .collect(),
            cwd: directory.into(),
            environment,
        },
        files: OutputFiles {
            stdout: Some(directory.join(format!("stage-{index}.stdout.log"))),
            stderr: Some(directory.join(format!("stage-{index}.stderr.log"))),
        },
        readiness_deadline: execution,
    })
}
fn lifecycle(report: &process::retained::Report<String>) -> Value {
    json!({"outcome":format!("{:?}",report.outcome),"members":report.members.iter().map(|m|json!({"member":String::from_utf8_lossy(m.member.name()),"disposition":m.disposition.label(),"status":m.process.status.as_ref().and_then(std::process::ExitStatus::code),"cleanup_complete":m.process.cleanup.complete,"forced":m.process.cleanup.forced,"graceful_signal_failed":m.process.cleanup.graceful_signal_failed,"cleanup_failure":m.process.cleanup.failure.as_ref().map(|e|format!("{e:?}")),"process_failure":m.process.failure.as_ref().map(|e|format!("{e:?}")),"stdout_complete":m.process.stdout.line_capture_complete,"stderr_complete":m.process.stderr.line_capture_complete})).collect::<Vec<_>>()})
}
fn run_owned(
    input: &Input,
    directory: &Path,
    tool: &Path,
    until: Instant,
    cancel: &Cancellation,
    reservations: Vec<TcpListener>,
) -> DynResult<Value> {
    let admitted = static_cell::admitted(tool, &input.arm, directory, until, cancel)?;
    let arm: identity::Input = serde_json::from_value(admitted["admitted"].clone())?;
    // Configs come from the shared admitted identity, then the exact mixed profile is applied.
    let count = if input.split { 2 } else { 1 };
    let mut configs = Vec::new();
    for index in 0..count {
        let config = mixed_workload::config(&arm, &input.shape, index, input.split)?;
        identity::fresh(
            &directory.join(format!("stage-{index}.json")),
            &serde_json::to_vec(&config)?,
        )?;
        configs.push(config);
    }
    let execution = static_cell::remaining(until, Duration::from_secs(3 * (count + 1) as u64))?;
    let mut worker = input.worker.clone();
    worker.base_url = format!("http://127.0.0.1:{}/v1", arm.openai_port);
    worker.model = arm.model_id.clone();
    worker.readiness_timeout_secs = input.startup_timeout_secs;
    worker
        .provenance
        .insert("admitted_arm".into(), admitted["admitted"].clone());
    worker
        .provenance
        .insert("model_identity".into(), admitted["model_identity"].clone());
    worker.validate()?;
    if Duration::from_secs(worker.timeout_secs) >= execution {
        return Err("mixed worker cannot fit retained execution budget".into());
    }
    let expected = identity::digest(&serde_json::to_vec(&worker)?);
    identity::fresh(
        &directory.join("worker-input.json"),
        &serde_json::to_vec(&worker)?,
    )?;
    let state = crate::automation::private_state::PrivateState::create(
        &std::env::temp_dir(),
        "mixed-cell",
    )?;
    let result = (|| -> DynResult<Value> {
        state.prepare()?;
        for index in 0..count {
            std::fs::create_dir(state.root().join("runtime").join(format!("stage-{index}")))?;
        }
        let mut environment = static_cell::static_environment(&state, &arm.native_build);
        environment.insert(
            "LLAMA_STAGE_BUILD_DIR".into(),
            Arg::Public(arm.native_build.clone().into_os_string()),
        );
        environment.insert("SKIPPY_TELEMETRY_STDERR".into(), Arg::Public("1".into()));
        environment.insert(
            "SKIPPY_NATIVE_MTP_GREEDY_SAMPLING_FASTPATH".into(),
            Arg::Public("1".into()),
        );
        let worker_launch = Launch {
            member: MemberId::WorkerTwo,
            spec: ProcessSpec {
                executable: tool.into(),
                arguments: vec![
                    Arg::Public("automation".into()),
                    Arg::Public("waiting-prefix".into()),
                    Arg::Public("mixed-worker".into()),
                    Arg::Public("--input".into()),
                    Arg::Public(directory.join("worker-input.json").into_os_string()),
                    Arg::Public("--output".into()),
                    Arg::Public(directory.join("requests.json").into_os_string()),
                ],
                cwd: directory.into(),
                environment: static_cell::static_environment(&state, &arm.native_build),
            },
            files: OutputFiles {
                stdout: Some(directory.join("worker.stdout.log")),
                stderr: Some(directory.join("worker.stderr.log")),
            },
            readiness_deadline: execution,
        };
        let mut owned_input: Input = serde_json::from_value(serde_json::to_value(input)?)?;
        owned_input.arm = arm.clone();
        let mut owner = Owner {
            downstream: if input.split {
                Some(launch(
                    &owned_input,
                    directory,
                    1,
                    environment.clone(),
                    execution,
                )?)
            } else {
                None
            },
            upstream: Some(launch(&owned_input, directory, 0, environment, execution)?),
            worker: Some(worker_launch),
            split: input.split,
            policy: ExpectedExit::new(&[0, 1], execution)?,
            telemetry: Observation::default(),
            stopping: 0,
        };
        let limits = Limits {
            execution,
            graceful_shutdown: Duration::from_secs(1),
            forced_shutdown: Duration::from_secs(1),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        };
        drop(reservations);
        let report = process::retained::run(&mut owner, &limits, cancel)?;
        let life = lifecycle(&report);
        publish(
            &directory.join("phase-observations.json"),
            &serde_json::to_vec_pretty(&owner.telemetry)?,
        )?;
        publish(
            &directory.join("lifecycle.json"),
            &serde_json::to_vec_pretty(&life)?,
        )?;
        let complete = report.members.len() == count + 1
            && report.recovery_success()
            && report.members.iter().all(|m| {
                static_cell::clean(&m.process)
                    && m.process.stdout.line_capture_complete
                    && m.process.stderr.line_capture_complete
            });
        let receipt: Value = serde_json::from_slice(&identity::bounded(
            &directory.join("requests.json"),
            16 * 1024 * 1024,
        )?)?;
        if receipt["schema_version"] != 1
            || receipt["input_sha256"] != expected
            || receipt["workload_sha256"] != worker.workload_sha256
            || !receipt["error"].is_null()
        {
            return Err(
                "mixed failed/uncorrelated request receipt; partial evidence retained".into(),
            );
        }
        if !report.members.iter().any(|m| {
            m.member == MemberId::WorkerTwo
                && m.process
                    .status
                    .as_ref()
                    .and_then(std::process::ExitStatus::code)
                    == Some(0)
        }) {
            return Err("mixed worker did not exit zero".into());
        }
        let measured = owner
            .telemetry
            .measured(&receipt, worker.requests.len(), complete)?;
        identity::fresh(
            &directory.join("measured-telemetry.json"),
            &serde_json::to_vec_pretty(&measured)?,
        )?;
        let mut summary = mixed_summary::requests(&receipt)?;
        mixed_summary::counters(
            &mut summary,
            &receipt,
            &measured,
            input.shape.n_batch,
            input.shape.prefills as usize,
        )?;
        let post = directory.join("postflight");
        std::fs::create_dir(&post)?;
        let current = static_cell::admitted(tool, &arm, &post, until, cancel)?;
        for key in ["admitted", "model_identity", "configs"] {
            if current[key] != admitted[key] {
                return Err("mixed static arm identity changed during execution".into());
            }
        }
        Ok(
            json!({"schema_version":1,"status":"mixed_cell_admitted","identity":admitted,"configs":configs,"shape":input.shape,"split":input.split,"lifecycle":life,"requests":receipt,"summary":summary,"scheduler_phase_scope":"same-host producer system-time spans with worker Instant clock guard","telemetry_drop_scope":"zero counters on observed events; no final sink-stat attestation"}),
        )
    })();
    let result = match state.retain_runtime_logs(&directory.join("native-runtime")) {
        Ok(()) => result,
        Err(error) => Err(format!(
            "mixed runtime log retention failure: {error}; prior {:?}",
            result.as_ref().err().map(ToString::to_string)
        )
        .into()),
    };
    state
        .finish(result)
        .map_err(|e| format!("mixed private-state cleanup/result: {e:?}").into())
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let opts = options(
        args,
        &["--input", "--output-directory"],
        &["--input", "--output-directory"],
    )?;
    let mut input: Input = serde_json::from_slice(&identity::bounded(
        Path::new(opts["--input"]),
        16 * 1024 * 1024,
    )?)?;
    input.validate()?;
    let expected = identity::digest(&serde_json::to_vec(&input)?);
    let directory = std::path::absolute(opts["--output-directory"])?;
    std::fs::create_dir(&directory)?;
    let directory = directory.canonicalize()?;
    let until = Instant::now() + Duration::from_secs(input.timeout_secs);
    let ports = (0..3)
        .map(|_| TcpListener::bind("127.0.0.1:0"))
        .collect::<std::io::Result<Vec<_>>>()?;
    input.arm.stage_ports = [ports[0].local_addr()?.port(), ports[1].local_addr()?.port()];
    input.arm.openai_port = ports[2].local_addr()?.port();
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let result = run_owned(
        &input,
        &directory,
        &std::env::current_exe()?,
        until,
        &cancellation,
        ports,
    );
    let finished: DynResult<()> = interrupt.finish().map_err(|error| error.to_string().into());
    let mut output = match &result {
        Ok(value) => value.clone(),
        Err(error) => {
            json!({"schema_version":1,"status":"mixed_cell_failed","error":error.to_string()})
        }
    };
    output["request_sha256"] = json!(expected);
    let admission = super::mixed_terminal::finalize(
        &mut output,
        finished,
        &cancellation,
        until,
        super::mixed_terminal::Kind::Cell,
    );
    publish(
        &directory.join("cell.json"),
        &serde_json::to_vec_pretty(&output)?,
    )?;
    admission?;
    result.map(|_| ())
}
