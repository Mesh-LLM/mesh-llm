//! Bounded native arm preflight, pair launch, serial worker and typed evidence admission.
use super::{
    adaptive_identity as identity, adaptive_owner::Owner, adaptive_telemetry::Observation, options,
    publish, sequential_cell,
};
use crate::process::retained::{ExpectedExit, Launch, MemberId};
use crate::{
    command::DynResult,
    process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Arg,
    },
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    net::TcpListener,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u64,
    pub arm: identity::Input,
    pub worker: Value,
    pub timeout_secs: u64,
    pub startup_timeout_secs: u64,
}
pub(super) fn clean(report: &process::ProcessReport) -> bool {
    report.failure.is_none()
        && report.cleanup.complete
        && !report.cleanup.forced
        && !report.cleanup.graceful_signal_failed
        && report.cleanup.failure.is_none()
}
pub(super) fn remaining(until: Instant, reserve: Duration) -> DynResult<Duration> {
    let budget = until
        .saturating_duration_since(Instant::now())
        .saturating_sub(reserve);
    if budget.is_zero() {
        Err("adaptive overall deadline cannot reserve owned cleanup".into())
    } else {
        Ok(budget)
    }
}
fn preflight_step(
    tool: &Path,
    args: Vec<std::ffi::OsString>,
    directory: &Path,
    name: &str,
    until: Instant,
    cancel: &Cancellation,
) -> DynResult<()> {
    let limits = Limits {
        execution: remaining(until, Duration::from_millis(750))?,
        graceful_shutdown: Duration::from_millis(250),
        forced_shutdown: Duration::from_millis(250),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let environment = ["PATH", "SYSTEMROOT", "WINDIR"]
        .into_iter()
        .filter_map(|k| std::env::var_os(k).map(|v| (k.into(), Arg::Public(v))))
        .collect();
    let report = process::supervise(
        &ProcessSpec {
            executable: tool.into(),
            arguments: args.into_iter().map(Arg::Public).collect(),
            cwd: directory.into(),
            environment,
        },
        &limits,
        cancel,
        OutputFiles {
            stdout: Some(directory.join(format!("{name}.stdout.log"))),
            stderr: Some(directory.join(format!("{name}.stderr.log"))),
        },
    )?;
    if !report.success()
        || !clean(&report)
        || report.stdout.truncated
        || report.stderr.truncated
        || report.stdout.suppressed_lines > 0
        || report.stderr.suppressed_lines > 0
    {
        return Err(format!("adaptive preflight {name} failed or lost diagnostics").into());
    }
    Ok(())
}
pub(super) fn admitted(
    tool: &Path,
    input: &identity::Input,
    directory: &Path,
    until: Instant,
    cancel: &Cancellation,
) -> DynResult<Value> {
    let bytes = serde_json::to_vec(input)?;
    let request = directory.join("identity-input.json");
    let receipt = directory.join("identity.json");
    identity::fresh(&request, &bytes)?;
    preflight_step(
        tool,
        vec![
            "automation".into(),
            "waiting-prefix".into(),
            "adaptive-identity-worker".into(),
            "--input".into(),
            request.into_os_string(),
            "--output".into(),
            receipt.clone().into_os_string(),
        ],
        directory,
        "identity",
        until,
        cancel,
    )?;
    let value: Value = serde_json::from_slice(&identity::bounded(&receipt, 128 * 1024)?)?;
    if value["schema_version"] != 1 || value["request_sha256"] != identity::digest(&bytes) {
        return Err("adaptive identity receipt correlation refused".into());
    }
    let arm: identity::Input = serde_json::from_value(value["admitted"].clone())?;
    arm.validate()?;
    if value["configs"] != json!([arm.config(0), arm.config(1)]) {
        return Err("adaptive identity/config receipt differs from admitted arm".into());
    }
    Ok(value)
}
pub(super) fn arguments(arm: &identity::Input, directory: &Path, index: usize) -> Vec<Arg> {
    let mut args = vec![
        "serve-binary".into(),
        "--config".into(),
        directory
            .join(format!("stage-{index}.json"))
            .into_os_string(),
        "--max-inflight".into(),
        "2".into(),
        "--reply-credit-limit".into(),
        "1".into(),
        "--async-prefill-forward".into(),
        "--telemetry-level".into(),
        "debug".into(),
    ];
    if index == 0 {
        args.extend([
            "--openai-bind-addr".into(),
            format!("127.0.0.1:{}", arm.openai_port).into(),
            "--openai-generation-concurrency".into(),
            "1".into(),
            "--openai-prefill-chunk-policy".into(),
            "adaptive-ramp".into(),
            "--openai-prefill-chunk-size".into(),
            "256".into(),
            "--openai-prefill-adaptive-start".into(),
            "128".into(),
            "--openai-prefill-adaptive-step".into(),
            "128".into(),
            "--openai-prefill-adaptive-max".into(),
            "384".into(),
        ]);
        if arm.version == super::acceptance::Version::New {
            args.extend([
                "--openai-prefill-adaptive-target-ms".into(),
                arm.adaptive_target_ms.to_string().into(),
            ]);
        }
    }
    args.into_iter().map(Arg::Public).collect()
}
fn launch(
    arm: &identity::Input,
    directory: &Path,
    index: usize,
    mut environment: BTreeMap<std::ffi::OsString, Arg>,
    execution: Duration,
) -> Launch {
    if let Some(Arg::Public(root)) = environment.get(std::ffi::OsStr::new("MESH_LLM_RUNTIME_ROOT"))
    {
        let path = PathBuf::from(root).join(format!("stage-{index}"));
        environment.insert(
            "MESH_LLM_RUNTIME_ROOT".into(),
            Arg::Public(path.into_os_string()),
        );
    }
    Launch {
        member: if index == 1 {
            MemberId::Seed
        } else {
            MemberId::WorkerOne
        },
        spec: ProcessSpec {
            executable: arm.binary.clone(),
            arguments: arguments(arm, directory, index),
            cwd: directory.into(),
            environment,
        },
        files: OutputFiles {
            stdout: Some(directory.join(format!("stage-{index}.stdout.log"))),
            stderr: Some(directory.join(format!("stage-{index}.stderr.log"))),
        },
        readiness_deadline: execution,
    }
}
fn session(
    input: &mut Input,
    directory: &Path,
    until: Instant,
    cancel: &Cancellation,
    reservations: Vec<TcpListener>,
) -> DynResult<Value> {
    let tool = std::env::current_exe()?;
    input
        .worker
        .as_object()
        .ok_or("adaptive worker input must be an object")?;
    input.worker["provenance"]
        .as_object()
        .ok_or("adaptive worker provenance must be an object")?;
    let identity = admitted(&tool, &input.arm, directory, until, cancel)?;
    let arm: identity::Input = serde_json::from_value(identity["admitted"].clone())?;
    for index in 0..2 {
        identity::fresh(
            &directory.join(format!("stage-{index}.json")),
            &serde_json::to_vec(&identity["configs"][index])?,
        )?;
    }
    let execution = remaining(until, Duration::from_secs(9))?;
    input.worker["round"] = json!(arm.round);
    input.worker["version"] = serde_json::to_value(arm.version)?;
    input.worker["model"] = json!(arm.model_id);
    input.worker["base_url"] = json!(format!("http://127.0.0.1:{}/v1", arm.openai_port));
    input.worker["readiness_timeout_secs"] = json!(input.startup_timeout_secs);
    let worker_budget = input.worker["timeout_secs"]
        .as_u64()
        .ok_or("worker timeout absent")?;
    if Duration::from_secs(worker_budget) >= execution {
        return Err("worker budget must fit retained pair execution".into());
    }
    input.worker["provenance"]["admitted_arm"] = identity["admitted"].clone();
    input.worker["provenance"]["model_identity"] = identity["model_identity"].clone();
    let typed: sequential_cell::Input = serde_json::from_value(input.worker.clone())?;
    typed.validate()?;
    let expected = identity::digest(&serde_json::to_vec(&typed)?);
    identity::fresh(
        &directory.join("worker-input.json"),
        &serde_json::to_vec(&typed)?,
    )?;
    let state = crate::automation::private_state::PrivateState::create(
        &std::env::temp_dir(),
        "adaptive-pair",
    )?;
    let result = (|| -> DynResult<Value> {
        state.prepare()?;
        for index in 0..2 {
            std::fs::create_dir(state.root().join("runtime").join(format!("stage-{index}")))?;
        }
        let mut environment = static_environment(&state, &arm.native_build);
        environment.insert(
            "LLAMA_STAGE_BUILD_DIR".into(),
            Arg::Public(arm.native_build.clone().into_os_string()),
        );
        environment.insert("SKIPPY_TELEMETRY_STDERR".into(), Arg::Public("1".into()));
        environment.insert(
            "SKIPPY_NATIVE_MTP_GREEDY_SAMPLING_FASTPATH".into(),
            Arg::Public("1".into()),
        );
        let worker = Launch {
            member: MemberId::WorkerTwo,
            spec: ProcessSpec {
                executable: tool,
                arguments: vec![
                    Arg::Public("automation".into()),
                    Arg::Public("waiting-prefix".into()),
                    Arg::Public("sequential-cell".into()),
                    Arg::Public("--input".into()),
                    Arg::Public(directory.join("worker-input.json").into_os_string()),
                    Arg::Public("--output".into()),
                    Arg::Public(directory.join("requests.json").into_os_string()),
                ],
                cwd: directory.into(),
                environment: static_environment(&state, &arm.native_build),
            },
            files: OutputFiles {
                stdout: Some(directory.join("worker.stdout.log")),
                stderr: Some(directory.join("worker.stderr.log")),
            },
            readiness_deadline: execution,
        };
        let mut owner = Owner {
            downstream: Some(launch(&arm, directory, 1, environment.clone(), execution)),
            upstream: Some(launch(&arm, directory, 0, environment, execution)),
            worker: Some(worker),
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
        let lifecycle = json!({"outcome":format!("{:?}",report.outcome),"members":report.members.iter().map(|member|json!({"member":String::from_utf8_lossy(member.member.name()),
            "disposition":member.disposition.label(),"status":member.process.status.as_ref().and_then(std::process::ExitStatus::code),
            "cleanup_complete":member.process.cleanup.complete,"forced":member.process.cleanup.forced,"graceful_signal_failed":member.process.cleanup.graceful_signal_failed,
            "cleanup_failure":member.process.cleanup.failure.as_ref().map(|e|format!("{e:?}")),"process_failure":member.process.failure.as_ref().map(|e|format!("{e:?}")),
            "stdout_complete":member.process.stdout.line_capture_complete,"stderr_complete":member.process.stderr.line_capture_complete})).collect::<Vec<_>>()});
        publish(
            &directory.join("lifecycle.json"),
            &serde_json::to_vec_pretty(&lifecycle)?,
        )?;
        publish(
            &directory.join("prefill-telemetry.json"),
            &serde_json::to_vec_pretty(
                &json!({"schema_version":1,"rows":owner.telemetry.rows,"error":owner.telemetry.error}),
            )?,
        )?;
        let complete = report.members.len() == 3
            && report.recovery_success()
            && report.members.iter().all(|m| {
                clean(&m.process)
                    && m.process.stdout.line_capture_complete
                    && m.process.stderr.line_capture_complete
            });
        let worker_ok = report
            .members
            .iter()
            .find(|m| m.member == MemberId::WorkerTwo)
            .is_some_and(|m| {
                m.process
                    .status
                    .as_ref()
                    .and_then(std::process::ExitStatus::code)
                    == Some(0)
            });
        let receipt: Value = serde_json::from_slice(&identity::bounded(
            &directory.join("requests.json"),
            16 * 1024 * 1024,
        )?)?;
        if !complete
            || !worker_ok
            || receipt["schema_version"] != 1
            || receipt["input_sha256"] != expected
            || !receipt["error"].is_null()
        {
            return Err("adaptive lifecycle or request receipt failed/unqualified".into());
        }
        let count = receipt["requests"]
            .as_array()
            .ok_or("measured request roster absent")?
            .len();
        if count
            != input.worker["manifest"]["prompts"]
                .as_array()
                .ok_or("input prompt roster absent")?
                .len()
            || receipt["successful_requests"].as_u64() != Some(count as u64)
        {
            return Err("adaptive request receipt incomplete".into());
        }
        let measured = owner.telemetry.measured(count, complete)?;
        let postflight = directory.join("postflight");
        std::fs::create_dir(&postflight)?;
        let current = admitted(&std::env::current_exe()?, &arm, &postflight, until, cancel)?;
        for key in ["admitted", "model_identity", "configs"] {
            if current[key] != identity[key] {
                return Err("adaptive arm identity changed across owned execution".into());
            }
        }
        Ok(
            json!({"schema_version":1,"status":"adaptive_cell_admitted","identity":identity,"lifecycle":lifecycle,"requests":receipt,"measured_prefill":measured,"latest_calibration":owner.telemetry.latest_calibration,"prefill_telemetry_available":true,"telemetry_correlation":"serial-owned-arm-event-order-excluding-calibration","context_qualification":"actual-gguf-native-context-and-exact-stage-config"}),
        )
    })();
    let result = match state.retain_runtime_logs(&directory.join("native-runtime")) {
        Ok(()) => result,
        Err(error) => Err(format!(
            "adaptive result {:?}; runtime log retention failed: {error}",
            result.as_ref().err().map(ToString::to_string)
        )
        .into()),
    };
    state
        .finish(result)
        .map_err(|e| format!("adaptive private-state cleanup/result: {e:?}").into())
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
    if input.schema_version != 1
        || !(10..=86400).contains(&input.timeout_secs)
        || input.startup_timeout_secs == 0
        || input.startup_timeout_secs >= input.timeout_secs
    {
        return Err("invalid adaptive cell deadlines".into());
    }
    let request_sha256 = identity::digest(&serde_json::to_vec(&input)?);
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
    let result = session(
        &mut input,
        &directory,
        until,
        &interrupt.cancellation(),
        ports,
    );
    let finished = interrupt.finish();
    let mut output = match &result {
        Ok(value) => value.clone(),
        Err(error) => {
            json!({"schema_version":1,"status":"adaptive_cell_failed","error":error.to_string()})
        }
    };
    output["request_sha256"] = json!(request_sha256);
    publish(
        &directory.join("cell.json"),
        &serde_json::to_vec_pretty(&output)?,
    )?;
    finished?;
    result.map(|_| ())
}

pub(super) fn static_environment(
    state: &crate::automation::private_state::PrivateState,
    native_build: &Path,
) -> std::collections::BTreeMap<std::ffi::OsString, Arg> {
    let mut environment = state.environment(native_build);
    environment.remove(std::ffi::OsStr::new("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR"));
    environment
}
