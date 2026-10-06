//! Existing process supervisor owns every hash/readiness/measurement/host child.
use super::{
    admission::{Receipt, hash},
    contract::{Host, Input},
    owner::Owner,
};
use crate::process::retained::{Launch, MemberId};
use crate::{
    command::DynResult,
    process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Arg,
    },
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
fn remaining(deadline: Instant, reserve: Duration) -> DynResult<Duration> {
    let left = deadline
        .saturating_duration_since(Instant::now())
        .saturating_sub(reserve);
    if left.is_zero() {
        return Err("cache cell cannot reserve owned cleanup before deadline".into());
    }
    Ok(left)
}
fn limits(execution: Duration) -> Limits {
    Limits {
        execution,
        graceful_shutdown: Duration::from_secs(1),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}
fn clean(report: &process::ProcessReport) -> bool {
    report.failure.is_none()
        && report.cleanup.complete
        && !report.cleanup.forced
        && !report.cleanup.graceful_signal_failed
        && report.cleanup.failure.is_none()
        && [&report.stdout, &report.stderr]
            .iter()
            .all(|s| s.line_capture_complete && !s.truncated && s.oversized_lines == 0)
}
fn worker(
    tool: &Path,
    directory: &Path,
    verb: &str,
    input: PathBuf,
    output: PathBuf,
) -> ProcessSpec {
    let mut args: Vec<OsString> = vec!["automation".into()];
    if verb == "measure" {
        args.push("cache-family-measure".into());
    } else {
        args.extend(["cache-family-cell".into(), verb.into()]);
    }
    args.extend([
        "--input".into(),
        input.into_os_string(),
        "--output".into(),
        output.into_os_string(),
    ]);
    let environment = ["PATH", "SYSTEMROOT", "WINDIR"]
        .into_iter()
        .filter_map(|k| std::env::var_os(k).map(|v| (k.into(), Arg::Public(v))))
        .collect();
    ProcessSpec {
        executable: tool.into(),
        arguments: args.into_iter().map(Arg::Public).collect(),
        cwd: directory.into(),
        environment,
    }
}
fn admit(
    tool: &Path,
    input: &Input,
    directory: &Path,
    name: &str,
    deadline: Instant,
    cancel: &Cancellation,
) -> DynResult<Receipt> {
    let bytes = serde_json::to_vec(input)?;
    let request = directory.join(format!("{name}-input.json"));
    let output = directory.join(format!("{name}-receipt.json"));
    crate::automation::waiting_prefix::adaptive_identity::fresh(&request, &bytes)?;
    let grace = if input.artifact.is_some() { 6 } else { 1 };
    let mut admission_limits = limits(remaining(deadline, Duration::from_secs(grace + 2))?);
    admission_limits.graceful_shutdown = Duration::from_secs(grace);
    let report = process::supervise(
        &worker(tool, directory, "admit-worker", request, output.clone()),
        &admission_limits,
        cancel,
        OutputFiles {
            stdout: Some(directory.join(format!("{name}.stdout.log"))),
            stderr: Some(directory.join(format!("{name}.stderr.log"))),
        },
    )?;
    if !report.success() || !clean(&report) {
        return Err("cache cell supervised identity failed".into());
    }
    let receipt: Receipt = serde_json::from_slice(
        &crate::automation::waiting_prefix::adaptive_identity::bounded(&output, 1024 * 1024)?,
    )?;
    if receipt.schema_version != 1 || receipt.request_sha256 != hash(&bytes) {
        return Err("cache cell identity receipt correlation refused".into());
    }
    receipt.admitted.validate()?;
    let mut expected = input.clone();
    expected.binary = expected.binary.canonicalize()?;
    expected.model = if let Some(artifact) = &mut expected.artifact {
        artifact.tool = artifact.tool.canonicalize()?;
        expected
            .model
            .parent()
            .ok_or("shard parent")?
            .canonicalize()?
            .join(expected.model.file_name().ok_or("primary name")?)
    } else {
        expected.model.canonicalize()?
    };
    expected.native_build = expected.native_build.canonicalize()?;
    for toolkit in expected.toolkit_directories.values_mut() {
        toolkit.path = toolkit.path.canonicalize()?;
    }
    if serde_json::to_value(&receipt.admitted)? != serde_json::to_value(&expected)? {
        return Err("cache cell identity changes declared profile".into());
    }
    Ok(receipt)
}
fn host_arguments(input: &Input, directory: &Path) -> Vec<Arg> {
    let args: Vec<OsString> = if input.host == Host::NativeBaseline {
        vec![
            "--model".into(),
            input.model.clone().into_os_string(),
            "--ctx-size".into(),
            input.ctx_size.to_string().into(),
            "--n-gpu-layers".into(),
            input.n_gpu_layers.to_string().into(),
            "--host".into(),
            "127.0.0.1".into(),
            "--port".into(),
            input.port.to_string().into(),
            "--parallel".into(),
            input.lane_count.to_string().into(),
            "--no-webui".into(),
        ]
    } else {
        vec![
            "serve-openai".into(),
            "--config".into(),
            directory.join("stage.json").into_os_string(),
            "--bind-addr".into(),
            format!("127.0.0.1:{}", input.port).into(),
            "--generation-concurrency".into(),
            input.lane_count.to_string().into(),
            "--telemetry-level".into(),
            "debug".into(),
        ]
    };
    args.into_iter().map(Arg::Public).collect()
}
fn signal(report: &process::ProcessReport) -> Option<i32> {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt as _;
        report
            .status
            .as_ref()
            .and_then(std::process::ExitStatus::signal)
    }
    #[cfg(not(unix))]
    {
        let _ = report;
        None
    }
}
fn projection(report: &process::retained::Report<String>) -> Value {
    json!({"outcome":format!("{:?}",report.outcome),"rejection":report.rejection,"members":report.members.iter().map(|m|json!({
        "member":String::from_utf8_lossy(m.member.name()),"disposition":m.disposition.label(),"status":m.process.status.as_ref().and_then(std::process::ExitStatus::code),
        "clean":clean(&m.process),"signal":signal(&m.process),"admitted_seconds":m.admitted.map(|d|d.as_secs_f64()),"forced":m.process.cleanup.forced,
        "stdout_suppressed":m.process.stdout.suppressed_lines,"stderr_suppressed":m.process.stderr.suppressed_lines,
        "stdout_bytes_seen":m.process.stdout.bytes_seen,"stderr_bytes_seen":m.process.stderr.bytes_seen})).collect::<Vec<_>>()})
}
fn launch(
    member: MemberId,
    spec: ProcessSpec,
    directory: &Path,
    name: &str,
    deadline: Duration,
) -> Launch {
    Launch {
        member,
        spec,
        files: OutputFiles {
            stdout: Some(directory.join(format!("{name}.stdout.log"))),
            stderr: Some(directory.join(format!("{name}.stderr.log"))),
        },
        readiness_deadline: deadline,
    }
}
pub(super) fn execute(input: &Input, directory: &Path, cancel: &Cancellation) -> DynResult<Value> {
    let until = Instant::now() + Duration::from_secs(input.execution_timeout_secs);
    let tool = std::env::current_exe()?;
    let before = admit(&tool, input, directory, "before", until, cancel)?;
    let input = before.admitted.clone();
    crate::automation::waiting_prefix::adaptive_identity::fresh(
        &directory.join("stage.json"),
        &serde_json::to_vec_pretty(&input.config())?,
    )?;
    // Bind refusal precedes launch. This reservation is released immediately before
    // the owned host starts; the subsequent owned post-bind marker and HTTP gate
    // are still required, never replaced by the reservation alone.
    let reservation = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, input.port))?;
    let state = crate::automation::private_state::PrivateState::create(
        &std::env::temp_dir(),
        "cache-cell",
    )?;
    let result = (|| -> DynResult<Value> {
        state.prepare()?;
        let mut environment = state.environment(&input.native_build);
        for (k, v) in &input.environment {
            environment.insert(k.into(), Arg::Public(v.into()));
        }
        for (name, toolkit) in &input.toolkit_directories {
            environment.insert(
                name.into(),
                Arg::Public(toolkit.path.clone().into_os_string()),
            );
        }
        environment.insert(
            "LLAMA_STAGE_BUILD_DIR".into(),
            Arg::Public(input.native_build.clone().into()),
        );
        environment.insert(
            "SKIPPY_NATIVE_MTP_GREEDY_SAMPLING_FASTPATH".into(),
            Arg::Public("1".into()),
        );
        environment.insert("SKIPPY_TELEMETRY_STDERR".into(), Arg::Public("1".into()));
        let effective = environment
            .iter()
            .map(|(k, v)| {
                (
                    k.to_string_lossy().into_owned(),
                    match v {
                        Arg::Public(v) => v.to_string_lossy().into_owned(),
                        Arg::Secret(_) => "redacted".into(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let execution = remaining(until, Duration::from_secs(9))?;
        let admitted = bound_measurement(&input, execution)?;
        let readiness_request = directory.join("readiness-input.json");
        let readiness_bytes = serde_json::to_vec(&admitted)?;
        crate::automation::waiting_prefix::adaptive_identity::fresh(
            &readiness_request,
            &readiness_bytes,
        )?;
        let measurement_request = directory.join("measurement-input.json");
        let measurement_bytes = if admitted.worker_sweep.is_empty() {
            serde_json::to_vec(&admitted.worker)?
        } else {
            serde_json::to_vec(
                &json!({"schema_version":1,"stages":admitted.worker_sweep,"execution_timeout_ms":admitted.worker.execution_timeout_ms}),
            )?
        };
        crate::automation::waiting_prefix::adaptive_identity::fresh(
            &measurement_request,
            &measurement_bytes,
        )?;
        let mut owner = Owner {
            server: Some(launch(
                MemberId::Seed,
                ProcessSpec {
                    executable: input.binary.clone(),
                    arguments: host_arguments(&input, directory),
                    cwd: directory.into(),
                    environment,
                },
                directory,
                "host",
                Duration::from_secs(input.startup_timeout_secs).min(execution),
            )),
            readiness: Some(launch(
                MemberId::WorkerOne,
                worker(
                    &tool,
                    directory,
                    "readiness-worker",
                    readiness_request,
                    directory.join("readiness.json"),
                ),
                directory,
                "readiness",
                execution,
            )),
            measurement: Some(launch(
                MemberId::WorkerTwo,
                worker(
                    &tool,
                    directory,
                    "measure",
                    measurement_request,
                    directory.join("measurement.json"),
                ),
                directory,
                "measurement",
                execution,
            )),
            input: admitted,
            marker: false,
            stopping: false,
        };
        drop(reservation);
        let report = process::retained::run(&mut owner, &limits(execution), cancel)?;
        let lifecycle = projection(&report);
        crate::automation::waiting_prefix::adaptive_identity::fresh(
            &directory.join("lifecycle.json"),
            &serde_json::to_vec_pretty(&lifecycle)?,
        )?;
        let mut measurement = None;
        let mut readiness = None;
        for (name, expected, target) in [
            (
                "measurement.json",
                hash(&measurement_bytes),
                &mut measurement,
            ),
            ("readiness.json", hash(&readiness_bytes), &mut readiness),
        ] {
            if let Ok(bytes) = crate::automation::waiting_prefix::adaptive_identity::bounded(
                &directory.join(name),
                64 * 1024 * 1024,
            ) && let Ok(value) = serde_json::from_slice::<Value>(&bytes)
                && value["schema_version"] == 1
                && value["request_sha256"] == expected
            {
                *target = Some(value);
            }
        }
        let owned_clean = report.recovery_success()
            && report.members.len() == 3
            && report.members.iter().all(|m| clean(&m.process))
            && [MemberId::WorkerOne, MemberId::WorkerTwo].iter().all(|id| {
                report.members.iter().any(|m| {
                    m.member == *id
                        && m.process
                            .status
                            .as_ref()
                            .and_then(std::process::ExitStatus::code)
                            == Some(0)
                })
            });
        let after = admit(&tool, &input, directory, "after", until, cancel);
        let identities_match = after.as_ref().is_ok_and(|value| {
            value.model_identity == before.model_identity
                && serde_json::to_value(&value.admitted).ok() == serde_json::to_value(&input).ok()
        });
        let completed = owned_clean
            && identities_match
            && readiness.as_ref().is_some_and(|r| r["ready"] == true)
            && measurement
                .as_ref()
                .is_some_and(|r| r["status"] == "completed")
            && !cancel.is_cancelled()
            && Instant::now() < until;
        Ok(
            json!({"schema_version":1,"status":if completed{"completed"}else{"incomplete"},"host":input.host,
            "before_identity":before,"after_identity_error":after.as_ref().err().map(|_|"owned post-run identity refusal"),"after_identity":after.ok(),"source_commit_provenance":"caller-declared-not-build-attested",
            "effective_environment":effective,"lifecycle":lifecycle,"readiness":readiness,"measurement":measurement}),
        )
    })();
    state.finish(result).map_err(|error| {
        format!("cache cell private-state cleanup/preceding failure: {error:?}").into()
    })
}

/// Keep readiness primary correlation while bounding every stage by this cell's
/// remaining absolute allowance; the sweep owner enforces the same total budget.
pub(super) fn bound_measurement(input: &Input, execution: Duration) -> DynResult<Input> {
    input.validate()?;
    let limit = u64::try_from(execution.as_millis())?;
    let mut admitted = input.clone();
    admitted.worker.execution_timeout_ms = admitted.worker.execution_timeout_ms.min(limit);
    for stage in &mut admitted.worker_sweep {
        stage.execution_timeout_ms = stage.execution_timeout_ms.min(limit);
    }
    admitted.validate()?;
    Ok(admitted)
}
