//! Retained original serve-openai startup with one current-exe HTTP worker per telemetry barrier.
use super::{
    adaptive_cell, adaptive_identity as io,
    radix_identity::Arm,
    radix_owner::{Next, Owner},
    radix_projection::Projection,
    radix_worker,
    radix_workload::{self, Shape},
};
use crate::{
    command::DynResult,
    process::{
        self, Cancellation, Value as Arg,
        retained::{Launch, MemberId},
    },
};
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
pub(super) fn config(arm: &Arm, shape: &Shape, warm: bool) -> Value {
    let mut value = json!({"run_id":"skippy-radix-cache-ab","topology_id":"skippy-radix-cache-ab-single-stage","model_id":arm.model_id,"model_path":arm.model,"source_model_sha256":arm.model_sha256,"stage_id":"stage-0","stage_index":0,"layer_start":0,"layer_end":arm.layer_end,"ctx_size":arm.ctx_size,"lane_count":shape.lanes,"n_gpu_layers":shape.n_gpu_layers,"load_mode":"runtime-slice","execution_contract":"","bind_addr":"127.0.0.1:0","upstream":null,"downstream":null});
    if warm {
        value["kv_cache"] = json!({"mode":"lookup-record","payload":arm.payload,"max_entries":64,"max_bytes":0,"min_tokens":64,"shared_prefix_stride_tokens":128,"shared_prefix_record_limit":4});
    }
    value
}
fn spec(
    executable: PathBuf,
    args: Vec<Arg>,
    directory: &Path,
    environment: std::collections::BTreeMap<std::ffi::OsString, Arg>,
) -> process::ProcessSpec {
    process::ProcessSpec {
        executable,
        arguments: args,
        cwd: directory.into(),
        environment,
    }
}
fn clean(report: &process::ProcessReport) -> bool {
    adaptive_cell::clean(report)
        && report.stdout.line_capture_complete
        && report.stderr.line_capture_complete
}
/// The first cohort owns initial model startup; later barriers recheck readiness.
pub(super) fn readiness_budget(index: usize, batch_secs: u64) -> u64 {
    if index == 0 {
        batch_secs.saturating_sub(1)
    } else {
        1
    }
}
#[derive(Clone, Copy)]
pub(super) struct Budget<'a> {
    pub batch_secs: u64,
    pub request_secs: f64,
    pub cell_secs: u64,
    pub until: Instant,
    pub cancel: &'a Cancellation,
}
pub(super) fn execute(
    arm: &Arm,
    shape: &Shape,
    warm: bool,
    directory: &Path,
    budget: &Budget<'_>,
) -> DynResult<Value> {
    let Budget {
        batch_secs,
        request_secs,
        cell_secs,
        until,
        cancel,
    } = *budget;
    std::fs::create_dir(directory)?;
    let plans = radix_workload::batches(shape, warm)?;
    let reservation = std::net::TcpListener::bind("127.0.0.1:0")?;
    let port = reservation.local_addr()?.port();
    let execution = Duration::from_secs(cell_secs)
        .min(adaptive_cell::remaining(until, Duration::from_secs(6))?);
    let cell_deadline = Instant::now() + execution;
    let state = crate::automation::private_state::PrivateState::create(
        &std::env::temp_dir(),
        "radix-cell",
    )?;
    let result = (|| -> DynResult<Value> {
        state.prepare()?;
        let mut env = adaptive_cell::static_environment(&state, &arm.native_build);
        for (name, value) in [
            (
                "LLAMA_STAGE_BUILD_DIR",
                Arg::Public(arm.native_build.clone().into_os_string()),
            ),
            ("SKIPPY_TELEMETRY_STDERR", Arg::Public("1".into())),
            (
                "SKIPPY_NATIVE_MTP_GREEDY_SAMPLING_FASTPATH",
                Arg::Public("1".into()),
            ),
        ] {
            env.insert(name.into(), value);
        }
        io::fresh(
            &directory.join("stage.json"),
            &serde_json::to_vec_pretty(&config(arm, shape, warm))?,
        )?;
        let mut server = Launch {
            member: MemberId::Seed,
            spec: spec(
                arm.binary.clone(),
                vec![
                    Arg::Public("serve-openai".into()),
                    Arg::Public("--config".into()),
                    Arg::Public(directory.join("stage.json").into_os_string()),
                    Arg::Public("--bind-addr".into()),
                    Arg::Public(format!("127.0.0.1:{port}").into()),
                    Arg::Public("--generation-concurrency".into()),
                    Arg::Public(shape.lanes.to_string().into()),
                    Arg::Public("--telemetry-level".into()),
                    Arg::Public("debug".into()),
                ],
                directory,
                env.clone(),
            ),
            files: process::OutputFiles {
                stdout: Some(directory.join("server.stdout.log")),
                stderr: Some(directory.join("server.stderr.log")),
            },
            readiness_deadline: execution,
        };
        let mut inputs = vec![];
        let mut batches = std::collections::VecDeque::new();
        for (index, batch) in plans.iter().enumerate() {
            let batchdir = directory.join(format!("batch-{index}"));
            std::fs::create_dir(&batchdir)?;
            let input = radix_worker::Input {
                schema_version: 1,
                batch: batch.clone(),
                model: arm.model_id.clone(),
                base_url: format!("http://127.0.0.1:{port}/v1"),
                output_tokens: shape.output_tokens,
                request_timeout_secs: request_secs,
                readiness_timeout_secs: readiness_budget(index, batch_secs),
                timeout_secs: batch_secs,
            };
            input.validate()?;
            io::fresh(&batchdir.join("input.json"), &serde_json::to_vec(&input)?)?;
            let launch = Launch {
                member: MemberId::new(&format!("batch-{index}"), 0)?,
                spec: spec(
                    std::env::current_exe()?,
                    vec![
                        Arg::Public("automation".into()),
                        Arg::Public("waiting-prefix".into()),
                        Arg::Public("radix-worker".into()),
                        Arg::Public("--input".into()),
                        Arg::Public(batchdir.join("input.json").into_os_string()),
                        Arg::Public("--output".into()),
                        Arg::Public(batchdir.join("receipt.json").into_os_string()),
                    ],
                    &batchdir,
                    env.clone(),
                ),
                files: process::OutputFiles {
                    stdout: Some(batchdir.join("stdout.log")),
                    stderr: Some(batchdir.join("stderr.log")),
                },
                readiness_deadline: execution,
            };
            inputs.push(input);
            batches.push_back(Next {
                launch,
                summaries: batch.prompts.len(),
                warmup: batch.warmup,
            });
        }
        crate::automation::skippy_cli_admission::prepare(
            &mut server.spec,
            &arm.binary_sha256,
            crate::automation::skippy_cli_admission::Role::Public,
            cell_deadline.min(until),
            cancel,
            &directory.join("cli-admission.json"),
        )?;
        let execution = cell_deadline
            .saturating_duration_since(Instant::now())
            .min(adaptive_cell::remaining(until, Duration::from_secs(6))?);
        if execution.is_zero() {
            return Err("radix CLI admission consumed cell budget".into());
        }
        if Duration::from_secs(batch_secs + 1) > execution {
            return Err(
                "radix batch cannot fit retained execution budget after CLI admission".into(),
            );
        }
        server.readiness_deadline = execution;
        for batch in &mut batches {
            batch.launch.readiness_deadline = execution;
        }
        let mut owner = Owner {
            server: Some(server),
            batches,
            active: None,
            projection: Projection::default(),
            boundaries: vec![],
            consumed: 0,
            batch_budget: Duration::from_secs(batch_secs + 1),
            telemetry_budget: Duration::from_secs(2),
            stopping: false,
        };
        drop(reservation);
        let report = process::retained::run(
            &mut owner,
            &process::Limits {
                execution,
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: process::Readiness::None,
                completion: process::Completion::Exit,
            },
            cancel,
        )?;
        let lifecycle = json!({"outcome":format!("{:?}",report.outcome),"session_failure":report.failure.as_ref().map(|error|format!("{error:?}")),"rejection":report.rejection,"members":report.members.iter().map(|m|json!({"member":String::from_utf8_lossy(m.member.name()),"status":m.process.status.as_ref().and_then(std::process::ExitStatus::code),"clean":clean(&m.process),"forced":m.process.cleanup.forced,"cleanup_failure":m.process.cleanup.failure.as_ref().map(|e|format!("{e:?}")),"process_failure":m.process.failure.as_ref().map(|e|format!("{e:?}")),"stdout_complete":m.process.stdout.line_capture_complete,"stderr_complete":m.process.stderr.line_capture_complete})).collect::<Vec<_>>()});
        io::fresh(
            &directory.join("lifecycle.json"),
            &serde_json::to_vec_pretty(&lifecycle)?,
        )?;
        io::fresh(
            &directory.join("telemetry.json"),
            &serde_json::to_vec_pretty(
                &json!({"schema_version":1,"rows":owner.projection.rows,"error":owner.projection.error,"boundaries":owner.boundaries,"suspect":owner.projection.suspect,"scope":"sole owned clients; exact summary count barrier before each next batch, including excluded warmup"}),
            )?,
        )?;
        let mut observations = vec![];
        let mut error = None;
        for (index, input) in inputs.iter().enumerate() {
            let batchdir = directory.join(format!("batch-{index}"));
            let bytes = io::bounded(&batchdir.join("receipt.json"), 4 * 1024 * 1024);
            let Ok(bytes) = bytes else {
                break;
            };
            let receipt: Value = serde_json::from_slice(&bytes)?;
            if receipt["schema_version"] != 1
                || receipt["request_sha256"] != io::digest(&serde_json::to_vec(input)?)
                || receipt["batch"] != serde_json::to_value(&input.batch)?
            {
                return Err("radix batch receipt correlation refused".into());
            }
            let requests = receipt["requests"]
                .as_array()
                .ok_or("radix request roster absent")?;
            let events = owner
                .boundaries
                .get(index)
                .and_then(|(start, end, _)| owner.projection.rows.get(*start..*end))
                .unwrap_or(&[]);
            if !input.batch.warmup {
                observations.push(json!({"scenario":input.batch.scenario,"concurrency":input.batch.concurrency,"makespan_ms":receipt["makespan_ms"],"requests":requests,"summary":super::radix_summary::summarize(requests,events)}));
            }
            if !receipt["error"].is_null() {
                error = Some("radix measured/warmup batch failed; receipt retained");
            }
        }
        let complete = report.recovery_success()
            && report.members.len() == inputs.len() + 1
            && report.members.iter().all(|m| clean(&m.process))
            && owner.boundaries.len() == inputs.len()
            && owner.projection.rows.len() == owner.consumed
            && owner.projection.error.is_none()
            && !owner.projection.suspect
            && error.is_none();
        let value = json!({"schema_version":1,"cache":if warm{"warm"}else{"cold"},"identity":arm,"config":config(arm,shape,warm),"observations":observations,"suspect_log":owner.projection.suspect,"lifecycle":lifecycle,"error":if complete{None}else{Some("radix serving/batch/telemetry/cleanup contract failed; partial evidence retained")}});
        io::fresh(
            &directory.join("cell.json"),
            &serde_json::to_vec_pretty(&value)?,
        )?;
        Ok(value)
    })();
    let result = match state.retain_runtime_logs(&directory.join("runtime-logs")) {
        Ok(()) => result,
        Err(e) => Err(e.into()),
    };
    state
        .finish(result)
        .map_err(|e| format!("radix private-state ownership/result: {e:?}").into())
}
