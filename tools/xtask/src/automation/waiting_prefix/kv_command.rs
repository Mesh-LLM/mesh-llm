//! Two owned default-startup sessions over one private durable/cache state directory.
use super::{
    adaptive_identity as io, kv_identity, kv_manifest, kv_metadata, kv_owner::Owner, kv_report,
    kv_worker, options,
};
use crate::process::retained::{ExpectedExit, Launch, MemberId};
use crate::{
    command::DynResult,
    process::{self, Cancellation, Value as Arg},
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    net::TcpListener,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub schema_version: u64,
    pub binary: PathBuf,
    pub model: PathBuf,
    pub turns: u32,
    pub turn_target_tokens: u32,
    pub system_tokens: u32,
    pub restore_repeats: u32,
    pub max_output_tokens: u64,
    pub request_timeout_secs: f64,
    pub ready_timeout_secs: u64,
    pub worker_timeout_secs: u64,
    pub timeout_secs: u64,
    #[serde(default)]
    pub serve_extra_args: Vec<String>,
}
impl Input {
    pub fn validate(&self) -> DynResult<()> {
        kv_manifest::build(self.turns, self.turn_target_tokens, self.system_tokens)?;
        server_args(&self.model, &self.serve_extra_args)?;
        let required = self
            .worker_timeout_secs
            .checked_mul(2)
            .and_then(|v| v.checked_add(20))
            .ok_or("restart total budget overflow")?;
        if self.schema_version != 1
            || !self.binary.is_absolute()
            || !self.model.is_absolute()
            || !(1..=128).contains(&self.restore_repeats)
            || !(1..=4096).contains(&self.max_output_tokens)
            || !self.request_timeout_secs.is_finite()
            || self.request_timeout_secs <= 0.0
            || self.request_timeout_secs > 86400.0
            || self.ready_timeout_secs == 0
            || self.ready_timeout_secs >= self.worker_timeout_secs
            || self.worker_timeout_secs < 2
            || self.timeout_secs < required
            || self.timeout_secs > 86400
        {
            return Err("restart identities, cohort counts or owned overall budget invalid".into());
        }
        Ok(())
    }
}
pub(super) fn server_args(model: &Path, extras: &[String]) -> DynResult<Vec<Arg>> {
    const FORBIDDEN: &[&str] = &[
        "--ctx-size",
        "--generation-concurrency",
        "--generation-queue-capacity",
        "--host",
        "--max-vram",
        "--parallel",
        "--port",
        "--model",
        "--log-format",
        "--bind-ip",
        "--bind-port",
        "--listen-all",
        "--config",
        "--kv-cache-disk-dir",
        "--owner-key",
        "--bin-dir",
        "--native-serving-plugin",
        "--native-serving-plugin-config",
        "--native-serving-plugin-state",
        "--plugin",
        "--plugin-arg",
        "--gguf",
        "--mmproj",
        "--draft",
        "--join",
        "--join-file",
        "--relay-auth",
    ];
    if extras.len() > 64
        || extras.iter().any(|v| {
            v.starts_with("-j")
                || v.is_empty()
                || v.len() > 4096
                || FORBIDDEN
                    .iter()
                    .any(|flag| v == flag || v.starts_with(&format!("{flag}=")))
        })
    {
        return Err("restart default startup refuses endpoint/tuning/identity overrides".into());
    }
    Ok(vec![
        Arg::Public("serve".into()),
        Arg::Public("--model".into()),
        Arg::Public(model.as_os_str().into()),
        Arg::Public("--log-format".into()),
        Arg::Public("json".into()),
    ]
    .into_iter()
    .chain(extras.iter().map(|v| Arg::Public(v.into())))
    .collect())
}
pub(super) fn reserve_default() -> DynResult<TcpListener> {
    let socket = tokio::net::TcpSocket::new_v4()?;
    // Unix reuseaddr permits reuse after TIME_WAIT, never an already-listening endpoint.
    #[cfg(not(windows))]
    socket.set_reuseaddr(true)?;
    socket.bind("127.0.0.1:9337".parse()?)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async { Ok(socket.listen(1)?.into_std()?) })
}
fn clean(p: &process::ProcessReport) -> bool {
    p.failure.is_none()
        && p.cleanup.complete
        && !p.cleanup.forced
        && !p.cleanup.graceful_signal_failed
        && p.cleanup.failure.is_none()
        && p.stdout.line_capture_complete
        && p.stderr.line_capture_complete
}
fn identity(
    input: &Input,
    directory: &Path,
    until: Instant,
    cancel: &Cancellation,
) -> DynResult<Value> {
    std::fs::create_dir(directory)?;
    let request = kv_identity::Input {
        schema_version: 1,
        binary: input.binary.clone(),
        model: input.model.clone(),
    };
    let bytes = serde_json::to_vec(&request)?;
    io::fresh(&directory.join("input.json"), &bytes)?;
    let execution = until
        .saturating_duration_since(Instant::now())
        .saturating_sub(Duration::from_millis(750));
    if execution.is_zero() {
        return Err("restart identity cannot reserve owned cleanup".into());
    }
    let p = process::supervise(
        &process::ProcessSpec {
            executable: std::env::current_exe()?,
            arguments: vec![
                Arg::Public("automation".into()),
                Arg::Public("waiting-prefix".into()),
                Arg::Public("kv-restart-identity".into()),
                Arg::Public("--input".into()),
                Arg::Public(directory.join("input.json").into_os_string()),
                Arg::Public("--output".into()),
                Arg::Public(directory.join("receipt.json").into_os_string()),
            ],
            cwd: directory.into(),
            environment: BTreeMap::new(),
        },
        &process::Limits {
            execution,
            graceful_shutdown: Duration::from_millis(250),
            forced_shutdown: Duration::from_millis(250),
            retained_bytes_per_stream: 65536,
            readiness: process::Readiness::None,
            completion: process::Completion::Exit,
        },
        cancel,
        process::OutputFiles {
            stdout: Some(directory.join("stdout.log")),
            stderr: Some(directory.join("stderr.log")),
        },
    )?;
    if p.outcome != process::Outcome::Exited
        || p.status.as_ref().and_then(std::process::ExitStatus::code) != Some(0)
        || !clean(&p)
    {
        return Err("restart identity worker failed".into());
    }
    let receipt: Value =
        serde_json::from_slice(&io::bounded(&directory.join("receipt.json"), 128 * 1024)?)?;
    if receipt["schema_version"] != 1 || receipt["request_sha256"] != io::digest(&bytes) {
        return Err("restart identity receipt correlation refused".into());
    }
    Ok(receipt)
}
fn life(report: &process::retained::Report<String>) -> Value {
    json!({"outcome":format!("{:?}",report.outcome),"members":report.members.iter().map(|m|json!({"member":String::from_utf8_lossy(m.member.name()),"disposition":m.disposition.label(),"status":m.process.status.as_ref().and_then(std::process::ExitStatus::code),"cleanup_complete":m.process.cleanup.complete,"forced":m.process.cleanup.forced,"graceful_signal_failed":m.process.cleanup.graceful_signal_failed,"cleanup_failure":m.process.cleanup.failure.as_ref().map(|e|format!("{e:?}")),"process_failure":m.process.failure.as_ref().map(|e|format!("{e:?}")),"stdout_complete":m.process.stdout.line_capture_complete,"stderr_complete":m.process.stderr.line_capture_complete})).collect::<Vec<_>>()})
}
struct SessionBudget<'a> {
    until: Instant,
    cancel: &'a Cancellation,
}
fn session(
    input: &Input,
    worker: &kv_worker::Input,
    state: &crate::automation::private_state::PrivateState,
    directory: &Path,
    budget: SessionBudget<'_>,
    port: TcpListener,
    run: &mut Value,
) -> DynResult<Value> {
    let SessionBudget { until, cancel } = budget;
    std::fs::create_dir(directory)?;
    worker.validate()?;
    let bytes = serde_json::to_vec(worker)?;
    io::fresh(&directory.join("worker-input.json"), &bytes)?;
    let execution = Duration::from_secs(input.worker_timeout_secs + 1);
    if until.saturating_duration_since(Instant::now()) < execution + Duration::from_secs(6) {
        return Err("restart session cannot reserve two-member cleanup".into());
    }
    let mut environment = state.environment(
        &input
            .binary
            .parent()
            .ok_or("restart binary parent absent")?
            .join("native-runtimes"),
    );
    environment.remove(std::ffi::OsStr::new("MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR"));
    let server = Launch {
        member: MemberId::Seed,
        spec: process::ProcessSpec {
            executable: input.binary.clone(),
            arguments: server_args(&input.model, &input.serve_extra_args)?,
            cwd: directory.into(),
            environment: environment.clone(),
        },
        files: process::OutputFiles {
            stdout: Some(directory.join("server.stdout.log")),
            stderr: Some(directory.join("server.stderr.log")),
        },
        readiness_deadline: execution,
    };
    let client = Launch {
        member: MemberId::WorkerOne,
        spec: process::ProcessSpec {
            executable: std::env::current_exe()?,
            arguments: vec![
                Arg::Public("automation".into()),
                Arg::Public("waiting-prefix".into()),
                Arg::Public("kv-restart-worker".into()),
                Arg::Public("--input".into()),
                Arg::Public(directory.join("worker-input.json").into_os_string()),
                Arg::Public("--output".into()),
                Arg::Public(directory.join("requests.json").into_os_string()),
            ],
            cwd: directory.into(),
            environment,
        },
        files: process::OutputFiles {
            stdout: Some(directory.join("worker.stdout.log")),
            stderr: Some(directory.join("worker.stderr.log")),
        },
        readiness_deadline: execution,
    };
    let mut owner = Owner {
        server: Some(server),
        worker: Some(client),
        policy: ExpectedExit::new(&[0, 1], execution)?,
        stopped: false,
        started: Instant::now(),
        request_sha256: io::digest(&bytes),
        ready_observed: None,
        ready_ambiguous: false,
    };
    drop(port);
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
    let lifecycle = life(&report);
    io::fresh(
        &directory.join("lifecycle.json"),
        &serde_json::to_vec_pretty(&lifecycle)?,
    )?;
    run["sessions"].as_array_mut().unwrap().push(lifecycle);
    let receipt_bytes = io::bounded(&directory.join("requests.json"), 16 * 1024 * 1024);
    if receipt_bytes.is_err() {
        // Only typed, request-correlated journal rows survive a forcibly interrupted worker.
        for index in 0..128 {
            let path = directory.join(format!("request-{index}.json"));
            match std::fs::symlink_metadata(&path) {
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => break,
                Err(e) => return Err(e.into()),
                Ok(_) => {}
            }
            let record: Value = serde_json::from_slice(&io::bounded(&path, 65536)?)?;
            if record["schema_version"] != 1
                || record["request_sha256"] != io::digest(&bytes)
                || record["manifest_sha256"] != worker.manifest_sha256
                || record["phase"] != serde_json::to_value(worker.phase)?
            {
                return Err("restart partial journal correlation refused".into());
            }
            run["requests"]
                .as_array_mut()
                .unwrap()
                .push(record["row"].clone());
        }
        return Err("restart worker final receipt absent; correlated journal rows retained".into());
    }
    let mut receipt: Value = serde_json::from_slice(&receipt_bytes?)?;
    if receipt["schema_version"] != 1
        || receipt["request_sha256"] != io::digest(&bytes)
        || receipt["manifest_sha256"] != worker.manifest_sha256
        || receipt["phase"] != serde_json::to_value(worker.phase)?
    {
        return Err("restart phase receipt schema/correlation differs".into());
    }
    run["requests"].as_array_mut().unwrap().extend(
        receipt["requests"]
            .as_array()
            .ok_or("restart request roster absent")?
            .iter()
            .cloned(),
    );
    if report.members.len() != 2
        || !report.recovery_success()
        || !report.members.iter().all(|m| clean(&m.process))
        || !report.members.iter().any(|m| {
            m.member == MemberId::WorkerOne
                && m.process
                    .status
                    .as_ref()
                    .and_then(std::process::ExitStatus::code)
                    == Some(0)
        })
        || !receipt["error"].is_null()
    {
        return Err("restart serving/client session failed or final shutdown incomplete; partial rows retained".into());
    }
    if owner.ready_ambiguous || owner.ready_observed.is_none() {
        return Err("restart worker readiness observation missing/ambiguous".into());
    }
    receipt["server_to_observed_model_ready_seconds"] =
        json!(owner.ready_observed.unwrap().as_secs_f64());
    let closed = reserve_default()?;
    drop(closed);
    Ok(receipt)
}
fn check_identity(before: &Value, after: &Value) -> DynResult<()> {
    for key in [
        "binary",
        "model",
        "model_metadata",
        "adjacent_runtime_root",
        "runtime_policy",
    ] {
        if before[key] != after[key] {
            return Err("restart file identity changed across sessions".into());
        }
    }
    Ok(())
}
fn execute(
    input: &Input,
    directory: &Path,
    until: Instant,
    cancel: &Cancellation,
    port: TcpListener,
) -> Value {
    let started_at = super::kv_terminal::wall_seconds();
    let mut run = json!({"started_at_unix_seconds":started_at,"completed_at_unix_seconds":null,"schema_version":1,"kind":"kv-restart-replay/run","requests":[],"sessions":[],"error":null,"binary":{"source_sha":"unknown","git_describe":"unknown"},"model":null,"hardware":null,"manifest":null,"manifest_sha256":null,"config":{"base_url":kv_worker::BASE,"turns":input.turns,"turn_target_tokens":input.turn_target_tokens,"system_tokens":input.system_tokens,"restore_repeats":input.restore_repeats,"max_output_tokens":input.max_output_tokens,"request_timeout_secs":input.request_timeout_secs,"ready_timeout_secs":input.ready_timeout_secs,"serve_extra_args":input.serve_extra_args,"endpoint_profile":"default-startup-no-endpoint-or-tuning-overrides"}});
    let result = (|| -> DynResult<()> {
        let first = identity(input, &directory.join("identity-before"), until, cancel)?;
        let mut admitted = input.clone();
        admitted.binary = PathBuf::from(
            first["binary"]["path"]
                .as_str()
                .ok_or("admitted binary path absent")?,
        );
        admitted.model = PathBuf::from(
            first["model"]["path"]
                .as_str()
                .ok_or("admitted model path absent")?,
        );
        run["binary"] = first["binary"].clone();
        run["model"] = first["model"].clone();
        run["hardware"] = first["hardware"].clone();
        run["runtime_policy"] = first["runtime_policy"].clone();
        run["adjacent_runtime_root"] = first["adjacent_runtime_root"].clone();
        let source = kv_metadata::checkout(
            Path::new("/usr/bin/git"),
            &std::env::current_dir()?,
            until,
            cancel,
        )?;
        run["binary"]
            .as_object_mut()
            .unwrap()
            .extend(source.as_object().unwrap().clone());
        if std::env::consts::OS == "macos" {
            let host =
                kv_metadata::darwin(Path::new("/usr/sbin/sysctl"), directory, until, cancel)?;
            run["hardware"]
                .as_object_mut()
                .unwrap()
                .extend(host.as_object().unwrap().clone());
        }
        let manifest =
            kv_manifest::build(input.turns, input.turn_target_tokens, input.system_tokens)?;
        let digest = io::digest(&serde_json::to_vec(&manifest)?);
        run["manifest"] = serde_json::to_value(&manifest)?;
        run["manifest_sha256"] = json!(digest);
        let state = crate::automation::private_state::PrivateState::create(
            &std::env::temp_dir(),
            "kv-restart",
        )?;
        let result = (|| -> DynResult<()> {
            state.prepare()?;
            let mut worker = kv_worker::Input {
                schema_version: 1,
                phase: kv_worker::Phase::Fill,
                manifest,
                manifest_sha256: digest,
                expected_model: None,
                baseline_sha256: None,
                restore_repeats: input.restore_repeats,
                max_output_tokens: input.max_output_tokens,
                request_timeout_secs: input.request_timeout_secs,
                ready_timeout_secs: input.ready_timeout_secs,
                timeout_secs: input.worker_timeout_secs,
            };
            let fill = session(
                &admitted,
                &worker,
                &state,
                &directory.join("fill"),
                SessionBudget { until, cancel },
                port,
                &mut run,
            )?;

            let reservation = reserve_default()?;
            let next = identity(input, &directory.join("identity-restart"), until, cancel)?;
            check_identity(&first, &next)?;
            worker.phase = kv_worker::Phase::Replay;
            worker.expected_model = Some(
                fill["model_id"]
                    .as_str()
                    .ok_or("restart fill model identity absent")?
                    .into(),
            );
            worker.baseline_sha256 = Some(
                fill["baseline_sha256"]
                    .as_str()
                    .ok_or("restart fill baseline absent")?
                    .into(),
            );
            let replay = session(
                &admitted,
                &worker,
                &state,
                &directory.join("replay"),
                SessionBudget { until, cancel },
                reservation,
                &mut run,
            )?;
            run["restart"] = json!({"method":"owned clean process stop and fresh serve on identical private HOME/config/cache/runtime paths","restart_to_ready_seconds":replay["server_to_observed_model_ready_seconds"],"readiness_scope":"retained session start to correlated worker model-ready marker observation; includes launch/poll latency","ready_seconds":replay["ready_seconds"],"baseline_sha256":worker.baseline_sha256,"model_id":worker.expected_model});
            let last = identity(input, &directory.join("identity-after"), until, cancel)?;
            check_identity(&first, &last)?;
            Ok(())
        })();
        let result = match state.retain_runtime_logs(&directory.join("runtime-logs")) {
            Ok(()) => result,
            Err(e) => Err(format!("restart native log retention failed: {e}").into()),
        };
        state
            .finish(result)
            .map_err(|e| format!("restart private-state cleanup/result: {e:?}"))?;
        Ok(())
    })();
    if let Err(error) = result {
        run["error"] = json!(error.to_string().chars().take(1024).collect::<String>());
    }
    run["cohorts"] = json!([
        kv_report::cohort("fill", run["requests"].as_array().unwrap()),
        kv_report::cohort("restore", run["requests"].as_array().unwrap()),
        kv_report::cohort("warm", run["requests"].as_array().unwrap())
    ]);
    run
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let flags = options(
        args,
        &["--input", "--output-directory"],
        &["--input", "--output-directory"],
    )?;
    let input: Input =
        serde_json::from_slice(&io::bounded(Path::new(flags["--input"]), 1024 * 1024)?)?;
    input.validate()?;
    let directory = std::path::absolute(flags["--output-directory"])?;
    match std::fs::symlink_metadata(&directory) {
        Ok(meta) if meta.is_dir() => {
            if std::fs::read_dir(&directory)?.next().is_some() {
                return Err("restart output directory is not empty".into());
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        _ => {
            return Err("restart output directory must be fresh or empty regular directory".into());
        }
    };
    // No child launch or output/private-state mutation happens before this occupancy guard.
    let port = reserve_default().map_err(|e| {
        format!("restart default TCP9337 occupied/unavailable; existing listener untouched: {e}")
    })?;
    std::fs::create_dir_all(&directory)?;
    let directory = directory.canonicalize()?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancellation = interrupt.cancellation();
    let deadline = Instant::now() + Duration::from_secs(input.timeout_secs);
    let mut value = execute(&input, &directory, deadline, &cancellation, port);
    let finished: DynResult<()> = interrupt.finish().map_err(|e| e.to_string().into());
    value["completed_at_unix_seconds"] = json!(super::kv_terminal::wall_seconds());
    let terminal = super::kv_terminal::finalize(&mut value, finished, &cancellation, deadline);
    let publication = (|| -> DynResult<()> {
        let rows = value["requests"]
            .as_array()
            .unwrap()
            .iter()
            .map(serde_json::to_string)
            .collect::<Result<Vec<_>, _>>()?
            .join("\n");
        io::fresh(
            &directory.join("requests.jsonl"),
            format!("{rows}\n").as_bytes(),
        )?;
        io::fresh(
            &directory.join("run.json"),
            &serde_json::to_vec_pretty(&value)?,
        )?;
        io::fresh(
            &directory.join("report.md"),
            kv_report::render(&value).as_bytes(),
        )?;
        Ok(())
    })();
    publication?;
    terminal
}
