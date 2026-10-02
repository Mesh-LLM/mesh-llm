use crate::automation::{private_state::PrivateState, retained_session};
use crate::command::DynResult;
use crate::process::retained::{
    Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState,
};
use crate::process::{ObservedLine, ProbeDecision, ProcessSpec, Value};
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use std::{net::TcpListener, path::Path, time::Duration};

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation replay-matrix server-cell --binary PATH --native-runtime-root PATH --model PATH --workload PATH --requests-output PATH --summary-output PATH --server-log PATH [--timeout SECONDS] [--startup-timeout SECONDS]",
    values: &[
        "--binary",
        "--native-runtime-root",
        "--model",
        "--hf-home",
        "--workload",
        "--requests-output",
        "--summary-output",
        "--server-log",
        "--timeout",
        "--startup-timeout",
    ],
    flags: &["--help"],
};

pub(in crate::automation) fn run(root: Option<&Path>, args: &[String]) -> DynResult<()> {
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let root = crate::repository::RepositoryRoot::resolve(root)?;
    let binary = Path::new(parsed.last("--binary").ok_or("missing --binary")?).canonicalize()?;
    let native = Path::new(
        parsed
            .last("--native-runtime-root")
            .ok_or("missing --native-runtime-root")?,
    )
    .canonicalize()?;
    let model = parsed.last("--model").ok_or("missing --model")?;
    let hf_home = parsed
        .last("--hf-home")
        .map(std::path::absolute)
        .transpose()?;
    let workload_path = parsed.last("--workload").ok_or("missing --workload")?;
    let workload: super::cell_execution::Workload =
        serde_json::from_slice(&std::fs::read(workload_path)?)?;
    let requests = std::path::absolute(
        parsed
            .last("--requests-output")
            .ok_or("missing --requests-output")?,
    )?;
    let summary = std::path::absolute(
        parsed
            .last("--summary-output")
            .ok_or("missing --summary-output")?,
    )?;
    let log = std::path::absolute(parsed.last("--server-log").ok_or("missing --server-log")?)?;
    super::pass_artifacts::prepare(&workload, (&requests, &summary), &log)?;
    if let Some(pin) = &workload.model_pin {
        if !pin.output.is_absolute() {
            return Err("model identity output must be absolute".into());
        }
        let identity = super::model_preflight::verify(
            Path::new(model),
            &pin.sha256,
            pin.minimum_context_tokens,
        )?;
        crate::command::write_json_file(&pin.output, &identity)?;
    } else if workload.runtime_context.is_some() {
        return Err("runtime-context qualification requires a model pin".into());
    }
    let endpoint: hyper::Uri = workload.base_url.parse()?;
    if endpoint.scheme_str() != Some("http")
        || endpoint.host() != Some("127.0.0.1")
        || endpoint.path() != "/v1"
        || endpoint.query().is_some()
    {
        return Err("server-cell requires http://127.0.0.1:<port>/v1".into());
    }
    let port = endpoint
        .port_u16()
        .ok_or("server-cell requires an explicit API port")?;
    let api_reservation = TcpListener::bind(("127.0.0.1", port))?;
    let console_reservation = TcpListener::bind(("127.0.0.1", 0))?;
    let console_port = console_reservation.local_addr()?.port();
    let execution = seconds(parsed.last("--timeout").unwrap_or("21600"))?;
    let startup = seconds(parsed.last("--startup-timeout").unwrap_or("1800"))?;
    if startup >= execution {
        return Err("startup deadline must precede execution deadline".into());
    }
    let state = PrivateState::create(&std::env::temp_dir(), "replay-server")?;
    state.prepare()?;
    let mut environment = state.environment(&native);
    environment.insert("SKIPPY_TELEMETRY_STDERR".into(), Value::Public("1".into()));
    if let Some(home) = hf_home
        .map(std::path::PathBuf::into_os_string)
        .or_else(|| std::env::var_os("HF_HOME"))
    {
        environment.insert("HF_HOME".into(), Value::Public(home));
    }
    let server = Launch {
        member: MemberId::Seed,
        spec: ProcessSpec {
            executable: binary,
            arguments: [
                "serve",
                "--model",
                model,
                "--log-format",
                "json",
                "--port",
                &port.to_string(),
                "--console",
                &console_port.to_string(),
            ]
            .into_iter()
            .map(|value| Value::Public(value.into()))
            .collect(),
            cwd: root.as_path().to_path_buf(),
            environment,
        },
        files: crate::process::OutputFiles {
            stdout: Some(log.clone()),
            stderr: Some(log.with_extension("stderr.log")),
        },
        readiness_deadline: execution,
    };
    let mut worker_args: Vec<std::ffi::OsString> = vec![
        "automation".into(),
        "replay-matrix".into(),
        "server-cell-worker".into(),
        startup.as_secs().to_string().into(),
        console_port.to_string().into(),
    ];
    for name in ["--workload", "--requests-output", "--summary-output"] {
        worker_args.push(
            std::path::absolute(parsed.last(name).ok_or_else(|| format!("missing {name}"))?)?
                .into_os_string(),
        );
    }
    let worker = Launch {
        member: MemberId::WorkerOne,
        spec: ProcessSpec {
            executable: std::env::current_exe()?,
            arguments: worker_args.into_iter().map(Value::Public).collect(),
            cwd: root.as_path().to_path_buf(),
            environment: state.environment(&native),
        },
        files: Default::default(),
        readiness_deadline: execution,
    };
    let result = std::thread::scope(|scope| -> DynResult<()> {
        let forwarder = super::progress::Forwarder::new(scope);
        let mut owner = Owner {
            progress: &forwarder,
            server: Some(server),
            worker: Some(worker),
            stop: false,
            policy: ExpectedExit::new(&[0], execution)?,
        };
        drop((api_reservation, console_reservation));
        let session =
            retained_session::run(&mut owner, &super::server_cell_worker::limits(execution));
        let output = forwarder.finish();
        let report = session?;
        output?;
        super::pass_lifecycle::retain(&report, log.parent().ok_or("missing log directory")?)?;
        if !report.recovery_success() {
            return Err(format!(
                "replay server/cell failed: {:?}; members={:?}",
                report.outcome, report.members
            )
            .into());
        }
        Ok(())
    });
    let retained = state.retain_runtime_logs(
        &log.parent()
            .ok_or("missing log directory")?
            .join("native-runtime"),
    );
    let result = match (result, retained) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) => Err(error),
        (Ok(()), Err(error)) => Err(error.into()),
        (Err(prior), Err(error)) => {
            Err(format!("{prior}; retaining runtime logs failed: {error}").into())
        }
    };
    let qualification = if workload.minimum_recurrent_restored_tokens.is_some()
        || workload
            .following_cells
            .iter()
            .any(|cell| cell.workload.minimum_recurrent_restored_tokens.is_some())
    {
        super::pass_recurrent::qualify(&workload, (&requests, &summary), &log)
    } else {
        Ok(())
    };
    let result = match (result, qualification) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
        (Err(prior), Err(error)) => Err(format!("{prior}; {error}").into()),
    };
    match state.finish(result) {
        Ok(()) => Ok(()),
        Err(error) => Err(format!("replay lifecycle: {error:?}").into()),
    }
}

fn seconds(text: &str) -> DynResult<Duration> {
    let value: u64 = text.parse()?;
    if !(1..=86400).contains(&value) {
        return Err("deadline must be in 1..=86400 seconds".into());
    }
    Ok(Duration::from_secs(value))
}

struct Owner<'a, 'scope> {
    progress: &'a super::progress::Forwarder<'scope>,
    server: Option<Launch>,
    worker: Option<Launch>,
    stop: bool,
    policy: ExpectedExit,
}
impl Coordinator for Owner<'_, '_> {
    type Rejection = String;
    fn line(&mut self, member: MemberId, line: ObservedLine<'_>) -> ProbeDecision<String> {
        if member == MemberId::WorkerOne
            && let Err(error) = self.progress.line(line.bytes)
        {
            return ProbeDecision::Rejected(error);
        }
        ProbeDecision::Pending
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
