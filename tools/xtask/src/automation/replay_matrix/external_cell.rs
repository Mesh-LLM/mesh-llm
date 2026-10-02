use crate::automation::retained_session;
use crate::command::DynResult;
use crate::process::retained::{
    Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState,
};
use crate::process::{ObservedLine, ProbeDecision, ProcessSpec, Value};
use serde::Deserialize;
use std::{
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Deserialize)]
pub(super) struct Input {
    pub build: super::external_probe::Verified,
    pub workload: PathBuf,
    pub requests_output: PathBuf,
    pub summary_output: PathBuf,
    pub server_log: PathBuf,
    pub startup_timeout_seconds: u64,
    pub timeout_seconds: u64,
}

pub(in crate::automation) fn run(root: Option<&Path>, args: &[String]) -> DynResult<()> {
    use crate::repository::{check_args::Grammar, check_report::CheckReport};
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix external-cell --input PATH",
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
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let input: Input = serde_json::from_slice(&std::fs::read(
        parsed.last("--input").ok_or("missing --input")?,
    )?)?;
    execute(root, &input)
}

pub(super) fn execute(root: Option<&Path>, input: &Input) -> DynResult<()> {
    let started = std::time::Instant::now();
    let root = crate::repository::RepositoryRoot::resolve(root)?;
    let workload: super::cell_execution::Workload =
        serde_json::from_slice(&std::fs::read(&input.workload)?)?;
    admit(input, &workload)?;
    let endpoint: hyper::Uri = workload.base_url.parse()?;
    if endpoint.scheme_str() != Some("http")
        || endpoint.host() != Some("127.0.0.1")
        || endpoint.path() != "/v1"
        || endpoint.query().is_some()
    {
        return Err("external cell requires http://127.0.0.1:<port>/v1".into());
    }
    let port = endpoint
        .port_u16()
        .ok_or("external cell requires explicit API port")?;
    let reservation = std::net::TcpListener::bind(("127.0.0.1", port))?;
    let total = Duration::from_secs(input.timeout_seconds);
    let fresh = super::external_probe::verify_with_budget(
        &input.build.external_engine,
        super::external_probe::VERSION_TIMEOUT.min(total.saturating_sub(started.elapsed())),
    )?;
    if fresh.version_sha256 != input.build.version_sha256
        || fresh.provenance.resolved_executable != input.build.provenance.resolved_executable
    {
        return Err("external engine version or executable changed before launch".into());
    }
    let command = super::external_command::server(
        &fresh.external_engine,
        &fresh.provenance.resolved_executable,
        port,
    )?;
    super::pass_artifacts::prepare(
        &workload,
        (&input.requests_output, &input.summary_output),
        &input.server_log,
    )?;
    crate::command::write_json_file(&input.server_log.with_extension("command.json"), &command)?;
    let execution = total.saturating_sub(started.elapsed());
    if execution.is_zero() {
        return Err("external cell execution deadline exceeded during version admission".into());
    }
    let server = Launch {
        member: MemberId::Seed,
        spec: ProcessSpec {
            executable: fresh.provenance.resolved_executable,
            arguments: command
                .into_iter()
                .skip(1)
                .map(|arg| Value::Public(arg.into()))
                .collect(),
            cwd: fresh.worktree,
            environment: super::external_probe::environment(),
        },
        files: crate::process::OutputFiles {
            stdout: Some(input.server_log.clone()),
            stderr: Some(input.server_log.with_extension("stderr.log")),
        },
        readiness_deadline: execution,
    };
    let worker = worker(input, root.as_path(), execution)?;
    std::thread::scope(|scope| -> DynResult<()> {
        let forwarder = super::progress::Forwarder::new(scope);
        let mut owner = Owner {
            progress: &forwarder,
            server: Some(server),
            worker: Some(worker),
            stop: false,
            policy: ExpectedExit::new(&[0], execution)?,
        };
        drop(reservation);
        let session =
            retained_session::run(&mut owner, &super::server_cell_worker::limits(execution));
        let output = forwarder.finish();
        let report = session?;
        output?;
        super::pass_lifecycle::retain(
            &report,
            input
                .server_log
                .parent()
                .ok_or("missing external log parent")?,
        )?;
        if !report.recovery_success() {
            return Err(format!(
                "external replay cell failed: {:?}; members={:?}",
                report.outcome, report.members
            )
            .into());
        }
        Ok(())
    })
}

fn admit(input: &Input, workload: &super::cell_execution::Workload) -> DynResult<()> {
    if !(1..=86400).contains(&input.timeout_seconds)
        || input.startup_timeout_seconds == 0
        || input.startup_timeout_seconds >= input.timeout_seconds
        || [
            &input.workload,
            &input.requests_output,
            &input.summary_output,
            &input.server_log,
        ]
        .iter()
        .any(|path| !path.is_absolute())
    {
        return Err(
            "external cell requires absolute paths and bounded startup/execution deadlines".into(),
        );
    }
    for cell in
        std::iter::once(workload).chain(workload.following_cells.iter().map(|cell| &cell.workload))
    {
        if cell.runtime_context.is_some()
            || cell.model_pin.is_some()
            || cell.eligibility.is_some()
            || cell.minimum_recurrent_restored_tokens.is_some()
            || cell.measured_prefix.is_some()
            || cell.qualification_probe
        {
            return Err("runtime context qualification currently requires mesh arms".into());
        }
        if cell.concurrency > input.build.external_engine.max_concurrency {
            return Err("cell concurrency exceeds external arm capacity".into());
        }
    }
    Ok(())
}

fn worker(input: &Input, root: &Path, execution: Duration) -> DynResult<Launch> {
    let mut arguments: Vec<std::ffi::OsString> = vec![
        "automation".into(),
        "replay-matrix".into(),
        "server-cell-worker".into(),
        input.startup_timeout_seconds.to_string().into(),
        "0".into(),
    ];
    arguments.extend(
        [
            &input.workload,
            &input.requests_output,
            &input.summary_output,
        ]
        .map(|path| path.as_os_str().to_owned()),
    );
    Ok(Launch {
        member: MemberId::WorkerOne,
        spec: ProcessSpec {
            executable: std::env::current_exe()?,
            arguments: arguments.into_iter().map(Value::Public).collect(),
            cwd: root.into(),
            environment: super::external_probe::environment(),
        },
        files: Default::default(),
        readiness_deadline: execution,
    })
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
