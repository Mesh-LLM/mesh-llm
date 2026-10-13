use super::{Source, battery, budget, http, limits};
use crate::automation::{private_state::PrivateState, retained_session};
use crate::command::DynResult;
use crate::process::retained::{
    Action, Context, Coordinator, ExpectedExit, Launch, MemberId, MemberState,
};
use crate::process::{ObservedLine, ProbeDecision, ProcessSpec, Value};
use crate::repository::check_args::Grammar;
use crate::repository::check_report::CheckReport;
use std::io::{Read, Seek, SeekFrom};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation laya product --mesh-binary PATH --model PATH --device NAME [--startup-timeout SECONDS] [--read-timeout SECONDS] [--json-out PATH]",
    values: &[
        "--mesh-binary",
        "--model",
        "--device",
        "--startup-timeout",
        "--read-timeout",
        "--json-out",
    ],
    flags: &["--help"],
};

pub(super) fn run(root: &Path, args: &[String]) -> DynResult<()> {
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
    let binary = Path::new(
        parsed
            .last("--mesh-binary")
            .ok_or("missing --mesh-binary")?,
    )
    .canonicalize()?;
    let model = Path::new(parsed.last("--model").ok_or("missing --model")?).canonicalize()?;
    let device = parsed.last("--device").ok_or("missing --device")?;
    let startup_text = parsed.last("--startup-timeout").unwrap_or("300");
    let read_text = parsed.last("--read-timeout").unwrap_or("300");
    let startup = budget(startup_text)?;
    let read = budget(read_text)?;
    let execution = startup
        .checked_add(read.saturating_mul(8))
        .ok_or("Laya budget overflow")?;
    if execution > Duration::from_secs(86400) {
        return Err("Laya total budget exceeds one day".into());
    }
    let state = PrivateState::create(&std::env::temp_dir(), "laya-smoke")?;
    state.prepare()?;
    let api = TcpListener::bind(("127.0.0.1", 0))?;
    let console = TcpListener::bind(("127.0.0.1", 0))?;
    let port = api.local_addr()?.port();
    let console_port = console.local_addr()?.port();
    let native = binary
        .parent()
        .ok_or("binary has no parent")?
        .join("native-runtimes");
    let mut environment = state.environment(&native);
    environment.insert(
        "MESH_LLM_NATIVE_RUNTIME_MANIFEST_URL".into(),
        Value::Public("http://127.0.0.1:9/native-runtimes.json".into()),
    );
    if device == "Vulkan0" {
        environment.insert(
            "MESH_LLM_VULKAN_AVAILABLE".into(),
            Value::Public("1".into()),
        );
    }
    let arguments = [
        "--log-format".into(),
        "json".into(),
        "serve".into(),
        "--gguf".into(),
        model.into_os_string(),
        "--no-draft".into(),
        "--device".into(),
        device.into(),
        "--ctx-size".into(),
        "1024".into(),
        "--port".into(),
        port.to_string().into(),
        "--console".into(),
        console_port.to_string().into(),
        "--headless".into(),
    ];
    let server = Launch {
        member: MemberId::Seed,
        spec: ProcessSpec {
            executable: binary,
            arguments: arguments.into_iter().map(Value::Public).collect(),
            cwd: root.canonicalize()?,
            environment,
        },
        files: state.output_files(),
        readiness_deadline: execution,
    };
    let log_files = state.output_files();
    let mut worker_arguments = vec![
        "automation".into(),
        "laya".into(),
        "product-worker".into(),
        port.to_string(),
        startup_text.into(),
        read_text.into(),
    ];
    if let Some(output) = parsed.last("--json-out") {
        worker_arguments.push(std::path::absolute(output)?.to_string_lossy().into_owned());
    }
    let worker = Launch {
        member: MemberId::WorkerOne,
        spec: ProcessSpec {
            executable: std::env::current_exe()?,
            arguments: worker_arguments
                .into_iter()
                .map(|value| Value::Public(value.into()))
                .collect(),
            cwd: root.canonicalize()?,
            environment: std::env::vars_os()
                .map(|(key, value)| (key, Value::Public(value)))
                .collect(),
        },
        files: Default::default(),
        readiness_deadline: execution,
    };
    drop((api, console));
    let result = (|| -> DynResult<()> {
        let mut owner = Owner {
            server: Some(server),
            worker: Some(worker),
            policy: ExpectedExit::new(&[0], execution)?,
            stopping: false,
        };
        let report = retained_session::run(&mut owner, &limits(execution))?;
        if !report.recovery_success() {
            let mut diagnostics = String::new();
            for path in [log_files.stdout.as_ref(), log_files.stderr.as_ref()]
                .into_iter()
                .flatten()
            {
                let mut log = std::fs::File::open(path)?;
                let length = log.metadata()?.len();
                log.seek(SeekFrom::Start(length.saturating_sub(12_000)))?;
                let mut bytes = Vec::new();
                log.take(12_000).read_to_end(&mut bytes)?;
                diagnostics.push_str(&String::from_utf8_lossy(&bytes));
            }
            return Err(format!(
                "Laya product smoke failed: {:?}; failure={:?}; members={:?}\n{diagnostics}",
                report.outcome, report.failure, report.members
            )
            .into());
        }
        Ok(())
    })();
    match state.finish(result) {
        Ok(()) => CheckReport::success("Laya product smoke passed\n".into()).emit(),
        Err(error) => Err(format!("Laya product smoke: {error:?}").into()),
    }
}

struct Owner {
    server: Option<Launch>,
    worker: Option<Launch>,
    policy: ExpectedExit,
    stopping: bool,
}
impl Coordinator for Owner {
    type Rejection = String;
    fn line(&mut self, _: MemberId, _: ObservedLine<'_>) -> ProbeDecision<String> {
        ProbeDecision::Pending
    }
    fn tick(&mut self, context: Context<'_>) -> Action<String> {
        if self.stopping {
            return Action::Complete;
        }
        if let Some(launch) = self.server.take() {
            return Action::Start(launch);
        }
        if context.members.iter().any(|member| {
            member.member == MemberId::Seed && matches!(member.state, MemberState::Starting)
        }) {
            return Action::Admit(MemberId::Seed);
        }
        if let Some(launch) = self.worker.take() {
            return Action::StartExpected {
                launch,
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

pub(super) fn worker(root: &Path, args: &[String]) -> DynResult<()> {
    #[derive(serde::Deserialize)]
    struct Models {
        data: Vec<Model>,
    }
    #[derive(serde::Deserialize)]
    struct Model {
        id: String,
    }
    let [port, startup, read, output @ ..] = args else {
        return Err("invalid product worker arguments".into());
    };
    let port: u16 = port.parse()?;
    let base_url = format!("http://127.0.0.1:{port}");
    let deadline = Instant::now() + budget(startup)?;
    let model = loop {
        if Instant::now() >= deadline {
            return Err("model startup deadline exceeded".into());
        }
        if let Ok(bytes) = http::request(
            &format!("{base_url}/v1/models"),
            None,
            Duration::from_secs(2),
        ) && let Ok(models) = serde_json::from_slice::<Models>(&bytes)
            && let Some(model) = models.data.into_iter().next()
            && !model.id.is_empty()
        {
            break model.id;
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    let output = output.first().map(PathBuf::from);
    if battery(
        root,
        &root.join("ci/llama-canary/fixtures/laya-golden"),
        &Source::Http { base_url, model },
        budget(read)?,
        output.as_deref(),
    )? {
        Ok(())
    } else {
        Err("Laya golden battery failed".into())
    }
}
