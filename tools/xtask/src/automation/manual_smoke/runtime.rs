use super::super::{command_interrupt::Interrupt, private_state::PrivateState};
use super::{http, text};
use crate::{
    command::DynResult,
    process::{
        self, Completion, Limits, OutputFiles, ProbeDecision, ProcessSpec, Readiness, Value,
        retained::{Action, Context, Coordinator, Launch, MemberId, MemberState},
    },
    repository::check_args::Grammar,
};
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    time::Duration,
};
const GRAMMAR: Grammar = Grammar {
    usage: "cargo xtool automation manual-smoke run --binary PATH --fixture PATH --model-path PATH --native-runtime-root PATH --output PATH [--draft-path PATH] [--mmproj-path PATH] [--api-port N] [--console-port N] [--max-wait N]",
    values: &[
        "--binary",
        "--fixture",
        "--model-path",
        "--native-runtime-root",
        "--output",
        "--draft-path",
        "--mmproj-path",
        "--api-port",
        "--console-port",
        "--max-wait",
    ],
    flags: &["--help"],
};
struct Owner {
    launch: Option<Launch>,
    runtime: tokio::runtime::Runtime,
    result: Option<tokio::task::JoinHandle<Result<serde_json::Value, String>>>,
    evidence: Option<Result<serde_json::Value, String>>,
}
impl Owner {
    fn poll(&mut self) {
        self.runtime.block_on(async {
            tokio::time::sleep(Duration::from_millis(5)).await;
        });
        if self
            .result
            .as_ref()
            .is_some_and(tokio::task::JoinHandle::is_finished)
        {
            let task = self.result.take().expect("finished task");
            self.evidence = Some(
                self.runtime
                    .block_on(task)
                    .map_err(|_| "smoke task failed".to_owned())
                    .and_then(|value| value),
            );
        }
    }
    fn stop_http(&mut self) {
        if let Some(task) = self.result.take() {
            task.abort();
            let _ = self.runtime.block_on(task);
        }
    }
}
impl Coordinator for Owner {
    type Rejection = String;
    fn line(&mut self, _: MemberId, _: process::ObservedLine<'_>) -> ProbeDecision<String> {
        ProbeDecision::Pending
    }
    fn tick(&mut self, context: Context<'_>) -> Action<String> {
        if let Some(launch) = self.launch.take() {
            return Action::Start(launch);
        }
        self.poll();
        match self.evidence.as_ref() {
            Some(Err(error)) => Action::Reject(error.clone()),
            Some(Ok(_)) => match context.members.first().map(|member| member.state) {
                Some(MemberState::Starting) => Action::Admit(MemberId::Seed),
                Some(MemberState::Ready { .. }) => Action::Stop(MemberId::Seed),
                Some(MemberState::IntentionalStop) => Action::Complete,
                _ => Action::Pending,
            },
            None => Action::Pending,
        }
    }
}
fn regular(path: &str) -> DynResult<PathBuf> {
    let path = Path::new(path).canonicalize()?;
    if !path.is_file() {
        return Err("manual smoke binary/model must resolve to a regular file".into());
    }
    Ok(path)
}
fn rewrite(
    fixture: &Path,
    model: &Path,
    draft: &Path,
    projector: Option<&Path>,
) -> DynResult<String> {
    let mut config: toml::Value = toml::from_str(&text(fixture)?)?;
    fn replace(value: &mut toml::Value, substitutions: &[(&str, &Path)]) -> DynResult<()> {
        match value {
            toml::Value::String(value) => {
                for (placeholder, path) in substitutions {
                    if value.contains(placeholder) {
                        *value = value.replace(
                            placeholder,
                            path.to_str().ok_or("model path must be Unicode")?,
                        );
                    }
                }
            }
            toml::Value::Array(values) => {
                for value in values {
                    replace(value, substitutions)?;
                }
            }
            toml::Value::Table(values) => {
                for (_, value) in values.iter_mut() {
                    replace(value, substitutions)?;
                }
            }
            _ => (),
        }
        Ok(())
    }
    let mut values = vec![
        ("__LOCAL_MODEL_PATH__", model),
        ("__LOCAL_INCOMPATIBLE_DRAFT_PATH__", draft),
    ];
    if let Some(projector) = projector {
        values.push(("__LOCAL_MMPROJ_PATH__", projector));
    }
    replace(&mut config, &values)?;
    Ok(toml::to_string(&config)?)
}
pub(super) fn run(args: &[String]) -> DynResult<()> {
    let parsed = GRAMMAR.parse(args).map_err(|error| format!("{error:?}"))?;
    if parsed.flag("--help") {
        println!("{}", GRAMMAR.usage);
        return Ok(());
    }
    if !parsed.positionals.is_empty() {
        return Err("manual smoke accepts named arguments".into());
    }
    let required = |name| parsed.last(name).ok_or_else(|| format!("missing {name}"));
    let binary = regular(required("--binary")?)?;
    let fixture = Path::new(required("--fixture")?).canonicalize()?;
    let model = regular(required("--model-path")?)?;
    let draft = parsed
        .last("--draft-path")
        .map(regular)
        .transpose()?
        .unwrap_or_else(|| model.clone());
    let projector = parsed.last("--mmproj-path").map(regular).transpose()?;
    let native = Path::new(required("--native-runtime-root")?).canonicalize()?;
    if !native.is_dir() {
        return Err("native runtime root must be directory".into());
    }
    let output = Path::new(required("--output")?);
    let parent = output
        .parent()
        .ok_or("output parent absent")?
        .canonicalize()?;
    let output = parent.join(output.file_name().ok_or("output leaf absent")?);

    let api: u16 = parsed.last("--api-port").unwrap_or("9437").parse()?;
    let console: u16 = parsed.last("--console-port").unwrap_or("3231").parse()?;
    let wait: u64 = parsed.last("--max-wait").unwrap_or("180").parse()?;
    if api == 0 || console == 0 || api == console || !(1..=3600).contains(&wait) {
        return Err("ports/wait refused".into());
    }
    // Refuse a currently occupied endpoint before launching a second host.
    let leases = [
        std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, api))?,
        std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, console))?,
    ];
    drop(leases);
    // Owned fresh evidence directory: prior receipts are never replaced.
    std::fs::create_dir(&output)?;
    let state = PrivateState::create(&parent, "manual-smoke")?;
    let result = execute(
        &state,
        &output,
        (
            &binary,
            &fixture,
            &model,
            &draft,
            projector.as_deref(),
            &native,
        ),
        (api, console, wait),
    );
    state
        .finish(result)
        .map_err(|error| format!("private smoke state finalization failed: {error:?}").into())
}
fn execute(
    state: &PrivateState,
    output: &Path,
    paths: (&Path, &Path, &Path, &Path, Option<&Path>, &Path),
    ports: (u16, u16, u64),
) -> DynResult<()> {
    let (binary, fixture, model, draft, projector, native) = paths;
    let (api, console, wait) = ports;
    state.prepare()?;
    let config = rewrite(fixture, model, draft, projector)?;
    let config_path = state.root().join("fixture.toml");
    std::fs::write(&config_path, &config)?;
    std::fs::write(output.join("applied-config.toml"), &config)?;
    let args: Vec<_> = [
        "serve".to_owned(),
        "--config".into(),
        config_path.to_str().ok_or("config path Unicode")?.into(),
        "--headless".into(),
        "--port".into(),
        api.to_string(),
        "--console".into(),
        console.to_string(),
    ]
    .into_iter()
    .map(|arg| Value::Public(arg.into()))
    .collect();
    let budget = Duration::from_secs(wait + 151);
    let launch = Launch {
        member: MemberId::Seed,
        spec: ProcessSpec {
            executable: binary.into(),
            arguments: args,
            cwd: state.root().into(),
            environment: state.environment(native),
        },
        files: OutputFiles {
            stdout: Some(output.join("stdout.log")),
            stderr: Some(output.join("stderr.log")),
        },
        readiness_deadline: budget,
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let task = runtime.spawn(http::smoke(api, console, Duration::from_secs(wait)));
    let mut owner = Owner {
        launch: Some(launch),
        runtime,
        result: Some(task),
        evidence: None,
    };
    let limits = Limits {
        execution: budget,
        graceful_shutdown: Duration::from_secs(10),
        forced_shutdown: Duration::from_secs(5),
        retained_bytes_per_stream: 1024 * 1024,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let interrupt = Interrupt::install()?;
    let result = process::retained::run(&mut owner, &limits, &interrupt.cancellation());
    owner.stop_http();
    let interruption = interrupt.finish();
    let report = result.map_err(|error| format!("retained smoke session: {error}"));
    let success = interruption.is_ok()
        && report.as_ref().is_ok_and(|report| {
            report.outcome == process::Outcome::Ready
                && report.failure.is_none()
                && report.rejection.is_none()
                && report.members.len() == 1
                && report.members.iter().all(|member| {
                    member.process.failure.is_none()
                        && member.process.status.is_some()
                        && member.process.cleanup.complete
                        && !member.process.cleanup.forced
                        && !member.process.cleanup.graceful_signal_failed
                        && member.process.cleanup.failure.is_none()
                })
        });
    let evidence = json!({"schema_version":1,"status":if success{"PASS"}else{"FAILED"},"fixture":fixture,"applied_config":output.join("applied-config.toml"),"http":owner.evidence,"process":retained_evidence(&report),"interruption":format!("{interruption:?}"),"runtime_logs_are_sanitized_diagnostics":true,"qualification":"this invocation only; not all matrix rows"});
    std::fs::write(
        output.join("receipt.json"),
        serde_json::to_vec_pretty(&evidence)?,
    )?;
    state.retain_runtime_logs(&output.join("runtime-logs"))?;
    println!(
        "COMMAND: {} serve --config {} --headless --port {api} --console {console}",
        binary.display(),
        config_path.display()
    );
    println!(
        "FIXTURE: {}\nTEMP_CONFIG: {}\nLOG_PATH: {}\nEVIDENCE: {}",
        fixture.display(),
        config_path.display(),
        output.join("stdout.log").display(),
        output.join("receipt.json").display()
    );
    if let Some(Ok(http)) = owner.evidence.as_ref() {
        for key in ["STATUS_JSON", "MODELS_JSON", "CHAT_JSON"] {
            println!("{key}: {}", serde_json::to_string_pretty(&http[key])?);
        }
    }
    println!("LOG_LINES_BEGIN");
    for name in ["stdout.log", "stderr.log"] {
        let logs = text(&output.join(name))?;
        let lines: Vec<_> = logs.lines().collect();
        for line in lines.iter().skip(lines.len().saturating_sub(80)) {
            println!("{line}");
        }
    }
    println!("LOG_LINES_END");
    if success {
        Ok(())
    } else {
        Err("runtime smoke failed; inspect retained receipt and logs".into())
    }
}

fn retained_evidence(
    report: &Result<process::retained::Report<String>, String>,
) -> serde_json::Value {
    match report {
        Ok(report) => json!({
            "outcome": format!("{:?}", report.outcome),
            "failure": report.failure.as_ref().map(ToString::to_string),
            "rejection": report.rejection,
            "members": report.members.iter().map(|member| json!({
                "member": format!("{:?}", member.member),
                "disposition": member.disposition.label(),
                "process": format!("{:?}", member.process),
            })).collect::<Vec<_>>(),
        }),
        Err(error) => json!({"failure": error}),
    }
}
