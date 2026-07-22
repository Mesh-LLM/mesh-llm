use std::{
    env, fs,
    net::{TcpStream, ToSocketAddrs},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use anyhow::{Context, Result, bail};
use serde::Serialize;
use serde_json::{Value, json};

use crate::cli::{
    EvalArgs, EvalCommandKind, EvalDoctorArgs, EvalId, EvalInfoArgs, EvalListArgs, EvalPack,
    EvalRunArgs, EvalSyncArgs,
};
use crate::telemetry_report::{self, BenchTelemetry};

const CORE_EVALS: [EvalId; 4] = [
    EvalId::SpeedBench,
    EvalId::TerminalBench,
    EvalId::SweBenchPro,
    EvalId::McpAtlas,
];
mod adapters;
mod doctor;
mod registry;
mod run;
mod sync;

pub fn eval_command(args: EvalArgs) -> Result<()> {
    match args.command {
        EvalCommandKind::List(args) => registry::list_evals(args),
        EvalCommandKind::Info(args) => registry::info_eval(args),
        EvalCommandKind::Sync(args) | EvalCommandKind::Install(args) => sync::sync_evals(args),
        EvalCommandKind::Doctor(args) => doctor::doctor_evals(args),
        EvalCommandKind::Run(args) => run::run_eval(args),
    }
}

#[derive(Clone, Copy)]
struct EvalDefinition {
    id: EvalId,
    name: &'static str,
    repo_url: &'static str,
    repo_ref: &'static str,
    cache_name: &'static str,
    description: &'static str,
    disk_estimate: &'static str,
    required_tools: &'static [&'static str],
    sync_notes: &'static [&'static str],
    run_notes: &'static [&'static str],
}

#[derive(Serialize)]
struct EvalView {
    id: &'static str,
    name: &'static str,
    pack: &'static str,
    repo_url: &'static str,
    repo_ref: &'static str,
    installed: bool,
    harness_dir: String,
    description: &'static str,
    disk_estimate: &'static str,
    required_tools: &'static [&'static str],
    sync_notes: &'static [&'static str],
    run_notes: &'static [&'static str],
}

#[derive(Serialize)]
struct DoctorView {
    eval_id: &'static str,
    installed: bool,
    harness_dir: String,
    checks: Vec<DoctorCheck>,
}

#[derive(Serialize)]
struct DoctorCheck {
    name: String,
    ok: bool,
    detail: String,
}

#[derive(Serialize)]
struct RunReport {
    run_id: String,
    eval_id: &'static str,
    model: String,
    base_url: String,
    endpoint_concurrency: usize,
    run_dir: String,
    harness_commit: Option<String>,
    dry_run: bool,
    command: String,
    exit_status: Option<i32>,
    success: bool,
    timed_out: bool,
    timeout_secs: u64,
    harness_timeout_secs: Option<u64>,
    stdout_path: Option<String>,
    stderr_path: Option<String>,
    metrics: EvalMetrics,
    telemetry: BenchTelemetry,
    artifacts: Vec<RunArtifact>,
}

#[derive(Default, Serialize)]
struct EvalMetrics {
    duration_ms: Option<f64>,
    request_count: Option<u64>,
    failed_count: Option<u64>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    total_tokens: Option<u64>,
    prompt_tok_s: Option<f64>,
    completion_tok_s: Option<f64>,
    total_tok_s: Option<f64>,
    avg_latency_ms: Option<f64>,
    draft_tokens: Option<u64>,
    draft_accepted_tokens: Option<u64>,
    draft_accept_rate: Option<f64>,
    pass_rate: Option<f64>,
}

#[derive(Serialize)]
struct RunArtifact {
    kind: &'static str,
    path: String,
}

#[derive(Clone)]
struct CommandSpec {
    program: String,
    args: Vec<String>,
    cwd: Option<PathBuf>,
    envs: Vec<(String, String)>,
    secret_envs: Vec<(String, String)>,
}

impl CommandSpec {
    fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            cwd: None,
            envs: Vec::new(),
            secret_envs: Vec::new(),
        }
    }

    fn args(mut self, args: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    fn cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.envs.push((key.into(), value.into()));
        self
    }

    fn secret_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.secret_envs.push((key.into(), value.into()));
        self
    }

    fn display(&self) -> String {
        let envs = self
            .envs
            .iter()
            .map(|(key, value)| format!("{key}={}", shell_quote(value)))
            .chain(
                self.secret_envs
                    .iter()
                    .map(|(key, _)| format!("{key}=<redacted>")),
            )
            .collect::<Vec<_>>()
            .join(" ");
        let command = std::iter::once(shell_quote(&self.program))
            .chain(self.args.iter().map(|arg| shell_quote(arg)))
            .collect::<Vec<_>>()
            .join(" ");
        let command = if envs.is_empty() {
            command
        } else {
            format!("{envs} {command}")
        };
        if let Some(cwd) = &self.cwd {
            format!(
                "(cd {} && {command})",
                shell_quote(&cwd.display().to_string())
            )
        } else {
            command
        }
    }

    fn command(&self) -> Command {
        let mut command = Command::new(&self.program);
        command.args(&self.args);
        if let Some(cwd) = &self.cwd {
            command.current_dir(cwd);
        }
        for (key, value) in &self.envs {
            command.env(key, value);
        }
        for (key, value) in &self.secret_envs {
            command.env(key, value);
        }
        command
    }
}
struct CommandOutcome {
    exit_status: Option<i32>,
    success: bool,
    timed_out: bool,
}

fn run_command_with_timeout(
    spec: &CommandSpec,
    timeout: Option<Duration>,
    stdout_path: &Path,
    stderr_path: &Path,
) -> Result<CommandOutcome> {
    let mut command = spec.command();
    configure_child_group(&mut command);
    command.stdout(Stdio::from(
        fs::File::create(stdout_path)
            .with_context(|| format!("create {}", stdout_path.display()))?,
    ));
    command.stderr(Stdio::from(
        fs::File::create(stderr_path)
            .with_context(|| format!("create {}", stderr_path.display()))?,
    ));

    let mut child = command
        .spawn()
        .with_context(|| format!("start {}", spec.program))?;
    wait_with_timeout(&mut child, timeout)
}

fn wait_with_timeout(child: &mut Child, timeout: Option<Duration>) -> Result<CommandOutcome> {
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait().context("poll harness command")? {
            return Ok(CommandOutcome {
                exit_status: status.code(),
                success: status.success(),
                timed_out: false,
            });
        }
        if timeout.is_some_and(|timeout| started.elapsed() >= timeout) {
            terminate_child(child)?;
            return Ok(CommandOutcome {
                exit_status: None,
                success: false,
                timed_out: true,
            });
        }
        thread::sleep(Duration::from_millis(250));
    }
}

fn configure_child_group(command: &mut Command) {
    #[cfg(unix)]
    {
        command.process_group(0);
    }
}

fn terminate_child(child: &mut Child) -> Result<()> {
    terminate_child_signal(child, ChildSignal::Terminate)?;
    let grace = Instant::now();
    while grace.elapsed() < Duration::from_secs(5) {
        if child
            .try_wait()
            .context("poll terminated harness command")?
            .is_some()
        {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    terminate_child_signal(child, ChildSignal::Kill)?;
    let _ = child.wait().context("wait for killed harness command")?;
    Ok(())
}

enum ChildSignal {
    Terminate,
    Kill,
}

#[cfg(unix)]
fn terminate_child_signal(child: &mut Child, signal: ChildSignal) -> Result<()> {
    let process_group = -libc::pid_t::try_from(child.id()).context("child PID overflow")?;
    let signal = match signal {
        ChildSignal::Terminate => libc::SIGTERM,
        ChildSignal::Kill => libc::SIGKILL,
    };
    // SAFETY: `kill` receives a process-group identifier and signal constant;
    // it does not dereference memory in this process.
    if unsafe { libc::kill(process_group, signal) } == -1 {
        child.kill().context("terminate harness command")?;
    }
    Ok(())
}

#[cfg(not(unix))]
fn terminate_child_signal(child: &mut Child, _signal: ChildSignal) -> Result<()> {
    child.kill().context("terminate harness command")
}

fn env_concurrency_matching_endpoint(
    var_name: &str,
    endpoint_concurrency: usize,
) -> Result<String> {
    match env::var(var_name) {
        Ok(value) => {
            let parsed = value.parse::<usize>().with_context(|| {
                format!("{var_name} must be a positive integer matching --endpoint-concurrency")
            })?;
            if parsed == 0 {
                bail!("{var_name} must be greater than zero");
            }
            if parsed != endpoint_concurrency {
                bail!(
                    "{var_name} ({parsed}) must equal --endpoint-concurrency ({endpoint_concurrency}) so native harness request concurrency matches the Skippy endpoint generation concurrency"
                );
            }
            Ok(value)
        }
        Err(env::VarError::NotPresent) => Ok(endpoint_concurrency.to_string()),
        Err(error) => Err(error).with_context(|| format!("read {var_name}")),
    }
}

fn cache_root(override_root: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(root) = override_root {
        return Ok(root);
    }
    let root = dirs::cache_dir()
        .or_else(|| dirs::home_dir().map(|home| home.join(".cache")))
        .context("cannot determine cache directory")?;
    Ok(root.join("mesh-llm").join("skippy-bench"))
}

fn absolute_path(path: PathBuf) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path);
    }
    Ok(env::current_dir()
        .context("read current directory")?
        .join(path))
}

fn harness_root(root: &Path) -> PathBuf {
    root.join("harnesses")
}

fn harness_dir(root: &Path, definition: EvalDefinition) -> PathBuf {
    harness_root(root).join(definition.cache_name)
}

fn output_dir(output_dir: Option<PathBuf>, eval_id: EvalId) -> Result<PathBuf> {
    if let Some(output_dir) = output_dir {
        return Ok(output_dir);
    }
    Ok(PathBuf::from("target")
        .join("skippy-bench")
        .join("evals")
        .join(format!("{}-{}", eval_id.as_str(), unix_millis()?)))
}

fn unix_millis() -> Result<u128> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system time is before Unix epoch")?
        .as_millis())
}

fn litellm_model_name(model: &str) -> String {
    const PROVIDER_PREFIXES: &[&str] = &[
        "openai/",
        "anthropic/",
        "azure/",
        "bedrock/",
        "gemini/",
        "hosted_vllm/",
        "ollama/",
        "openrouter/",
        "together_ai/",
        "vllm/",
    ];
    if PROVIDER_PREFIXES
        .iter()
        .any(|prefix| model.starts_with(prefix))
    {
        model.to_string()
    } else {
        format!("openai/{model}")
    }
}

fn model_label(model: &str) -> String {
    let label = model
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string();
    if label.is_empty() {
        "model".to_string()
    } else {
        label
    }
}

fn command_exists(program: &str) -> bool {
    let Some(path) = env::var_os("PATH") else {
        return false;
    };
    env::split_paths(&path).any(|dir| {
        let candidate = dir.join(program);
        candidate.is_file()
    })
}

fn print_notes(label: &str, notes: &[&str]) {
    if notes.is_empty() {
        return;
    }
    println!("{label}:");
    for note in notes {
        println!("  - {note}");
    }
}

fn shell_quote(raw: &str) -> String {
    if raw
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || "-_./:=@".contains(ch))
    {
        return raw.to_string();
    }
    format!("'{}'", raw.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::{
        adapters::{
            mcp_atlas_command, speed_bench_command, swe_bench_pro_command, terminal_bench_command,
        },
        doctor::preflight_eval_run,
        registry::{definition, selected_evals},
        run::{
            fill_client_rates, resolved_harness_commit, run_artifacts, speed_bench_metrics,
            speed_bench_output_path, speed_bench_response_timings_path, swe_bench_pro_metrics,
            swe_bench_pro_output_path, telemetry_or_unavailable, terminal_bench_metrics,
            terminal_bench_output_path,
        },
        sync::existing_repo_sync_steps,
    };

    #[test]
    fn default_pack_selects_core_evals() {
        let ids = selected_evals(&[], EvalPack::Core)
            .into_iter()
            .map(|definition| definition.id)
            .collect::<Vec<_>>();
        assert_eq!(ids, CORE_EVALS);
    }

    #[test]
    fn explicit_selection_overrides_pack() {
        let ids = selected_evals(&[EvalId::McpAtlas], EvalPack::Core)
            .into_iter()
            .map(|definition| definition.id)
            .collect::<Vec<_>>();
        assert_eq!(ids, vec![EvalId::McpAtlas]);
    }

    #[test]
    fn speed_bench_command_points_at_openai_endpoint() {
        let args = EvalRunArgs {
            eval: EvalId::SpeedBench,
            base_url: "http://127.0.0.1:9337/v1".to_string(),
            model: "tiny-local".to_string(),
            api_key: "test".to_string(),
            cache_root: None,
            output_dir: None,
            timeout_secs: 30,
            harness_timeout_secs: None,
            endpoint_concurrency: 1,
            run_id: None,
            metrics_http: "http://127.0.0.1:18080".to_string(),
            metrics_run_id: None,
            metrics_finalize_only: false,
            dry_run: true,
        };
        let root = PathBuf::from("/tmp/skippy-cache");
        let run_dir = temp_run_dir("speed-command");
        fs::create_dir_all(run_dir.join("raw")).unwrap();
        let command =
            speed_bench_command(definition(EvalId::SpeedBench), &args, &root, &run_dir).unwrap();
        assert!(command.args.contains(&"--url".to_string()));
        assert!(
            command
                .args
                .contains(&"http://127.0.0.1:9337/v1".to_string())
        );
        assert!(command.args.contains(&"tiny-local".to_string()));
        assert!(
            command.args.contains(
                &run_dir
                    .join("raw/speed-bench-auth.py")
                    .display()
                    .to_string()
            )
        );
        assert!(
            command
                .secret_envs
                .contains(&("SKIPPY_BENCH_API_KEY".to_string(), "test".to_string()))
        );
        assert!(command.envs.contains(&(
            "XDG_CACHE_HOME".to_string(),
            root.join("speed-cache/xdg").display().to_string()
        )));
        assert!(
            command
                .envs
                .contains(&("SKIPPY_BENCH_BASE_URL".to_string(), args.base_url.clone()))
        );
        assert!(
            command.envs.contains(&(
                "SKIPPY_BENCH_RESPONSE_TIMINGS_PATH".to_string(),
                speed_bench_response_timings_path(&run_dir)
                    .display()
                    .to_string(),
            ))
        );
        assert!(!command.display().contains("test"));
        assert!(
            command
                .display()
                .contains("SKIPPY_BENCH_API_KEY=<redacted>")
        );
        let launcher = fs::read_to_string(run_dir.join("raw/speed-bench-auth.py")).unwrap();
        assert!(launcher.contains("request_origin(url) == benchmark_origin"));
        assert!(launcher.contains("headers.setdefault(\"Authorization\""));
        assert!(launcher.contains("capture_response_timings"));
        assert!(launcher.contains("response.get(\"timings\")"));
        assert!(launcher.contains("safe_timings"));
        let _ = fs::remove_dir_all(run_dir);
    }

    #[test]
    fn speed_bench_reports_the_response_timing_artifact() {
        let run_dir = temp_run_dir("speed-artifacts");
        let artifacts = run_artifacts(definition(EvalId::SpeedBench), &run_dir);

        assert!(artifacts.iter().any(|artifact| {
            artifact.kind == "speed-bench-response-timings"
                && artifact.path
                    == speed_bench_response_timings_path(&run_dir)
                        .display()
                        .to_string()
        }));
        let _ = fs::remove_dir_all(run_dir);
    }

    #[test]
    fn terminal_bench_command_uses_full_dataset_without_task_filter() {
        let args = EvalRunArgs {
            eval: EvalId::TerminalBench,
            base_url: "http://127.0.0.1:9337/v1".to_string(),
            model: "tiny-local".to_string(),
            api_key: "terminal-secret-value".to_string(),
            cache_root: None,
            output_dir: None,
            timeout_secs: 30,
            harness_timeout_secs: None,
            endpoint_concurrency: 1,
            run_id: None,
            metrics_http: "http://127.0.0.1:18080".to_string(),
            metrics_run_id: None,
            metrics_finalize_only: false,
            dry_run: true,
        };
        let command = terminal_bench_command(&args, Path::new("/tmp/skippy-run"));
        assert!(
            command
                .args
                .contains(&"terminal-bench-core==0.1.1".to_string())
        );
        assert!(!command.args.contains(&"--task-id".to_string()));
        assert!(command.args.contains(&"--output-path".to_string()));
        assert!(
            command
                .envs
                .contains(&("OPENAI_BASE_URL".to_string(), args.base_url))
        );
        assert!(command.secret_envs.contains(&(
            "OPENAI_API_KEY".to_string(),
            "terminal-secret-value".to_string()
        )));
        assert!(!command.display().contains("terminal-secret-value"));
    }

    #[test]
    fn generated_eval_scripts_do_not_persist_api_keys() {
        let root = temp_run_dir("script-secrets-cache");
        let run_dir = temp_run_dir("script-secrets-run");
        fs::create_dir_all(run_dir.join("raw")).unwrap();

        for eval in [EvalId::McpAtlas, EvalId::SweBenchPro] {
            let args = eval_run_args(eval, "literal-secret");
            let command = match eval {
                EvalId::McpAtlas => mcp_atlas_command(&args, &root, &run_dir).unwrap(),
                EvalId::SweBenchPro => swe_bench_pro_command(&args, &root, &run_dir).unwrap(),
                _ => unreachable!(),
            };
            let script = fs::read_to_string(&command.args[0]).unwrap();

            assert!(command.secret_envs.contains(&(
                "SKIPPY_BENCH_API_KEY".to_string(),
                "literal-secret".to_string()
            )));
            assert!(!command.display().contains("literal-secret"));
            assert!(!script.contains("literal-secret"));
            assert!(script.contains("SKIPPY_BENCH_API_KEY"));
            assert!(!script.contains("--agent.model.api_key"));
        }

        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(run_dir);
    }

    #[test]
    fn telemetry_failure_is_reported_as_unavailable() {
        let telemetry = telemetry_or_unavailable(
            "http://127.0.0.1:18080",
            "run-id",
            Err(anyhow::anyhow!("collector offline")),
        );

        assert_eq!(telemetry.status, "unavailable");
        assert_eq!(telemetry.detail.as_deref(), Some("collector offline"));
    }

    #[test]
    fn preflight_eval_run_reports_missing_prerequisites() {
        let definition = EvalDefinition {
            id: EvalId::TerminalBench,
            name: "Test Eval",
            repo_url: "https://example.invalid/repo.git",
            repo_ref: "main",
            cache_name: "test-eval",
            description: "test",
            disk_estimate: "none",
            required_tools: &["definitely-not-a-real-skippybench-tool"],
            sync_notes: &[],
            run_notes: &[],
        };

        let error = preflight_eval_run(definition).unwrap_err().to_string();

        assert!(error.contains("terminal-bench prerequisites failed"));
        assert!(error.contains("definitely-not-a-real-skippybench-tool"));
        assert!(error.contains("skippy-bench eval doctor terminal-bench"));
    }

    #[test]
    fn existing_repo_sync_fetches_ref_and_checks_out_fetch_head() {
        let steps = existing_repo_sync_steps(Path::new("/tmp/harness"), "main");

        assert_eq!(
            steps[0].args,
            ["-C", "/tmp/harness", "fetch", "--prune", "origin", "main"]
        );
        assert_eq!(
            steps[1].args,
            ["-C", "/tmp/harness", "checkout", "--detach", "FETCH_HEAD"]
        );
    }

    #[test]
    fn resolved_harness_commit_reads_checked_out_revision() {
        let root = temp_run_dir("harness-revision");
        let definition = EvalDefinition {
            id: EvalId::SpeedBench,
            name: "test",
            repo_url: "unused",
            repo_ref: "main",
            cache_name: "test-harness",
            description: "test",
            disk_estimate: "none",
            required_tools: &[],
            sync_notes: &[],
            run_notes: &[],
        };
        let harness = harness_dir(&root, definition);
        fs::create_dir_all(&harness).unwrap();
        for args in [
            vec!["init"],
            vec!["config", "user.email", "skippy-bench@example.invalid"],
            vec!["config", "user.name", "Skippy Bench"],
        ] {
            assert!(
                Command::new("git")
                    .args(args)
                    .current_dir(&harness)
                    .status()
                    .unwrap()
                    .success()
            );
        }
        fs::write(harness.join("README"), "fixture").unwrap();
        assert!(
            Command::new("git")
                .args(["add", "README"])
                .current_dir(&harness)
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new("git")
                .args(["commit", "-m", "fixture"])
                .current_dir(&harness)
                .status()
                .unwrap()
                .success()
        );
        let expected = String::from_utf8(
            Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(&harness)
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap()
        .trim()
        .to_string();

        assert_eq!(
            resolved_harness_commit(&root, definition).unwrap(),
            Some(expected)
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn run_command_with_timeout_captures_successful_output() {
        let run_dir = temp_run_dir("command-success");
        fs::create_dir_all(&run_dir).unwrap();
        let stdout_path = run_dir.join("stdout.log");
        let stderr_path = run_dir.join("stderr.log");
        let command = CommandSpec::new("sh").args(["-c", "echo stdout-line; echo stderr-line >&2"]);

        let outcome = run_command_with_timeout(
            &command,
            Some(Duration::from_secs(5)),
            &stdout_path,
            &stderr_path,
        )
        .unwrap();

        assert!(outcome.success);
        assert_eq!(outcome.exit_status, Some(0));
        assert!(!outcome.timed_out);
        assert_eq!(
            fs::read_to_string(stdout_path).unwrap().trim(),
            "stdout-line"
        );
        assert_eq!(
            fs::read_to_string(stderr_path).unwrap().trim(),
            "stderr-line"
        );
        let _ = fs::remove_dir_all(run_dir);
    }

    #[test]
    fn run_command_with_timeout_marks_hung_command_timed_out() {
        let run_dir = temp_run_dir("command-timeout");
        fs::create_dir_all(&run_dir).unwrap();
        let stdout_path = run_dir.join("stdout.log");
        let stderr_path = run_dir.join("stderr.log");
        let command = CommandSpec::new("sh").args(["-c", "echo before-sleep; sleep 30"]);

        let started = Instant::now();
        let outcome = run_command_with_timeout(
            &command,
            Some(Duration::from_secs(1)),
            &stdout_path,
            &stderr_path,
        )
        .unwrap();

        assert!(!outcome.success);
        assert_eq!(outcome.exit_status, None);
        assert!(outcome.timed_out);
        assert!(started.elapsed() < Duration::from_secs(10));
        assert_eq!(
            fs::read_to_string(stdout_path).unwrap().trim(),
            "before-sleep"
        );
        let _ = fs::remove_dir_all(run_dir);
    }

    #[test]
    fn speed_bench_metrics_extract_token_rates_and_acceptance() {
        let run_dir = temp_run_dir("speed-metrics");
        fs::create_dir_all(run_dir.join("raw")).unwrap();
        fs::write(
            speed_bench_output_path(&run_dir),
            r#"{
              "completed_samples": 2,
              "failed_samples": 1,
              "summary": [
                {
                  "category": "overall",
                  "avg_prompt_t_s": 123.5,
                  "avg_pred_t_s": 45.25,
                  "avg_latency": 1.5,
                  "draft_n": 12,
                  "accepted": 9,
                  "accept_rate": 0.75
                }
              ],
              "results": [
                {
                  "prompt_tokens": 10,
                  "completion_tokens": 5,
                  "total_tokens": 15,
                  "draft_n": 8,
                  "draft_n_accepted": 6
                },
                {
                  "prompt_tokens": 20,
                  "completion_tokens": 7,
                  "total_tokens": 27,
                  "draft_n": 4,
                  "draft_n_accepted": 3
                }
              ]
            }"#,
        )
        .unwrap();

        let metrics = speed_bench_metrics(&run_dir).unwrap();
        assert_eq!(metrics.request_count, Some(2));
        assert_eq!(metrics.failed_count, Some(1));
        assert_eq!(metrics.prompt_tokens, Some(30));
        assert_eq!(metrics.completion_tokens, Some(12));
        assert_eq!(metrics.total_tokens, Some(42));
        assert_eq!(metrics.draft_tokens, Some(12));
        assert_eq!(metrics.draft_accepted_tokens, Some(9));
        assert_eq!(metrics.prompt_tok_s, Some(123.5));
        assert_eq!(metrics.completion_tok_s, Some(45.25));
        assert_eq!(metrics.avg_latency_ms, Some(1500.0));
        assert_eq!(metrics.draft_accept_rate, Some(0.75));
        let _ = fs::remove_dir_all(run_dir);
    }

    #[test]
    fn swe_bench_pro_metrics_extract_openai_usage() {
        let run_dir = temp_run_dir("swe-metrics");
        fs::create_dir_all(swe_bench_pro_output_path(&run_dir).parent().unwrap()).unwrap();
        fs::write(
            swe_bench_pro_output_path(&run_dir),
            r#"{
              "rows": [
                {"usage": {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14}},
                {"usage": {"prompt_tokens": 17, "completion_tokens": 4, "total_tokens": 21}}
              ]
            }"#,
        )
        .unwrap();

        let metrics = swe_bench_pro_metrics(&run_dir).unwrap();
        assert_eq!(metrics.request_count, Some(2));
        assert_eq!(metrics.failed_count, None);
        assert_eq!(metrics.prompt_tokens, Some(28));
        assert_eq!(metrics.completion_tokens, Some(7));
        assert_eq!(metrics.total_tokens, Some(35));
        let _ = fs::remove_dir_all(run_dir);
    }

    #[test]
    fn terminal_bench_metrics_extract_accuracy_and_tokens() {
        let run_dir = temp_run_dir("terminal-metrics");
        let result_dir = terminal_bench_output_path(&run_dir).join("run-id");
        fs::create_dir_all(&result_dir).unwrap();
        fs::write(
            result_dir.join("results.json"),
            r#"{
              "results": [
                {
                  "task_id": "hello-world",
                  "is_resolved": true,
                  "total_input_tokens": 100,
                  "total_output_tokens": 25
                }
              ],
              "n_resolved": 1,
              "n_unresolved": 0,
              "accuracy": 1.0
            }"#,
        )
        .unwrap();

        let metrics = terminal_bench_metrics(&run_dir).unwrap();
        assert_eq!(metrics.request_count, Some(1));
        assert_eq!(metrics.failed_count, Some(0));
        assert_eq!(metrics.pass_rate, Some(1.0));
        assert_eq!(metrics.prompt_tokens, Some(100));
        assert_eq!(metrics.completion_tokens, Some(25));
        assert_eq!(metrics.total_tokens, Some(125));
        let _ = fs::remove_dir_all(run_dir);
    }

    #[test]
    fn fill_client_rates_computes_missing_rates() {
        let mut metrics = EvalMetrics {
            prompt_tokens: Some(20),
            completion_tokens: Some(30),
            total_tokens: Some(50),
            ..EvalMetrics::default()
        };
        fill_client_rates(&mut metrics, 2_000.0);
        assert_eq!(metrics.prompt_tok_s, Some(10.0));
        assert_eq!(metrics.completion_tok_s, Some(15.0));
        assert_eq!(metrics.total_tok_s, Some(25.0));
    }

    fn temp_run_dir(label: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "skippy-bench-{label}-{}-{}",
            std::process::id(),
            unix_millis().unwrap()
        ))
    }

    fn eval_run_args(eval: EvalId, api_key: &str) -> EvalRunArgs {
        EvalRunArgs {
            eval,
            base_url: "http://127.0.0.1:9337/v1".to_string(),
            model: "tiny-local".to_string(),
            api_key: api_key.to_string(),
            cache_root: None,
            output_dir: None,
            timeout_secs: 30,
            harness_timeout_secs: None,
            endpoint_concurrency: 1,
            run_id: None,
            metrics_http: "http://127.0.0.1:18080".to_string(),
            metrics_run_id: None,
            metrics_finalize_only: false,
            dry_run: true,
        }
    }
}
