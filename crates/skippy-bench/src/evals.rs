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
pub fn eval_command(args: EvalArgs) -> Result<()> {
    match args.command {
        EvalCommandKind::List(args) => list_evals(args),
        EvalCommandKind::Info(args) => info_eval(args),
        EvalCommandKind::Sync(args) | EvalCommandKind::Install(args) => sync_evals(args),
        EvalCommandKind::Doctor(args) => doctor_evals(args),
        EvalCommandKind::Run(args) => run_eval(args),
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
}

impl CommandSpec {
    fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            cwd: None,
            envs: Vec::new(),
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

    fn display(&self) -> String {
        let envs = self
            .envs
            .iter()
            .map(|(key, value)| format!("{key}={}", shell_quote(value)))
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
        command
    }
}

fn list_evals(args: EvalListArgs) -> Result<()> {
    let root = cache_root(args.cache_root.clone())?;
    let views = selected_evals(&[], EvalPack::Core)
        .into_iter()
        .map(|definition| eval_view(definition, &root))
        .collect::<Vec<_>>();
    if args.json {
        println!("{}", serde_json::to_string_pretty(&views)?);
        return Ok(());
    }

    println!("SkippyBench external evals");
    println!("cache: {}", root.display());
    for view in views {
        let status = if view.installed {
            "installed"
        } else {
            "not installed"
        };
        println!(
            "  {:<16} {:<13} {}",
            view.id,
            format!("[{status}]"),
            view.description
        );
    }
    Ok(())
}

fn info_eval(args: EvalInfoArgs) -> Result<()> {
    let root = cache_root(args.cache_root.clone())?;
    let definition = definition(args.eval);
    let view = eval_view(definition, &root);
    if args.json {
        println!("{}", serde_json::to_string_pretty(&view)?);
        return Ok(());
    }

    println!("{} ({})", view.name, view.id);
    println!("description: {}", view.description);
    println!("repo: {} @ {}", view.repo_url, view.repo_ref);
    println!("pack: {}", view.pack);
    println!("disk: {}", view.disk_estimate);
    println!("installed: {}", view.installed);
    println!("harness: {}", view.harness_dir);
    println!("requires: {}", view.required_tools.join(", "));
    print_notes("sync", view.sync_notes);
    print_notes("run", view.run_notes);
    Ok(())
}

fn sync_evals(args: EvalSyncArgs) -> Result<()> {
    let root = cache_root(args.cache_root)?;
    fs::create_dir_all(harness_root(&root)).with_context(|| {
        format!(
            "create eval harness cache {}",
            harness_root(&root).display()
        )
    })?;

    for definition in selected_evals(&args.evals, args.pack) {
        println!("sync {}", definition.id.as_str());
        sync_repo(definition, &root, args.dry_run)?;
        for step in sync_steps(definition, &root) {
            run_step(&step, args.dry_run)?;
        }
    }
    Ok(())
}

fn doctor_evals(args: EvalDoctorArgs) -> Result<()> {
    let root = cache_root(args.cache_root)?;
    let views = selected_evals(&args.evals, args.pack)
        .into_iter()
        .map(|definition| doctor_view(definition, &root))
        .collect::<Vec<_>>();
    if args.json {
        println!("{}", serde_json::to_string_pretty(&views)?);
        return Ok(());
    }

    println!("SkippyBench eval doctor");
    println!("cache: {}", root.display());
    for view in views {
        println!();
        println!("{}:", view.eval_id);
        println!("  installed: {}", view.installed);
        println!("  harness: {}", view.harness_dir);
        for check in view.checks {
            let status = if check.ok { "ok" } else { "missing" };
            println!("  {status:<7} {} - {}", check.name, check.detail);
        }
    }
    Ok(())
}

fn run_eval(args: EvalRunArgs) -> Result<()> {
    if args.endpoint_concurrency == 0 {
        bail!("--endpoint-concurrency must be greater than zero");
    }
    let root = absolute_path(cache_root(args.cache_root.clone())?)?;
    let definition = definition(args.eval);
    let run_dir = absolute_path(output_dir(args.output_dir.clone(), args.eval)?)?;
    let run_id = args
        .run_id
        .clone()
        .unwrap_or_else(|| eval_run_id(args.eval));
    let metrics_run_id = args
        .metrics_run_id
        .clone()
        .unwrap_or_else(|| run_id.clone());
    let metrics_http = args.metrics_http.trim_end_matches('/').to_string();

    if !args.dry_run {
        let harness = harness_dir(&root, definition);
        if !harness.exists() {
            bail!(
                "{} is not installed at {}; run `skippy-bench eval sync {}` first",
                definition.id.as_str(),
                harness.display(),
                definition.id.as_str()
            );
        }
        preflight_eval_run(definition)?;
    }

    fs::create_dir_all(run_dir.join("raw"))
        .with_context(|| format!("create eval run dir {}", run_dir.display()))?;

    let command = run_command(definition, &args, &root, &run_dir)?;
    let display = command.display();
    println!("{display}");

    let mut report = RunReport {
        run_id: run_id.clone(),
        eval_id: definition.id.as_str(),
        model: args.model.clone(),
        base_url: args.base_url.clone(),
        endpoint_concurrency: args.endpoint_concurrency,
        run_dir: run_dir.display().to_string(),
        dry_run: args.dry_run,
        command: display,
        exit_status: None,
        success: args.dry_run,
        timed_out: false,
        timeout_secs: args.timeout_secs,
        harness_timeout_secs: args.harness_timeout_secs,
        stdout_path: None,
        stderr_path: None,
        metrics: EvalMetrics::default(),
        telemetry: telemetry_report::pending(&metrics_http, &metrics_run_id),
        artifacts: run_artifacts(definition, &run_dir),
    };

    if !args.dry_run {
        create_metrics_run(&args, &run_id, &metrics_run_id)?;
        let started = Instant::now();
        let stdout_path = run_dir.join("raw").join("stdout.log");
        let stderr_path = run_dir.join("raw").join("stderr.log");
        let outcome = run_command_with_timeout(
            &command,
            args.harness_timeout_secs.map(Duration::from_secs),
            &stdout_path,
            &stderr_path,
        )
        .with_context(|| format!("run {}", definition.id.as_str()))?;
        let duration_ms = started.elapsed().as_secs_f64() * 1000.0;
        report.exit_status = outcome.exit_status;
        report.success = outcome.success;
        report.timed_out = outcome.timed_out;
        report.stdout_path = Some(stdout_path.display().to_string());
        report.stderr_path = Some(stderr_path.display().to_string());
        report.metrics = collect_metrics(definition, &run_dir, duration_ms);
        if let Err(error) = collect_telemetry(&metrics_http, &metrics_run_id, &run_dir)
            .map(|telemetry| report.telemetry = telemetry)
        {
            report.telemetry =
                telemetry_report::unavailable(&metrics_http, &metrics_run_id, &error);
            report.success = false;
        }
    }

    let report_path = run_dir.join("run.json");
    fs::write(&report_path, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write {}", report_path.display()))?;
    if !report.success {
        bail!(
            "{} failed; see {}",
            definition.id.as_str(),
            report_path.display()
        );
    }
    Ok(())
}

fn preflight_eval_run(definition: EvalDefinition) -> Result<()> {
    let failed = definition
        .required_tools
        .iter()
        .map(|tool| tool_check(tool))
        .filter(|check| !check.ok)
        .map(|check| format!("{} - {}", check.name, check.detail))
        .collect::<Vec<_>>();
    if failed.is_empty() {
        return Ok(());
    }
    bail!(
        "{} prerequisites failed; run `skippy-bench eval doctor {}` for details:\n  {}",
        definition.id.as_str(),
        definition.id.as_str(),
        failed.join("\n  ")
    )
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
    terminate_child_signal(child, "TERM");
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
    terminate_child_signal(child, "KILL");
    let _ = child.wait().context("wait for killed harness command")?;
    Ok(())
}

fn terminate_child_signal(child: &mut Child, signal: &str) {
    #[cfg(unix)]
    {
        let process_group = format!("-{}", child.id());
        let _ = Command::new("kill")
            .args([format!("-{signal}"), process_group])
            .status();
    }
    #[cfg(not(unix))]
    {
        let _ = child.kill();
        let _ = signal;
    }
}

fn definition(id: EvalId) -> EvalDefinition {
    match id {
        EvalId::SpeedBench => EvalDefinition {
            id,
            name: "llama.cpp SPEED-Bench",
            repo_url: "https://github.com/ggml-org/llama.cpp.git",
            repo_ref: "master",
            cache_name: "llama.cpp",
            description: "OpenAI-compatible serving latency and throughput benchmark.",
            disk_estimate: "1-3GB including clone and Python dataset cache",
            required_tools: &["git", "uv", "python3"],
            sync_notes: &["Clones llama.cpp; SPEED-Bench data is fetched by the Python runner."],
            run_notes: &[
                "Runs the upstream SPEED-Bench client with category=all and no sample limit.",
            ],
        },
        EvalId::TerminalBench => EvalDefinition {
            id,
            name: "Terminal-Bench",
            repo_url: "https://github.com/harbor-framework/terminal-bench.git",
            repo_ref: "main",
            cache_name: "terminal-bench",
            description: "Agent benchmark for real terminal tasks in Docker sandboxes.",
            disk_estimate: "5-20GB depending on Docker images",
            required_tools: &["git", "uv", "python3.12", "tb", "docker"],
            sync_notes: &["Clones task repo and installs the terminal-bench uv tool."],
            run_notes: &["Runs the Terminal-Bench CLI against the full selected dataset."],
        },
        EvalId::SweBenchPro => EvalDefinition {
            id,
            name: "SWE-Bench Pro",
            repo_url: "https://github.com/scaleapi/SWE-bench_Pro-os.git",
            repo_ref: "main",
            cache_name: "swe-bench-pro",
            description: "Long-horizon software-engineering patch benchmark.",
            disk_estimate: "10GB+ before task Docker images",
            required_tools: &["git", "uv", "python3", "docker"],
            sync_notes: &["Clones the official repo and initializes submodules."],
            run_notes: &[
                "Generates SWE-agent instances from the full SWE-Bench Pro test split.",
                "Runs the synced SWE-agent scaffold, gathers .pred patches, then invokes swe_bench_pro_eval.py.",
            ],
        },
        EvalId::McpAtlas => EvalDefinition {
            id,
            name: "MCP-Atlas",
            repo_url: "https://github.com/scaleapi/mcp-atlas.git",
            repo_ref: "main",
            cache_name: "mcp-atlas",
            description: "Tool-use benchmark over real MCP servers and tasks.",
            disk_estimate: "10GB+ including Docker image",
            required_tools: &["git", "uv", "python3", "docker", "make", "curl"],
            sync_notes: &["Clones repo and pulls the prebuilt MCP-Atlas Docker image."],
            run_notes: &[
                "Starts the MCP environment and completion service when they are not already running.",
                "Runs the MCP-Atlas completion and scoring scripts with --no-filter and without --num-tasks or tool_choice overrides.",
            ],
        },
    }
}

fn selected_evals(requested: &[EvalId], pack: EvalPack) -> Vec<EvalDefinition> {
    let ids = if requested.is_empty() {
        match pack {
            EvalPack::Core => CORE_EVALS.to_vec(),
        }
    } else {
        requested.to_vec()
    };
    ids.into_iter().map(definition).collect()
}

fn eval_view(definition: EvalDefinition, root: &Path) -> EvalView {
    let harness_dir = harness_dir(root, definition);
    EvalView {
        id: definition.id.as_str(),
        name: definition.name,
        pack: "core",
        repo_url: definition.repo_url,
        repo_ref: definition.repo_ref,
        installed: harness_dir.exists(),
        harness_dir: harness_dir.display().to_string(),
        description: definition.description,
        disk_estimate: definition.disk_estimate,
        required_tools: definition.required_tools,
        sync_notes: definition.sync_notes,
        run_notes: definition.run_notes,
    }
}

fn doctor_view(definition: EvalDefinition, root: &Path) -> DoctorView {
    let harness = harness_dir(root, definition);
    let mut checks = definition
        .required_tools
        .iter()
        .map(|tool| tool_check(tool))
        .collect::<Vec<_>>();
    if definition.id == EvalId::McpAtlas {
        checks.push(port_check("mcp-atlas agent environment", 1984));
        checks.push(port_check("mcp-atlas completion service", 3000));
    }
    DoctorView {
        eval_id: definition.id.as_str(),
        installed: harness.exists(),
        harness_dir: harness.display().to_string(),
        checks,
    }
}

fn tool_check(tool: &str) -> DoctorCheck {
    if tool == "docker" {
        return docker_check();
    }
    let ok = command_exists(tool);
    DoctorCheck {
        name: tool.to_string(),
        ok,
        detail: if ok {
            "found on PATH".to_string()
        } else {
            "not found on PATH".to_string()
        },
    }
}

fn docker_check() -> DoctorCheck {
    let installed = command_exists("docker");
    if !installed {
        return DoctorCheck {
            name: "docker".to_string(),
            ok: false,
            detail: "docker CLI not found on PATH".to_string(),
        };
    }

    let Some(info) = run_probe_command(
        CommandSpec::new("docker").args(["info"]),
        "docker-info",
        Duration::from_secs(10),
    ) else {
        return DoctorCheck {
            name: "docker".to_string(),
            ok: false,
            detail: "docker CLI found, but daemon did not answer within 10s".to_string(),
        };
    };
    if !info.outcome.success {
        return DoctorCheck {
            name: "docker".to_string(),
            ok: false,
            detail: format!(
                "docker CLI found, but daemon is not reachable{}",
                probe_detail_suffix(&info)
            ),
        };
    }

    let container_name = format!("skippybench-doctor-{}", unix_millis().unwrap_or_default());
    let start = run_probe_command(
        CommandSpec::new("docker").args([
            "run",
            "--rm",
            "--name",
            &container_name,
            "--pull=missing",
            "hello-world",
        ]),
        "docker-run",
        Duration::from_secs(60),
    );
    cleanup_docker_container(&container_name);
    let Some(start) = start else {
        return DoctorCheck {
            name: "docker".to_string(),
            ok: false,
            detail:
                "docker daemon is reachable, but a hello-world container did not finish within 60s"
                    .to_string(),
        };
    };
    let ok = start.outcome.success;
    DoctorCheck {
        name: "docker".to_string(),
        ok,
        detail: if ok {
            "docker daemon is reachable and can start containers".to_string()
        } else {
            format!(
                "docker daemon is reachable, but hello-world container start failed{}",
                probe_detail_suffix(&start)
            )
        },
    }
}

struct ProbeCommandResult {
    outcome: CommandOutcome,
    stdout: String,
    stderr: String,
}

fn run_probe_command(
    spec: CommandSpec,
    label: &str,
    timeout: Duration,
) -> Option<ProbeCommandResult> {
    let run_dir = temp_probe_dir(label)?;
    let stdout_path = run_dir.join("stdout.log");
    let stderr_path = run_dir.join("stderr.log");
    let outcome =
        run_command_with_timeout(&spec, Some(timeout), &stdout_path, &stderr_path).ok()?;
    let stdout = fs::read_to_string(&stdout_path).unwrap_or_default();
    let stderr = fs::read_to_string(&stderr_path).unwrap_or_default();
    let _ = fs::remove_dir_all(run_dir);
    Some(ProbeCommandResult {
        outcome,
        stdout,
        stderr,
    })
}

fn temp_probe_dir(label: &str) -> Option<PathBuf> {
    let millis = unix_millis().ok()?;
    let dir = env::temp_dir().join(format!(
        "skippy-bench-{label}-{}-{millis}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

fn cleanup_docker_container(name: &str) {
    let _ = Command::new("docker").args(["rm", "-f", name]).status();
}

fn probe_detail_suffix(result: &ProbeCommandResult) -> String {
    if result.outcome.timed_out {
        return " (timed out)".to_string();
    }
    let detail =
        first_nonempty_line(&result.stderr).or_else(|| first_nonempty_line(&result.stdout));
    detail.map(|line| format!(": {line}")).unwrap_or_default()
}

fn first_nonempty_line(text: &str) -> Option<String> {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(str::to_string)
}

fn port_check(name: &str, port: u16) -> DoctorCheck {
    let ok = localhost_port_ready(port);
    DoctorCheck {
        name: name.to_string(),
        ok,
        detail: if ok {
            format!("localhost:{port} is reachable")
        } else {
            format!("expected on localhost:{port} when running this eval")
        },
    }
}

fn localhost_port_ready(port: u16) -> bool {
    let timeout = Duration::from_millis(500);
    ("127.0.0.1", port)
        .to_socket_addrs()
        .ok()
        .into_iter()
        .flatten()
        .any(|addr| TcpStream::connect_timeout(&addr, timeout).is_ok())
}

fn sync_repo(definition: EvalDefinition, root: &Path, dry_run: bool) -> Result<()> {
    let target = harness_dir(root, definition);
    if target.exists() {
        run_step(
            &CommandSpec::new("git").args(["-C", &target.display().to_string(), "fetch", "origin"]),
            dry_run,
        )?;
        run_step(
            &CommandSpec::new("git").args([
                "-C",
                &target.display().to_string(),
                "checkout",
                definition.repo_ref,
            ]),
            dry_run,
        )?;
        return Ok(());
    }

    run_step(
        &CommandSpec::new("git").args([
            "clone",
            "--recurse-submodules",
            "--branch",
            definition.repo_ref,
            definition.repo_url,
            &target.display().to_string(),
        ]),
        dry_run,
    )
}

fn sync_steps(definition: EvalDefinition, root: &Path) -> Vec<CommandSpec> {
    let harness = harness_dir(root, definition);
    match definition.id {
        EvalId::SpeedBench => Vec::new(),
        EvalId::TerminalBench => {
            vec![CommandSpec::new("uv").args([
                "tool",
                "install",
                "--python",
                "3.12",
                "terminal-bench",
            ])]
        }
        EvalId::SweBenchPro => vec![
            CommandSpec::new("git")
                .args(["submodule", "update", "--init", "--recursive"])
                .cwd(harness),
        ],
        EvalId::McpAtlas => {
            vec![CommandSpec::new("docker").args(["pull", "ghcr.io/scaleapi/mcp-atlas:1.2.5"])]
        }
    }
}

fn run_artifacts(definition: EvalDefinition, run_dir: &Path) -> Vec<RunArtifact> {
    let mut artifacts = vec![
        RunArtifact {
            kind: "stdout",
            path: run_dir.join("raw/stdout.log").display().to_string(),
        },
        RunArtifact {
            kind: "stderr",
            path: run_dir.join("raw/stderr.log").display().to_string(),
        },
    ];
    match definition.id {
        EvalId::SpeedBench => artifacts.push(RunArtifact {
            kind: "speed-bench-json",
            path: speed_bench_output_path(run_dir).display().to_string(),
        }),
        EvalId::SweBenchPro => artifacts.extend([
            RunArtifact {
                kind: "swe-bench-pro-sweagent-results",
                path: swe_bench_pro_sweagent_output_path(run_dir)
                    .display()
                    .to_string(),
            },
            RunArtifact {
                kind: "swe-bench-pro-patches-json",
                path: swe_bench_pro_patches_path(run_dir).display().to_string(),
            },
            RunArtifact {
                kind: "swe-bench-pro-eval-json",
                path: swe_bench_pro_output_path(run_dir).display().to_string(),
            },
        ]),
        EvalId::McpAtlas => artifacts.extend([
            RunArtifact {
                kind: "mcp-atlas-completion-results-csv",
                path: mcp_atlas_output_path(run_dir).display().to_string(),
            },
            RunArtifact {
                kind: "mcp-atlas-score-dir",
                path: mcp_atlas_score_dir(run_dir).display().to_string(),
            },
        ]),
        EvalId::TerminalBench => artifacts.push(RunArtifact {
            kind: "terminal-bench-results",
            path: terminal_bench_output_path(run_dir).display().to_string(),
        }),
    }
    artifacts
}

fn collect_metrics(definition: EvalDefinition, run_dir: &Path, duration_ms: f64) -> EvalMetrics {
    let mut metrics = match definition.id {
        EvalId::SpeedBench => speed_bench_metrics(run_dir).unwrap_or_default(),
        EvalId::SweBenchPro => swe_bench_pro_metrics(run_dir).unwrap_or_default(),
        EvalId::McpAtlas => mcp_atlas_metrics(run_dir).unwrap_or_default(),
        EvalId::TerminalBench => terminal_bench_metrics(run_dir).unwrap_or_default(),
    };
    metrics.duration_ms = Some(duration_ms);
    fill_client_rates(&mut metrics, duration_ms);
    metrics
}

fn create_metrics_run(args: &EvalRunArgs, eval_run_id: &str, metrics_run_id: &str) -> Result<()> {
    let config = json!({
        "eval_run_id": eval_run_id,
        "mode": "skippy-bench-external-eval",
        "eval_id": args.eval.as_str(),
        "model": args.model,
        "base_url": args.base_url,
    });
    telemetry_report::create_run(&args.metrics_http, metrics_run_id, &config)
}

fn collect_telemetry(
    metrics_http: &str,
    metrics_run_id: &str,
    run_dir: &Path,
) -> Result<BenchTelemetry> {
    telemetry_report::finalize_and_collect(
        metrics_http,
        metrics_run_id,
        &metrics_report_path(run_dir),
    )
}

fn speed_bench_metrics(run_dir: &Path) -> Result<EvalMetrics> {
    let value = read_json(&speed_bench_output_path(run_dir))?;
    let overall = value
        .get("summary")
        .and_then(Value::as_array)
        .and_then(|rows| {
            rows.iter()
                .find(|row| string_field(row, "category") == Some("overall"))
        });
    let results = value
        .get("results")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mut metrics = EvalMetrics {
        request_count: value.get("completed_samples").and_then(Value::as_u64),
        failed_count: value.get("failed_samples").and_then(Value::as_u64),
        ..EvalMetrics::default()
    };
    metrics.prompt_tokens = sum_u64_field(results, "prompt_tokens");
    metrics.completion_tokens = sum_u64_field(results, "completion_tokens");
    metrics.total_tokens = sum_u64_field(results, "total_tokens");
    metrics.draft_tokens = sum_u64_field(results, "draft_n");
    metrics.draft_accepted_tokens = sum_u64_field(results, "draft_n_accepted");
    if let Some(overall) = overall {
        metrics.prompt_tok_s = numeric_field(overall, "avg_prompt_t_s");
        metrics.completion_tok_s = numeric_field(overall, "avg_pred_t_s");
        metrics.avg_latency_ms =
            numeric_field(overall, "avg_latency").map(|seconds| seconds * 1000.0);
        metrics.draft_accept_rate = numeric_field(overall, "accept_rate");
    }
    Ok(metrics)
}

fn swe_bench_pro_metrics(run_dir: &Path) -> Result<EvalMetrics> {
    let value = read_json(&swe_bench_pro_output_path(run_dir))?;
    let rows = value
        .get("rows")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mut metrics = EvalMetrics {
        request_count: swe_bench_pro_request_count(&value, rows),
        failed_count: swe_bench_pro_failed_count(&value),
        pass_rate: swe_bench_pro_pass_rate(&value),
        ..EvalMetrics::default()
    };
    metrics.prompt_tokens = sum_nested_usage(rows, "prompt_tokens");
    metrics.completion_tokens = sum_nested_usage(rows, "completion_tokens");
    metrics.total_tokens = sum_nested_usage(rows, "total_tokens");
    Ok(metrics)
}

fn swe_bench_pro_request_count(value: &Value, rows: &[Value]) -> Option<u64> {
    value
        .get("total_instances")
        .or_else(|| value.get("total"))
        .or_else(|| value.get("n_total"))
        .and_then(Value::as_u64)
        .or_else(|| (!rows.is_empty()).then_some(rows.len() as u64))
}

fn swe_bench_pro_failed_count(value: &Value) -> Option<u64> {
    value
        .get("failed_instances")
        .or_else(|| value.get("unresolved_instances"))
        .or_else(|| value.get("n_unresolved"))
        .and_then(Value::as_u64)
}

fn swe_bench_pro_pass_rate(value: &Value) -> Option<f64> {
    if let Some(rate) = value
        .get("pass_rate")
        .or_else(|| value.get("resolved_rate"))
        .or_else(|| value.get("accuracy"))
        .and_then(Value::as_f64)
    {
        return Some(rate);
    }
    let resolved = value
        .get("resolved_instances")
        .or_else(|| value.get("n_resolved"))
        .and_then(Value::as_u64)?;
    let total = value
        .get("total_instances")
        .or_else(|| value.get("total"))
        .or_else(|| value.get("n_total"))
        .and_then(Value::as_u64)?;
    (total > 0).then_some(resolved as f64 / total as f64)
}

fn terminal_bench_metrics(run_dir: &Path) -> Result<EvalMetrics> {
    let value = read_json(&terminal_bench_results_path(run_dir)?)?;
    let results = value
        .get("results")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mut metrics = EvalMetrics {
        request_count: Some(results.len() as u64),
        failed_count: value.get("n_unresolved").and_then(Value::as_u64),
        pass_rate: value.get("accuracy").and_then(Value::as_f64),
        ..EvalMetrics::default()
    };
    metrics.prompt_tokens = sum_u64_field(results, "total_input_tokens");
    metrics.completion_tokens = sum_u64_field(results, "total_output_tokens");
    metrics.total_tokens = match (metrics.prompt_tokens, metrics.completion_tokens) {
        (Some(prompt), Some(completion)) => Some(prompt + completion),
        _ => None,
    };
    Ok(metrics)
}

fn terminal_bench_results_path(run_dir: &Path) -> Result<PathBuf> {
    let root = terminal_bench_output_path(run_dir);
    for entry in fs::read_dir(&root).with_context(|| format!("read {}", root.display()))? {
        let entry = entry?;
        let path = entry.path().join("results.json");
        if path.is_file() {
            return Ok(path);
        }
    }
    bail!("no Terminal-Bench results.json under {}", root.display())
}

fn mcp_atlas_metrics(run_dir: &Path) -> Result<EvalMetrics> {
    let mut reader = csv::Reader::from_path(mcp_atlas_output_path(run_dir))?;
    let data_rows = reader.records().filter(|record| record.is_ok()).count();
    Ok(EvalMetrics {
        request_count: Some(data_rows as u64),
        ..EvalMetrics::default()
    })
}

fn fill_client_rates(metrics: &mut EvalMetrics, duration_ms: f64) {
    if duration_ms <= 0.0 {
        return;
    }
    let seconds = duration_ms / 1000.0;
    if metrics.prompt_tok_s.is_none() {
        metrics.prompt_tok_s = metrics.prompt_tokens.map(|tokens| tokens as f64 / seconds);
    }
    if metrics.completion_tok_s.is_none() {
        metrics.completion_tok_s = metrics
            .completion_tokens
            .map(|tokens| tokens as f64 / seconds);
    }
    metrics.total_tok_s = metrics.total_tokens.map(|tokens| tokens as f64 / seconds);
}

fn read_json(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn sum_u64_field(rows: &[Value], key: &str) -> Option<u64> {
    let mut total = 0;
    let mut found = false;
    for row in rows {
        if let Some(value) = row.get(key).and_then(Value::as_u64) {
            total += value;
            found = true;
        }
    }
    found.then_some(total)
}

fn sum_nested_usage(rows: &[Value], key: &str) -> Option<u64> {
    let mut total = 0;
    let mut found = false;
    for row in rows {
        if let Some(value) = row
            .get("usage")
            .and_then(|usage| usage.get(key))
            .and_then(Value::as_u64)
        {
            total += value;
            found = true;
        }
    }
    found.then_some(total)
}

fn numeric_field(value: &Value, key: &str) -> Option<f64> {
    value.get(key).and_then(Value::as_f64)
}

fn string_field<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
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

fn run_command(
    definition: EvalDefinition,
    args: &EvalRunArgs,
    root: &Path,
    run_dir: &Path,
) -> Result<CommandSpec> {
    Ok(match definition.id {
        EvalId::SpeedBench => speed_bench_command(definition, args, root, run_dir),
        EvalId::TerminalBench => terminal_bench_command(args, run_dir),
        EvalId::SweBenchPro => swe_bench_pro_command(args, root, run_dir)?,
        EvalId::McpAtlas => mcp_atlas_command(args, root, run_dir)?,
    })
}

fn speed_bench_command(
    definition: EvalDefinition,
    args: &EvalRunArgs,
    root: &Path,
    run_dir: &Path,
) -> CommandSpec {
    let harness = harness_dir(root, definition);
    let requirements = harness.join("tools/server/bench/speed-bench/requirements.txt");
    let script = harness.join("tools/server/bench/speed-bench/speed_bench.py");
    let cache_root = env::temp_dir().join("skippy-bench-speed-cache");
    CommandSpec::new("uv")
        .args([
            "run".to_string(),
            "--with-requirements".to_string(),
            requirements.display().to_string(),
            "python".to_string(),
            script.display().to_string(),
            "--url".to_string(),
            args.base_url.clone(),
            "--model".to_string(),
            args.model.clone(),
            "--bench".to_string(),
            "qualitative".to_string(),
            "--category".to_string(),
            "all".to_string(),
            "--osl".to_string(),
            "1024".to_string(),
            "--concurrency".to_string(),
            args.endpoint_concurrency.to_string(),
            "--timeout".to_string(),
            args.timeout_secs.to_string(),
            "--output".to_string(),
            speed_bench_output_path(run_dir).display().to_string(),
        ])
        .env(
            "XDG_CACHE_HOME",
            cache_root.join("xdg").display().to_string(),
        )
        .env("HF_HOME", cache_root.join("hf").display().to_string())
        .env(
            "HF_DATASETS_CACHE",
            cache_root.join("hf-datasets").display().to_string(),
        )
        .env("UV_CACHE_DIR", cache_root.join("uv").display().to_string())
}

fn terminal_bench_command(args: &EvalRunArgs, run_dir: &Path) -> CommandSpec {
    let model = litellm_model_name(&args.model);
    CommandSpec::new("tb")
        .args([
            "run".to_string(),
            "--dataset".to_string(),
            "terminal-bench-core==0.1.1".to_string(),
            "--agent".to_string(),
            "terminus".to_string(),
            "--model".to_string(),
            model,
            "--n-concurrent".to_string(),
            args.endpoint_concurrency.to_string(),
            "--output-path".to_string(),
            terminal_bench_output_path(run_dir).display().to_string(),
            "--global-agent-timeout-sec".to_string(),
            args.timeout_secs.to_string(),
            "--global-test-timeout-sec".to_string(),
            "60".to_string(),
            "--no-upload-results".to_string(),
            "--no-livestream".to_string(),
        ])
        .env("OPENAI_BASE_URL", args.base_url.clone())
        .env("OPENAI_API_KEY", args.api_key.clone())
}

fn swe_bench_pro_command(args: &EvalRunArgs, root: &Path, run_dir: &Path) -> Result<CommandSpec> {
    let harness = harness_dir(root, definition(EvalId::SweBenchPro));
    let script = run_dir.join("raw/swe-bench-pro-run.sh");
    write_swe_bench_pro_run_script(&script, args, root, &harness, run_dir)?;
    Ok(CommandSpec::new("zsh").args([script.display().to_string()]))
}

fn mcp_atlas_command(args: &EvalRunArgs, root: &Path, run_dir: &Path) -> Result<CommandSpec> {
    let harness = harness_dir(root, definition(EvalId::McpAtlas));
    let script = run_dir.join("raw/mcp-atlas-run.sh");
    write_mcp_atlas_run_script(&script, args, root, &harness, run_dir)?;
    Ok(CommandSpec::new("zsh").args([script.display().to_string()]))
}

fn speed_bench_output_path(run_dir: &Path) -> PathBuf {
    run_dir.join("raw/speed-bench.json")
}

fn swe_bench_pro_output_path(run_dir: &Path) -> PathBuf {
    run_dir.join("raw/swe-bench-pro/eval/eval_results.json")
}

fn swe_bench_pro_sweagent_output_path(run_dir: &Path) -> PathBuf {
    run_dir.join("raw/swe-bench-pro/sweagent-results")
}

fn swe_bench_pro_patches_path(run_dir: &Path) -> PathBuf {
    run_dir.join("raw/swe-bench-pro/patches.json")
}

fn mcp_atlas_output_path(run_dir: &Path) -> PathBuf {
    run_dir.join("raw/mcp-atlas-completion-results.csv")
}

fn mcp_atlas_score_dir(run_dir: &Path) -> PathBuf {
    run_dir.join("raw/mcp-atlas-evaluation-results")
}

fn terminal_bench_output_path(run_dir: &Path) -> PathBuf {
    run_dir.join("raw/terminal-bench")
}

fn metrics_report_path(run_dir: &Path) -> PathBuf {
    run_dir.join("raw/metrics-report.json")
}

fn eval_run_id(eval: EvalId) -> String {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    format!("skippy-eval-{}-{millis}", eval.as_str())
}

fn write_mcp_atlas_run_script(
    path: &Path,
    args: &EvalRunArgs,
    cache_root: &Path,
    harness: &Path,
    run_dir: &Path,
) -> Result<()> {
    let raw_dir = run_dir.join("raw");
    let completion_dir = harness.join("services/mcp_eval");
    let default_output_name = format!("skippybench-{}-completion-results.csv", unix_millis()?);
    let model_label = model_label(&args.model);
    let score_dir = mcp_atlas_score_dir(run_dir);
    let completion_concurrency = env_concurrency_matching_endpoint(
        "MCP_ATLAS_COMPLETION_CONCURRENCY",
        args.endpoint_concurrency,
    )?;
    let script = format!(
        r#"#!/usr/bin/env zsh
set -euo pipefail

HARNESS={harness}
COMPLETION_DIR={completion_dir}
RAW_DIR={raw_dir}
BASE_URL={base_url}
API_KEY={api_key}
MODEL={model}
EVAL_MODEL=${{EVAL_LLM_MODEL:-$MODEL}}
OUTPUT={output}
OUTPUT_NAME=${{MCP_ATLAS_COMPLETION_OUTPUT_NAME:-{output_name}}}
MODEL_LABEL={model_label}
SCORE_DIR={score_dir}
COMPLETION_CONCURRENCY={completion_concurrency}
SCORE_CONCURRENCY=${{MCP_ATLAS_SCORE_CONCURRENCY:-5}}
HF_HOME_DIR={hf_home}
HF_DATASETS_CACHE_DIR={hf_datasets_cache}
UV_CACHE_DIR_LOCAL={uv_cache_dir}
XDG_CACHE_HOME_DIR={xdg_cache_home}
agent_started=0
completion_started=0
completion_pid=""

mkdir -p \
  "$HF_HOME_DIR" \
  "$HF_DATASETS_CACHE_DIR" \
  "$UV_CACHE_DIR_LOCAL" \
  "$XDG_CACHE_HOME_DIR"
export HF_HOME="$HF_HOME_DIR"
export HF_DATASETS_CACHE="$HF_DATASETS_CACHE_DIR"
export UV_CACHE_DIR="$UV_CACHE_DIR_LOCAL"
export XDG_CACHE_HOME="$XDG_CACHE_HOME_DIR"

port_ready() {{
  python3 - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
}}

wait_url() {{
  local name="$1"
  local url="$2"
  local log="$3"
  for _ in {{1..90}}; do
    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
      return 0
    fi
    tail -20 "$log" 2>/dev/null || true
    sleep 2
  done
  echo "timed out waiting for $name at $url" >&2
  return 1
}}

cleanup() {{
  if [[ "$completion_started" == "1" && -n "$completion_pid" ]]; then
    kill "$completion_pid" >/dev/null 2>&1 || true
    wait "$completion_pid" >/dev/null 2>&1 || true
  fi
  if [[ "$agent_started" == "1" ]]; then
    docker rm -f skippy-bench-mcp-atlas-agent-env >/dev/null 2>&1 || true
  fi
}}
trap cleanup EXIT

mkdir -p "$RAW_DIR" "$SCORE_DIR"
cd "$HARNESS"
cp -n env.template .env >/dev/null 2>&1 || true
if ! docker image inspect agent-environment:latest >/dev/null 2>&1; then
  docker tag ghcr.io/scaleapi/mcp-atlas:1.2.5 agent-environment:latest
fi

if ! port_ready 1984; then
  docker rm -f skippy-bench-mcp-atlas-agent-env >/dev/null 2>&1 || true
  docker run --rm \
    --name skippy-bench-mcp-atlas-agent-env \
    -p 1984:1984 \
    --env-file .env \
    agent-environment:latest \
    > "$RAW_DIR/mcp-agent-env.log" 2>&1 &
  agent_started=1
fi
wait_url "MCP-Atlas agent environment" \
  "http://localhost:1984/enabled-servers" \
  "$RAW_DIR/mcp-agent-env.log"

if ! port_ready 3000; then
  (
    cd "$COMPLETION_DIR"
    LLM_BASE_URL="$BASE_URL" \
      LLM_API_KEY="$API_KEY" \
      OPENAI_BASE_URL="$BASE_URL" \
      OPENAI_API_KEY="$API_KEY" \
      uv run python -m mcp_completion.main
  ) > "$RAW_DIR/mcp-completion.log" 2>&1 &
  completion_pid="$!"
  completion_started=1
fi
wait_url "MCP-Atlas completion service" \
  "http://localhost:3000/docs" \
  "$RAW_DIR/mcp-completion.log"

cd "$COMPLETION_DIR"
LLM_BASE_URL="$BASE_URL" \
  LLM_API_KEY="$API_KEY" \
  OPENAI_BASE_URL="$BASE_URL" \
  OPENAI_API_KEY="$API_KEY" \
uv run python mcp_completion_script.py \
    --model "$MODEL" \
    --input_huggingface ScaleAI/MCP-Atlas \
    --output "$OUTPUT_NAME" \
    --no-filter \
    --concurrency "$COMPLETION_CONCURRENCY"
cp "completion_results/$OUTPUT_NAME" "$OUTPUT"
EVAL_LLM_BASE_URL="${{EVAL_LLM_BASE_URL:-$BASE_URL}}" \
  EVAL_LLM_API_KEY="${{EVAL_LLM_API_KEY:-$API_KEY}}" \
uv run python mcp_evals_scores.py \
    --input-file "completion_results/$OUTPUT_NAME" \
    --model-label "$MODEL_LABEL" \
    --evaluator-model "$EVAL_MODEL" \
    --output-dir "$SCORE_DIR" \
    --concurrency "$SCORE_CONCURRENCY"
"#,
        harness = shell_quote(&harness.display().to_string()),
        completion_dir = shell_quote(&completion_dir.display().to_string()),
        raw_dir = shell_quote(&raw_dir.display().to_string()),
        base_url = shell_quote(&args.base_url),
        api_key = shell_quote(&args.api_key),
        model = shell_quote(&litellm_model_name(&args.model)),
        output = shell_quote(&mcp_atlas_output_path(run_dir).display().to_string()),
        output_name = shell_quote(&default_output_name),
        model_label = shell_quote(&model_label),
        score_dir = shell_quote(&score_dir.display().to_string()),
        completion_concurrency = shell_quote(&completion_concurrency),
        hf_home = shell_quote(&cache_root.join("hf").display().to_string()),
        hf_datasets_cache = shell_quote(&cache_root.join("hf-datasets").display().to_string()),
        uv_cache_dir = shell_quote(&cache_root.join("uv").display().to_string()),
        xdg_cache_home = shell_quote(&cache_root.join("xdg").display().to_string()),
    );
    fs::write(path, script).with_context(|| format!("write {}", path.display()))
}

fn write_swe_bench_pro_run_script(
    path: &Path,
    args: &EvalRunArgs,
    cache_root: &Path,
    harness: &Path,
    run_dir: &Path,
) -> Result<()> {
    let raw_dir = run_dir.join("raw/swe-bench-pro");
    let instances = raw_dir.join("instances.yaml");
    let expert_instances = raw_dir.join("instances-expert.yaml");
    let sweagent_output = raw_dir.join("sweagent-results");
    let patches = raw_dir.join("patches.json");
    let eval_dir = raw_dir.join("eval");
    let dockerhub_username =
        env::var("SWE_BENCH_PRO_DOCKERHUB_USERNAME").unwrap_or_else(|_| "jefzda".to_string());
    let deployment_type =
        env::var("SWE_BENCH_PRO_DEPLOYMENT_TYPE").unwrap_or_else(|_| "docker".to_string());
    let num_workers =
        env_concurrency_matching_endpoint("SWE_BENCH_PRO_NUM_WORKERS", args.endpoint_concurrency)?;
    let eval_workers = env::var("SWE_BENCH_PRO_EVAL_WORKERS").unwrap_or_else(|_| "100".to_string());
    let docker_platform =
        env::var("SWE_BENCH_PRO_DOCKER_PLATFORM").unwrap_or_else(|_| "linux/amd64".to_string());
    let parse_function = env::var("SWE_BENCH_PRO_PARSE_FUNCTION").ok();
    let sweagent_python = env::var("SWE_BENCH_PRO_PYTHON").unwrap_or_else(|_| "3.12".to_string());
    let swerex_spec = env::var("SWE_BENCH_PRO_SWEREX_SPEC").ok();
    let swerex_pip_index_url = env::var("SWE_BENCH_PRO_SWEREX_PIP_INDEX_URL")
        .unwrap_or_else(|_| "https://pypi.org/simple".to_string());
    let use_local_eval = env::var("SWE_BENCH_PRO_USE_LOCAL_DOCKER")
        .map(|value| matches!(value.as_str(), "1" | "true" | "yes"))
        .unwrap_or(true);
    let local_eval_flag = if use_local_eval {
        "--use_local_docker"
    } else {
        ""
    };
    let model = litellm_model_name(&args.model);
    let script = format!(
        r#"#!/usr/bin/env zsh
set -euo pipefail

HARNESS={harness}
RAW_DIR={raw_dir}
INSTANCES={instances}
EXPERT_INSTANCES={expert_instances}
SWEAGENT_OUTPUT={sweagent_output}
PATCHES={patches}
EVAL_DIR={eval_dir}
MODEL={model}
BASE_URL={base_url}
API_KEY={api_key}
DOCKERHUB_USERNAME={dockerhub_username}
DEPLOYMENT_TYPE={deployment_type}
NUM_WORKERS={num_workers}
EVAL_WORKERS={eval_workers}
LOCAL_EVAL_FLAG={local_eval_flag}
DOCKER_PLATFORM={docker_platform}
PARSE_FUNCTION={parse_function}
SWEAGENT_PYTHON={sweagent_python}
SWEREX_SPEC={swerex_spec}
SWEREX_PIP_INDEX_URL={swerex_pip_index_url}
HF_HOME_DIR={hf_home}
HF_DATASETS_CACHE_DIR={hf_datasets_cache}
UV_CACHE_DIR_LOCAL={uv_cache_dir}
XDG_CACHE_HOME_DIR={xdg_cache_home}

mkdir -p \
  "$RAW_DIR" \
  "$SWEAGENT_OUTPUT" \
  "$EVAL_DIR" \
  "$HF_HOME_DIR" \
  "$HF_DATASETS_CACHE_DIR" \
  "$UV_CACHE_DIR_LOCAL" \
  "$XDG_CACHE_HOME_DIR"
export HF_HOME="$HF_HOME_DIR"
export HF_DATASETS_CACHE="$HF_DATASETS_CACHE_DIR"
export UV_CACHE_DIR="$UV_CACHE_DIR_LOCAL"
export XDG_CACHE_HOME="$XDG_CACHE_HOME_DIR"
deployment_timeout_args=()
if [[ "$DEPLOYMENT_TYPE" == "modal" ]]; then
  deployment_timeout_args=(
    --instances.deployment.startup_timeout 1800
    --instances.deployment.runtime_timeout 3600
  )
fi
deployment_platform_args=()
if [[ "$DEPLOYMENT_TYPE" == "docker" && -n "$DOCKER_PLATFORM" ]]; then
  deployment_platform_args=(
    --instances.deployment.platform "$DOCKER_PLATFORM"
  )
fi
deployment_type_args=(
  --instances.deployment.type "$DEPLOYMENT_TYPE"
)
expert_instance_args=(
  --instances.type file
  --instances.path "$INSTANCES"
)
parse_function_args=()
if [[ -n "$PARSE_FUNCTION" ]]; then
  parse_function_args=(
    --agent.tools.parse_function.type "$PARSE_FUNCTION"
  )
fi
if [[ -z "$SWEREX_SPEC" && "$DEPLOYMENT_TYPE" == "docker" ]]; then
  SWEREX_SPEC="swe-rex[modal]==1.4.0"
fi

cd "$HARNESS"

uv run \
  --with-requirements requirements.txt \
  --with pyyaml \
  python helper_code/generate_sweagent_instances.py \
    --dockerhub_username "$DOCKERHUB_USERNAME" \
    --output_path "$INSTANCES"

(
  cd SWE-agent
  uv venv --clear --python "$SWEAGENT_PYTHON" .venv
  uv pip install --python .venv/bin/python -e .
  if [[ -n "$SWEREX_SPEC" ]]; then
    uv pip install --python .venv/bin/python --upgrade "$SWEREX_SPEC"
  fi
  if [[ "$DEPLOYMENT_TYPE" == "docker" && -n "$SWEREX_PIP_INDEX_URL" ]]; then
    .venv/bin/python - "$SWEREX_PIP_INDEX_URL" <<'PY'
import sys
from pathlib import Path

import swerex.deployment.docker as docker

path = Path(docker.__file__)
text = path.read_text()
old = 'f"RUN /root/python3.11/bin/pip3 install --no-cache-dir {{PACKAGE_NAME}}\\n\\n"'
new = (
    f'f"RUN /root/python3.11/bin/pip3 install --index-url {{sys.argv[1]}} '
    '--no-cache-dir {{PACKAGE_NAME}}\\n\\n"'
)
if old in text:
    path.write_text(text.replace(old, new))
elif new not in text:
    raise RuntimeError(f"could not patch SWE-ReX Docker pip index in {{path}}")
PY
  fi
  if [[ "$DEPLOYMENT_TYPE" == "modal" ]]; then
    .venv/bin/python swerex_patches/patch.py --yes
  fi
)

if [[ "$DEPLOYMENT_TYPE" == "docker" ]]; then
  (
    cd SWE-agent
    .venv/bin/python - "$INSTANCES" "$EXPERT_INSTANCES" "$DOCKER_PLATFORM" <<'PY'
import sys

import yaml
from sweagent.agent.problem_statement import TextProblemStatement
from sweagent.environment.repo import PreExistingRepoConfig
from sweagent.environment.swe_env import EnvironmentConfig
from sweagent.run.batch_instances import BatchInstance
from swerex.deployment.config import DockerDeploymentConfig

source, target, platform = sys.argv[1:4]
with open(source) as handle:
    simple_instances = yaml.safe_load(handle)

docker_args = [
    "--entrypoint",
    "",
]
instances = []
for item in simple_instances:
    deployment = DockerDeploymentConfig(
        image=item["image_name"],
        docker_args=docker_args,
        platform=platform or None,
        python_standalone_dir="/root",
        startup_timeout=1800,
    )
    instance = BatchInstance(
        env=EnvironmentConfig(
            deployment=deployment,
            repo=PreExistingRepoConfig(
                repo_name=item.get("repo_name") or "app",
                base_commit=item.get("base_commit") or "HEAD",
            ),
        ),
        problem_statement=TextProblemStatement(
            text=item["problem_statement"],
            id=item["instance_id"],
            extra_fields=item.get("extra_fields") or {{}},
        ),
    )
    instances.append(instance.model_dump(mode="json", exclude_none=True))

with open(target, "w") as handle:
    yaml.safe_dump(instances, handle, sort_keys=False)
PY
  )
  expert_instance_args=(
    --instances.type expert_file
    --instances.path "$EXPERT_INSTANCES"
  )
  deployment_type_args=()
  deployment_platform_args=()
fi

(
  cd SWE-agent
  OPENAI_BASE_URL="$BASE_URL" \
  OPENAI_API_KEY="$API_KEY" \
  .venv/bin/sweagent run-batch \
    --config config/tool_use.yaml \
    --output_dir "$SWEAGENT_OUTPUT" \
    --num_workers "$NUM_WORKERS" \
    --random_delay_multiplier 1 \
    "${{expert_instance_args[@]}}" \
    --instances.shuffle=False \
    "${{deployment_type_args[@]}}" \
    "${{deployment_timeout_args[@]}}" \
    "${{deployment_platform_args[@]}}" \
    "${{parse_function_args[@]}}" \
    --agent.model.name "$MODEL" \
    --agent.model.api_base "$BASE_URL" \
    --agent.model.api_key "$API_KEY" \
    --agent.model.max_input_tokens 0 \
    --agent.model.per_instance_cost_limit 0 \
    --agent.model.total_cost_limit 0
)

uv run \
  --with-requirements requirements.txt \
  python helper_code/gather_patches.py \
    --directory "$SWEAGENT_OUTPUT" \
    --prefix skippybench \
    --output "$PATCHES"

uv run \
  --with-requirements requirements.txt \
  --with docker \
  --with modal \
  python swe_bench_pro_eval.py \
    --raw_sample_path helper_code/sweap_eval_full_v2.jsonl \
    --patch_path "$PATCHES" \
    --output_dir "$EVAL_DIR" \
    --scripts_dir run_scripts \
    --num_workers "$EVAL_WORKERS" \
    --dockerhub_username "$DOCKERHUB_USERNAME" \
    $LOCAL_EVAL_FLAG
"#,
        harness = shell_quote(&harness.display().to_string()),
        raw_dir = shell_quote(&raw_dir.display().to_string()),
        instances = shell_quote(&instances.display().to_string()),
        expert_instances = shell_quote(&expert_instances.display().to_string()),
        sweagent_output = shell_quote(&sweagent_output.display().to_string()),
        patches = shell_quote(&patches.display().to_string()),
        eval_dir = shell_quote(&eval_dir.display().to_string()),
        model = shell_quote(&model),
        base_url = shell_quote(&args.base_url),
        api_key = shell_quote(&args.api_key),
        dockerhub_username = shell_quote(&dockerhub_username),
        deployment_type = shell_quote(&deployment_type),
        num_workers = shell_quote(&num_workers),
        eval_workers = shell_quote(&eval_workers),
        docker_platform = shell_quote(&docker_platform),
        parse_function = shell_quote(parse_function.as_deref().unwrap_or("")),
        sweagent_python = shell_quote(&sweagent_python),
        swerex_spec = shell_quote(swerex_spec.as_deref().unwrap_or("")),
        swerex_pip_index_url = shell_quote(&swerex_pip_index_url),
        local_eval_flag = shell_quote(local_eval_flag),
        hf_home = shell_quote(&cache_root.join("hf").display().to_string()),
        hf_datasets_cache = shell_quote(&cache_root.join("hf-datasets").display().to_string()),
        uv_cache_dir = shell_quote(&cache_root.join("uv").display().to_string()),
        xdg_cache_home = shell_quote(&cache_root.join("xdg").display().to_string()),
    );
    fs::write(path, script).with_context(|| format!("write {}", path.display()))
}

fn run_step(step: &CommandSpec, dry_run: bool) -> Result<()> {
    println!("{}", step.display());
    if dry_run {
        return Ok(());
    }

    let status = step
        .command()
        .status()
        .with_context(|| format!("start {}", step.program))?;
    if !status.success() {
        bail!("command failed with status {status}: {}", step.display());
    }
    Ok(())
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
            dry_run: true,
        };
        let root = PathBuf::from("/tmp/skippy-cache");
        let run_dir = PathBuf::from("/tmp/skippy-run");
        let command = speed_bench_command(definition(EvalId::SpeedBench), &args, &root, &run_dir);
        assert!(command.args.contains(&"--url".to_string()));
        assert!(
            command
                .args
                .contains(&"http://127.0.0.1:9337/v1".to_string())
        );
        assert!(command.args.contains(&"tiny-local".to_string()));
    }

    #[test]
    fn terminal_bench_command_uses_full_dataset_without_task_filter() {
        let args = EvalRunArgs {
            eval: EvalId::TerminalBench,
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
}
