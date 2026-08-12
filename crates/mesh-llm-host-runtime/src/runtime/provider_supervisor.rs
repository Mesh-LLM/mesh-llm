use super::{add_runtime_local_target, remove_runtime_local_target, upsert_dashboard_process};
use crate::{api, inference::election};
use anyhow::{Context, Result, bail};
use mesh_llm_events::{OutputEvent, emit_event};
use mesh_llm_provider_runtime::{
    InstalledProviderRuntime, PROVIDER_RUNTIME_MANIFEST_FILE, PROVIDER_RUNTIME_SCHEMA_VERSION,
    ProviderRuntimeBundlePolicy, ProviderRuntimeCache, ProviderRuntimeHost,
    ProviderRuntimeInstallOptions, ProviderRuntimeReleaseManifest, ProviderRuntimeRequest,
    install_provider_runtime,
};
use serde::Deserialize;
#[cfg(target_os = "macos")]
use std::ffi::OsString;
use std::{
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    time::Duration,
};
use tokio::{
    io::{AsyncBufReadExt, BufReader, Lines},
    process::{Child, ChildStdout, Command},
    sync::watch,
    task::JoinHandle,
};

const APPLE_MODEL_ID: &str = "apple/system";
const APPLE_PROVIDER_KIND: &str = "apple";
const APPLE_PROVIDER_PROTOCOL: &str = "0.1";
const PROVIDER_INSTANCE_ID: &str = "provider:apple/system";
const PROVIDER_HEALTH_INTERVAL: Duration = Duration::from_secs(5);
const PROVIDER_READY_TIMEOUT: Duration = Duration::from_secs(30);
const PROVIDER_SHUTDOWN_GRACE: Duration = Duration::from_secs(5);
const PROVIDER_MAX_HEALTH_FAILURES: u8 = 3;
const PROVIDER_MAX_RESTART_BACKOFF_SECS: u64 = 30;

pub(super) struct ProviderSupervisorContext {
    pub(super) target_tx: Arc<watch::Sender<election::ModelTargets>>,
    pub(super) dashboard_processes: Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    pub(super) console_state: Option<api::MeshApi>,
}

pub(super) struct ProviderSupervisorHandle {
    shutdown_tx: watch::Sender<bool>,
    task: JoinHandle<()>,
}

#[derive(Clone)]
struct ProviderRuntimeContext {
    runtime: InstalledProviderRuntime,
    model_id: String,
}

#[derive(Debug)]
enum ProviderRunOutcome {
    Shutdown,
    Restart(String),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProviderAvailability {
    available: bool,
    context_length: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ProviderModelsResponse {
    data: Vec<ProviderModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ProviderModelEntry {
    id: String,
    #[serde(default)]
    availability: Option<String>,
    #[serde(default)]
    context_length: Option<u32>,
}

impl ProviderSupervisorHandle {
    pub(super) async fn shutdown(self) {
        let _ = self.shutdown_tx.send(true);
        let mut task = self.task;
        if tokio::time::timeout(PROVIDER_SHUTDOWN_GRACE + Duration::from_secs(2), &mut task)
            .await
            .is_err()
        {
            task.abort();
            let _ = task.await;
        }
    }
}

pub(super) async fn start_apple_provider_supervisor(
    context: ProviderSupervisorContext,
) -> Option<ProviderSupervisorHandle> {
    let resolved = match resolve_apple_provider_runtime().await {
        Ok(Some(runtime)) => runtime,
        Ok(None) => return None,
        Err(error) => {
            emit_provider_warning(
                "Apple provider runtime was discovered but could not be resolved",
                &error,
            );
            return None;
        }
    };
    if let Err(error) = validate_provider_platform_policy(&resolved) {
        emit_provider_warning("Apple provider runtime failed platform policy", &error);
        return None;
    }
    let runtime = ProviderRuntimeContext {
        runtime: resolved,
        model_id: APPLE_MODEL_ID.to_string(),
    };
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let task = tokio::spawn(supervise_provider_runtime(runtime, context, shutdown_rx));
    Some(ProviderSupervisorHandle { shutdown_tx, task })
}

async fn resolve_apple_provider_runtime() -> Result<Option<InstalledProviderRuntime>> {
    let discovery = ProviderDiscovery::from_environment()?;
    if !discovery.has_candidates()? {
        return Ok(None);
    }
    let outcome = install_provider_runtime(ProviderRuntimeInstallOptions {
        host: ProviderRuntimeHost::current(),
        request: ProviderRuntimeRequest {
            provider_kind: Some(APPLE_PROVIDER_KIND.to_string()),
            model_id: Some(APPLE_MODEL_ID.to_string()),
            protocol_version: Some(APPLE_PROVIDER_PROTOCOL.to_string()),
            ..ProviderRuntimeRequest::default()
        },
        release_manifest: discovery.release_manifest,
        bundle_dirs: discovery.bundle_dirs,
        cache_dir: Some(discovery.cache_dir),
        bundle_policy: ProviderRuntimeBundlePolicy::UseInPlace,
        allow_download: discovery.allow_download,
    })
    .await?;
    Ok(Some(outcome.runtime))
}

struct ProviderDiscovery {
    bundle_dirs: Vec<PathBuf>,
    release_manifest: ProviderRuntimeReleaseManifest,
    cache_dir: PathBuf,
    allow_download: bool,
}

impl ProviderDiscovery {
    fn from_environment() -> Result<Self> {
        let cache_dir = provider_cache_dir()?;
        let mut roots = configured_bundle_roots();
        roots.extend(default_bundle_roots());
        let bundle_dirs = discover_bundle_dirs(&roots)?;
        let release_manifest = configured_release_manifest()?;
        Ok(Self {
            bundle_dirs,
            release_manifest,
            cache_dir,
            allow_download: environment_flag("MESH_LLM_PROVIDER_RUNTIME_DOWNLOAD"),
        })
    }

    fn has_candidates(&self) -> Result<bool> {
        if !self.bundle_dirs.is_empty() || !self.release_manifest.artifacts.is_empty() {
            return Ok(true);
        }
        Ok(!ProviderRuntimeCache::new(self.cache_dir.clone())
            .list()?
            .is_empty())
    }
}

fn configured_bundle_roots() -> Vec<PathBuf> {
    std::env::var_os("MESH_LLM_PROVIDER_RUNTIME_BUNDLE_DIR")
        .map(|value| std::env::split_paths(&value).collect())
        .unwrap_or_default()
}

fn default_bundle_roots() -> Vec<PathBuf> {
    let Ok(executable) = std::env::current_exe() else {
        return Vec::new();
    };
    let Some(binary_dir) = executable.parent() else {
        return Vec::new();
    };
    let mut roots = vec![
        binary_dir.join("runtimes/apple"),
        binary_dir.join("provider-runtimes/apple"),
    ];
    if let Some(product_root) = binary_dir.parent() {
        roots.push(product_root.join("Resources/provider-runtimes/apple"));
    }
    roots
}

fn discover_bundle_dirs(roots: &[PathBuf]) -> Result<Vec<PathBuf>> {
    let mut bundles = Vec::new();
    for root in roots {
        if root.join(PROVIDER_RUNTIME_MANIFEST_FILE).is_file() {
            bundles.push(root.clone());
            continue;
        }
        if !root.is_dir() {
            continue;
        }
        for entry in std::fs::read_dir(root)
            .with_context(|| format!("read provider runtime bundle root {}", root.display()))?
        {
            let entry = entry?;
            if entry.file_type()?.is_dir()
                && entry.path().join(PROVIDER_RUNTIME_MANIFEST_FILE).is_file()
            {
                bundles.push(entry.path());
            }
        }
    }
    bundles.sort();
    bundles.dedup();
    Ok(bundles)
}

fn configured_release_manifest() -> Result<ProviderRuntimeReleaseManifest> {
    let Some(path) = std::env::var_os("MESH_LLM_PROVIDER_RUNTIME_INDEX") else {
        return Ok(empty_release_manifest());
    };
    ProviderRuntimeReleaseManifest::read_from_path(Path::new(&path))
}

fn empty_release_manifest() -> ProviderRuntimeReleaseManifest {
    ProviderRuntimeReleaseManifest {
        schema_version: PROVIDER_RUNTIME_SCHEMA_VERSION,
        artifacts: Vec::new(),
    }
}

fn provider_cache_dir() -> Result<PathBuf> {
    if let Some(path) = std::env::var_os("MESH_LLM_PROVIDER_RUNTIME_CACHE_DIR") {
        return Ok(PathBuf::from(path));
    }
    dirs::cache_dir()
        .or_else(|| dirs::home_dir().map(|home| home.join(".cache")))
        .context("cannot determine executable provider runtime cache directory")
        .map(|root| root.join("mesh-llm/provider-runtimes"))
}

fn environment_flag(name: &str) -> bool {
    std::env::var(name).is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

async fn supervise_provider_runtime(
    runtime: ProviderRuntimeContext,
    context: ProviderSupervisorContext,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    let mut restart_count = 0_u32;
    loop {
        if *shutdown_rx.borrow() {
            break;
        }
        match run_provider_process(&runtime, &context, &mut shutdown_rx).await {
            ProviderRunOutcome::Shutdown => break,
            ProviderRunOutcome::Restart(detail) => {
                remove_provider_process(&context).await;
                restart_count = restart_count.saturating_add(1);
                let delay = restart_backoff(restart_count);
                let _ = emit_event(OutputEvent::Warning {
                    message: format!("Apple provider exited; restarting in {}s", delay.as_secs()),
                    context: Some(detail),
                });
                if wait_for_restart_or_shutdown(delay, &mut shutdown_rx).await {
                    break;
                }
            }
        }
    }
    remove_provider_process(&context).await;
}

async fn wait_for_restart_or_shutdown(
    delay: Duration,
    shutdown_rx: &mut watch::Receiver<bool>,
) -> bool {
    tokio::select! {
        () = tokio::time::sleep(delay) => false,
        changed = shutdown_rx.changed() => changed.is_err() || *shutdown_rx.borrow(),
    }
}

fn restart_backoff(restart_count: u32) -> Duration {
    let exponent = restart_count.saturating_sub(1).min(5);
    Duration::from_secs(
        1_u64
            .checked_shl(exponent)
            .unwrap_or(PROVIDER_MAX_RESTART_BACKOFF_SECS)
            .min(PROVIDER_MAX_RESTART_BACKOFF_SECS),
    )
}

async fn run_provider_process(
    runtime: &ProviderRuntimeContext,
    context: &ProviderSupervisorContext,
    shutdown_rx: &mut watch::Receiver<bool>,
) -> ProviderRunOutcome {
    let mut child = match spawn_provider_process(runtime) {
        Ok(child) => child,
        Err(error) => return ProviderRunOutcome::Restart(format!("launch failed: {error:#}")),
    };
    let pid = child.id().unwrap_or_default();
    let stderr_task = child.stderr.take().map(spawn_provider_stderr_drain);
    let Some(stdout) = child.stdout.take() else {
        let _ = terminate_provider_process(&mut child).await;
        return ProviderRunOutcome::Restart("provider stdout was not captured".to_string());
    };
    let mut stdout = BufReader::new(stdout).lines();
    let port = match wait_for_provider_ready(&mut child, &mut stdout, shutdown_rx).await {
        Ok(Some(port)) => port,
        Ok(None) => {
            let _ = terminate_provider_process(&mut child).await;
            abort_log_tasks(None, stderr_task);
            return ProviderRunOutcome::Shutdown;
        }
        Err(error) => {
            let _ = terminate_provider_process(&mut child).await;
            abort_log_tasks(None, stderr_task);
            return ProviderRunOutcome::Restart(format!("readiness failed: {error:#}"));
        }
    };
    let stdout_task = Some(spawn_provider_stdout_drain(stdout));
    let outcome =
        monitor_provider_process(runtime, context, &mut child, pid, port, shutdown_rx).await;
    abort_log_tasks(stdout_task, stderr_task);
    outcome
}

fn spawn_provider_process(runtime: &ProviderRuntimeContext) -> Result<Child> {
    let executable = runtime.runtime.entrypoint();
    let mut command = Command::new(&executable);
    command
        .arg("serve")
        .arg("--port")
        .arg("0")
        .arg("--parent-pid")
        .arg(std::process::id().to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    remove_provider_secret_environment(&mut command);
    command
        .spawn()
        .with_context(|| format!("launch executable provider {}", executable.display()))
}

fn remove_provider_secret_environment(command: &mut Command) {
    const SECRET_NAMES: &[&str] = &[
        "ANTHROPIC_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "MESH_LLM_TOKEN",
        "OPENAI_API_KEY",
    ];
    for name in SECRET_NAMES {
        command.env_remove(name);
    }
}

async fn wait_for_provider_ready(
    child: &mut Child,
    stdout: &mut Lines<BufReader<ChildStdout>>,
    shutdown_rx: &mut watch::Receiver<bool>,
) -> Result<Option<u16>> {
    let deadline = tokio::time::sleep(PROVIDER_READY_TIMEOUT);
    tokio::pin!(deadline);
    loop {
        tokio::select! {
            () = &mut deadline => bail!("provider did not report readiness within {}s", PROVIDER_READY_TIMEOUT.as_secs()),
            changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                    return Ok(None);
                }
            }
            status = child.wait() => {
                bail!("provider exited before readiness: {}", status?);
            }
            line = stdout.next_line() => {
                let Some(line) = line? else {
                    bail!("provider closed stdout before readiness");
                };
                if let Some(port) = provider_ready_port(&line)? {
                    return Ok(Some(port));
                }
                tracing::debug!(provider = APPLE_PROVIDER_KIND, output = %line, "provider startup output");
            }
        }
    }
}

fn provider_ready_port(line: &str) -> Result<Option<u16>> {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
        return Ok(None);
    };
    if value.get("type").and_then(serde_json::Value::as_str) != Some("ready") {
        if value.get("type").and_then(serde_json::Value::as_str) == Some("error") {
            bail!("provider reported startup error: {value}");
        }
        return Ok(None);
    }
    let port = value
        .get("port")
        .and_then(|port| {
            port.as_u64()
                .and_then(|port| u16::try_from(port).ok())
                .or_else(|| port.as_str()?.parse::<u16>().ok())
        })
        .filter(|port| *port != 0)
        .context("provider readiness event contained no valid port")?;
    Ok(Some(port))
}

async fn monitor_provider_process(
    runtime: &ProviderRuntimeContext,
    context: &ProviderSupervisorContext,
    child: &mut Child,
    pid: u32,
    port: u16,
    shutdown_rx: &mut watch::Receiver<bool>,
) -> ProviderRunOutcome {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
    {
        Ok(client) => client,
        Err(error) => {
            let _ = terminate_provider_process(child).await;
            return ProviderRunOutcome::Restart(format!("health client failed: {error}"));
        }
    };
    let mut health_tick = tokio::time::interval(PROVIDER_HEALTH_INTERVAL);
    health_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut failures = 0_u8;
    let mut routed = false;
    loop {
        tokio::select! {
            changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                    withdraw_provider(&runtime.model_id, port, context).await;
                    let _ = terminate_provider_process(child).await;
                    return ProviderRunOutcome::Shutdown;
                }
            }
            status = child.wait() => {
                withdraw_provider(&runtime.model_id, port, context).await;
                return ProviderRunOutcome::Restart(match status {
                    Ok(status) => format!("provider process exited with {status}"),
                    Err(error) => format!("provider process wait failed: {error}"),
                });
            }
            _ = health_tick.tick() => {
                match probe_provider(&client, port, &runtime.model_id).await {
                    Ok(availability) => {
                        failures = 0;
                        publish_provider_state(runtime, context, pid, port, &availability).await;
                        if availability.available && !routed {
                            add_runtime_local_target(&context.target_tx, &runtime.model_id, port);
                            routed = true;
                            emit_provider_ready(runtime, port, pid);
                        } else if !availability.available && routed {
                            remove_runtime_local_target(&context.target_tx, &runtime.model_id, port);
                            routed = false;
                        }
                    }
                    Err(error) => {
                        failures = failures.saturating_add(1);
                        publish_provider_unhealthy(runtime, context, pid, port).await;
                        if routed {
                            remove_runtime_local_target(&context.target_tx, &runtime.model_id, port);
                            routed = false;
                        }
                        if failures >= PROVIDER_MAX_HEALTH_FAILURES {
                            let _ = terminate_provider_process(child).await;
                            return ProviderRunOutcome::Restart(format!(
                                "provider failed {failures} consecutive health checks: {error:#}"
                            ));
                        }
                    }
                }
            }
        }
    }
}

async fn probe_provider(
    client: &reqwest::Client,
    port: u16,
    model_id: &str,
) -> Result<ProviderAvailability> {
    let base = format!("http://127.0.0.1:{port}");
    client
        .get(format!("{base}/health"))
        .send()
        .await
        .context("request provider health")?
        .error_for_status()
        .context("provider health returned an error")?;
    let models = client
        .get(format!("{base}/v1/models"))
        .send()
        .await
        .context("request provider models")?
        .error_for_status()
        .context("provider models returned an error")?
        .json::<ProviderModelsResponse>()
        .await
        .context("decode provider models")?;
    let model = models
        .data
        .into_iter()
        .find(|candidate| candidate.id == model_id)
        .with_context(|| format!("provider does not report requested model {model_id}"))?;
    Ok(ProviderAvailability {
        available: model
            .availability
            .as_deref()
            .is_none_or(|status| status.eq_ignore_ascii_case("available")),
        context_length: model.context_length,
    })
}

async fn publish_provider_state(
    runtime: &ProviderRuntimeContext,
    context: &ProviderSupervisorContext,
    pid: u32,
    port: u16,
    availability: &ProviderAvailability,
) {
    let status = if availability.available {
        "ready"
    } else {
        "unavailable"
    };
    upsert_provider_process(
        runtime,
        context,
        pid,
        port,
        status,
        availability.context_length,
    )
    .await;
}

async fn publish_provider_unhealthy(
    runtime: &ProviderRuntimeContext,
    context: &ProviderSupervisorContext,
    pid: u32,
    port: u16,
) {
    upsert_provider_process(runtime, context, pid, port, "unhealthy", None).await;
}

async fn upsert_provider_process(
    runtime: &ProviderRuntimeContext,
    context: &ProviderSupervisorContext,
    pid: u32,
    port: u16,
    status: &str,
    context_length: Option<u32>,
) {
    let process = api::RuntimeProcessPayload {
        name: runtime.model_id.clone(),
        instance_id: Some(PROVIDER_INSTANCE_ID.to_string()),
        profile: String::new(),
        backend: runtime.runtime.manifest.runtime.provider_kind.clone(),
        status: status.to_string(),
        port,
        pid,
        slots: 1,
        context_length,
    };
    upsert_dashboard_process(&context.dashboard_processes, process.clone()).await;
    if let Some(console_state) = &context.console_state {
        console_state.upsert_local_process(process).await;
    }
}

async fn withdraw_provider(model_id: &str, port: u16, context: &ProviderSupervisorContext) {
    remove_runtime_local_target(&context.target_tx, model_id, port);
    remove_provider_process(context).await;
}

async fn remove_provider_process(context: &ProviderSupervisorContext) {
    super::remove_dashboard_process(&context.dashboard_processes, PROVIDER_INSTANCE_ID).await;
    if let Some(console_state) = &context.console_state {
        console_state
            .remove_local_process(PROVIDER_INSTANCE_ID)
            .await;
    }
}

fn emit_provider_ready(runtime: &ProviderRuntimeContext, port: u16, pid: u32) {
    let _ = emit_event(OutputEvent::Info {
        message: format!(
            "Apple system model is available through the MeshLLM OpenAI API ({})",
            runtime.model_id
        ),
        context: Some(format!(
            "provider={} version={} pid={pid} port={port}",
            runtime.runtime.manifest.runtime.id, runtime.runtime.manifest.runtime.version
        )),
    });
}

fn emit_provider_warning(message: &str, error: &anyhow::Error) {
    let _ = emit_event(OutputEvent::Warning {
        message: message.to_string(),
        context: Some(format!("{error:#}")),
    });
}

fn spawn_provider_stdout_drain(mut lines: Lines<BufReader<ChildStdout>>) -> JoinHandle<()> {
    tokio::spawn(async move {
        while let Ok(Some(line)) = lines.next_line().await {
            tracing::debug!(provider = APPLE_PROVIDER_KIND, output = %line, "provider output");
        }
    })
}

fn spawn_provider_stderr_drain(stderr: tokio::process::ChildStderr) -> JoinHandle<()> {
    tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            tracing::warn!(provider = APPLE_PROVIDER_KIND, output = %line, "provider diagnostic");
        }
    })
}

fn abort_log_tasks(stdout: Option<JoinHandle<()>>, stderr: Option<JoinHandle<()>>) {
    if let Some(task) = stdout {
        task.abort();
    }
    if let Some(task) = stderr {
        task.abort();
    }
}

async fn terminate_provider_process(child: &mut Child) -> Result<()> {
    if child.try_wait()?.is_some() {
        return Ok(());
    }
    request_provider_termination(child)?;
    if tokio::time::timeout(PROVIDER_SHUTDOWN_GRACE, child.wait())
        .await
        .is_ok()
    {
        return Ok(());
    }
    child.kill().await.context("force-stop provider process")?;
    let _ = child.wait().await;
    Ok(())
}

#[cfg(unix)]
fn request_provider_termination(child: &mut Child) -> Result<()> {
    let pid = child.id().context("provider process has no pid")?;
    let pid = i32::try_from(pid).context("provider pid exceeds platform range")?;
    // SAFETY: `pid` is the live child PID returned by Tokio and `SIGTERM` is a
    // valid signal. No pointer or shared-memory access crosses this boundary.
    let result = unsafe { libc::kill(pid, libc::SIGTERM) };
    if result == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error()).context("send SIGTERM to provider process")
    }
}

#[cfg(not(unix))]
fn request_provider_termination(child: &mut Child) -> Result<()> {
    child.start_kill().context("stop provider process")
}

#[cfg(target_os = "macos")]
fn validate_provider_platform_policy(runtime: &InstalledProviderRuntime) -> Result<()> {
    if runtime.manifest.runtime.provider_kind != APPLE_PROVIDER_KIND {
        return Ok(());
    }
    let executable = runtime.entrypoint();
    run_policy_command(
        "codesign",
        &[
            OsString::from("--verify"),
            OsString::from("--strict"),
            executable.clone().into(),
        ],
        "verify Apple provider code signature",
    )?;
    let details = run_policy_command(
        "codesign",
        &[
            OsString::from("-dv"),
            OsString::from("--verbose=4"),
            executable.clone().into(),
        ],
        "inspect Apple provider code signature",
    )?;
    let team_identifier = signing_detail(&details, "TeamIdentifier");
    let signing_identifier = signing_detail(&details, "Identifier");
    let is_ad_hoc = team_identifier
        .as_deref()
        .is_none_or(|team| team == "not set");
    if is_ad_hoc && !environment_flag("MESH_LLM_APPLE_PROVIDER_ALLOW_AD_HOC") {
        bail!(
            "Apple provider is ad-hoc signed; set MESH_LLM_APPLE_PROVIDER_ALLOW_AD_HOC=1 only for local experimental builds"
        );
    }
    if let Some(signature) = &runtime.manifest.runtime.signature {
        compare_signing_detail(
            "team identifier",
            signature.team_identifier.as_deref(),
            team_identifier.as_deref(),
        )?;
        compare_signing_detail(
            "signing identifier",
            signature.signing_identifier.as_deref(),
            signing_identifier.as_deref(),
        )?;
        validate_declared_entitlements(&executable, &signature.entitlements)?;
        if signature.notarized == Some(true) {
            run_policy_command(
                "spctl",
                &[
                    OsString::from("--assess"),
                    OsString::from("--type"),
                    OsString::from("execute"),
                    executable.into(),
                ],
                "assess Apple provider notarization",
            )?;
        }
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn validate_provider_platform_policy(runtime: &InstalledProviderRuntime) -> Result<()> {
    if runtime.manifest.runtime.provider_kind == APPLE_PROVIDER_KIND {
        bail!("Apple provider runtimes may only be launched on macOS");
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn run_policy_command(program: &str, arguments: &[OsString], label: &str) -> Result<String> {
    let output = std::process::Command::new(program)
        .args(arguments)
        .output()
        .with_context(|| label.to_string())?;
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !output.status.success() {
        bail!("{label} failed: {}", combined.trim());
    }
    Ok(combined)
}

#[cfg(target_os = "macos")]
fn signing_detail(details: &str, name: &str) -> Option<String> {
    details.lines().find_map(|line| {
        line.strip_prefix(&format!("{name}="))
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
}

#[cfg(target_os = "macos")]
fn compare_signing_detail(label: &str, expected: Option<&str>, actual: Option<&str>) -> Result<()> {
    if let Some(expected) = expected
        && actual != Some(expected)
    {
        bail!(
            "Apple provider {label} mismatch: expected {expected}, got {}",
            actual.unwrap_or("missing")
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn validate_declared_entitlements(executable: &Path, declared: &[String]) -> Result<()> {
    if declared.is_empty() {
        return Ok(());
    }
    let output = run_policy_command(
        "codesign",
        &[
            OsString::from("-d"),
            OsString::from("--entitlements"),
            OsString::from(":-"),
            executable.to_path_buf().into(),
        ],
        "inspect Apple provider entitlements",
    )?;
    for entitlement in declared {
        if !output.contains(&format!("<key>{entitlement}</key>")) {
            bail!("Apple provider is missing declared entitlement {entitlement}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readiness_accepts_string_and_numeric_ports() {
        assert_eq!(
            provider_ready_port(r#"{"type":"ready","port":"11435"}"#).unwrap(),
            Some(11_435)
        );
        assert_eq!(
            provider_ready_port(r#"{"type":"ready","port":11436}"#).unwrap(),
            Some(11_436)
        );
        assert_eq!(provider_ready_port(r#"{"type":"status"}"#).unwrap(), None);
    }

    #[test]
    fn readiness_rejects_zero_and_error_events() {
        assert!(provider_ready_port(r#"{"type":"ready","port":0}"#).is_err());
        assert!(provider_ready_port(r#"{"type":"error","error":{"code":"failed"}}"#).is_err());
    }

    #[test]
    fn restart_backoff_is_bounded() {
        assert_eq!(restart_backoff(1), Duration::from_secs(1));
        assert_eq!(restart_backoff(2), Duration::from_secs(2));
        assert_eq!(restart_backoff(6), Duration::from_secs(30));
        assert_eq!(restart_backoff(100), Duration::from_secs(30));
    }

    #[test]
    fn bundle_discovery_accepts_a_bundle_or_parent_directory() {
        let temp = tempfile::tempdir().unwrap();
        let direct = temp.path().join("direct");
        let parent = temp.path().join("parent");
        let nested = parent.join("nested");
        std::fs::create_dir_all(&direct).unwrap();
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(direct.join(PROVIDER_RUNTIME_MANIFEST_FILE), "{}").unwrap();
        std::fs::write(nested.join(PROVIDER_RUNTIME_MANIFEST_FILE), "{}").unwrap();

        let discovered = discover_bundle_dirs(&[direct.clone(), parent]).unwrap();
        assert_eq!(discovered, vec![direct, nested]);
    }

    #[tokio::test]
    async fn provider_target_is_withdrawn_without_touching_other_models() {
        let mut targets = election::ModelTargets::default();
        targets.targets.insert(
            APPLE_MODEL_ID.to_string(),
            vec![
                election::InferenceTarget::Local(11_435),
                election::InferenceTarget::Local(11_436),
            ],
        );
        targets.targets.insert(
            "other/model".to_string(),
            vec![election::InferenceTarget::Local(12_345)],
        );
        let (target_tx, _target_rx) = watch::channel(targets);
        let context = ProviderSupervisorContext {
            target_tx: Arc::new(target_tx),
            dashboard_processes: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            console_state: None,
        };

        withdraw_provider(APPLE_MODEL_ID, 11_435, &context).await;

        assert_eq!(
            context.target_tx.borrow().candidates(APPLE_MODEL_ID),
            vec![election::InferenceTarget::Local(11_436)]
        );
        assert_eq!(
            context.target_tx.borrow().candidates("other/model"),
            vec![election::InferenceTarget::Local(12_345)]
        );
    }
}
