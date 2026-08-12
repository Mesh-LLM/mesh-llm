mod advertisement;
mod platform_policy;

use crate::{api, inference::election, mesh};
use advertisement::*;
use anyhow::{Context, Result, bail};
use mesh_llm_events::{OutputEvent, emit_event};
use mesh_llm_provider_runtime::{
    InstalledProviderRuntime, PROVIDER_RUNTIME_MANIFEST_FILE, PROVIDER_RUNTIME_SCHEMA_VERSION,
    ProviderRuntimeBundlePolicy, ProviderRuntimeCache, ProviderRuntimeHost,
    ProviderRuntimeInstallOptions, ProviderRuntimeReleaseManifest, ProviderRuntimeRequest,
    install_provider_runtime,
};
use platform_policy::validate_provider_platform_policy;
use serde::Deserialize;
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
// Provider load changes at request timescale. Poll fast enough for another Mac
// to avoid a one-slot runtime while retaining a three-second health grace.
const PROVIDER_HEALTH_INTERVAL: Duration = Duration::from_millis(250);
const PROVIDER_READY_TIMEOUT: Duration = Duration::from_secs(30);
const PROVIDER_SHUTDOWN_GRACE: Duration = Duration::from_secs(5);
const PROVIDER_MAX_HEALTH_FAILURES: u8 = 12;
const PROVIDER_MAX_RESTART_BACKOFF_SECS: u64 = 30;

pub(crate) struct ProviderSupervisorContext {
    pub(super) target_tx: Arc<watch::Sender<election::ModelTargets>>,
    pub(super) dashboard_processes: Arc<tokio::sync::Mutex<Vec<api::RuntimeProcessPayload>>>,
    pub(super) console_state: Option<api::MeshApi>,
    pub(super) node: mesh::Node,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ProviderRuntimeDiscoveryOptions {
    pub(crate) bundle_roots: Vec<PathBuf>,
    pub(crate) release_manifest: Option<PathBuf>,
    pub(crate) cache_dir: Option<PathBuf>,
    pub(crate) allow_download: bool,
    pub(crate) inherit_environment: bool,
}

pub(crate) struct ProviderSupervisorHandle {
    shutdown_tx: watch::Sender<bool>,
    task: JoinHandle<()>,
    pub(super) model_id: String,
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
    unavailable_reason: Option<String>,
    context_length: Option<u32>,
    model_version: String,
    versioned_model_id: String,
    capabilities: Vec<String>,
    max_concurrent_requests: u32,
    active_requests: u32,
    queued_requests: u32,
}

#[derive(Debug, Deserialize)]
struct ProviderModelsResponse {
    data: Vec<ProviderModelEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct ProviderModelEntry {
    id: String,
    #[serde(default)]
    availability: Option<String>,
    #[serde(default)]
    unavailable_reason: Option<String>,
    #[serde(default)]
    context_length: Option<u32>,
    model_version: String,
    version_source: String,
    versioned_model_id: String,
    #[serde(default)]
    capabilities: Vec<String>,
    max_concurrent_requests: u32,
    active_requests: u32,
    queued_requests: u32,
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

pub(crate) async fn start_apple_provider_supervisor(
    context: ProviderSupervisorContext,
    discovery_options: Option<&ProviderRuntimeDiscoveryOptions>,
    requested_model_id: Option<&str>,
) -> Option<ProviderSupervisorHandle> {
    let resolved = match resolve_apple_provider_runtime(discovery_options, requested_model_id).await
    {
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
    let model_id = resolved
        .manifest
        .runtime
        .models
        .first()
        .map(|model| model.id.clone())
        .unwrap_or_else(|| APPLE_MODEL_ID.to_string());
    let runtime = ProviderRuntimeContext {
        runtime: resolved,
        model_id: model_id.clone(),
    };
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let task = tokio::spawn(supervise_provider_runtime(runtime, context, shutdown_rx));
    Some(ProviderSupervisorHandle {
        shutdown_tx,
        task,
        model_id,
    })
}

async fn resolve_apple_provider_runtime(
    options: Option<&ProviderRuntimeDiscoveryOptions>,
    requested_model_id: Option<&str>,
) -> Result<Option<InstalledProviderRuntime>> {
    let discovery = ProviderDiscovery::from_options(options)?;
    if !discovery.has_candidates()? {
        return Ok(None);
    }
    let outcome = install_provider_runtime(ProviderRuntimeInstallOptions {
        host: ProviderRuntimeHost::current(),
        request: ProviderRuntimeRequest {
            provider_kind: Some(APPLE_PROVIDER_KIND.to_string()),
            // The selected Apple artifact owns the model identity. Filtering
            // on apple/system would make packaged Core AI artifacts invisible.
            model_id: requested_model_id
                .filter(|model| model.starts_with("apple/") || model.contains('/'))
                .map(str::to_string),
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
    fn from_options(options: Option<&ProviderRuntimeDiscoveryOptions>) -> Result<Self> {
        let inherit_environment = options.is_none_or(|options| options.inherit_environment);
        let cache_dir = provider_cache_dir(
            options.and_then(|options| options.cache_dir.clone()),
            inherit_environment,
        )?;
        let roots = discovery_roots(
            options
                .map(|options| options.bundle_roots.clone())
                .unwrap_or_default(),
            inherit_environment.then(configured_bundle_roots),
            default_bundle_roots(),
            inherit_environment,
        );
        let bundle_dirs = discover_bundle_dirs(&roots)?;
        let release_manifest = provider_release_manifest(
            options.and_then(|options| options.release_manifest.as_deref()),
            inherit_environment,
        )?;
        Ok(Self {
            bundle_dirs,
            release_manifest,
            cache_dir,
            allow_download: options.is_some_and(|options| options.allow_download)
                || (inherit_environment && environment_flag("MESH_LLM_PROVIDER_RUNTIME_DOWNLOAD")),
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

fn discovery_roots(
    mut explicit: Vec<PathBuf>,
    environment: Option<Vec<PathBuf>>,
    defaults: Vec<PathBuf>,
    inherit_environment: bool,
) -> Vec<PathBuf> {
    if inherit_environment {
        explicit.extend(environment.unwrap_or_default());
    }
    explicit.extend(defaults);
    explicit
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

fn provider_release_manifest(
    configured: Option<&Path>,
    inherit_environment: bool,
) -> Result<ProviderRuntimeReleaseManifest> {
    let path = configured.map(Path::to_path_buf).or_else(|| {
        inherit_environment
            .then(|| std::env::var_os("MESH_LLM_PROVIDER_RUNTIME_INDEX"))
            .flatten()
            .map(PathBuf::from)
    });
    let Some(path) = path else {
        return Ok(empty_release_manifest());
    };
    ProviderRuntimeReleaseManifest::read_from_path(&path)
}

fn empty_release_manifest() -> ProviderRuntimeReleaseManifest {
    ProviderRuntimeReleaseManifest {
        schema_version: PROVIDER_RUNTIME_SCHEMA_VERSION,
        artifacts: Vec::new(),
    }
}

fn provider_cache_dir(configured: Option<PathBuf>, inherit_environment: bool) -> Result<PathBuf> {
    if let Some(path) = configured {
        return Ok(path);
    }
    if inherit_environment
        && let Some(path) = std::env::var_os("MESH_LLM_PROVIDER_RUNTIME_CACHE_DIR")
    {
        return Ok(path.into());
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
    let mut state = ProviderRoutingState::default();
    loop {
        tokio::select! {
            changed = shutdown_rx.changed() => {
                if changed.is_err() || *shutdown_rx.borrow() {
                    withdraw_provider_routes(&mut state.routed_model_ids, port, context);
                    withdraw_provider_advertisement(&mut state.advertised_model_ids, context).await;
                    remove_provider_process(context).await;
                    let _ = terminate_provider_process(child).await;
                    return ProviderRunOutcome::Shutdown;
                }
            }
            status = child.wait() => {
                withdraw_provider_routes(&mut state.routed_model_ids, port, context);
                withdraw_provider_advertisement(&mut state.advertised_model_ids, context).await;
                remove_provider_process(context).await;
                return ProviderRunOutcome::Restart(match status {
                    Ok(status) => format!("provider process exited with {status}"),
                    Err(error) => format!("provider process wait failed: {error}"),
                });
            }
            _ = health_tick.tick() => {
                if let Some(outcome) = observe_provider_health(
                    runtime, context, child, &client, pid, port, &mut state,
                )
                .await
                {
                    return outcome;
                }
            }
        }
    }
}

#[derive(Default)]
struct ProviderRoutingState {
    failures: u8,
    routed_model_ids: Vec<String>,
    advertised_model_ids: Vec<String>,
}

async fn observe_provider_health(
    runtime: &ProviderRuntimeContext,
    context: &ProviderSupervisorContext,
    child: &mut Child,
    client: &reqwest::Client,
    pid: u32,
    port: u16,
    state: &mut ProviderRoutingState,
) -> Option<ProviderRunOutcome> {
    match probe_provider(client, port, &runtime.model_id).await {
        Ok(availability) => {
            state.failures = 0;
            publish_provider_state(runtime, context, pid, port, &availability).await;
            let was_unrouted = state.routed_model_ids.is_empty();
            reconcile_provider_routes(
                runtime,
                &availability,
                &mut state.routed_model_ids,
                port,
                context,
            );
            reconcile_provider_advertisement(
                &runtime.model_id,
                &availability,
                &mut state.advertised_model_ids,
                context,
            )
            .await;
            if was_unrouted && !state.routed_model_ids.is_empty() {
                emit_provider_ready(runtime, &availability, port, pid);
            }
            None
        }
        Err(error) => {
            state.failures = state.failures.saturating_add(1);
            publish_provider_unhealthy(runtime, context, pid, port).await;
            withdraw_provider_routes(&mut state.routed_model_ids, port, context);
            withdraw_provider_advertisement(&mut state.advertised_model_ids, context).await;
            if state.failures >= PROVIDER_MAX_HEALTH_FAILURES {
                let _ = terminate_provider_process(child).await;
                Some(ProviderRunOutcome::Restart(format!(
                    "provider failed {} consecutive health checks: {error:#}",
                    state.failures
                )))
            } else {
                None
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
    let versioned_model_id = validated_versioned_model_id(&model)?;
    if model.max_concurrent_requests == 0 {
        bail!("provider returned zero max_concurrent_requests");
    }
    Ok(ProviderAvailability {
        available: model
            .availability
            .as_deref()
            .is_none_or(|status| status.eq_ignore_ascii_case("available")),
        unavailable_reason: model.unavailable_reason,
        context_length: model.context_length,
        model_version: model.model_version,
        versioned_model_id,
        capabilities: model.capabilities,
        max_concurrent_requests: model.max_concurrent_requests,
        active_requests: model.active_requests,
        queued_requests: model.queued_requests,
    })
}

fn validated_versioned_model_id(model: &ProviderModelEntry) -> Result<String> {
    if !matches!(
        model.version_source.as_str(),
        "apple_os_release_band" | "coreai_model_artifact"
    ) {
        bail!("provider returned unsupported Apple model version source");
    }
    let expected = format!("{}@{}", model.id, model.model_version);
    if model.versioned_model_id != expected {
        bail!(
            "provider versioned model id mismatch: expected {expected}, got {}",
            model.versioned_model_id
        );
    }
    Ok(model.versioned_model_id.clone())
}

fn emit_provider_ready(
    runtime: &ProviderRuntimeContext,
    availability: &ProviderAvailability,
    port: u16,
    pid: u32,
) {
    let _ = emit_event(OutputEvent::Info {
        message: format!(
            "Apple system model is available through the MeshLLM OpenAI API ({})",
            runtime.model_id
        ),
        context: Some(format!(
            "provider={} version={} model_generation={} pid={pid} port={port}",
            runtime.runtime.manifest.runtime.id,
            runtime.runtime.manifest.runtime.version,
            availability.model_version
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

    #[test]
    fn embedded_discovery_roots_do_not_inherit_process_environment() {
        let explicit = PathBuf::from("/sdk/resources/provider-runtimes/apple");
        let environment = PathBuf::from("/process/provider-runtimes");
        let adjacent = PathBuf::from("/app/Resources/provider-runtimes/apple");

        let roots = discovery_roots(
            vec![explicit.clone()],
            Some(vec![environment]),
            vec![adjacent.clone()],
            false,
        );

        assert_eq!(roots, vec![explicit, adjacent]);
    }

    #[test]
    fn cli_discovery_roots_retain_environment_and_adjacent_defaults() {
        let explicit = PathBuf::from("/configured/provider-runtimes");
        let environment = PathBuf::from("/process/provider-runtimes");
        let adjacent = PathBuf::from("/bin/provider-runtimes/apple");

        let roots = discovery_roots(
            vec![explicit.clone()],
            Some(vec![environment.clone()]),
            vec![adjacent.clone()],
            true,
        );

        assert_eq!(roots, vec![explicit, environment, adjacent]);
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
        targets.targets.insert(
            "apple/system@27.0".to_string(),
            vec![election::InferenceTarget::Local(11_435)],
        );
        let (target_tx, _target_rx) = watch::channel(targets);
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Host { http_port: 9_337 })
            .await
            .unwrap();
        let context = ProviderSupervisorContext {
            target_tx: Arc::new(target_tx),
            dashboard_processes: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            console_state: None,
            node,
        };

        let mut routed_model_ids =
            vec![APPLE_MODEL_ID.to_string(), "apple/system@27.0".to_string()];
        withdraw_provider_routes(&mut routed_model_ids, 11_435, &context);

        assert_eq!(
            context.target_tx.borrow().candidates(APPLE_MODEL_ID),
            vec![election::InferenceTarget::Local(11_436)]
        );
        assert_eq!(
            context.target_tx.borrow().candidates("other/model"),
            vec![election::InferenceTarget::Local(12_345)]
        );
        assert!(
            context
                .target_tx
                .borrow()
                .candidates("apple/system@27.0")
                .is_empty()
        );
    }

    #[test]
    fn validates_documented_version_metadata_and_builds_both_routes() {
        let entry = ProviderModelEntry {
            id: APPLE_MODEL_ID.to_string(),
            availability: Some("available".to_string()),
            unavailable_reason: None,
            context_length: Some(4_096),
            model_version: "27.0".to_string(),
            version_source: "apple_os_release_band".to_string(),
            versioned_model_id: "apple/system@27.0".to_string(),
            capabilities: vec!["tool_calling".to_string()],
            max_concurrent_requests: 1,
            active_requests: 0,
            queued_requests: 0,
        };
        let versioned_model_id = validated_versioned_model_id(&entry).unwrap();
        let coreai_base = entry.clone();
        let availability = ProviderAvailability {
            available: true,
            unavailable_reason: None,
            context_length: entry.context_length,
            model_version: entry.model_version,
            versioned_model_id,
            capabilities: entry.capabilities,
            max_concurrent_requests: entry.max_concurrent_requests,
            active_requests: entry.active_requests,
            queued_requests: entry.queued_requests,
        };
        assert_eq!(
            desired_provider_routes(APPLE_MODEL_ID, &availability),
            vec!["apple/system".to_string(), "apple/system@27.0".to_string()]
        );

        let coreai = ProviderModelEntry {
            id: "apple/coreai/qwen3-4b".to_string(),
            model_version: "qwen3-4b-2026-08-01".to_string(),
            version_source: "coreai_model_artifact".to_string(),
            versioned_model_id: "apple/coreai/qwen3-4b@qwen3-4b-2026-08-01".to_string(),
            ..coreai_base
        };
        assert_eq!(
            validated_versioned_model_id(&coreai).unwrap(),
            "apple/coreai/qwen3-4b@qwen3-4b-2026-08-01"
        );
    }

    #[test]
    fn rejects_unsupported_or_mismatched_version_metadata() {
        let unsupported = ProviderModelEntry {
            id: APPLE_MODEL_ID.to_string(),
            availability: Some("available".to_string()),
            unavailable_reason: None,
            context_length: Some(4_096),
            model_version: "27.0".to_string(),
            version_source: "unknown".to_string(),
            versioned_model_id: "apple/system@27.0".to_string(),
            capabilities: vec![],
            max_concurrent_requests: 1,
            active_requests: 0,
            queued_requests: 0,
        };
        assert!(validated_versioned_model_id(&unsupported).is_err());

        let mismatched = ProviderModelEntry {
            version_source: "apple_os_release_band".to_string(),
            versioned_model_id: "apple/system@26.4".to_string(),
            ..unsupported
        };
        assert!(validated_versioned_model_id(&mismatched).is_err());
    }

    fn available_provider() -> ProviderAvailability {
        ProviderAvailability {
            available: true,
            unavailable_reason: None,
            context_length: Some(4_096),
            model_version: "27.0".to_string(),
            versioned_model_id: "apple/system@27.0".to_string(),
            capabilities: vec!["tool_calling".to_string(), "reasoning".to_string()],
            max_concurrent_requests: 1,
            active_requests: 1,
            queued_requests: 2,
        }
    }

    #[tokio::test]
    async fn private_mesh_advertises_provider_load_and_withdraws_it() {
        let node = mesh::Node::new_for_tests(mesh::NodeRole::Host { http_port: 9_337 })
            .await
            .unwrap();
        let (target_tx, _target_rx) = watch::channel(election::ModelTargets::default());
        let context = ProviderSupervisorContext {
            target_tx: Arc::new(target_tx),
            dashboard_processes: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            console_state: None,
            node: node.clone(),
        };
        let mut advertised = Vec::new();

        reconcile_provider_advertisement(
            APPLE_MODEL_ID,
            &available_provider(),
            &mut advertised,
            &context,
        )
        .await;

        assert_eq!(
            advertised,
            vec!["apple/system".to_string(), "apple/system@27.0".to_string()]
        );
        assert_eq!(node.hosted_models().await, advertised);
        let runtimes = node.all_model_runtime_descriptors().await;
        assert_eq!(runtimes.len(), 2);
        assert!(runtimes.iter().all(|runtime| {
            runtime.provider_kind.as_deref() == Some("apple")
                && runtime.max_concurrent_requests == Some(1)
                && runtime.active_requests == Some(1)
                && runtime.queued_requests == Some(2)
        }));

        withdraw_provider_advertisement(&mut advertised, &context).await;
        assert!(advertised.is_empty());
        assert!(node.hosted_models().await.is_empty());
        assert!(node.all_model_runtime_descriptors().await.is_empty());
    }

    #[tokio::test]
    async fn public_mesh_does_not_advertise_apple_system() {
        let mut node = mesh::Node::new_for_tests(mesh::NodeRole::Host { http_port: 9_337 })
            .await
            .unwrap();
        node.public_mesh = true;
        let (target_tx, _target_rx) = watch::channel(election::ModelTargets::default());
        let context = ProviderSupervisorContext {
            target_tx: Arc::new(target_tx),
            dashboard_processes: Arc::new(tokio::sync::Mutex::new(Vec::new())),
            console_state: None,
            node: node.clone(),
        };
        let mut advertised = Vec::new();

        reconcile_provider_advertisement(
            APPLE_MODEL_ID,
            &available_provider(),
            &mut advertised,
            &context,
        )
        .await;

        assert!(advertised.is_empty());
        assert!(node.hosted_models().await.is_empty());
    }
}
