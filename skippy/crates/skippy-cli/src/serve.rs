//! One public serving entry point with binary internal stage transport.

use std::{
    future::Future,
    io::IsTerminal,
    net::SocketAddr,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use skippy_commands::console::{self, OutputMode};
use skippy_runtime::{ModelOpenEventQueue, RuntimeEventKind, RuntimeEventProgressUnit};
use tokio::{process::Command, sync::oneshot, task::JoinHandle};

use crate::{
    cli::{ServeBinaryArgs, ServeCommandArgs, ServeOpenAiArgs, StageTransport},
    conversion, shutdown_signal,
};

pub(crate) fn default_public_bind_addr() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 9337))
}

pub async fn run(mut args: ServeCommandArgs) -> Result<()> {
    validate(&args)?;
    if let Some(model) = args.model.take() {
        console::status(&format!("🔎 Checking local model cache for {model}"))?;
        let existing = installed_model_path(&model);
        let catalog_model = (!Path::new(&model).exists())
            .then(|| {
                let exact = if existing.is_some() && model.contains('/') {
                    skippy_model_hf::remote_catalog::find_loaded_model_exact(&model)
                } else {
                    skippy_model_hf::remote_catalog::find_model_exact(&model)
                };
                exact.or_else(|| {
                    (!model.contains('/'))
                        .then(|| skippy_model_hf::remote_catalog::resolve_model_download(&model))
                        .flatten()
                        .and_then(|resolved| {
                            skippy_model_hf::remote_catalog::matching_model_for_huggingface(
                                &resolved.repo,
                                resolved.revision.as_deref(),
                                &resolved.file,
                            )
                        })
                })
            })
            .flatten();
        let (path, downloaded_projector) = if let Some(local_path) = existing {
            console::status("📂 Using cached model")?;
            (local_path, None)
        } else {
            let cache = skippy_commands::models::model_cache_dir();
            let downloaded =
                skippy_commands::models::download_model(&cache, &model, None, None).await?;
            (downloaded.load_path, downloaded.projector_path)
        };
        args.public.model_path = Some(path);
        if args.public.mmproj.is_none() {
            args.public.mmproj = downloaded_projector;
        }
        if args.public.mmproj.is_none()
            && let Some(asset) = catalog_model.and_then(|entry| entry.mmproj)
        {
            let projector_ref = asset.download_ref();
            let cache = skippy_commands::models::model_cache_dir();
            args.public.mmproj = Some(
                skippy_commands::models::download_model(&cache, &projector_ref, None, None)
                    .await?
                    .primary_path,
            );
        }
        if !Path::new(&model).exists() {
            args.public.model_id.get_or_insert(model);
        }
    }
    match args.stage_transport {
        Some(StageTransport::Binary) => serve_binary_stage(args).await,
        None => serve_public(args).await,
    }
}

fn installed_model_path(model: &str) -> Option<std::path::PathBuf> {
    let path = Path::new(model);
    if path.is_file() || path.is_dir() {
        return Some(path.to_path_buf());
    }
    // Mesh prefers catalog resolution for short curated names, then falls
    // back to local cache stems. Exact Hub refs can use the cache directly.
    if model.contains('/') || skippy_model_hf::remote_catalog::find_model_exact(model).is_none() {
        let cached = skippy_model_hf::store::local::find_model_path(model);
        if cached.exists() {
            return Some(cached);
        }
    }
    None
}

pub(crate) fn validate(args: &ServeCommandArgs) -> Result<()> {
    args.public.settings.validate_mode(args)?;
    args.public.settings.tuning(args.public.openai_guardrails)?;
    let mut stage = skippy_protocol::StageConfig::default();
    args.public.settings.apply_stage(&mut stage)?;
    if args.stage_transport.is_some() && args.public.config.is_none() {
        bail!("--stage-transport requires --config, supplied directly or through serving settings");
    }
    if args.worker_only && args.prompt {
        bail!("--prompt requires a public inference API");
    }
    if args.worker_only && args.stage_transport.is_none() {
        bail!("--worker-only requires --stage-transport");
    }
    if args.prompt && (!std::io::stdin().is_terminal() || console::mode() != OutputMode::Human) {
        bail!("--prompt requires an interactive terminal and human output");
    }
    if console::mode() == OutputMode::Json && !args.print_effective_config {
        bail!("serving produces a stream of events; use --output jsonl or --output human");
    }
    if args.public.config.is_none() && args.public.model_path.is_none() && args.model.is_none() {
        bail!("provide --model, --model-path, or --config");
    }
    Ok(())
}

async fn serve_public(args: ServeCommandArgs) -> Result<()> {
    let bind_addr = args
        .public
        .bind_addr
        .unwrap_or_else(default_public_bind_addr);
    let startup_timeout = Duration::from_secs(args.public.startup_timeout_secs.max(1));
    console::status("🧠 Preparing model")?;
    let settings = args.public.settings.clone();
    let mut options = conversion::local_openai_options(args.public)?;
    if args.print_effective_config {
        let frontend = options.resolved_openai_options()?;
        return console::write_json(&settings.report(
            &options.config,
            Some(&frontend),
            &options.tuning,
        ));
    }
    let model_open_events = ModelOpenEventQueue::new(skippy_runtime::next_operation_id());
    options.model_open_events = Some(Arc::clone(&model_open_events));
    let model_id = options
        .model_id
        .clone()
        .unwrap_or_else(|| options.config.model_id.clone());
    let (stop, stopped) = oneshot::channel();
    let shutdown = shutdown_signal()?;
    let server = skippy_api::serving::serve_local_openai_with_shutdown(options, async move {
        tokio::select! { _ = shutdown => {}, _ = stopped => {} }
    });
    serve_with_readiness(
        server,
        bind_addr,
        model_id,
        args.prompt,
        stop,
        startup_timeout,
        Some(model_open_events),
    )
    .await
}

async fn serve_binary_stage(mut args: ServeCommandArgs) -> Result<()> {
    let disk_cache = crate::disk_cache::from_public_settings(
        args.public.kv_cache_disk.as_deref(),
        args.public.kv_cache_disk_dir.clone(),
        args.public.kv_cache_min_free.as_deref(),
    )?;
    apply_public_frontend_tuning(&args.public, &mut args.stage);
    args.stage.settings = args.public.settings.clone();
    let startup_timeout = Duration::from_secs(args.public.startup_timeout_secs.max(1));
    args.stage.config = args.public.config.clone().context("--config is required")?;
    args.stage.topology = args.public.topology.clone();
    args.stage.metrics_otlp_grpc = args.public.metrics_otlp_grpc.clone();
    args.stage.telemetry_queue_capacity = args.public.telemetry_queue_capacity;
    args.stage.telemetry_level = args.public.telemetry_level;
    args.stage.worker_only = args.worker_only;
    if args.worker_only {
        args.stage.bind_addr = args.public.bind_addr;
    }
    args.stage.api_bind_addr = Some(
        args.public
            .bind_addr
            .unwrap_or_else(default_public_bind_addr),
    );
    let mut options = conversion::binary_stage_options(args.stage)?;
    args.public
        .settings
        .validate_cache_dependencies(&options.config, disk_cache.is_some())?;
    if args.print_effective_config {
        let frontend = options.openai.as_ref().map(|stage| {
            let mut frontend = skippy_api::serving::OpenAiOptions::embedded_stage_defaults(
                stage.model_id.clone(),
                stage.default_max_tokens,
                stage.generation_concurrency,
                0,
                options.native_mtp_enabled,
            );
            frontend.request_defaults = options.tuning.request_defaults.clone();
            frontend.continuous_batching = options.continuous_batching;
            frontend.pipeline_decode_groups = stage.pipeline_decode_groups;
            frontend.adaptive_generation_min_concurrency =
                stage.adaptive_generation_min_concurrency;
            frontend.generation_queue_capacity = stage.generation_queue_capacity;
            frontend.generation_admission_timeout_secs = stage.generation_admission_timeout_secs;
            frontend.prefill_chunk_size = stage.prefill_chunk_size;
            frontend.prefill_chunk_policy = stage.prefill_chunk_policy.clone();
            frontend.prefill_chunk_schedule = stage.prefill_chunk_schedule.clone();
            frontend.prefill_adaptive_start = stage.prefill_adaptive_start;
            frontend.prefill_adaptive_step = stage.prefill_adaptive_step;
            frontend.prefill_adaptive_max = stage.prefill_adaptive_max;
            frontend.prefill_adaptive_target_ms = stage.prefill_adaptive_target_ms;
            frontend.draft_model_path = stage.draft_model_path.clone();
            frontend.speculative_window = stage.speculative_window;
            frontend.adaptive_speculative_window = stage.adaptive_speculative_window;
            frontend.draft_n_gpu_layers = stage.draft_n_gpu_layers;
            frontend.speculative = stage.speculative.clone();
            frontend.native_mtp_draft_model_path = stage.native_mtp_draft_model_path.clone();
            frontend.native_mtp_max_tokens = stage.native_mtp_max_tokens;
            frontend.native_mtp_min_tokens = stage.native_mtp_min_tokens;
            frontend.reply_credit_limit = options.reply_credit_limit;
            frontend.downstream_connect_timeout_secs = options.downstream_connect_timeout_secs;
            frontend
        });
        return console::write_json(&args.public.settings.report(
            &options.config,
            frontend.as_ref(),
            &options.tuning,
        ));
    }
    options.l3_manager = disk_cache.and_then(skippy_api::serving::LocalDiskCacheOptions::acquire);
    let Some(openai) = options.openai.as_ref() else {
        if args.prompt {
            bail!("--prompt requires stage 0 to expose the public inference API");
        }
        if !args.worker_only {
            bail!("this stage has no public API; add --worker-only or serve stage 0");
        }
        let bind_addr = options.bind_addr;
        let model_id = options.config.model_id.clone();
        let server = skippy_serving::binary_transport::serve_binary_stage_with_shutdown(
            options,
            shutdown_signal()?,
        );
        return serve_worker_with_readiness(server, bind_addr, model_id, "binary", startup_timeout)
            .await;
    };
    let bind_addr = openai.bind_addr;
    let model_id = openai
        .model_id
        .clone()
        .unwrap_or_else(|| options.config.model_id.clone());
    let (stop, stopped) = oneshot::channel();
    let shutdown = shutdown_signal()?;
    let server =
        skippy_serving::binary_transport::serve_binary_stage_with_shutdown(options, async move {
            tokio::select! { _ = shutdown => {}, _ = stopped => {} }
        });
    serve_with_readiness(
        server,
        bind_addr,
        model_id,
        args.prompt,
        stop,
        startup_timeout,
        None,
    )
    .await
}

pub(crate) fn apply_public_frontend_tuning(public: &ServeOpenAiArgs, stage: &mut ServeBinaryArgs) {
    stage.openai_model_id = public.model_id.clone();
    stage.openai_default_max_tokens = public.default_max_tokens;
    stage.openai_generation_concurrency = public.generation_concurrency;
    stage.openai_adaptive_generation_concurrency = public.adaptive_generation_concurrency;
    stage.openai_adaptive_generation_min_concurrency = public.adaptive_generation_min_concurrency;
    stage.openai_generation_queue_capacity = public.generation_queue_capacity;
    stage.openai_generation_admission_timeout_secs = public.generation_admission_timeout_secs;
    stage.openai_prefill_chunk_size = public.prefill_chunk_size;
    stage.openai_prefill_chunk_policy = public.prefill_chunk_policy.clone();
    stage.openai_prefill_chunk_schedule = public.prefill_chunk_schedule.clone();
    stage.openai_prefill_adaptive_start = public.prefill_adaptive_start;
    stage.openai_prefill_adaptive_step = public.prefill_adaptive_step;
    stage.openai_prefill_adaptive_max = public.prefill_adaptive_max;
    stage.openai_prefill_adaptive_target_ms = public.prefill_adaptive_target_ms;
    stage.openai_speculative_config = public.speculative_config.clone();
}

async fn serve_worker_with_readiness(
    server: impl Future<Output = Result<()>> + Send + 'static,
    bind_addr: SocketAddr,
    model_id: String,
    transport: &str,
    startup_timeout: Duration,
) -> Result<()> {
    if bind_addr.port() == 0 {
        bail!("stage listener must use a fixed port so readiness can be reported");
    }
    let mut server = tokio::spawn(server);
    console::status(&format!("🧠 Loading {transport} stage for {model_id}"))?;
    let deadline = Instant::now() + startup_timeout;
    let probe = readiness_addr(bind_addr);
    loop {
        tokio::select! {
            outcome = &mut server => {
                outcome.context("join serving task")??;
                bail!("stage exited before its listener became ready");
            }
            connection = tokio::net::TcpStream::connect(probe) => {
                if connection.is_ok() && !server.is_finished() { break; }
            }
        }
        if Instant::now() >= deadline {
            bail!(
                "stage did not become ready at {bind_addr} within {} seconds",
                startup_timeout.as_secs()
            );
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
    console::event(
        "ready",
        &serde_json::json!({
            "model_id": model_id, "transport": transport, "bind_addr": bind_addr,
        }),
    )?;
    if console::mode() == OutputMode::Human {
        console::status(&format!("✅ {transport} stage ready at {bind_addr}"))?;
    }
    server.await.context("join serving task")?
}

async fn serve_with_readiness(
    server: impl Future<Output = Result<()>> + Send + 'static,
    bind_addr: SocketAddr,
    model_id: String,
    prompt: bool,
    stop: oneshot::Sender<()>,
    startup_timeout: Duration,
    model_open_events: Option<Arc<ModelOpenEventQueue>>,
) -> Result<()> {
    if bind_addr.port() == 0 {
        bail!("--bind-addr must use a fixed port so readiness can be reported");
    }
    let api_base = format!("http://{}/v1", readiness_addr(bind_addr));
    let mut server = tokio::spawn(server);
    console::status("🧠 Loading model and starting the API")?;
    wait_for_ready(
        &api_base,
        &model_id,
        &mut server,
        startup_timeout,
        model_open_events.as_deref(),
    )
    .await?;
    console::event(
        "ready",
        &serde_json::json!({"model_id":model_id,"api_base":api_base}),
    )?;
    if console::mode() == OutputMode::Human {
        console::status(&format!("✅ {model_id} ready at {api_base}"))?;
    }
    if !prompt {
        return server.await.context("join serving task")?;
    }
    // Readline cannot interrupt a blocked terminal read. Keep it in a child
    // process so server shutdown can terminate and join the prompt reliably.
    let mut prompt_process =
        Command::new(std::env::current_exe().context("locate skippy executable")?)
            .args([
                "--output",
                "human",
                "prompt",
                "--endpoint",
                &api_base,
                "--model",
                &model_id,
            ])
            .kill_on_drop(true)
            .spawn()
            .context("start interactive prompt")?;
    tokio::select! {
        result = prompt_process.wait() => {
            let status = result.context("join interactive prompt")?;
            let _ = stop.send(());
            server.await.context("join serving task")??;
            if !status.success() {
                bail!("interactive prompt exited with {status}");
            }
            Ok(())
        }
        result = &mut server => {
            let _ = prompt_process.start_kill();
            prompt_process.wait().await.context("join interactive prompt after server shutdown")?;
            result.context("join serving task")??;
            Ok(())
        }
    }
}

fn readiness_addr(bind_addr: SocketAddr) -> SocketAddr {
    // A wildcard listener accepts loopback; a LAN-only listener does not.
    if bind_addr.ip().is_unspecified() {
        SocketAddr::new(
            if bind_addr.is_ipv4() {
                std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST)
            } else {
                std::net::IpAddr::V6(std::net::Ipv6Addr::LOCALHOST)
            },
            bind_addr.port(),
        )
    } else {
        bind_addr
    }
}

async fn wait_for_ready(
    api_base: &str,
    model_id: &str,
    server: &mut JoinHandle<Result<()>>,
    startup_timeout: Duration,
    model_open_events: Option<&ModelOpenEventQueue>,
) -> Result<()> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(1))
        .build()?;
    let deadline = Instant::now() + startup_timeout;
    loop {
        if let Some(queue) = model_open_events {
            report_model_open_events(queue)?;
        }
        tokio::select! {
            outcome = &mut *server => {
                outcome.context("join serving task")??;
                bail!("server exited before the API became ready");
            }
            response = client.get(format!("{api_base}/models")).send() => {
                if let Ok(response) = response && response.status().is_success() {
                    let body: serde_json::Value = response.json().await.context("read ready model list")?;
                    let listed = body["data"].as_array().is_some_and(|items| items.iter().any(|item| item["id"] == model_id));
                    if listed && !server.is_finished() {
                        if let Some(queue) = model_open_events {
                            report_model_open_events(queue)?;
                        }
                        return Ok(());
                    }
                }
            }
        }
        if Instant::now() >= deadline {
            bail!(
                "API did not become ready within {} seconds at {api_base}",
                startup_timeout.as_secs()
            );
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn report_model_open_events(queue: &ModelOpenEventQueue) -> Result<()> {
    let mut records = Vec::new();
    queue.drain(&mut records, usize::MAX);
    for record in records {
        let event = record.to_event();
        match event.kind {
            RuntimeEventKind::ModelOpenProgress if event.progress_total > 0 => {
                let unit = match event.progress_unit {
                    RuntimeEventProgressUnit::Bytes => "bytes",
                    RuntimeEventProgressUnit::Items => "items",
                    RuntimeEventProgressUnit::Tensors => "tensors",
                    RuntimeEventProgressUnit::Steps => "steps",
                    RuntimeEventProgressUnit::None | RuntimeEventProgressUnit::Unknown(_) => {
                        "units"
                    }
                };
                console::progress_with_unit(
                    "Model load",
                    event.progress_current,
                    event.progress_total,
                    unit,
                )?;
            }
            RuntimeEventKind::BackendDeviceSelected => {
                console::event(
                    "backend_device_selected",
                    &serde_json::json!({"detail": String::from_utf8_lossy(&event.detail_bytes)}),
                )?;
            }
            RuntimeEventKind::ModelOpenFailedHandled => {
                console::event(
                    "model_open_failed_handled",
                    &serde_json::json!({"detail": String::from_utf8_lossy(&event.detail_bytes)}),
                )?;
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Command};
    use clap::Parser;

    #[test]
    fn binary_worker_bind_override_is_explicit() {
        for (flags, expected) in [
            (Vec::<&str>::new(), None),
            (
                vec!["--bind-addr", "192.0.2.10:9400"],
                Some("192.0.2.10:9400".parse().unwrap()),
            ),
        ] {
            let mut argv = vec![
                "skippy",
                "serve",
                "--config",
                "stage.json",
                "--worker-only",
                "--stage-transport",
                "binary",
            ];
            argv.extend(flags);
            let cli = Cli::try_parse_from(argv).unwrap();
            let Command::Serve(args) = cli.command else {
                panic!("expected serve");
            };
            assert_eq!(args.public.bind_addr, expected);
        }
    }

    #[test]
    fn readiness_uses_the_bound_interface_for_lan_only_listeners() {
        for (bound, expected) in [
            ("0.0.0.0:9337", "127.0.0.1:9337"),
            ("[::]:9337", "[::1]:9337"),
            ("192.0.2.10:9337", "192.0.2.10:9337"),
            ("[2001:db8::10]:9337", "[2001:db8::10]:9337"),
        ] {
            assert_eq!(
                readiness_addr(bound.parse().unwrap()),
                expected.parse().unwrap()
            );
        }
    }

    #[test]
    fn existing_local_model_path_is_used_without_hub_resolution() {
        let temp = tempfile::tempdir().unwrap();
        let model = temp.path().join("model.gguf");
        std::fs::write(&model, b"GGUF").unwrap();
        assert_eq!(installed_model_path(model.to_str().unwrap()), Some(model));
    }

    #[test]
    fn common_frontend_flags_reach_binary_stage_zero() {
        let cli = Cli::try_parse_from([
            "skippy",
            "serve",
            "--config",
            "stage-0.json",
            "--stage-transport",
            "binary",
            "--model-id",
            "served-model",
            "--default-max-tokens",
            "64",
            "--generation-concurrency",
            "2",
            "--prefill-chunk-size",
            "512",
        ])
        .unwrap();
        let Command::Serve(mut args) = cli.command else {
            panic!("expected serve");
        };
        apply_public_frontend_tuning(&args.public, &mut args.stage);
        args.stage.settings = args.public.settings.clone();
        assert_eq!(args.stage.openai_model_id.as_deref(), Some("served-model"));
        assert_eq!(args.stage.openai_default_max_tokens, 64);
        assert_eq!(args.stage.openai_generation_concurrency, Some(2));
        assert_eq!(args.stage.openai_prefill_chunk_size, 512);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn binary_stage_defaults_reach_config_loading() {
        let directory = tempfile::tempdir().unwrap();
        let config = directory.path().join("missing-stage.json");
        let cli = Cli::try_parse_from([
            "skippy",
            "serve",
            "--config",
            config.to_str().unwrap(),
            "--stage-transport",
            "binary",
            "--worker-only",
        ])
        .unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("expected serve");
        };
        let error = serve_binary_stage(*args).await.unwrap_err();
        assert!(error.to_string().contains("load stage config"), "{error:#}");
    }
}
