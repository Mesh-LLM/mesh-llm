//! Standalone HTTP serving for Laya's decision-only native model.

use std::{
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail};
use skippy_commands::console;
use skippy_serving::EmbeddedState;

use crate::{cli::ServeCommandArgs, shutdown_signal};

pub(crate) fn is_laya(path: &Path) -> bool {
    skippy_model_artifact::gguf::scan_gguf_compact_meta(path)
        .is_some_and(|metadata| metadata.architecture == "laya")
}

pub(crate) async fn run(args: ServeCommandArgs) -> Result<()> {
    let public = args.public;
    if args.prompt || args.print_effective_config {
        bail!("Laya decision-only serving does not support prompt or effective stage config");
    }
    if public.config.is_some() || public.mmproj.is_some() || public.n_gpu_layers.is_some() {
        bail!(
            "Laya decision-only serving requires a model path and explicit --device; stage and GPU-layer options do not apply"
        );
    }
    let path = public.model_path.context("Laya model path is required")?;
    let selected_device = selected_device(&public.settings)?;
    let model_id = public.model_id.unwrap_or_else(|| {
        path.file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("laya")
            .to_owned()
    });
    let bind_addr = public
        .bind_addr
        .unwrap_or_else(super::serve::default_public_bind_addr);
    if bind_addr.port() == 0 {
        bail!("--bind-addr must use a fixed port so readiness can be reported");
    }
    console::status("🧠 Loading Laya decision model")?;
    let threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(4);
    let model = tokio::task::spawn_blocking(move || {
        skippy_runtime::LayaModel::open(&path, threads, selected_device.as_deref())
    })
    .await
    .context("join Laya model load")??;
    let backend = Arc::new(skippy_serving::LayaSystemOneBackend::new(
        model_id.clone(),
        Arc::new(model),
    ));
    let server = skippy_serving::start_openai_backend(bind_addr, backend);
    let shutdown = shutdown_signal()?;
    tokio::pin!(shutdown);
    let deadline = Instant::now() + Duration::from_secs(public.startup_timeout_secs.max(1));
    loop {
        let status = server.status();
        match status.state {
            EmbeddedState::Ready => break,
            EmbeddedState::Failed | EmbeddedState::Stopped => {
                bail!(
                    "Laya API failed before readiness: {}",
                    status.last_error.unwrap_or_default()
                )
            }
            EmbeddedState::Starting | EmbeddedState::Stopping => {}
        }
        if Instant::now() >= deadline {
            bail!("Laya API did not become ready before the startup timeout");
        }
        tokio::select! {
            _ = &mut shutdown => return server.shutdown().await,
            _ = tokio::time::sleep(Duration::from_millis(100)) => {}
        }
    }
    let api_base = format!("http://{}/v1", super::serve::readiness_addr(bind_addr));
    console::event(
        "ready",
        &serde_json::json!({"model_id": model_id, "api_base": api_base}),
    )?;
    loop {
        tokio::select! {
            _ = &mut shutdown => return server.shutdown().await,
            _ = tokio::time::sleep(Duration::from_millis(200)) => {
                let status = server.status();
                if matches!(status.state, EmbeddedState::Failed | EmbeddedState::Stopped) {
                    return server.shutdown().await.context("Laya API stopped unexpectedly");
                }
            }
        }
    }
}

fn selected_device(settings: &crate::serve_settings::ServeSettings) -> Result<Option<String>> {
    let device = settings
        .values
        .get("device")
        .and_then(serde_json::Value::as_str)
        .context(
            "Laya decision-only serving requires explicit --device CPU or a backend device name",
        )?;
    Ok((device != "CPU").then_some(device.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Command;
    use std::ffi::OsString;

    fn tiny_gguf(architecture: &str) -> Vec<u8> {
        let mut bytes = Vec::from(*b"GGUF");
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        let key = "general.architecture";
        bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
        bytes.extend_from_slice(key.as_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&(architecture.len() as u64).to_le_bytes());
        bytes.extend_from_slice(architecture.as_bytes());
        bytes
    }

    #[test]
    fn only_laya_metadata_selects_decision_only_server() {
        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("model.gguf");
        std::fs::write(&model, tiny_gguf("laya")).unwrap();
        assert!(is_laya(&model));
        std::fs::write(&model, tiny_gguf("llama")).unwrap();
        assert!(!is_laya(&model));
    }

    #[test]
    fn laya_requires_explicit_device_and_cpu_never_selects_gpu() {
        for (device, expected) in [("CPU", None), ("MTL0", Some("MTL0".to_owned()))] {
            let cli = crate::serve_settings::parse(
                [
                    "skippy",
                    "serve",
                    "--model-path",
                    "model.gguf",
                    "--device",
                    device,
                ]
                .map(OsString::from),
            )
            .unwrap();
            let Command::Serve(args) = cli.command else {
                panic!("expected serve")
            };
            assert_eq!(selected_device(&args.public.settings).unwrap(), expected);
        }
        let cli = crate::serve_settings::parse(
            ["skippy", "serve", "--model-path", "model.gguf"].map(OsString::from),
        )
        .unwrap();
        let Command::Serve(args) = cli.command else {
            panic!("expected serve")
        };
        assert!(selected_device(&args.public.settings).is_err());
    }
}
