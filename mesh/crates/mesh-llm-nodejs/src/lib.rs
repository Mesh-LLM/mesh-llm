// napi registration glue requires generated unsafe code; handwritten unsafe remains denied.
#![deny(unsafe_code)]

use mesh_llm_sdk::embedded_node::{SseDecoder, SseFrame};
use mesh_llm_sdk::{MeshNode, MeshNodeBuilder, OpenAiClient};
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;

fn parse_json(source: &str) -> Result<Value> {
    serde_json::from_str(source).map_err(to_napi_error)
}

#[napi(js_name = "currentMeshVersion")]
pub fn current_mesh_version() -> String {
    mesh_llm_sdk::native_runtime::CURRENT_MESH_VERSION.to_string()
}

#[napi(js_name = "currentSkippyAbiVersion")]
pub fn current_skippy_abi_version() -> String {
    mesh_llm_sdk::native_runtime::current_skippy_abi_version()
}

#[napi(js_name = "installNativeRuntimeJson")]
pub async fn install_native_runtime_json(
    options_json: String,
    progress: Option<ThreadsafeFunction<String>>,
) -> Result<String> {
    let mut options = parse_native_runtime_install_options(&options_json)?;
    if let Some(progress) = progress {
        options.progress = Some(native_runtime_progress_callback(progress));
    }
    let outcome = mesh_llm_sdk::native_runtime::install_native_runtime(options)
        .await
        .map_err(to_napi_error)?;
    Ok(native_runtime_install_outcome_json(outcome).to_string())
}

#[napi(js_name = "installedNativeRuntimesJson")]
pub fn installed_native_runtimes_json(cache_dir: Option<String>) -> Result<String> {
    let cache = native_runtime_cache(cache_dir)?;
    let runtimes = mesh_llm_sdk::native_runtime::discover_local_native_runtimes(&[], &cache)
        .map_err(to_napi_error)?;
    Ok(Value::Array(
        runtimes
            .into_iter()
            .map(installed_native_runtime_json)
            .collect(),
    )
    .to_string())
}

#[napi(js_name = "removeNativeRuntime")]
pub fn remove_native_runtime(
    cache_dir: Option<String>,
    mesh_version: String,
    native_runtime_id: String,
) -> Result<bool> {
    native_runtime_cache(cache_dir)?
        .remove(&mesh_version, &native_runtime_id)
        .map_err(to_napi_error)
}

#[napi(js_name = "pruneNativeRuntimesJson")]
pub fn prune_native_runtimes_json(
    cache_dir: Option<String>,
    active_mesh_version: Option<String>,
    mode: Option<String>,
) -> Result<String> {
    let active_mesh_version = active_mesh_version
        .unwrap_or_else(|| mesh_llm_sdk::native_runtime::current_runtime_release().to_string());
    let mode = parse_native_runtime_prune_mode(mode.as_deref())?;
    let plan = native_runtime_cache(cache_dir)?
        .prune(&active_mesh_version, mode)
        .map_err(to_napi_error)?;
    Ok(json!({
        "removedDirs": plan.remove_dirs.into_iter().map(path_to_string).collect::<Vec<_>>()
    })
    .to_string())
}

#[napi]
pub struct Node {
    builder: MeshNodeBuilder,
    node: tokio::sync::Mutex<Option<MeshNode>>,
    mode: String,
    streams: Arc<Mutex<HashMap<String, oneshot::Sender<()>>>>,
}

#[napi]
impl Node {
    #[napi(factory)]
    pub fn create(
        mode: String,
        join_tokens: Vec<String>,
        models: Vec<String>,
        auto_join: bool,
        owner_key_path: Option<String>,
        api_port: u16,
        console_port: u16,
    ) -> Result<Self> {
        let mut builder = MeshNode::builder()
            .join_tokens(join_tokens)
            .models(models)
            .auto_join(auto_join)
            .api_port(api_port)
            .console_port(console_port);
        builder = match mode.as_str() {
            "client" => builder.client(),
            "serve" => builder.serve_only(),
            "combined" => builder.serve(),
            _ => return Err(Error::from_reason(format!("unknown node mode: {mode}"))),
        };
        if let Some(path) = owner_key_path {
            builder = builder.owner_key(path);
        }
        Ok(Self {
            builder,
            node: tokio::sync::Mutex::new(None),
            mode,
            streams: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    #[napi]
    pub async fn start(&self) -> Result<()> {
        let mut node = self.node.lock().await;
        if node.is_none() {
            *node = Some(self.builder.clone().start().await.map_err(to_napi_error)?);
        }
        Ok(())
    }

    #[napi]
    pub async fn stop(&self) -> Result<()> {
        for (_, cancel) in self.streams.lock().map_err(to_napi_error)?.drain() {
            let _ = cancel.send(());
        }
        if let Some(node) = self.node.lock().await.take() {
            node.stop().await.map_err(to_napi_error)?;
        }
        Ok(())
    }

    #[napi(js_name = "statusJson")]
    pub async fn status_json(&self) -> Result<String> {
        let node = self.node.lock().await;
        let Some(node) = node.as_ref() else {
            return Ok(json!({"running": false, "mode": self.mode}).to_string());
        };
        let status = node.status().await.map_err(to_napi_error)?;
        Ok(json!({
            "running": true,
            "mode": self.mode,
            "apiBaseUrl": status.api_base_url,
            "consoleUrl": status.console_url,
            "payload": status.payload,
        })
        .to_string())
    }

    #[napi(js_name = "joinToken")]
    pub async fn join_token(&self, token: String) -> Result<()> {
        let node = self.node.lock().await;
        node.as_ref()
            .ok_or_else(|| Error::from_reason("node is not running"))?
            .join_token(token)
            .await
            .map_err(to_napi_error)
    }

    #[napi(js_name = "listModelsJson")]
    pub async fn list_models_json(&self) -> Result<String> {
        self.openai_client()
            .await?
            .models()
            .await
            .map(|value| {
                value
                    .get("data")
                    .cloned()
                    .unwrap_or_else(|| json!([]))
                    .to_string()
            })
            .map_err(to_napi_error)
    }

    #[napi(js_name = "openaiRequestJson")]
    pub async fn openai_request_json(&self, path: String, body_json: String) -> Result<String> {
        let response = self
            .openai_client()
            .await?
            .request(&path, body_json)
            .await
            .map_err(to_napi_error)?;
        Ok(json!({
            "statusCode": response.status_code,
            "contentType": response.content_type,
            "body": response.body,
        })
        .to_string())
    }

    #[napi(js_name = "openaiStream")]
    pub async fn openai_stream(
        &self,
        path: String,
        body_json: String,
        callback: ThreadsafeFunction<String>,
    ) -> Result<String> {
        let response = self
            .openai_client()
            .await?
            .stream(&path, body_json)
            .await
            .map_err(to_napi_error)?;
        let id = format!("stream-{}", uuid::Uuid::new_v4());
        let task_id = id.clone();
        let streams = self.streams.clone();
        let (cancel_tx, cancel_rx) = oneshot::channel();
        let (ready_tx, ready_rx) = oneshot::channel();
        tokio::spawn(async move {
            if ready_rx.await.is_err() {
                return;
            }
            tokio::select! {
                biased;
                () = emit_stream(response, task_id.clone(), &callback) => {}
                _ = cancel_rx => emit(&callback, json!({
                    "type": "failed", "requestId": task_id,
                    "statusCode": null, "error": "stream cancelled", "body": null,
                })),
            }
            if let Ok(mut guard) = streams.lock() {
                guard.remove(&task_id);
            }
        });
        self.streams
            .lock()
            .map_err(to_napi_error)?
            .insert(id.clone(), cancel_tx);
        let _ = ready_tx.send(());
        Ok(id)
    }

    #[napi]
    pub async fn cancel(&self, request_id: String) -> Result<()> {
        if let Some(cancel) = self
            .streams
            .lock()
            .map_err(to_napi_error)?
            .remove(&request_id)
        {
            let _ = cancel.send(());
        }
        Ok(())
    }
}

impl Node {
    async fn openai_client(&self) -> Result<OpenAiClient> {
        self.node
            .lock()
            .await
            .as_ref()
            .ok_or_else(|| Error::from_reason("node is not running"))?
            .openai_client()
            .map_err(to_napi_error)
    }
}

async fn emit_stream(
    mut response: reqwest::Response,
    request_id: String,
    callback: &ThreadsafeFunction<String>,
) {
    let status_code = response.status().as_u16();
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(ToString::to_string);
    emit(
        callback,
        json!({
            "type": "started", "requestId": request_id,
            "statusCode": status_code, "contentType": content_type,
        }),
    );
    if !(200..300).contains(&status_code)
        || !content_type
            .as_deref()
            .is_some_and(|value| value.starts_with("text/event-stream"))
    {
        let body = response.text().await.ok();
        emit(
            callback,
            json!({
                "type": "failed", "requestId": request_id,
                "statusCode": status_code, "error": format!("expected streaming response; HTTP {status_code}"), "body": body,
            }),
        );
        return;
    }
    let mut decoder = SseDecoder::default();
    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => match decoder.push(&chunk) {
                Ok(frames) => {
                    for frame in frames {
                        emit_frame(callback, &request_id, frame);
                    }
                }
                Err(error) => {
                    emit_stream_error(callback, &request_id, status_code, error.to_string());
                    return;
                }
            },
            Ok(None) => {
                match decoder.finish() {
                    Ok(Some(frame)) => emit_frame(callback, &request_id, frame),
                    Ok(None) => {}
                    Err(error) => {
                        emit_stream_error(callback, &request_id, status_code, error.to_string());
                        return;
                    }
                }
                emit(
                    callback,
                    json!({"type": "completed", "requestId": request_id}),
                );
                return;
            }
            Err(error) => {
                emit_stream_error(callback, &request_id, status_code, error.to_string());
                return;
            }
        }
    }
}

fn emit_frame(callback: &ThreadsafeFunction<String>, request_id: &str, frame: SseFrame) {
    emit(
        callback,
        json!({
            "type": "sse", "requestId": request_id,
            "event": frame.event_type, "data": frame.data, "raw": frame.raw,
        }),
    );
}

fn emit_stream_error(
    callback: &ThreadsafeFunction<String>,
    request_id: &str,
    status_code: u16,
    error: String,
) {
    emit(
        callback,
        json!({
            "type": "failed", "requestId": request_id,
            "statusCode": status_code, "error": error, "body": null,
        }),
    );
}

fn emit(callback: &ThreadsafeFunction<String>, value: Value) {
    let _ = callback.call(
        Ok(value.to_string()),
        ThreadsafeFunctionCallMode::NonBlocking,
    );
}

fn parse_native_runtime_install_options(
    source: &str,
) -> Result<mesh_llm_sdk::native_runtime::NativeRuntimeInstallOptions> {
    let value = parse_json(source)?;
    Ok(mesh_llm_sdk::native_runtime::NativeRuntimeInstallOptions {
        catalog: mesh_llm_sdk::native_runtime::mesh_native_runtime_catalog(),
        release_version: optional_string(&value, "meshVersion")
            .unwrap_or_else(|| mesh_llm_sdk::native_runtime::current_runtime_release().to_string()),
        skippy_abi_version: optional_string(&value, "skippyAbiVersion"),
        selection: mesh_llm_sdk::native_runtime::RuntimeSelection::parse(
            optional_string(&value, "selection").as_deref(),
        )
        .map_err(to_napi_error)?,
        manifest_path: optional_string(&value, "manifestPath").map(PathBuf::from),
        manifest_url: optional_string(&value, "manifestUrl"),
        bundle_dirs: string_array(&value, "bundleDirs")
            .into_iter()
            .map(PathBuf::from)
            .collect(),
        cache_dir: optional_string(&value, "cacheDir").map(PathBuf::from),
        verification_policy: parse_native_runtime_verification_policy(
            optional_string(&value, "verificationPolicy").as_deref(),
        )?,
        bundle_install_policy: Default::default(),
        progress: None,
        allow_download: value
            .get("allowDownload")
            .and_then(Value::as_bool)
            .unwrap_or(true),
    })
}

fn optional_string(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(ToOwned::to_owned)
}

fn string_array(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect()
}

fn parse_native_runtime_verification_policy(
    value: Option<&str>,
) -> Result<mesh_llm_sdk::native_runtime::NativeRuntimeVerificationPolicy> {
    match value.unwrap_or("require_checksum") {
        "require_checksum" | "RequireChecksum" => {
            Ok(mesh_llm_sdk::native_runtime::NativeRuntimeVerificationPolicy::RequireChecksum)
        }
        "require_checksum_and_signature" | "RequireChecksumAndSignature" => Ok(
            mesh_llm_sdk::native_runtime::NativeRuntimeVerificationPolicy::RequireChecksumAndSignature,
        ),
        other => Err(Error::from_reason(format!(
            "unsupported native runtime verification policy: {other}"
        ))),
    }
}

fn parse_native_runtime_prune_mode(
    value: Option<&str>,
) -> Result<mesh_llm_sdk::native_runtime::NativeRuntimePruneMode> {
    match value.unwrap_or("keep_active_and_previous") {
        "keep_active_and_previous" | "KeepActiveAndPrevious" => {
            Ok(mesh_llm_sdk::native_runtime::NativeRuntimePruneMode::KeepActiveAndPrevious)
        }
        "active_only" | "ActiveOnly" => {
            Ok(mesh_llm_sdk::native_runtime::NativeRuntimePruneMode::ActiveOnly)
        }
        other => Err(Error::from_reason(format!(
            "unsupported native runtime prune mode: {other}"
        ))),
    }
}

fn native_runtime_cache(
    cache_dir: Option<String>,
) -> Result<mesh_llm_sdk::native_runtime::NativeRuntimeCache> {
    let cache_dir = cache_dir.map(PathBuf::from);
    mesh_llm_sdk::native_runtime::native_runtime_cache(cache_dir.as_deref()).map_err(to_napi_error)
}

fn native_runtime_install_outcome_json(
    outcome: mesh_llm_sdk::native_runtime::NativeRuntimeInstallOutcome,
) -> Value {
    json!({
        "status": match outcome.status {
            mesh_llm_sdk::native_runtime::NativeRuntimeInstallStatus::AlreadyInstalled => "already_installed",
            mesh_llm_sdk::native_runtime::NativeRuntimeInstallStatus::Installed => "installed",
        },
        "runtime": installed_native_runtime_json(outcome.runtime),
        "selectedNativeRuntimeId": outcome.resolution.selected.id,
        "selectedSource": native_runtime_source_name(&outcome.resolution.source),
    })
}

fn native_runtime_progress_json(
    event: mesh_llm_sdk::native_runtime::NativeRuntimeDownloadProgress,
) -> Value {
    json!({
        "nativeRuntimeId": event.native_runtime_id,
        "url": event.url,
        "downloadedBytes": event.downloaded_bytes,
        "totalBytes": event.total_bytes,
        "finished": event.finished,
    })
}

fn native_runtime_progress_callback(
    progress: ThreadsafeFunction<String>,
) -> mesh_llm_sdk::native_runtime::NativeRuntimeDownloadProgressCallback {
    Arc::new(move |event| {
        let _ = progress.call(
            Ok(native_runtime_progress_json(event).to_string()),
            ThreadsafeFunctionCallMode::NonBlocking,
        );
    })
}

fn installed_native_runtime_json(
    runtime: mesh_llm_sdk::native_runtime::InstalledNativeRuntime,
) -> Value {
    json!({
        "meshVersion": runtime.release_version,
        "nativeRuntimeId": runtime.native_runtime_id,
        "flavor": runtime.flavor,
        "path": path_to_string(runtime.path),
        "skippyAbiVersion": runtime.manifest.runtime.skippy_abi,
    })
}

fn native_runtime_source_name(source: &mesh_llm_sdk::native_runtime::NativeRuntimeSource) -> &str {
    match source {
        mesh_llm_sdk::native_runtime::NativeRuntimeSource::Installed { .. } => "installed",
        mesh_llm_sdk::native_runtime::NativeRuntimeSource::Bundle { .. } => "bundle",
        mesh_llm_sdk::native_runtime::NativeRuntimeSource::Download { .. } => "download",
        mesh_llm_sdk::native_runtime::NativeRuntimeSource::Missing => "missing",
    }
}

fn path_to_string(path: PathBuf) -> String {
    path.display().to_string()
}

fn to_napi_error(error: impl ToString) -> Error {
    Error::from_reason(error.to_string())
}
