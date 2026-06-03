#![forbid(unsafe_code)]

#[cfg(feature = "embedded-runtime")]
use mesh_llm_sdk::embedded_runtime::{EmbeddedChatMessage, EmbeddedServingController};
use mesh_llm_sdk::events::{Event, EventListener};
use mesh_llm_sdk::node as sdk_node;
use mesh_llm_sdk::node::{
    ChatMessage, ChatRequest, DevicePolicy, DownloadOptions, InviteToken, LoadModelOptions,
    MeshNode, OwnerKeypair, ResponsesRequest, UnloadModelOptions, UnloadTarget,
};
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

#[napi]
pub fn generate_owner_keypair_hex() -> String {
    OwnerKeypair::generate().to_hex()
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
    let runtimes = native_runtime_cache(cache_dir)?
        .installed()
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
        .unwrap_or_else(|| mesh_llm_sdk::native_runtime::CURRENT_MESH_VERSION.to_string());
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
    node: MeshNode,
    #[cfg(feature = "embedded-runtime")]
    local_serving: Option<Arc<EmbeddedServingController>>,
}

#[napi]
pub struct ConsoleHandle {
    inner: Arc<Mutex<Option<mesh_llm_sdk::console::ConsoleServerHandle>>>,
    url: String,
}

#[napi]
impl ConsoleHandle {
    #[napi(getter)]
    pub fn url(&self) -> String {
        self.url.clone()
    }

    #[napi]
    pub async fn stop(&self) -> Result<()> {
        let handle = self
            .inner
            .lock()
            .map_err(|error| Error::from_reason(error.to_string()))?
            .take();
        if let Some(handle) = handle {
            handle.stop().await;
        }
        Ok(())
    }
}

#[napi]
impl Node {
    #[napi(factory)]
    pub fn create(
        owner_keypair_hex: String,
        invite_token: String,
        cache_dir: Option<String>,
        runtime_dir: Option<String>,
        serving_enabled: Option<bool>,
    ) -> Result<Self> {
        let owner = parse_owner_keypair(&owner_keypair_hex)?;
        let token = invite_token
            .parse::<InviteToken>()
            .map_err(|error| Error::from_reason(format!("invalid invite token: {error}")))?;
        let serving_enabled = serving_enabled.unwrap_or(false);

        #[cfg(not(feature = "embedded-runtime"))]
        if serving_enabled {
            return Err(Error::from_reason(
                "serving is unsupported: native addon was built without embedded-runtime",
            ));
        }

        let mut builder = MeshNode::builder().identity(owner).join(token);
        #[cfg(feature = "embedded-runtime")]
        let local_serving = if serving_enabled {
            let controller = Arc::new(EmbeddedServingController::new());
            builder = builder.serving_controller(controller.clone());
            Some(controller)
        } else {
            builder = builder.serving_enabled(false);
            None
        };
        #[cfg(not(feature = "embedded-runtime"))]
        {
            builder = builder.serving_enabled(serving_enabled);
        }

        if let Some(path) = non_empty(cache_dir) {
            builder = builder.cache_dir(path);
        }
        if let Some(path) = non_empty(runtime_dir) {
            builder = builder.runtime_dir(path);
        }

        let node = builder.build().map_err(to_napi_error)?;
        Ok(Self {
            node,
            #[cfg(feature = "embedded-runtime")]
            local_serving,
        })
    }

    #[napi]
    pub async fn start(&self) -> Result<()> {
        self.node.start().await.map_err(to_napi_error)
    }

    #[napi]
    pub async fn stop(&self) -> Result<()> {
        self.node.stop().await.map_err(to_napi_error)
    }

    #[napi]
    pub async fn reconnect(&self) -> Result<()> {
        self.node.reconnect().await.map_err(to_napi_error)
    }

    #[napi(js_name = "statusJson")]
    pub async fn status_json(&self) -> Result<String> {
        let status = self.node.status().node().await.map_err(to_napi_error)?;
        Ok(json!({
            "connected": status.connected,
            "peerCount": status.peer_count,
        })
        .to_string())
    }

    #[napi(js_name = "listModelsJson")]
    pub async fn list_models_json(&self) -> Result<String> {
        #[cfg(feature = "embedded-runtime")]
        if let Some(controller) = &self.local_serving {
            let models = controller.model_list().await;
            if !models.is_empty() {
                return Ok(Value::Array(
                    models
                        .into_iter()
                        .map(|(id, name)| json!({ "id": id, "name": name }))
                        .collect(),
                )
                .to_string());
            }
        }

        let models = self
            .node
            .inference()
            .list_models()
            .await
            .map_err(to_napi_error)?;
        Ok(Value::Array(
            models
                .into_iter()
                .map(|model| json!({ "id": model.id, "name": model.name }))
                .collect(),
        )
        .to_string())
    }

    #[napi(js_name = "chatJson")]
    pub async fn chat_json(&self, request_json: String, timeout_ms: Option<u32>) -> Result<String> {
        let request = parse_chat_request(&request_json)?;

        #[cfg(feature = "embedded-runtime")]
        if let Some(controller) = &self.local_serving {
            let request_id = new_request_id();
            let messages = request
                .messages
                .iter()
                .map(|message| EmbeddedChatMessage {
                    role: message.role.clone(),
                    content: message.content.clone(),
                })
                .collect();
            let content = controller
                .chat_completion_text(&request.model, messages)
                .await
                .map_err(|error| Error::from_reason(error.to_string()))?;
            return Ok(json!({
                "requestId": request_id,
                "content": content,
                "events": [
                    { "type": "tokenDelta", "requestId": request_id, "delta": content },
                    { "type": "completed", "requestId": request_id }
                ]
            })
            .to_string());
        }

        let collector = Arc::new(EventCollector::default());
        let request_id = self
            .node
            .inference()
            .chat(request, collector.clone())
            .await
            .map_err(to_napi_error)?
            .0;
        let snapshot = collector.wait(timeout_ms.unwrap_or(120_000)).await;
        Ok(json!({
            "requestId": request_id,
            "content": snapshot.content,
            "events": snapshot.events,
        })
        .to_string())
    }

    #[napi(js_name = "responsesJson")]
    pub async fn responses_json(
        &self,
        request_json: String,
        timeout_ms: Option<u32>,
    ) -> Result<String> {
        let value = parse_json(&request_json)?;
        let model = required_string(&value, "model")?;
        let input = required_string(&value, "input")?;

        #[cfg(feature = "embedded-runtime")]
        if let Some(controller) = &self.local_serving {
            let request_id = new_request_id();
            let content = controller
                .chat_completion_text(
                    &model,
                    vec![EmbeddedChatMessage {
                        role: "user".to_string(),
                        content: input,
                    }],
                )
                .await
                .map_err(|error| Error::from_reason(error.to_string()))?;
            return Ok(json!({
                "requestId": request_id,
                "content": content,
                "events": [
                    { "type": "tokenDelta", "requestId": request_id, "delta": content },
                    { "type": "completed", "requestId": request_id }
                ]
            })
            .to_string());
        }

        let collector = Arc::new(EventCollector::default());
        let request_id = self
            .node
            .inference()
            .responses(ResponsesRequest { model, input }, collector.clone())
            .await
            .map_err(to_napi_error)?
            .0;
        let snapshot = collector.wait(timeout_ms.unwrap_or(120_000)).await;
        Ok(json!({
            "requestId": request_id,
            "content": snapshot.content,
            "events": snapshot.events,
        })
        .to_string())
    }

    #[napi]
    pub async fn cancel(&self, request_id: String) -> Result<()> {
        self.node
            .inference()
            .cancel(sdk_node::RequestId(request_id))
            .await
            .map_err(to_napi_error)
    }

    #[napi(js_name = "recommendedModelsJson")]
    pub async fn recommended_models_json(&self) -> Result<String> {
        let models = self
            .node
            .models()
            .recommended()
            .await
            .map_err(to_napi_error)?;
        Ok(Value::Array(models.into_iter().map(model_summary_json).collect()).to_string())
    }

    #[napi(js_name = "searchModelsJson")]
    pub async fn search_models_json(&self, query: String, limit: Option<u32>) -> Result<String> {
        let models = self
            .node
            .models()
            .search(sdk_node::ModelSearchQuery {
                query,
                limit: limit.map(|value| value as usize),
            })
            .await
            .map_err(to_napi_error)?;
        Ok(Value::Array(models.into_iter().map(model_summary_json).collect()).to_string())
    }

    #[napi(js_name = "showModelJson")]
    pub async fn show_model_json(&self, model_ref: String) -> Result<String> {
        let model = self
            .node
            .models()
            .show(model_ref)
            .await
            .map_err(to_napi_error)?;
        Ok(json!({
            "id": model.id,
            "name": model.name,
            "modelRef": model.model_ref,
            "downloadRef": model.download_ref,
            "path": model.path.map(|path| path.display().to_string()),
            "sizeBytes": model.size_bytes,
            "sizeLabel": model.size_label,
            "description": model.description,
            "draft": model.draft,
            "installed": model.installed,
            "capabilities": capabilities_json(model.capabilities),
        })
        .to_string())
    }

    #[napi(js_name = "installedModelsJson")]
    pub async fn installed_models_json(&self) -> Result<String> {
        let models = self
            .node
            .models()
            .installed()
            .await
            .map_err(to_napi_error)?;
        Ok(Value::Array(
            models
                .into_iter()
                .map(|model| {
                    json!({
                        "modelRef": model.model_ref,
                        "path": model.path.display().to_string(),
                        "sizeBytes": model.size_bytes,
                        "capabilities": capabilities_json(model.capabilities),
                    })
                })
                .collect(),
        )
        .to_string())
    }

    #[napi(js_name = "downloadModelJson")]
    pub async fn download_model_json(&self, model_ref: String) -> Result<String> {
        let model = self
            .node
            .models()
            .download(model_ref, DownloadOptions)
            .await
            .map_err(to_napi_error)?;
        Ok(json!({
            "modelRef": model.model_ref,
            "paths": model.paths.into_iter().map(|path| path.display().to_string()).collect::<Vec<_>>(),
            "primaryPath": model.primary_path.map(|path| path.display().to_string()),
        })
        .to_string())
    }

    #[napi(js_name = "servingStatusJson")]
    pub async fn serving_status_json(&self) -> Result<String> {
        let status = self.node.serving().status().await.map_err(to_napi_error)?;
        Ok(json!({
            "enabled": status.enabled,
            "models": status.models.into_iter().map(served_model_json).collect::<Vec<_>>(),
        })
        .to_string())
    }

    #[napi(js_name = "loadServingModelJson")]
    pub async fn load_serving_model_json(
        &self,
        model_ref: String,
        options_json: Option<String>,
    ) -> Result<String> {
        let options = parse_load_options(options_json)?;
        let served = self
            .node
            .serving()
            .load(model_ref, options)
            .await
            .map_err(to_napi_error)?;
        Ok(served_model_json(served).to_string())
    }

    #[napi(js_name = "unloadServingModel")]
    pub async fn unload_serving_model(
        &self,
        target_json: String,
        options_json: Option<String>,
    ) -> Result<()> {
        self.node
            .serving()
            .unload(
                parse_unload_target(&target_json)?,
                parse_unload_options(options_json)?,
            )
            .await
            .map_err(to_napi_error)
    }

    #[napi(js_name = "startConsole")]
    pub async fn start_console(
        &self,
        asset_dir: String,
        port: Option<u32>,
        listen_all: Option<bool>,
    ) -> Result<ConsoleHandle> {
        let port = port
            .map(u16::try_from)
            .transpose()
            .map_err(|_| Error::from_reason("console port must be between 0 and 65535"))?
            .unwrap_or(0);
        let handle = mesh_llm_sdk::console::start_file_console(
            mesh_llm_sdk::console::ConsoleServerOptions {
                asset_dir: asset_dir.into(),
                port,
                listen_all: listen_all.unwrap_or(false),
            },
        )
        .await
        .map_err(|error| Error::from_reason(error.to_string()))?;
        let url = handle.url().to_string();
        Ok(ConsoleHandle {
            inner: Arc::new(Mutex::new(Some(handle))),
            url,
        })
    }
}

#[derive(Default)]
struct EventCollector {
    state: Mutex<EventState>,
    wake: Notify,
}

#[derive(Default)]
struct EventState {
    events: Vec<Value>,
    content: String,
    done: bool,
}

struct EventSnapshot {
    events: Vec<Value>,
    content: String,
}

impl EventCollector {
    async fn wait(&self, timeout_ms: u32) -> EventSnapshot {
        let deadline = tokio::time::Instant::now() + Duration::from_millis(timeout_ms as u64);
        loop {
            {
                let state = self.state.lock().expect("event collector lock");
                if state.done {
                    return EventSnapshot {
                        events: state.events.clone(),
                        content: state.content.clone(),
                    };
                }
            }

            if tokio::time::timeout_at(deadline, self.wake.notified())
                .await
                .is_err()
            {
                let mut state = self.state.lock().expect("event collector lock");
                if !state.done {
                    state.events.push(json!({ "type": "timeout" }));
                }
                return EventSnapshot {
                    events: state.events.clone(),
                    content: state.content.clone(),
                };
            }
        }
    }
}

impl EventListener for EventCollector {
    fn on_event(&self, event: Event) {
        let mut state = self.state.lock().expect("event collector lock");
        match event {
            Event::Connecting => state.events.push(json!({ "type": "connecting" })),
            Event::Joined { node_id } => state
                .events
                .push(json!({ "type": "joined", "nodeId": node_id })),
            Event::ModelsUpdated { models } => state.events.push(json!({
                "type": "modelsUpdated",
                "models": models.into_iter().map(|model| json!({ "id": model.id, "name": model.name })).collect::<Vec<_>>()
            })),
            Event::TokenDelta { request_id, delta } => {
                state.content.push_str(&delta);
                state.events.push(json!({
                    "type": "tokenDelta",
                    "requestId": request_id,
                    "delta": delta,
                }));
            }
            Event::Completed { request_id } => {
                state.done = true;
                state
                    .events
                    .push(json!({ "type": "completed", "requestId": request_id }));
                self.wake.notify_waiters();
            }
            Event::Failed { request_id, error } => {
                state.done = true;
                state.events.push(json!({
                    "type": "failed",
                    "requestId": request_id,
                    "error": error,
                }));
                self.wake.notify_waiters();
            }
            Event::Disconnected { reason } => state
                .events
                .push(json!({ "type": "disconnected", "reason": reason })),
        }
    }
}

fn parse_owner_keypair(value: &str) -> Result<OwnerKeypair> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return Err(Error::from_reason("owner keypair must not be empty"));
    }
    OwnerKeypair::from_hex(trimmed).map_err(Error::from_reason)
}

fn non_empty(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}

fn parse_json(source: &str) -> Result<Value> {
    serde_json::from_str(source).map_err(|error| Error::from_reason(error.to_string()))
}

fn required_string(value: &Value, key: &str) -> Result<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| Error::from_reason(format!("missing string field: {key}")))
}

fn parse_chat_request(source: &str) -> Result<ChatRequest> {
    let value = parse_json(source)?;
    let model = required_string(&value, "model")?;
    let messages = value
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::from_reason("missing array field: messages"))?
        .iter()
        .map(|message| {
            Ok(ChatMessage {
                role: required_string(message, "role")?,
                content: required_string(message, "content")?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ChatRequest { model, messages })
}

fn parse_load_options(source: Option<String>) -> Result<LoadModelOptions> {
    let policy = source
        .as_deref()
        .map(parse_json)
        .transpose()?
        .as_ref()
        .and_then(|value| value.get("devicePolicy"))
        .map(parse_device_policy)
        .transpose()?
        .unwrap_or(DevicePolicy::Auto);
    Ok(LoadModelOptions {
        device_policy: policy,
    })
}

fn parse_unload_options(source: Option<String>) -> Result<UnloadModelOptions> {
    let value = source.as_deref().map(parse_json).transpose()?;
    Ok(UnloadModelOptions {
        drain_timeout: value
            .as_ref()
            .and_then(|value| value.get("drainTimeoutMs"))
            .and_then(Value::as_u64)
            .map(Duration::from_millis)
            .unwrap_or_else(|| Duration::from_secs(30)),
        force: value
            .as_ref()
            .and_then(|value| value.get("force"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

fn parse_unload_target(source: &str) -> Result<UnloadTarget> {
    let value = parse_json(source)?;
    if let Some(instance_id) = value.get("instanceId").and_then(Value::as_str) {
        return Ok(UnloadTarget::Instance(instance_id.to_string()));
    }
    if let Some(model_id) = value.get("modelId").and_then(Value::as_str) {
        return Ok(UnloadTarget::Model(model_id.to_string()));
    }
    Err(Error::from_reason(
        "unload target requires instanceId or modelId",
    ))
}

fn parse_device_policy(value: &Value) -> Result<DevicePolicy> {
    match value.as_str() {
        Some("auto") | Some("Auto") => Ok(DevicePolicy::Auto),
        Some("cpu") | Some("Cpu") => Ok(DevicePolicy::Cpu),
        Some("gpu") | Some("Gpu") => Ok(DevicePolicy::Gpu {
            device_ids: Vec::new(),
        }),
        _ => {
            if let Some(ids) = value.get("gpu").and_then(Value::as_array) {
                return Ok(DevicePolicy::Gpu {
                    device_ids: ids
                        .iter()
                        .filter_map(Value::as_str)
                        .map(ToOwned::to_owned)
                        .collect(),
                });
            }
            Err(Error::from_reason("unsupported device policy"))
        }
    }
}

fn parse_native_runtime_install_options(
    source: &str,
) -> Result<mesh_llm_sdk::native_runtime::NativeRuntimeInstallOptions> {
    let value = parse_json(source)?;
    Ok(mesh_llm_sdk::native_runtime::NativeRuntimeInstallOptions {
        mesh_version: optional_string(&value, "meshVersion")
            .unwrap_or_else(|| mesh_llm_sdk::native_runtime::CURRENT_MESH_VERSION.to_string()),
        skippy_abi_version: optional_string(&value, "skippyAbiVersion")
            .unwrap_or_else(mesh_llm_sdk::native_runtime::current_skippy_abi_version),
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

fn model_summary_json(model: sdk_node::ModelSummary) -> Value {
    json!({
        "id": model.id,
        "name": model.name,
        "sizeLabel": model.size_label,
        "description": model.description,
        "capabilities": capabilities_json(model.capabilities),
    })
}

fn served_model_json(model: sdk_node::ServedModel) -> Value {
    json!({
        "modelRef": model.model_ref,
        "modelId": model.model_id,
        "instanceId": model.instance_id,
        "state": serving_model_state_json(model.state),
        "backend": model.backend,
        "capabilities": capabilities_json(model.capabilities),
        "contextLength": model.context_length,
        "error": model.error,
    })
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
        "meshVersion": runtime.mesh_version,
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

fn capabilities_json(value: sdk_node::ModelCapabilities) -> Value {
    json!({
        "multimodal": value.multimodal,
        "vision": capability_level_json(value.vision),
        "audio": capability_level_json(value.audio),
        "reasoning": capability_level_json(value.reasoning),
        "toolUse": capability_level_json(value.tool_use),
        "moe": value.moe,
    })
}

fn serving_model_state_json(value: sdk_node::ServingModelState) -> Value {
    match value {
        sdk_node::ServingModelState::Loading => json!({ "type": "Loading" }),
        sdk_node::ServingModelState::Ready => json!({ "type": "Ready" }),
        sdk_node::ServingModelState::Failed => json!({ "type": "Failed" }),
        sdk_node::ServingModelState::Unloading => json!({ "type": "Unloading" }),
        sdk_node::ServingModelState::Stopped => json!({ "type": "Stopped" }),
        sdk_node::ServingModelState::Unknown(value) => {
            json!({ "type": "Unknown", "value": value })
        }
    }
}

fn capability_level_json(value: sdk_node::CapabilityLevel) -> Value {
    match value {
        sdk_node::CapabilityLevel::None => json!({ "type": "None" }),
        sdk_node::CapabilityLevel::Likely => json!({ "type": "Likely" }),
        sdk_node::CapabilityLevel::Supported => json!({ "type": "Supported" }),
    }
}

fn to_napi_error(error: impl ToString) -> Error {
    Error::from_reason(error.to_string())
}

#[cfg(feature = "embedded-runtime")]
fn new_request_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);
    format!(
        "node-local-{}",
        NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed)
    )
}
