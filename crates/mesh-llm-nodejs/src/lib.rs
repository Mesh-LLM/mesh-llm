#![forbid(unsafe_code)]

use mesh_llm_api_server::events::{Event, EventListener};
use mesh_llm_api_server::{
    ChatMessage, ChatRequest, DevicePolicy, DownloadOptions, InviteToken, LoadModelOptions,
    MeshApiError, MeshNode, OwnerKeypair, ResponsesRequest, UnloadModelOptions, UnloadTarget,
};
#[cfg(feature = "embedded-runtime")]
use mesh_llm_host_runtime::sdk::{EmbeddedChatMessage, EmbeddedServingController};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

#[napi]
pub fn generate_owner_keypair_hex() -> String {
    OwnerKeypair::generate().to_hex()
}

#[napi]
pub struct Node {
    node: MeshNode,
    #[cfg(feature = "embedded-runtime")]
    local_serving: Option<Arc<EmbeddedServingController>>,
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
            .cancel(mesh_llm_api_server::RequestId(request_id))
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
            .search(mesh_llm_api_server::ModelSearchQuery {
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

fn model_summary_json(model: mesh_llm_api_server::ModelSummary) -> Value {
    json!({
        "id": model.id,
        "name": model.name,
        "sizeLabel": model.size_label,
        "description": model.description,
        "capabilities": capabilities_json(model.capabilities),
    })
}

fn served_model_json(model: mesh_llm_api_server::ServedModel) -> Value {
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

fn capabilities_json(value: mesh_llm_api_server::ModelCapabilities) -> Value {
    json!({
        "multimodal": value.multimodal,
        "vision": capability_level_json(value.vision),
        "audio": capability_level_json(value.audio),
        "reasoning": capability_level_json(value.reasoning),
        "toolUse": capability_level_json(value.tool_use),
        "moe": value.moe,
    })
}

fn serving_model_state_json(value: mesh_llm_api_server::ServingModelState) -> Value {
    match value {
        mesh_llm_api_server::ServingModelState::Loading => json!({ "type": "Loading" }),
        mesh_llm_api_server::ServingModelState::Ready => json!({ "type": "Ready" }),
        mesh_llm_api_server::ServingModelState::Failed => json!({ "type": "Failed" }),
        mesh_llm_api_server::ServingModelState::Unloading => json!({ "type": "Unloading" }),
        mesh_llm_api_server::ServingModelState::Stopped => json!({ "type": "Stopped" }),
        mesh_llm_api_server::ServingModelState::Unknown(value) => {
            json!({ "type": "Unknown", "value": value })
        }
    }
}

fn capability_level_json(value: mesh_llm_api_server::CapabilityLevel) -> Value {
    match value {
        mesh_llm_api_server::CapabilityLevel::None => json!({ "type": "None" }),
        mesh_llm_api_server::CapabilityLevel::Likely => json!({ "type": "Likely" }),
        mesh_llm_api_server::CapabilityLevel::Supported => json!({ "type": "Supported" }),
    }
}

fn to_napi_error(error: MeshApiError) -> Error {
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
