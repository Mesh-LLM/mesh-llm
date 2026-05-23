use crate::inference::skippy::{SkippyDeviceDescriptor, SkippyModelHandle, SkippyModelLoadOptions};
use crate::models;
use anyhow::{Context, Result};
use mesh_llm_node::serving::{
    DevicePolicy, LoadModelRequest, ServedModel, ServingController, ServingFuture,
    ServingModelState, ServingStatus, UnloadModelRequest, UnloadTarget,
};
use mesh_llm_system::hardware::{self, Metric};
use openai_frontend::{ChatCompletionRequest, ChatMessage, MessageContent, OpenAiBackend};
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Clone, Debug)]
pub struct EmbeddedChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone)]
pub struct EmbeddedServingController {
    inner: Arc<Mutex<EmbeddedServingState>>,
}

struct EmbeddedServingState {
    next_instance_id: u64,
    default_device_policy: DevicePolicy,
    models: HashMap<String, Arc<EmbeddedServedModel>>,
}

struct EmbeddedServedModel {
    served: ServedModel,
    handle: SkippyModelHandle,
}

impl Default for EmbeddedServingController {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddedServingController {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(EmbeddedServingState {
                next_instance_id: 1,
                default_device_policy: DevicePolicy::Auto,
                models: HashMap::new(),
            })),
        }
    }

    pub async fn chat_completion_text(
        &self,
        model: &str,
        messages: Vec<EmbeddedChatMessage>,
    ) -> Result<String> {
        let loaded = self.loaded_model(model).await?;
        let request = ChatCompletionRequest {
            model: loaded.served.model_id.clone(),
            messages: messages
                .into_iter()
                .map(|message| ChatMessage {
                    role: message.role,
                    content: Some(MessageContent::Text(message.content)),
                    extra: BTreeMap::new(),
                })
                .collect(),
            stream: false,
            max_tokens: None,
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            n: None,
            logprobs: None,
            top_logprobs: None,
            presence_penalty: None,
            frequency_penalty: None,
            logit_bias: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            user: None,
            stop: None,
            seed: None,
            reasoning: None,
            reasoning_effort: None,
            prompt_cache_key: None,
            prompt_cache_retention: None,
            stream_options: None,
            extra: BTreeMap::new(),
        };
        let response = loaded
            .handle
            .chat_completion(request)
            .await
            .map_err(|error| anyhow::anyhow!(error.to_string()))?;
        Ok(response
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .unwrap_or_default())
    }

    pub async fn model_list(&self) -> Vec<(String, String)> {
        self.inner
            .lock()
            .await
            .models
            .values()
            .fold(BTreeMap::new(), |mut models, model| {
                models.insert(
                    model.served.model_id.clone(),
                    model.served.model_ref.clone(),
                );
                models
            })
            .into_iter()
            .collect()
    }

    async fn loaded_model(&self, model: &str) -> Result<Arc<EmbeddedServedModel>> {
        let state = self.inner.lock().await;
        state
            .models
            .values()
            .find(|loaded| {
                loaded.served.model_id == model
                    || loaded.served.model_ref == model
                    || loaded.served.instance_id.as_deref() == Some(model)
            })
            .cloned()
            .with_context(|| format!("model is not loaded for local serving: {model}"))
    }
}

impl ServingController for EmbeddedServingController {
    fn load<'a>(&'a self, request: LoadModelRequest) -> ServingFuture<'a, ServedModel> {
        Box::pin(async move {
            let model_path =
                models::resolve_model_spec_with_progress(Path::new(&request.model_ref), true)
                    .await
                    .with_context(|| format!("resolve model {}", request.model_ref))?;
            let model_id = models::model_ref_for_path(&model_path);
            let device_policy = self.effective_device_policy(&request.device_policy).await;
            reject_obvious_vram_overcommit(&model_path, &device_policy)?;
            let options = apply_device_policy(
                SkippyModelLoadOptions::for_direct_gguf(&model_id, &model_path),
                &device_policy,
            )?;
            let handle = tokio::task::spawn_blocking(move || SkippyModelHandle::load(options))
                .await
                .context("join embedded model load task")??;
            let capabilities = models::runtime_verified_model_capabilities(
                &model_id,
                &model_path,
                models::RuntimeMediaCapabilityEvidence {
                    vision_projector_loaded: false,
                },
            );

            let mut state = self.inner.lock().await;
            let instance_id = format!("embedded-{}", state.next_instance_id);
            state.next_instance_id += 1;
            let served = ServedModel {
                model_ref: request.model_ref,
                model_id: model_id.clone(),
                instance_id: Some(instance_id),
                state: ServingModelState::Ready,
                backend: Some("skippy".to_string()),
                capabilities,
                context_length: Some(handle.status().ctx_size),
                error: None,
            };
            state.models.insert(
                model_id,
                Arc::new(EmbeddedServedModel {
                    served: served.clone(),
                    handle,
                }),
            );
            Ok(served)
        })
    }

    fn unload<'a>(&'a self, request: UnloadModelRequest) -> ServingFuture<'a, ()> {
        Box::pin(async move {
            let target = request.target.as_runtime_target().to_string();
            let mut state = self.inner.lock().await;
            let key = state.models.iter().find_map(|(key, loaded)| {
                let matches = loaded.served.model_id == target
                    || loaded.served.model_ref == target
                    || loaded.served.instance_id.as_deref() == Some(target.as_str());
                matches.then(|| key.clone())
            });
            if let Some(key) = key {
                state.models.remove(&key);
                return Ok(());
            }
            match request.target {
                UnloadTarget::Model(model_ref) => {
                    anyhow::bail!("model is not loaded for local serving: {model_ref}")
                }
                UnloadTarget::Instance(instance_id) => {
                    anyhow::bail!("instance is not loaded for local serving: {instance_id}")
                }
            }
        })
    }

    fn served_models<'a>(&'a self) -> ServingFuture<'a, Vec<ServedModel>> {
        Box::pin(async move {
            Ok(self
                .inner
                .lock()
                .await
                .models
                .values()
                .map(|model| model.served.clone())
                .collect())
        })
    }

    fn status<'a>(&'a self) -> ServingFuture<'a, ServingStatus> {
        Box::pin(async move {
            let models = self.served_models().await?;
            Ok(ServingStatus {
                enabled: true,
                models,
            })
        })
    }

    fn set_device_policy<'a>(&'a self, policy: DevicePolicy) -> ServingFuture<'a, ()> {
        Box::pin(async move {
            self.inner.lock().await.default_device_policy = policy;
            Ok(())
        })
    }
}

impl EmbeddedServingController {
    async fn effective_device_policy(&self, request_policy: &DevicePolicy) -> DevicePolicy {
        match request_policy {
            DevicePolicy::Auto => self.inner.lock().await.default_device_policy.clone(),
            explicit => explicit.clone(),
        }
    }
}

fn reject_obvious_vram_overcommit(model_path: &Path, policy: &DevicePolicy) -> Result<()> {
    if matches!(policy, DevicePolicy::Cpu) {
        return Ok(());
    }
    let survey = hardware::query(&[Metric::GpuFacts]);
    let total_vram_bytes = survey.gpus.iter().map(|gpu| gpu.vram_bytes).sum::<u64>();
    if total_vram_bytes == 0 {
        return Ok(());
    }
    let model_size_bytes = std::fs::metadata(model_path)
        .with_context(|| format!("read model metadata {}", model_path.display()))?
        .len();
    anyhow::ensure!(
        model_size_bytes <= total_vram_bytes,
        "model file is larger than detected total GPU VRAM: model={} bytes, vram={} bytes",
        model_size_bytes,
        total_vram_bytes
    );
    Ok(())
}

fn apply_device_policy(
    mut options: SkippyModelLoadOptions,
    policy: &DevicePolicy,
) -> Result<SkippyModelLoadOptions> {
    match policy {
        DevicePolicy::Auto => Ok(options),
        DevicePolicy::Cpu => {
            options.n_gpu_layers = 0;
            Ok(options)
        }
        DevicePolicy::Gpu { device_ids } => {
            if device_ids.is_empty() {
                return Ok(options);
            }
            anyhow::ensure!(
                device_ids.len() == 1,
                "embedded serving can pin one GPU per loaded model; got {} device ids",
                device_ids.len()
            );
            let survey = hardware::query(&[Metric::GpuFacts]);
            let gpu =
                hardware::resolve_pinned_gpu_strict(Some(device_ids[0].as_str()), &survey.gpus)
                    .with_context(|| {
                        format!(
                            "resolve requested serving GPU '{}' from local hardware",
                            device_ids[0]
                        )
                    })?;
            let backend_device = gpu.backend_device.clone().with_context(|| {
                format!(
                    "requested serving GPU '{}' has no backend device name",
                    device_ids[0]
                )
            })?;
            Ok(options.with_selected_device(SkippyDeviceDescriptor {
                backend_device,
                stable_id: gpu.stable_id.clone(),
                index: Some(gpu.index),
                vram_bytes: Some(gpu.vram_bytes),
            }))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn explicit_load_policy_overrides_stored_default() {
        let controller = EmbeddedServingController::new();
        controller
            .set_device_policy(DevicePolicy::Cpu)
            .await
            .unwrap();

        assert_eq!(
            controller
                .effective_device_policy(&DevicePolicy::Gpu {
                    device_ids: vec!["metal:0".to_string()],
                })
                .await,
            DevicePolicy::Gpu {
                device_ids: vec!["metal:0".to_string()],
            }
        );
    }

    #[tokio::test]
    async fn auto_load_policy_uses_stored_default() {
        let controller = EmbeddedServingController::new();
        controller
            .set_device_policy(DevicePolicy::Cpu)
            .await
            .unwrap();

        assert_eq!(
            controller
                .effective_device_policy(&DevicePolicy::Auto)
                .await,
            DevicePolicy::Cpu
        );
    }

    #[test]
    fn cpu_policy_forces_cpu_only_runtime_load() {
        let options =
            apply_device_policy(test_load_options(), &DevicePolicy::Cpu).expect("cpu policy");

        assert_eq!(options.n_gpu_layers, 0);
        assert!(options.selected_device.is_none());
    }

    #[test]
    fn multi_gpu_policy_is_rejected_instead_of_ignored() {
        let err = apply_device_policy(
            test_load_options(),
            &DevicePolicy::Gpu {
                device_ids: vec!["metal:0".to_string(), "metal:1".to_string()],
            },
        )
        .expect_err("multi-gpu policy should be rejected");

        assert!(
            err.to_string().contains("can pin one GPU per loaded model"),
            "{err}"
        );
    }

    fn test_load_options() -> SkippyModelLoadOptions {
        SkippyModelLoadOptions::for_direct_gguf("test-model", PathBuf::from("/tmp/test.gguf"))
    }
}
