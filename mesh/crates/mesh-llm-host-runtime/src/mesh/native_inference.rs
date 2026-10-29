//! Provider-bound native inference. No loopback sockets or HTTP messages.
use super::*;
use crate::runtime::activity_policy::{AdmissionResult, IngressType};
use crate::sdk::native::{ChatCompletionStream, Model, ModelTarget, NativeChatRequest};
use skippy_inference_api::{ChatCompletionRequest, InferenceBackend, InferenceRequestContext};
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use std::time::Duration;

mod streaming;
#[cfg(test)]
mod tests;

/// Bound for a request that has no progress signal of its own: one unary
/// response, or the request frame that opens a stream.
const NATIVE_CHAT_TIMEOUT: Duration = Duration::from_secs(300);
/// Bound for one frame write blocked by QUIC flow control, and for the request
/// frame a provider reads before it starts work. This is not a stream lifetime:
/// a reader that stops consuming can block a single write for this long.
const NATIVE_READ_TIMEOUT: Duration = Duration::from_secs(15);
/// Streaming inactivity window. It restarts with every chunk, so a generation
/// that keeps producing is never cut off by its total duration. A generation
/// that produces nothing for this long fails as a stalled stream. Tests lower
/// it through [`Node::set_native_stream_idle`].
const NATIVE_STREAM_IDLE: Duration = Duration::from_secs(300);
const MAX_NATIVE_REQUESTS: usize = 64;

#[derive(Clone)]
struct NativeBackend {
    port: u16,
    backend: Arc<dyn InferenceBackend>,
    efforts: Option<Vec<String>>,
}

type NativeBackends = HashMap<String, BTreeMap<String, NativeBackend>>;

pub(crate) struct NativeInferenceState {
    backends: tokio::sync::RwLock<NativeBackends>,
    slots: Arc<tokio::sync::Semaphore>,
    stream_slots: Arc<tokio::sync::Semaphore>,
    stream_idle_millis: std::sync::atomic::AtomicU64,
}

impl Default for NativeInferenceState {
    fn default() -> Self {
        Self {
            backends: Default::default(),
            slots: Arc::new(tokio::sync::Semaphore::new(MAX_NATIVE_REQUESTS)),
            stream_slots: Arc::new(tokio::sync::Semaphore::new(MAX_NATIVE_REQUESTS)),
            stream_idle_millis: std::sync::atomic::AtomicU64::new(
                NATIVE_STREAM_IDLE.as_millis() as u64
            ),
        }
    }
}

impl NativeInferenceState {
    /// The inactivity window every native chunk wait is bounded by.
    fn stream_idle(&self) -> Duration {
        Duration::from_millis(self.stream_idle_millis.load(Ordering::Relaxed))
    }
}

/// Keeps unload/drain admission and backend cancellation tied to the call.
struct PreparedNativeChat {
    backend: Arc<dyn InferenceBackend>,
    request: ChatCompletionRequest,
    context: InferenceRequestContext,
    _slot: tokio::sync::OwnedSemaphorePermit,
    _instance: crate::runtime::InstanceRequestGuard,
    _inflight: InflightRequestGuard,
    _cancel: CancelNativeOnDrop,
}

impl Node {
    pub(crate) async fn register_native_backend(
        &self,
        model: &str,
        instance: &str,
        port: u16,
        backend: Option<Arc<dyn InferenceBackend>>,
        efforts: Option<Vec<String>>,
    ) {
        let mut guard = self.native_inference.backends.write().await;
        let instances = guard.entry(model.into()).or_default();
        if let Some(backend) = backend {
            instances.insert(
                instance.into(),
                NativeBackend {
                    port,
                    backend,
                    efforts,
                },
            );
        } else {
            instances.remove(instance);
        }
        if instances.is_empty() {
            guard.remove(model);
        }
    }

    pub(crate) async fn unregister_native_backend(&self, model: &str, instance: &str) {
        let mut guard = self.native_inference.backends.write().await;
        if let Some(instances) = guard.get_mut(model) {
            instances.remove(instance);
            if instances.is_empty() {
                guard.remove(model);
            }
        }
    }

    /// Lower the streaming inactivity window so a test can exercise both an
    /// active stream that outlives the window and a stalled one that fails.
    #[cfg(test)]
    pub(crate) fn set_native_stream_idle(&self, idle: Duration) {
        self.native_inference
            .stream_idle_millis
            .store(idle.as_millis() as u64, Ordering::Relaxed);
    }

    pub(crate) async fn native_models(&self) -> Result<Vec<Model>> {
        let mut models = Vec::new();
        let local = self.hosted_models().await;
        append_models(
            &mut models,
            self.endpoint.id(),
            &local,
            self.served_model_descriptors().await,
        );
        // Keep provider records separate. Merging by model name can advertise
        // an effort unsupported by the provider that actually receives work.
        for peer in self.routable_peers().await {
            if !matches!(peer.role, NodeRole::Host { .. }) {
                continue;
            }
            // Public HTTP aliases must not replace exact native serving IDs.
            let hosted = if peer.hosted_models_known {
                peer.hosted_models
            } else {
                peer.serving_models
            };
            append_models(&mut models, peer.id, &hosted, peer.served_model_descriptors);
        }
        models.sort_by(|a, b| {
            (&a.target.serving_id, &a.target.provider_node_id)
                .cmp(&(&b.target.serving_id, &b.target.provider_node_id))
        });
        Ok(models)
    }

    pub(crate) async fn native_chat_completion(
        &self,
        mut call: NativeChatRequest,
    ) -> Result<serde_json::Value> {
        normalize_target(&mut call)?;
        anyhow::ensure!(
            !call.request.stream,
            "use chat_completion_stream for streaming requests"
        );
        let provider: EndpointId = call
            .target
            .provider_node_id
            .parse()
            .context("invalid native inference provider ID")?;
        tokio::time::timeout(NATIVE_CHAT_TIMEOUT, async {
            if provider == self.endpoint.id() {
                return self.execute_native_chat(call, false).await;
            }
            let (mut send, mut recv) = self.open_native_chat(provider, &call).await?;
            let response = read_native_frame(&mut recv).await?;
            let _ = send.finish();
            if let Some(error) = response.error {
                anyhow::bail!("native inference provider: {error}");
            }
            serde_json::from_slice(&response.payload_json).context("decode native chat response")
        })
        .await
        .context("native inference timed out")?
    }

    async fn open_native_chat(
        &self,
        provider: EndpointId,
        call: &NativeChatRequest,
    ) -> Result<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)> {
        anyhow::ensure!(
            self.routable_peers()
                .await
                .iter()
                .any(|peer| peer.id == provider),
            "native inference provider is unavailable or blocked"
        );
        let conn = self.connection_to_peer(provider).await?;
        let (mut send, recv) = conn.open_bi().await?;
        send.write_all(&[STREAM_NATIVE_CHAT]).await?;
        let frame = crate::proto::node::NativeChatFrame {
            generation: 1,
            payload_json: serde_json::to_vec(call)?,
            error: None,
        };
        write_native_frame(&mut send, frame).await?;
        send.finish()?;
        Ok((send, recv))
    }

    async fn prepare_native_chat(
        &self,
        mut call: NativeChatRequest,
        remote: bool,
    ) -> Result<PreparedNativeChat> {
        normalize_target(&mut call)?;
        anyhow::ensure!(
            call.target.provider_node_id == self.endpoint.id().to_string(),
            "native inference target names a different provider"
        );
        let ingress = if remote {
            IngressType::RemoteQuicHttp
        } else {
            IngressType::LocalOpenAi
        };
        if let AdmissionResult::Paused { reason, .. } =
            self.activity_policy_guard.check_admission(ingress)
        {
            anyhow::bail!("native inference paused: {reason}");
        }
        #[cfg(feature = "payments")]
        if remote {
            let prices = self.advertised_payment_offers().await?;
            let public_id = self
                .served_model_descriptors()
                .await
                .iter()
                .find(|descriptor| descriptor.identity.model_name == call.target.serving_id)
                .and_then(|descriptor| public_model_id_from_identity(&descriptor.identity));
            anyhow::ensure!(
                !prices.contains_key(&call.target.serving_id)
                    && !public_id.is_some_and(|id| prices.contains_key(&id)),
                "native chat does not support this provider's payment protocol"
            );
        }
        let slot = self
            .native_inference
            .slots
            .clone()
            .try_acquire_owned()
            .context("native inference capacity exhausted")?;
        let effort = requested_effort(&call.request)?;
        let candidates = self
            .native_inference
            .backends
            .read()
            .await
            .get(&call.target.serving_id)
            .cloned()
            .context("native model is no longer loaded on this provider")?;
        for (instance_id, candidate) in &candidates {
            if !effort_supported(candidate.efforts.as_deref(), effort.as_deref()) {
                continue;
            }
            let Ok(Some(instance)) = self.begin_runtime_instance_request(candidate.port).await
            else {
                continue;
            };
            // The admission await can race unload/reload (including port
            // reuse). Never dispatch a retired snapshot through a new load's
            // lifecycle guard.
            let current = self.native_inference.backends.read().await;
            if !current
                .get(&call.target.serving_id)
                .and_then(|instances| instances.get(instance_id))
                .is_some_and(|active| Arc::ptr_eq(&active.backend, &candidate.backend))
            {
                continue;
            }
            drop(current);
            let context = InferenceRequestContext::new();
            return Ok(PreparedNativeChat {
                backend: candidate.backend.clone(),
                request: call.request,
                _cancel: CancelNativeOnDrop(context.cancellation_token()),
                context,
                _slot: slot,
                _instance: instance,
                _inflight: self.begin_inflight_request(),
            });
        }
        anyhow::bail!(
            "no active instance supports the requested thinking effort; refresh model metadata"
        )
    }

    async fn execute_native_chat(
        &self,
        call: NativeChatRequest,
        remote: bool,
    ) -> Result<serde_json::Value> {
        let mut prepared = self.prepare_native_chat(call, remote).await?;
        let request = std::mem::take(&mut prepared.request);
        let response = prepared
            .backend
            .chat_completion_with_context(request, prepared.context.clone())
            .await?;
        serde_json::to_value(response).context("encode native chat response")
    }

    pub(crate) fn spawn_native_chat_stream(
        &self,
        remote: EndpointId,
        mut send: iroh::endpoint::SendStream,
        mut recv: iroh::endpoint::RecvStream,
    ) {
        let Ok(slot) = self
            .native_inference
            .stream_slots
            .clone()
            .try_acquire_owned()
        else {
            let _ = send.reset(0u32.into());
            let _ = recv.stop(0u32.into());
            return;
        };
        let node = self.clone();
        tokio::spawn(async move {
            let _slot = slot;
            let result = tokio::time::timeout(NATIVE_READ_TIMEOUT, async {
                let frame = read_native_frame(&mut recv).await?;
                anyhow::ensure!(frame.error.is_none(), "request contains response error");
                serde_json::from_slice::<NativeChatRequest>(&frame.payload_json)
                    .context("decode native chat request")
            })
            .await
            .context("native chat request read timed out")
            .and_then(|r| r);
            let result = match result {
                Ok(call) => {
                    // A stream is bounded per chunk inside `serve_native_chat`;
                    // an unary response has no progress signal of its own, so it
                    // keeps the whole-request bound.
                    let streaming = call.request.stream;
                    let serving = node.serve_native_chat(call, &mut send);
                    if streaming {
                        serving.await
                    } else {
                        tokio::time::timeout(NATIVE_CHAT_TIMEOUT, serving)
                            .await
                            .context("native inference timed out")
                            .and_then(|r| r)
                    }
                }
                Err(error) => Err(error),
            };
            if let Err(error) = result {
                let frame = crate::proto::node::NativeChatFrame {
                    generation: 1,
                    payload_json: Vec::new(),
                    error: Some(error.to_string()),
                };
                if let Err(error) = write_native_frame(&mut send, frame).await {
                    tracing::debug!(%remote, %error, "native chat response stream closed");
                }
            }
            let _ = send.finish();
        });
    }
}

struct CancelNativeOnDrop(skippy_inference_api::CancellationToken);
impl Drop for CancelNativeOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

fn append_models(
    models: &mut Vec<Model>,
    provider: EndpointId,
    hosted: &[String],
    descriptors: Vec<ServedModelDescriptor>,
) {
    for descriptor in descriptors {
        if !hosted.contains(&descriptor.identity.model_name) {
            continue;
        }
        models.push(Model {
            target: ModelTarget {
                provider_node_id: provider.to_string(),
                serving_id: descriptor.identity.model_name.clone(),
            },
            thinking_efforts: descriptor
                .metadata
                .as_ref()
                .and_then(|m| m.thinking_efforts.clone()),
            descriptor,
        });
    }
}

fn normalize_target(call: &mut NativeChatRequest) -> Result<()> {
    anyhow::ensure!(
        !call.target.serving_id.is_empty(),
        "native inference requires an exact serving ID"
    );
    anyhow::ensure!(
        call.request.model.is_empty() || call.request.model == call.target.serving_id,
        "request model differs from the selected native target"
    );
    call.request.model.clone_from(&call.target.serving_id);
    Ok(())
}

fn requested_effort(request: &ChatCompletionRequest) -> Result<Option<String>> {
    let options = skippy_inference_api::common::normalize_reasoning_template_options(
        request.reasoning.as_ref(),
        request.reasoning_effort,
        &request.extra,
    )?;
    Ok(options
        .chat_template_kwargs
        .get("reasoning_effort")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned))
}

fn effort_supported(observed: Option<&[String]>, requested: Option<&str>) -> bool {
    requested.is_none_or(|effort| {
        observed.is_some_and(|levels| levels.iter().any(|level| level == effort))
    })
}

fn decode_frame(bytes: &[u8]) -> Result<crate::proto::node::NativeChatFrame> {
    let frame = crate::proto::node::NativeChatFrame::decode(bytes)?;
    anyhow::ensure!(frame.generation == 1, "unsupported native chat generation");
    Ok(frame)
}

async fn read_native_frame(
    recv: &mut iroh::endpoint::RecvStream,
) -> Result<crate::proto::node::NativeChatFrame> {
    let bytes = read_len_prefixed(recv)
        .await
        .context("native inference stream closed; provider may not support native chat")?;
    decode_frame(&bytes)
}

async fn write_native_frame(
    send: &mut iroh::endpoint::SendStream,
    frame: crate::proto::node::NativeChatFrame,
) -> Result<()> {
    tokio::time::timeout(
        NATIVE_READ_TIMEOUT,
        write_len_prefixed(send, &frame.encode_to_vec()),
    )
    .await
    .context("native chat response write timed out")?
}
