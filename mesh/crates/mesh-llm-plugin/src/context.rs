use anyhow::{Result, bail};
use serde::Serialize;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};

mod model_assets;

use crate::{
    PROTOCOL_VERSION,
    helpers::{channel_message, json_channel_message},
    io::{LocalStream, connect_side_stream},
    proto,
};

static NEXT_HOST_REQUEST_ID: AtomicU64 = AtomicU64::new(1);
/// How long [`PluginContext::request_peer_block`] waits for the host. The host
/// bounds its own handling at 30 s and then replies with an error; this only
/// catches a host that never replies.
const PEER_BLOCK_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(35);
const PLUGIN_ORIGINATED_REQUEST_BIT: u64 = 1 << 63;

pub(crate) type PendingHostResponses =
    Arc<Mutex<HashMap<u64, oneshot::Sender<Result<proto::Envelope>>>>>;

struct PendingHostResponseGuard {
    request_id: u64,
    pending_host_responses: PendingHostResponses,
    active: bool,
}

impl PendingHostResponseGuard {
    fn new(request_id: u64, pending_host_responses: PendingHostResponses) -> Self {
        Self {
            request_id,
            pending_host_responses,
            active: true,
        }
    }

    fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for PendingHostResponseGuard {
    fn drop(&mut self) {
        if self.active {
            remove_pending_host_response(&self.pending_host_responses, self.request_id);
        }
    }
}

#[derive(Clone)]
pub struct PluginContext<'a> {
    pub(crate) outbound_tx: mpsc::Sender<proto::Envelope>,
    pub(crate) pending_host_responses: PendingHostResponses,
    pub(crate) plugin_id: String,
    pub(crate) invocation_id: Option<u64>,
    /// What the host listed in its `InitializeRequest`.
    pub(crate) host_capabilities: Arc<[String]>,
    pub(crate) inference_observation: crate::inference_observation::FeedSlot,
    pub(crate) _marker: PhantomData<&'a mut ()>,
}

impl<'a> PluginContext<'a> {
    pub(crate) fn new(
        plugin_id: String,
        outbound_tx: mpsc::Sender<proto::Envelope>,
        pending_host_responses: PendingHostResponses,
    ) -> Self {
        Self {
            outbound_tx,
            pending_host_responses,
            plugin_id,
            invocation_id: None,
            host_capabilities: Arc::from([]),
            inference_observation: Default::default(),
            _marker: PhantomData,
        }
    }

    pub fn owned(&self) -> PluginContext<'static> {
        PluginContext {
            outbound_tx: self.outbound_tx.clone(),
            pending_host_responses: self.pending_host_responses.clone(),
            plugin_id: self.plugin_id.clone(),
            invocation_id: self.invocation_id,
            host_capabilities: self.host_capabilities.clone(),
            inference_observation: self.inference_observation.clone(),
            _marker: PhantomData,
        }
    }

    pub(crate) fn with_inference_observation(
        mut self,
        slot: crate::inference_observation::FeedSlot,
    ) -> Self {
        self.inference_observation = slot;
        self
    }

    pub(crate) fn with_host_capabilities(mut self, host_capabilities: Arc<[String]>) -> Self {
        self.host_capabilities = host_capabilities;
        self
    }

    pub(crate) fn with_invocation_id(mut self, invocation_id: Option<u64>) -> Self {
        self.invocation_id = invocation_id;
        self
    }

    /// Whether the host listed `capability` (see [`crate::host_capabilities`])
    /// when it initialized this plugin. An older host lists none.
    pub fn host_supports(&self, capability: &str) -> bool {
        self.host_capabilities
            .iter()
            .any(|listed| listed == capability)
    }

    pub async fn send_channel(&mut self, message: proto::ChannelMessage) -> Result<()> {
        self.send_channel_message(message).await
    }

    pub async fn send_channel_message(&mut self, message: proto::ChannelMessage) -> Result<()> {
        self.send_payload(proto::envelope::Payload::ChannelMessage(message), 0)
            .await
    }

    pub async fn send_text_channel(
        &mut self,
        channel: impl Into<String>,
        target_peer_id: impl Into<String>,
        message_kind: impl Into<String>,
        text: impl Into<String>,
    ) -> Result<()> {
        self.send_channel_message(channel_message(
            channel,
            target_peer_id,
            "text/plain",
            text.into().into_bytes(),
            message_kind,
        ))
        .await
    }

    pub async fn send_json_channel<T: Serialize>(
        &mut self,
        channel: impl Into<String>,
        target_peer_id: impl Into<String>,
        message_kind: impl Into<String>,
        payload: &T,
    ) -> Result<()> {
        self.send_channel_message(json_channel_message(
            channel,
            target_peer_id,
            message_kind,
            payload,
        )?)
        .await
    }

    pub async fn send_bulk(&mut self, message: proto::BulkTransferMessage) -> Result<()> {
        self.send_bulk_transfer_message(message).await
    }

    pub async fn send_bulk_transfer_message(
        &mut self,
        message: proto::BulkTransferMessage,
    ) -> Result<()> {
        self.send_payload(proto::envelope::Payload::BulkTransferMessage(message), 0)
            .await
    }

    pub async fn notify_host<P>(&mut self, method: &str, params: P) -> Result<()>
    where
        P: Serialize,
    {
        self.send_payload(
            proto::envelope::Payload::RpcNotification(proto::RpcNotification {
                invocation_id: self.invocation_id,
                method: method.to_string(),
                params_json: serde_json::to_string(&params)?,
            }),
            0,
        )
        .await
    }

    pub async fn request_host<P, R>(&self, method: &str, params: P) -> Result<R>
    where
        P: Serialize,
        R: serde::de::DeserializeOwned,
    {
        let request_id = next_host_request_id();
        let (tx, rx) = oneshot::channel();
        insert_pending_host_response(&self.pending_host_responses, request_id, tx);
        let mut pending_guard =
            PendingHostResponseGuard::new(request_id, self.pending_host_responses.clone());

        self.send_payload(
            proto::envelope::Payload::RpcRequest(proto::RpcRequest {
                invocation_id: self.invocation_id,
                method: method.to_string(),
                params_json: serde_json::to_string(&params)?,
            }),
            request_id,
        )
        .await?;

        let response = rx.await??;
        pending_guard.disarm();
        match response.payload {
            Some(proto::envelope::Payload::RpcResponse(response)) => {
                Ok(serde_json::from_str(&response.result_json)?)
            }
            Some(proto::envelope::Payload::ErrorResponse(error)) => bail!(error.message),
            _ => bail!("Host returned an unexpected RPC response"),
        }
    }

    pub async fn infer(
        &self,
        request: crate::HostInferenceRequest,
    ) -> Result<crate::HostInferenceResponse> {
        self.request_host("inference/chat_completions", request)
            .await
    }

    pub async fn open_mesh_stream(
        &mut self,
        request: proto::OpenMeshStreamRequest,
    ) -> Result<proto::OpenMeshStreamResponse> {
        let request_id = next_host_request_id();
        let (tx, rx) = oneshot::channel();
        insert_pending_host_response(&self.pending_host_responses, request_id, tx);
        let mut pending_guard =
            PendingHostResponseGuard::new(request_id, self.pending_host_responses.clone());

        self.send_payload(
            proto::envelope::Payload::OpenMeshStreamRequest(request),
            request_id,
        )
        .await?;

        let response = rx.await??;
        pending_guard.disarm();
        match response.payload {
            Some(proto::envelope::Payload::OpenMeshStreamResponse(response)) => Ok(response),
            Some(proto::envelope::Payload::ErrorResponse(error)) => bail!(error.message),
            _ => bail!("Host returned an unexpected open_mesh_stream response"),
        }
    }

    /// Ask the host to stop (or resume) routing to a peer. The host applies
    /// it as a local block requested by this plugin; see `PeerBlockRequest`.
    ///
    /// Fails at once, without sending anything, if the host does not list
    /// [`crate::host_capabilities::PEER_BLOCKS`]; otherwise fails if the host
    /// has not answered within 35 seconds. The host refuses the request unless
    /// its operator set `allow_peer_blocks = true` for this plugin.
    pub async fn request_peer_block(
        &mut self,
        request: proto::PeerBlockRequest,
    ) -> Result<proto::PeerBlockResponse> {
        self.request_peer_block_within(request, PEER_BLOCK_REQUEST_TIMEOUT)
            .await
    }

    async fn request_peer_block_within(
        &mut self,
        request: proto::PeerBlockRequest,
        timeout: std::time::Duration,
    ) -> Result<proto::PeerBlockResponse> {
        if !self.host_supports(crate::host_capabilities::PEER_BLOCKS) {
            bail!(
                "peer blocks are unsupported by this host: it does not list the `{}` capability",
                crate::host_capabilities::PEER_BLOCKS
            );
        }
        let request_id = next_host_request_id();
        let (tx, rx) = oneshot::channel();
        insert_pending_host_response(&self.pending_host_responses, request_id, tx);
        let mut pending_guard =
            PendingHostResponseGuard::new(request_id, self.pending_host_responses.clone());

        self.send_payload(
            proto::envelope::Payload::PeerBlockRequest(request),
            request_id,
        )
        .await?;

        let Ok(response) = tokio::time::timeout(timeout, rx).await else {
            bail!("peer block request: the host did not answer within {timeout:?}");
        };
        let response = response??;
        pending_guard.disarm();
        match response.payload {
            Some(proto::envelope::Payload::PeerBlockResponse(response)) => Ok(response),
            Some(proto::envelope::Payload::ErrorResponse(error)) => bail!(error.message),
            _ => bail!("Host returned an unexpected peer block response"),
        }
    }

    /// Ask the host to announce `public_key` (32-byte Ed25519) as the key this
    /// plugin signs with, bound to this node by the node's own key, so peers'
    /// plugins can check what this plugin signs. An empty key withdraws it. The
    /// key is announced under the host's name for this plugin; a later request
    /// replaces it. See `PluginKeyRequest`.
    ///
    /// Fails at once, without sending anything, if the host does not list
    /// [`crate::host_capabilities::PLUGIN_KEYS`].
    pub async fn announce_plugin_key(
        &mut self,
        public_key: Vec<u8>,
    ) -> Result<proto::PluginKeyResponse> {
        self.announce_plugin_key_within(public_key, PEER_BLOCK_REQUEST_TIMEOUT)
            .await
    }

    async fn announce_plugin_key_within(
        &mut self,
        public_key: Vec<u8>,
        timeout: std::time::Duration,
    ) -> Result<proto::PluginKeyResponse> {
        if !self.host_supports(crate::host_capabilities::PLUGIN_KEYS) {
            bail!(
                "plugin keys are unsupported by this host: it does not list the `{}` capability",
                crate::host_capabilities::PLUGIN_KEYS
            );
        }
        let request_id = next_host_request_id();
        let (tx, rx) = oneshot::channel();
        insert_pending_host_response(&self.pending_host_responses, request_id, tx);
        let mut pending_guard =
            PendingHostResponseGuard::new(request_id, self.pending_host_responses.clone());

        self.send_payload(
            proto::envelope::Payload::PluginKeyRequest(proto::PluginKeyRequest { public_key }),
            request_id,
        )
        .await?;

        let Ok(response) = tokio::time::timeout(timeout, rx).await else {
            bail!("plugin key request: the host did not answer within {timeout:?}");
        };
        let response = response??;
        pending_guard.disarm();
        match response.payload {
            Some(proto::envelope::Payload::PluginKeyResponse(response)) => Ok(response),
            Some(proto::envelope::Payload::ErrorResponse(error)) => bail!(error.message),
            _ => bail!("Host returned an unexpected plugin key response"),
        }
    }

    /// Publish `data`, a JSON object of at most
    /// [`crate::plugin_status::MAX_DATA_BYTES`], as this plugin's snapshot in
    /// `GET /api/status`; it replaces the previous one. The host shows it as
    /// `ready` for `fresh_for` (0 for the default), then `stale`. See
    /// [`crate::plugin_status`].
    ///
    /// Fails at once, without sending anything, if the host does not list
    /// [`crate::host_capabilities::PLUGIN_STATUS`]. The host refuses a snapshot
    /// from a plugin whose manifest does not declare `plugin_status`, one that
    /// is too large or not an object, and one sent sooner than
    /// [`crate::plugin_status::MIN_PUBLISH_INTERVAL`] after the last.
    pub async fn publish_status<T: Serialize>(
        &mut self,
        data: &T,
        fresh_for: std::time::Duration,
    ) -> Result<proto::PublishPluginStatusResponse> {
        let request = proto::PublishPluginStatusRequest {
            data_json: serde_json::to_string(data)?,
            fresh_for_ms: u64::try_from(fresh_for.as_millis()).unwrap_or(u64::MAX),
        };
        self.publish_status_within(request, PEER_BLOCK_REQUEST_TIMEOUT)
            .await
    }

    /// Remove this plugin's snapshot from `GET /api/status`.
    pub async fn withdraw_status(&mut self) -> Result<()> {
        self.publish_status_within(
            proto::PublishPluginStatusRequest::default(),
            PEER_BLOCK_REQUEST_TIMEOUT,
        )
        .await
        .map(|_| ())
    }

    async fn publish_status_within(
        &mut self,
        request: proto::PublishPluginStatusRequest,
        timeout: std::time::Duration,
    ) -> Result<proto::PublishPluginStatusResponse> {
        if !self.host_supports(crate::host_capabilities::PLUGIN_STATUS) {
            bail!(
                "status contributions are unsupported by this host: it does not list the `{}` capability",
                crate::host_capabilities::PLUGIN_STATUS
            );
        }
        let request_id = next_host_request_id();
        let (tx, rx) = oneshot::channel();
        insert_pending_host_response(&self.pending_host_responses, request_id, tx);
        let mut pending_guard =
            PendingHostResponseGuard::new(request_id, self.pending_host_responses.clone());

        self.send_payload(
            proto::envelope::Payload::PublishPluginStatusRequest(request),
            request_id,
        )
        .await?;

        let Ok(response) = tokio::time::timeout(timeout, rx).await else {
            bail!("status publication: the host did not answer within {timeout:?}");
        };
        let response = response??;
        pending_guard.disarm();
        match response.payload {
            Some(proto::envelope::Payload::PublishPluginStatusResponse(response)) => Ok(response),
            Some(proto::envelope::Payload::ErrorResponse(error)) => bail!(error.message),
            _ => bail!("Host returned an unexpected status publication response"),
        }
    }

    /// Subscribe to the local inference observation feed: a snapshot of what
    /// this node is executing, then its events in order. A later call
    /// replaces the subscription. See [`crate::inference_observation`].
    ///
    /// Fails at once, without sending anything, if the host does not list
    /// [`crate::host_capabilities::LOCAL_INFERENCE_OBSERVATION`]. The host
    /// refuses a plugin whose manifest does not declare
    /// `inference_observation`, or whose operator has not set
    /// `allow_inference_observation = true` for it.
    pub async fn subscribe_inference_observation(
        &mut self,
    ) -> Result<crate::inference_observation::InferenceObservationFeed> {
        self.subscribe_inference_observation_within(PEER_BLOCK_REQUEST_TIMEOUT)
            .await
    }

    async fn subscribe_inference_observation_within(
        &mut self,
        timeout: std::time::Duration,
    ) -> Result<crate::inference_observation::InferenceObservationFeed> {
        if !self.host_supports(crate::host_capabilities::LOCAL_INFERENCE_OBSERVATION) {
            bail!(
                "local inference observation is unsupported by this host: it does not list the `{}` capability",
                crate::host_capabilities::LOCAL_INFERENCE_OBSERVATION
            );
        }
        let request_id = next_host_request_id();
        let (tx, rx) = oneshot::channel();
        insert_pending_host_response(&self.pending_host_responses, request_id, tx);
        let mut pending_guard =
            PendingHostResponseGuard::new(request_id, self.pending_host_responses.clone());
        // Open before asking, so a batch the host sends right after its answer
        // is kept even if it is read first.
        let feed = self.inference_observation.open();
        let feed_guard = FeedCloseGuard(self.inference_observation.clone(), true);

        self.send_payload(
            proto::envelope::Payload::SubscribeInferenceObservationRequest(
                proto::SubscribeInferenceObservationRequest {},
            ),
            request_id,
        )
        .await?;

        let Ok(response) = tokio::time::timeout(timeout, rx).await else {
            bail!("inference observation subscription: the host did not answer within {timeout:?}");
        };
        let response = response??;
        pending_guard.disarm();
        match response.payload {
            Some(proto::envelope::Payload::SubscribeInferenceObservationResponse(response)) => {
                feed_guard.disarm();
                Ok(feed.start(response.snapshot.unwrap_or_default()))
            }
            Some(proto::envelope::Payload::ErrorResponse(error)) => bail!(error.message),
            _ => bail!("Host returned an unexpected inference observation response"),
        }
    }

    pub async fn connect_mesh_stream(
        &mut self,
        request: proto::OpenMeshStreamRequest,
    ) -> Result<LocalStream> {
        let response = self.open_mesh_stream(request).await?;
        if !response.accepted {
            bail!(
                "Host rejected mesh stream: {}",
                response
                    .message
                    .unwrap_or_else(|| "no reason provided".into())
            );
        }
        let endpoint = response
            .endpoint
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Host accepted mesh stream without an endpoint"))?;
        connect_side_stream(endpoint, response.transport_kind).await
    }

    async fn send_payload(&self, payload: proto::envelope::Payload, request_id: u64) -> Result<()> {
        self.outbound_tx
            .send(proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: self.plugin_id.clone(),
                request_id,
                payload: Some(payload),
            })
            .await
            .map_err(|_| anyhow::anyhow!("plugin host connection is closed"))
    }
}

/// Closes a subscription the host did not grant, whatever the way out.
struct FeedCloseGuard(crate::inference_observation::FeedSlot, bool);

impl FeedCloseGuard {
    fn disarm(mut self) {
        self.1 = false;
    }
}

impl Drop for FeedCloseGuard {
    fn drop(&mut self) {
        if self.1 {
            self.0.close();
        }
    }
}

pub(crate) fn next_host_request_id() -> u64 {
    PLUGIN_ORIGINATED_REQUEST_BIT | NEXT_HOST_REQUEST_ID.fetch_add(1, Ordering::Relaxed)
}

pub(crate) fn insert_pending_host_response(
    pending_host_responses: &PendingHostResponses,
    request_id: u64,
    sender: oneshot::Sender<Result<proto::Envelope>>,
) {
    pending_host_responses
        .lock()
        .expect("pending host response map poisoned")
        .insert(request_id, sender);
}

pub(crate) fn remove_pending_host_response(
    pending_host_responses: &PendingHostResponses,
    request_id: u64,
) -> Option<oneshot::Sender<Result<proto::Envelope>>> {
    pending_host_responses
        .lock()
        .expect("pending host response map poisoned")
        .remove(&request_id)
}

pub(crate) fn drain_pending_host_responses(
    pending_host_responses: &PendingHostResponses,
) -> Vec<oneshot::Sender<Result<proto::Envelope>>> {
    pending_host_responses
        .lock()
        .expect("pending host response map poisoned")
        .drain()
        .map(|(_, sender)| sender)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn owned_invocation_context_echoes_id_on_callbacks() {
        let (context, mut outbound, pending) =
            context(&[crate::host_capabilities::INVOCATION_CALLBACKS]);
        let mut context = context.with_invocation_id(Some(73)).owned();
        context
            .notify_host("notifications/message", serde_json::json!({}))
            .await
            .unwrap();
        let notification = outbound.recv().await.unwrap();
        let Some(proto::envelope::Payload::RpcNotification(notification)) = notification.payload
        else {
            panic!("expected notification");
        };
        assert_eq!(notification.invocation_id, Some(73));

        let callback = tokio::spawn(async move {
            context
                .request_host::<_, serde_json::Value>("roots/list", serde_json::json!({}))
                .await
        });
        let request = outbound.recv().await.unwrap();
        let Some(proto::envelope::Payload::RpcRequest(rpc)) = request.payload else {
            panic!("expected request");
        };
        assert_eq!(rpc.invocation_id, Some(73));
        remove_pending_host_response(&pending, request.request_id)
            .unwrap()
            .send(Ok(proto::Envelope {
                payload: Some(proto::envelope::Payload::RpcResponse(proto::RpcResponse {
                    result_json: "{}".into(),
                })),
                ..Default::default()
            }))
            .unwrap();
        callback.await.unwrap().unwrap();
    }

    #[test]
    fn invocation_ids_survive_protobuf_roundtrip_and_legacy_defaults() {
        use prost::Message;
        let request = proto::RpcRequest {
            method: "roots/list".into(),
            params_json: "{}".into(),
            invocation_id: Some(91),
        };
        assert_eq!(
            proto::RpcRequest::decode(request.encode_to_vec().as_slice())
                .unwrap()
                .invocation_id,
            Some(91)
        );
        let notification = proto::RpcNotification {
            method: "notifications/message".into(),
            params_json: "{}".into(),
            invocation_id: Some(92),
        };
        assert_eq!(
            proto::RpcNotification::decode(notification.encode_to_vec().as_slice())
                .unwrap()
                .invocation_id,
            Some(92)
        );
        let service = proto::InvokeServiceRequest {
            invocation_id: Some(93),
            ..Default::default()
        };
        assert_eq!(
            proto::InvokeServiceRequest::decode(service.encode_to_vec().as_slice())
                .unwrap()
                .invocation_id,
            Some(93)
        );
        assert_eq!(
            proto::RpcRequest::decode(&[][..]).unwrap().invocation_id,
            None
        );
    }

    fn context(
        host_capabilities: &[&str],
    ) -> (
        PluginContext<'static>,
        mpsc::Receiver<proto::Envelope>,
        PendingHostResponses,
    ) {
        let (outbound_tx, outbound_rx) = mpsc::channel(8);
        let pending = PendingHostResponses::default();
        let context = PluginContext::new("demo".into(), outbound_tx, pending.clone())
            .with_host_capabilities(host_capabilities.iter().map(|c| c.to_string()).collect());
        (context, outbound_rx, pending)
    }

    #[tokio::test]
    async fn a_peer_block_request_to_an_older_host_fails_at_once() {
        let (mut context, mut outbound_rx, pending) = context(&[]);
        let error = context
            .request_peer_block(proto::PeerBlockRequest::default())
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("unsupported by this host"),
            "{error}"
        );
        assert!(outbound_rx.try_recv().is_err(), "nothing was sent");
        assert!(pending.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_peer_block_request_the_host_never_answers_times_out() {
        let (mut context, mut outbound_rx, pending) =
            context(&[crate::host_capabilities::PEER_BLOCKS]);
        let error = context
            .request_peer_block_within(
                proto::PeerBlockRequest::default(),
                std::time::Duration::from_millis(20),
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("did not answer"), "{error}");
        let sent = outbound_rx.try_recv().expect("the request was sent");
        assert!(matches!(
            sent.payload,
            Some(proto::envelope::Payload::PeerBlockRequest(_))
        ));
        assert!(
            pending.lock().unwrap().is_empty(),
            "a timed-out request leaves nothing pending"
        );
    }

    #[tokio::test]
    async fn a_peer_block_request_returns_the_host_answer() {
        let (mut context, mut outbound_rx, pending) =
            context(&[crate::host_capabilities::PEER_BLOCKS]);
        let host = tokio::spawn(async move {
            let request = outbound_rx.recv().await.unwrap();
            let sender = remove_pending_host_response(&pending, request.request_id).unwrap();
            sender
                .send(Ok(proto::Envelope {
                    request_id: request.request_id,
                    payload: Some(proto::envelope::Payload::PeerBlockResponse(
                        proto::PeerBlockResponse {
                            choice_json: "{}".into(),
                        },
                    )),
                    ..Default::default()
                }))
                .unwrap();
        });
        let response = context
            .request_peer_block(proto::PeerBlockRequest::default())
            .await
            .unwrap();
        assert_eq!(response.choice_json, "{}");
        host.await.unwrap();
    }

    #[tokio::test]
    async fn a_plugin_key_to_an_older_host_fails_at_once() {
        let (mut context, mut outbound_rx, pending) = context(&[]);
        let error = context.announce_plugin_key(vec![1; 32]).await.unwrap_err();
        assert!(
            error.to_string().contains("unsupported by this host"),
            "{error}"
        );
        assert!(outbound_rx.try_recv().is_err(), "nothing was sent");
        assert!(pending.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_plugin_key_request_the_host_never_answers_times_out() {
        let (mut context, mut outbound_rx, pending) =
            context(&[crate::host_capabilities::PLUGIN_KEYS]);
        let error = context
            .announce_plugin_key_within(vec![1; 32], std::time::Duration::from_millis(20))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("did not answer"), "{error}");
        let sent = outbound_rx.try_recv().expect("the request was sent");
        assert!(matches!(
            sent.payload,
            Some(proto::envelope::Payload::PluginKeyRequest(ref request)) if request.public_key == vec![1; 32]
        ));
        assert!(pending.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_plugin_key_request_returns_the_host_answer() {
        let (mut context, mut outbound_rx, pending) =
            context(&[crate::host_capabilities::PLUGIN_KEYS]);
        let host = tokio::spawn(async move {
            let request = outbound_rx.recv().await.unwrap();
            let sender = remove_pending_host_response(&pending, request.request_id).unwrap();
            sender
                .send(Ok(proto::Envelope {
                    request_id: request.request_id,
                    payload: Some(proto::envelope::Payload::PluginKeyResponse(
                        proto::PluginKeyResponse {
                            node_id: "ab".repeat(32),
                            binding_signature: vec![2; 64],
                        },
                    )),
                    ..Default::default()
                }))
                .unwrap();
        });
        let response = context.announce_plugin_key(vec![1; 32]).await.unwrap();
        assert_eq!(response.node_id, "ab".repeat(32));
        assert_eq!(response.binding_signature, vec![2; 64]);
        host.await.unwrap();
    }

    #[tokio::test]
    async fn a_status_publication_to_an_older_host_fails_at_once() {
        let (mut context, mut outbound_rx, pending) = context(&[]);
        let error = context
            .publish_status(&serde_json::json!({"watts": 8.2}), Duration::ZERO)
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("unsupported by this host"),
            "{error}"
        );
        assert!(outbound_rx.try_recv().is_err(), "nothing was sent");
        assert!(pending.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_status_publication_the_host_never_answers_times_out() {
        let (mut context, mut outbound_rx, pending) =
            context(&[crate::host_capabilities::PLUGIN_STATUS]);
        let error = context
            .publish_status_within(
                proto::PublishPluginStatusRequest {
                    data_json: "{}".into(),
                    fresh_for_ms: 0,
                },
                Duration::from_millis(20),
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("did not answer"), "{error}");
        assert!(outbound_rx.try_recv().is_ok(), "the publication was sent");
        assert!(pending.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_status_publication_sends_the_data_and_returns_the_host_answer() {
        let (mut context, mut outbound_rx, pending) =
            context(&[crate::host_capabilities::PLUGIN_STATUS]);
        let host = tokio::spawn(async move {
            let request = outbound_rx.recv().await.unwrap();
            let Some(proto::envelope::Payload::PublishPluginStatusRequest(publication)) =
                request.payload
            else {
                panic!("expected a status publication");
            };
            assert_eq!(publication.data_json, r#"{"watts":8.2}"#);
            assert_eq!(publication.fresh_for_ms, 5_000);
            let sender = remove_pending_host_response(&pending, request.request_id).unwrap();
            sender
                .send(Ok(proto::Envelope {
                    request_id: request.request_id,
                    payload: Some(proto::envelope::Payload::PublishPluginStatusResponse(
                        proto::PublishPluginStatusResponse {
                            received_at_unix_ms: 1_000,
                            fresh_until_unix_ms: 6_000,
                        },
                    )),
                    ..Default::default()
                }))
                .unwrap();
        });
        let response = context
            .publish_status(&serde_json::json!({"watts": 8.2}), Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(response.fresh_until_unix_ms, 6_000);
        host.await.unwrap();
    }

    #[tokio::test]
    async fn a_withdrawal_sends_empty_data() {
        let (mut context, mut outbound_rx, pending) =
            context(&[crate::host_capabilities::PLUGIN_STATUS]);
        let host = tokio::spawn(async move {
            let request = outbound_rx.recv().await.unwrap();
            assert!(matches!(
                request.payload,
                Some(proto::envelope::Payload::PublishPluginStatusRequest(ref publication))
                    if publication.data_json.is_empty()
            ));
            let sender = remove_pending_host_response(&pending, request.request_id).unwrap();
            sender
                .send(Ok(proto::Envelope {
                    request_id: request.request_id,
                    payload: Some(proto::envelope::Payload::PublishPluginStatusResponse(
                        proto::PublishPluginStatusResponse::default(),
                    )),
                    ..Default::default()
                }))
                .unwrap();
        });
        context.withdraw_status().await.unwrap();
        host.await.unwrap();
    }

    #[tokio::test]
    async fn an_observation_subscription_to_an_older_host_fails_at_once() {
        let (mut context, mut outbound_rx, pending) = context(&[]);
        let error = context
            .subscribe_inference_observation()
            .await
            .err()
            .expect("refused");
        assert!(
            error.to_string().contains("unsupported by this host"),
            "{error}"
        );
        assert!(outbound_rx.try_recv().is_err(), "nothing was sent");
        assert!(pending.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn an_observation_subscription_keeps_the_batches_sent_with_its_answer() {
        let (mut context, mut outbound_rx, pending) =
            context(&[crate::host_capabilities::LOCAL_INFERENCE_OBSERVATION]);
        let slot = context.inference_observation.clone();
        let host = tokio::spawn(async move {
            let request = outbound_rx.recv().await.unwrap();
            assert!(matches!(
                request.payload,
                Some(proto::envelope::Payload::SubscribeInferenceObservationRequest(_))
            ));
            // The first batch is read before the answer reaches the plugin.
            slot.deliver(proto::InferenceObservationBatch {
                host_generation: 5,
                watermark_unix_ns: 99,
                ..Default::default()
            });
            remove_pending_host_response(&pending, request.request_id)
                .unwrap()
                .send(Ok(proto::Envelope {
                    request_id: request.request_id,
                    payload: Some(
                        proto::envelope::Payload::SubscribeInferenceObservationResponse(
                            proto::SubscribeInferenceObservationResponse {
                                snapshot: Some(proto::InferenceObservationSnapshot {
                                    host_generation: 5,
                                    next_sequence: 1,
                                    ..Default::default()
                                }),
                            },
                        ),
                    ),
                    ..Default::default()
                }))
                .unwrap();
        });
        let mut feed = context.subscribe_inference_observation().await.unwrap();
        host.await.unwrap();
        assert_eq!(feed.snapshot().host_generation, 5);
        let Some(crate::inference_observation::InferenceObservationItem::Batch(batch)) =
            feed.next().await
        else {
            panic!("expected the early batch");
        };
        assert_eq!(batch.watermark_unix_ns, 99);
    }

    #[tokio::test]
    async fn a_refused_observation_subscription_receives_nothing() {
        let (mut context, mut outbound_rx, pending) =
            context(&[crate::host_capabilities::LOCAL_INFERENCE_OBSERVATION]);
        let slot = context.inference_observation.clone();
        let host = tokio::spawn(async move {
            let request = outbound_rx.recv().await.unwrap();
            remove_pending_host_response(&pending, request.request_id)
                .unwrap()
                .send(Ok(proto::Envelope {
                    request_id: request.request_id,
                    payload: Some(proto::envelope::Payload::ErrorResponse(
                        proto::ErrorResponse {
                            message: "not granted".into(),
                            ..Default::default()
                        },
                    )),
                    ..Default::default()
                }))
                .unwrap();
            slot
        });
        let error = context
            .subscribe_inference_observation()
            .await
            .err()
            .expect("refused");
        assert!(error.to_string().contains("not granted"), "{error}");
        let slot = host.await.unwrap();
        // A batch arriving now has nowhere to go.
        let mut probe = slot.open().start(Default::default());
        slot.close();
        assert!(probe.next().await.is_none());
    }
}
