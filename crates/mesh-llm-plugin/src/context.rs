use anyhow::{Result, bail};
use serde::Serialize;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, oneshot};

use crate::{
    PROTOCOL_VERSION,
    helpers::{channel_message, json_channel_message},
    io::{LocalStream, connect_side_stream},
    proto,
};

static NEXT_HOST_REQUEST_ID: AtomicU64 = AtomicU64::new(1);
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

pub struct PluginContext<'a> {
    pub(crate) outbound_tx: mpsc::Sender<proto::Envelope>,
    pub(crate) pending_host_responses: PendingHostResponses,
    pub(crate) plugin_id: String,
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
            _marker: PhantomData,
        }
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
                method: method.to_string(),
                params_json: serde_json::to_string(&params)?,
            }),
            0,
        )
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
