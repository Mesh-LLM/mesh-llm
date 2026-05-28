use anyhow::{Result, bail};
use serde::Serialize;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::{Mutex, mpsc, oneshot};

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
        self.pending_host_responses
            .lock()
            .await
            .insert(request_id, tx);

        if let Err(err) = self
            .send_payload(
                proto::envelope::Payload::OpenMeshStreamRequest(request),
                request_id,
            )
            .await
        {
            self.pending_host_responses.lock().await.remove(&request_id);
            return Err(err);
        }

        let response = rx.await??;
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
