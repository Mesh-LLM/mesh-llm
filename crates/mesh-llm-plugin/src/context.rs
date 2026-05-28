use anyhow::{Result, bail};
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    PROTOCOL_VERSION,
    helpers::{channel_message, json_channel_message},
    io::{
        LocalStream, connect_side_stream, read_envelope, send_bulk_transfer_message,
        send_channel_message, write_envelope,
    },
    proto,
};

static NEXT_HOST_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

pub struct PluginContext<'a> {
    pub(crate) stream: &'a mut LocalStream,
    pub(crate) plugin_id: &'a str,
}

impl<'a> PluginContext<'a> {
    pub async fn send_channel(&mut self, message: proto::ChannelMessage) -> Result<()> {
        self.send_channel_message(message).await
    }

    pub async fn send_channel_message(&mut self, message: proto::ChannelMessage) -> Result<()> {
        send_channel_message(self.stream, self.plugin_id, message).await
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
        send_bulk_transfer_message(self.stream, self.plugin_id, message).await
    }

    pub async fn notify_host<P>(&mut self, method: &str, params: P) -> Result<()>
    where
        P: Serialize,
    {
        write_envelope(
            self.stream,
            &proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: self.plugin_id.to_string(),
                request_id: 0,
                payload: Some(proto::envelope::Payload::RpcNotification(
                    proto::RpcNotification {
                        method: method.to_string(),
                        params_json: serde_json::to_string(&params)?,
                    },
                )),
            },
        )
        .await
    }

    pub async fn open_mesh_stream(
        &mut self,
        request: proto::OpenMeshStreamRequest,
    ) -> Result<proto::OpenMeshStreamResponse> {
        let request_id = NEXT_HOST_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
        write_envelope(
            self.stream,
            &proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: self.plugin_id.to_string(),
                request_id,
                payload: Some(proto::envelope::Payload::OpenMeshStreamRequest(request)),
            },
        )
        .await?;

        let response = read_envelope(self.stream).await?;
        if response.request_id != request_id {
            bail!(
                "Received host response id {} while waiting for {}",
                response.request_id,
                request_id
            );
        }
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
}
