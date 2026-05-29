use anyhow::{Context, Result};
use iroh::endpoint::{Connection, RecvStream, SendStream};
use prost::Message;
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};

use super::{Node, endpoint_id_hex};
use crate::protocol::{STREAM_PLUGIN_MESH_STREAM, read_len_prefixed, write_len_prefixed};

fn plugin_mesh_stream_error(message: impl Into<String>) -> crate::plugin::proto::ErrorResponse {
    crate::plugin::proto::ErrorResponse {
        code: rmcp::model::ErrorCode::INTERNAL_ERROR.0,
        message: message.into(),
        data_json: String::new(),
    }
}

fn open_stream_request_from_mesh_request(
    request: &crate::plugin::proto::OpenMeshStreamRequest,
) -> crate::plugin::proto::OpenStreamRequest {
    crate::plugin::proto::OpenStreamRequest {
        stream_id: request.stream_id.clone(),
        purpose: request.purpose,
        mode: request.mode,
        bidirectional: request.bidirectional,
        content_type: request.content_type.clone(),
        correlation_id: request.correlation_id.clone(),
        metadata_json: request.metadata_json.clone(),
        expected_bytes: request.expected_bytes,
        idle_timeout_ms: request.idle_timeout_ms,
    }
}

impl Node {
    pub(super) async fn open_outbound_plugin_mesh_stream(
        &self,
        plugin_id: String,
        mut request: crate::plugin::proto::OpenMeshStreamRequest,
    ) -> Result<crate::plugin::proto::OpenMeshStreamResponse, crate::plugin::proto::ErrorResponse>
    {
        if request.stream_id.is_empty() {
            return Err(plugin_mesh_stream_error("stream_id is required"));
        }
        if request.target_peer_id.is_empty() {
            return Err(plugin_mesh_stream_error("target_peer_id is required"));
        }
        if request.channel.is_empty() {
            return Err(plugin_mesh_stream_error("channel is required"));
        }
        if !self
            .plugin_event_channel_declared(&plugin_id, &request.channel, "mesh stream")
            .await
        {
            return Err(plugin_mesh_stream_error(
                "plugin does not declare mesh channel",
            ));
        }

        request.plugin_id = plugin_id;
        let Some(conn) = self.connection_for_peer_hex(&request.target_peer_id).await else {
            return Err(plugin_mesh_stream_error("target peer is not connected"));
        };
        let listener = match crate::plugin::bind_local_listener(
            &crate::plugin::make_instance_id(),
            "mesh-stream",
        )
        .await
        {
            Ok(listener) => listener,
            Err(error) => return Err(plugin_mesh_stream_error(error.to_string())),
        };
        let response = crate::plugin::proto::OpenMeshStreamResponse {
            stream_id: request.stream_id.clone(),
            accepted: true,
            transport_kind: listener.transport_kind(),
            endpoint: Some(listener.endpoint()),
            token: None,
            expires_at_unix_ms: None,
            message: None,
        };

        tokio::spawn(async move {
            if let Err(error) = bridge_outbound_plugin_mesh_stream(listener, conn, request).await {
                tracing::debug!("Plugin mesh stream bridge failed: {error}");
            }
        });
        Ok(response)
    }

    pub(super) async fn handle_plugin_mesh_stream(
        &self,
        _remote: iroh::EndpointId,
        send: SendStream,
        mut recv: RecvStream,
    ) -> Result<()> {
        let buf = read_len_prefixed(&mut recv).await?;
        let request = crate::plugin::proto::OpenMeshStreamRequest::decode(buf.as_slice())?;
        if request.plugin_id.is_empty() || request.channel.is_empty() {
            anyhow::bail!("Plugin mesh stream is missing plugin_id or channel");
        }
        if !self
            .plugin_event_channel_declared(
                &request.plugin_id,
                &request.channel,
                "inbound mesh stream",
            )
            .await
        {
            return Ok(());
        }

        let plugin_manager = self
            .plugin_manager
            .lock()
            .await
            .clone()
            .context("No plugin manager is available for mesh stream")?;
        let local = plugin_manager
            .connect_stream(
                &request.plugin_id,
                open_stream_request_from_mesh_request(&request),
            )
            .await?;

        if request.bidirectional {
            bridge_local_stream_bidirectional(local, send, recv).await
        } else {
            bridge_local_stream_to_quic(local, send).await
        }
    }

    async fn connection_for_peer_hex(&self, peer_id: &str) -> Option<Connection> {
        let state = self.state.lock().await;
        state
            .connections
            .iter()
            .find(|(id, _)| endpoint_id_hex(**id) == peer_id)
            .map(|(_, conn)| conn.clone())
    }
}

async fn bridge_outbound_plugin_mesh_stream(
    listener: crate::plugin::LocalListener,
    conn: Connection,
    request: crate::plugin::proto::OpenMeshStreamRequest,
) -> Result<()> {
    let local = listener.accept().await?;
    let (mut send, recv) = conn.open_bi().await?;
    send.write_all(&[STREAM_PLUGIN_MESH_STREAM]).await?;
    write_len_prefixed(&mut send, &request.encode_to_vec()).await?;
    if request.bidirectional {
        bridge_local_stream_bidirectional(local, send, recv).await
    } else {
        send.finish()?;
        bridge_quic_to_local_stream(recv, local).await
    }
}

async fn bridge_quic_to_local_stream(
    recv: RecvStream,
    local: crate::plugin::LocalStream,
) -> Result<()> {
    match local {
        #[cfg(unix)]
        crate::plugin::LocalStream::Unix(stream) => copy_quic_to_local_write(recv, stream).await,
        #[cfg(windows)]
        crate::plugin::LocalStream::PipeClient(stream) => {
            copy_quic_to_local_write(recv, stream).await
        }
        #[cfg(windows)]
        crate::plugin::LocalStream::PipeServer(stream) => {
            copy_quic_to_local_write(recv, stream).await
        }
    }
}

async fn bridge_local_stream_to_quic(
    local: crate::plugin::LocalStream,
    send: SendStream,
) -> Result<()> {
    match local {
        #[cfg(unix)]
        crate::plugin::LocalStream::Unix(stream) => copy_local_read_to_quic(stream, send).await,
        #[cfg(windows)]
        crate::plugin::LocalStream::PipeClient(stream) => {
            copy_local_read_to_quic(stream, send).await
        }
        #[cfg(windows)]
        crate::plugin::LocalStream::PipeServer(stream) => {
            copy_local_read_to_quic(stream, send).await
        }
    }
}

async fn bridge_local_stream_bidirectional(
    local: crate::plugin::LocalStream,
    send: SendStream,
    recv: RecvStream,
) -> Result<()> {
    match local {
        #[cfg(unix)]
        crate::plugin::LocalStream::Unix(stream) => {
            bridge_stream_bidirectional(stream, send, recv).await
        }
        #[cfg(windows)]
        crate::plugin::LocalStream::PipeClient(stream) => {
            bridge_stream_bidirectional(stream, send, recv).await
        }
        #[cfg(windows)]
        crate::plugin::LocalStream::PipeServer(stream) => {
            bridge_stream_bidirectional(stream, send, recv).await
        }
    }
}

async fn copy_quic_to_local_write<S>(mut recv: RecvStream, stream: S) -> Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let (_read_half, mut write_half) = tokio::io::split(stream);
    tokio::io::copy(&mut recv, &mut write_half).await?;
    write_half.shutdown().await?;
    Ok(())
}

async fn copy_local_read_to_quic<S>(stream: S, mut send: SendStream) -> Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let (mut read_half, _write_half) = tokio::io::split(stream);
    tokio::io::copy(&mut read_half, &mut send).await?;
    send.finish()?;
    Ok(())
}

async fn bridge_stream_bidirectional<S>(
    stream: S,
    mut send: SendStream,
    mut recv: RecvStream,
) -> Result<()>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let (mut local_read, mut local_write) = tokio::io::split(stream);
    let to_mesh = async {
        tokio::io::copy(&mut local_read, &mut send).await?;
        send.finish()?;
        Ok::<_, anyhow::Error>(())
    };
    let from_mesh = async {
        tokio::io::copy(&mut recv, &mut local_write).await?;
        local_write.shutdown().await?;
        Ok::<_, anyhow::Error>(())
    };
    tokio::try_join!(to_mesh, from_mesh)?;
    Ok(())
}
