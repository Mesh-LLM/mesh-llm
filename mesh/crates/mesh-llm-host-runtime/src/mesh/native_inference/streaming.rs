//! Native chunk streams with bounded backpressure and cancellation on drop.
use super::*;
use futures_util::StreamExt;
use serde_json::Value;
use tokio::sync::mpsc;

type ChunkSender = mpsc::Sender<Result<Value>>;

/// Await one piece of stream progress under the inactivity window.
///
/// The window is per wait, not per request: every chunk that arrives starts a
/// fresh window, so the total duration of an active generation is unbounded and
/// only a generation that stops producing fails.
async fn progress_within<T>(idle: Duration, progress: impl Future<Output = T>) -> Result<T> {
    tokio::time::timeout(idle, progress)
        .await
        .context("native inference timed out")
}

impl Node {
    pub(crate) async fn native_chat_completion_stream(
        &self,
        mut call: NativeChatRequest,
    ) -> Result<ChatCompletionStream> {
        normalize_target(&mut call)?;
        call.request.stream = true;
        let provider: EndpointId = call
            .target
            .provider_node_id
            .parse()
            .context("invalid native inference provider ID")?;
        let (tx, rx) = mpsc::channel(16);
        let node = self.clone();
        let idle = self.native_inference.stream_idle();
        let task = tokio::spawn(async move {
            let result = tokio::select! {
                _ = tx.closed() => return,
                result = async {
                    if provider == node.endpoint.id() {
                        node.produce_local_chunks(call, &tx, idle).await
                    } else {
                        node.produce_remote_chunks(provider, call, &tx, idle).await
                    }
                } => result,
            };
            if let Err(error) = result {
                let _ = tx.send(Err(error)).await;
            }
        });
        Ok(ChatCompletionStream::new(rx, task))
    }

    async fn produce_local_chunks(
        &self,
        call: NativeChatRequest,
        tx: &ChunkSender,
        idle: Duration,
    ) -> Result<()> {
        let mut prepared = self.prepare_native_chat(call, false).await?;
        let request = std::mem::take(&mut prepared.request);
        let mut chunks = tokio::select! {
            _ = tx.closed() => return Ok(()),
            started = progress_within(
                idle,
                prepared.backend.chat_completion_stream(request, prepared.context.clone()),
            ) => started??,
        };
        loop {
            let chunk = tokio::select! {
                _ = tx.closed() => return Ok(()),
                chunk = progress_within(idle, chunks.next()) => chunk?,
            };
            let Some(chunk) = chunk else {
                break;
            };
            tx.send(Ok(serde_json::to_value(chunk?)?))
                .await
                .context("native chunk consumer closed")?;
        }
        Ok(())
    }

    async fn produce_remote_chunks(
        &self,
        provider: EndpointId,
        call: NativeChatRequest,
        tx: &ChunkSender,
        idle: Duration,
    ) -> Result<()> {
        let (_send, mut recv) = tokio::time::timeout(idle, self.open_native_chat(provider, &call))
            .await
            .context("native inference timed out opening the provider stream")??;
        loop {
            let frame = tokio::select! {
                _ = tx.closed() => return Ok(()),
                frame = progress_within(idle, read_native_frame(&mut recv)) => frame??,
            };
            if let Some(error) = frame.error {
                anyhow::bail!("native inference provider: {error}");
            }
            // Only an explicit empty terminal frame is completion. A lost
            // connection before it is an error, not a truncated success.
            if frame.payload_json.is_empty() {
                return Ok(());
            }
            let chunk =
                serde_json::from_slice(&frame.payload_json).context("decode native chat chunk")?;
            tx.send(Ok(chunk))
                .await
                .context("native chunk consumer closed")?;
        }
    }

    pub(super) async fn serve_native_chat(
        &self,
        call: NativeChatRequest,
        send: &mut iroh::endpoint::SendStream,
    ) -> Result<()> {
        let idle = self.native_inference.stream_idle();
        let stream = call.request.stream;
        let mut prepared = self.prepare_native_chat(call, true).await?;
        let request = std::mem::take(&mut prepared.request);
        if stream {
            let mut chunks = tokio::select! {
                _ = send.stopped() => return Ok(()),
                started = progress_within(
                    idle,
                    prepared.backend.chat_completion_stream(request, prepared.context.clone()),
                ) => started??,
            };
            loop {
                let chunk = tokio::select! {
                    _ = send.stopped() => return Ok(()),
                    chunk = progress_within(idle, chunks.next()) => chunk?,
                };
                let Some(chunk) = chunk else {
                    break;
                };
                send_value(send, &chunk?).await?;
            }
            write_native_frame(
                send,
                crate::proto::node::NativeChatFrame {
                    generation: 1,
                    payload_json: Vec::new(),
                    error: None,
                },
            )
            .await
        } else {
            let response = tokio::select! {
                _ = send.stopped() => return Ok(()),
                response = prepared.backend.chat_completion_with_context(request, prepared.context.clone()) => response?,
            };
            send_value(send, &response).await
        }
    }
}

async fn send_value(send: &mut iroh::endpoint::SendStream, value: &impl Serialize) -> Result<()> {
    write_native_frame(
        send,
        crate::proto::node::NativeChatFrame {
            generation: 1,
            payload_json: serde_json::to_vec(value)?,
            error: None,
        },
    )
    .await
}
