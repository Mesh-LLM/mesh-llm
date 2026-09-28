//! MoA wire formatting delegates to the shared Messages protocol adapter.
use crate::network::openai::{
    client_stream::ClientStream, response_adapter::write_chunked_sse_event,
};
use tokio::io::AsyncWriteExt;

pub(super) async fn send(
    mut stream: ClientStream,
    body: &serde_json::Value,
    headers: &[(&str, String)],
    header_sent: bool,
) -> std::io::Result<()> {
    if !header_sent {
        super::streaming::write_sse_response_headers(&mut stream, headers).await?;
    }
    let events =
        openai_frontend::anthropic::completion_events(body).map_err(std::io::Error::other)?;
    for event in events {
        let data = serde_json::to_string(&event).map_err(std::io::Error::other)?;
        write_chunked_sse_event(&mut stream, Some(event.event_name()), &data).await?;
    }
    stream.write_all(b"0\r\n\r\n").await?;
    stream.shutdown().await
}
