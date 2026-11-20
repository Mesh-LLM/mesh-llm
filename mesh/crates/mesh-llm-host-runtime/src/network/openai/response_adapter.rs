use crate::network::openai::client_stream::ClientStream;
use tokio::io::AsyncWriteExt;

pub(crate) use mesh_openresponses::translate_chat_completion_to_responses;

pub(crate) fn sse_frame(event: Option<&str>, data: &str) -> Vec<u8> {
    let mut frame = Vec::new();
    if let Some(event_name) = event {
        frame.extend_from_slice(format!("event: {event_name}\n").as_bytes());
    }
    for line in data.lines() {
        frame.extend_from_slice(b"data: ");
        frame.extend_from_slice(line.as_bytes());
        frame.extend_from_slice(b"\n");
    }
    if data.is_empty() {
        frame.extend_from_slice(b"data: \n");
    }
    frame.extend_from_slice(b"\n");
    frame
}

async fn write_chunked_bytes(stream: &mut ClientStream, bytes: &[u8]) -> std::io::Result<()> {
    let header = format!("{:x}\r\n", bytes.len());
    stream.write_all(header.as_bytes()).await?;
    stream.write_all(bytes).await?;
    stream.write_all(b"\r\n").await
}

pub(crate) async fn write_chunked_sse_event(
    stream: &mut ClientStream,
    event: Option<&str>,
    data: &str,
) -> std::io::Result<()> {
    let frame = sse_frame(event, data);
    write_chunked_bytes(stream, &frame).await
}
