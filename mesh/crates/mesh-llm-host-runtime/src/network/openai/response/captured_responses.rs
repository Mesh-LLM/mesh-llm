//! Decode the bounded in-process HTTP/SSE response produced for a Responses
//! WebSocket turn.

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use tokio::{io::AsyncRead, sync::mpsc};

use super::{
    body_reader::BodyReader,
    probe::{probe_http_response, response_is_event_stream, try_parse_response_headers},
    shared_responses_stream::{MAX_SSE_FRAME_BYTES, frame_end, normalize_crlf, sse_frame_data},
};

const MAX_ERROR_BODY_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CapturedResponsesOutcome {
    Completed,
    Failed,
}

pub(crate) async fn capture_responses_events<R>(
    mut reader: R,
    sender: mpsc::Sender<Value>,
) -> Result<CapturedResponsesOutcome>
where
    R: AsyncRead + Unpin,
{
    let probe = probe_http_response(&mut reader).await?;
    let headers = try_parse_response_headers(&probe.buffered)?
        .context("captured Mesh response has incomplete HTTP headers")?;
    let mut body = BodyReader::new(
        reader,
        probe.buffered[headers.header_end..].to_vec(),
        headers.chunked,
        headers.content_length,
    );
    if !(200..300).contains(&headers.status_code) {
        let value = captured_http_error(&mut body, headers.status_code).await;
        sender
            .send(value)
            .await
            .map_err(|_| anyhow::anyhow!("Responses WebSocket writer stopped"))?;
        return Ok(CapturedResponsesOutcome::Failed);
    }
    if !response_is_event_stream(&headers) {
        bail!("captured Responses turn did not produce an event stream");
    }

    let mut state = CaptureState::default();
    let result = capture_sse_body(&mut body, &sender, &mut state).await;
    match result {
        Ok(()) if state.terminal == Some(CapturedResponsesOutcome::Completed) => {
            Ok(CapturedResponsesOutcome::Completed)
        }
        Ok(()) if state.terminal == Some(CapturedResponsesOutcome::Failed) => {
            Ok(CapturedResponsesOutcome::Failed)
        }
        Ok(()) => {
            let error = anyhow::anyhow!("captured Responses stream ended before a terminal event");
            finish_capture_failure(&sender, &mut state, &error).await
        }
        Err(error) => finish_capture_failure(&sender, &mut state, &error).await,
    }
}

#[derive(Default)]
struct CaptureState {
    created_response: Option<Value>,
    next_sequence: u64,
    terminal: Option<CapturedResponsesOutcome>,
}

async fn capture_sse_body<R: AsyncRead + Unpin>(
    body: &mut BodyReader<R>,
    sender: &mpsc::Sender<Value>,
    state: &mut CaptureState,
) -> Result<()> {
    let mut carry = Vec::new();
    while let Some(bytes) = body.next().await? {
        carry.extend_from_slice(&bytes);
        normalize_crlf(&mut carry);
        while let Some(end) = frame_end(&carry, 0) {
            if end > MAX_SSE_FRAME_BYTES {
                bail!("captured Responses SSE frame exceeded the frame limit");
            }
            let frame = std::str::from_utf8(&carry[..end])
                .context("captured Responses SSE frame was not UTF-8")?;
            let data = sse_frame_data(frame);
            carry.drain(..end + 2);
            if data.is_empty() || data == "[DONE]" {
                continue;
            }
            let value: Value =
                serde_json::from_str(&data).context("captured Responses SSE data was not JSON")?;
            state.observe(&value);
            sender
                .send(value)
                .await
                .map_err(|_| anyhow::anyhow!("Responses WebSocket writer stopped"))?;
        }
        if carry.len() > MAX_SSE_FRAME_BYTES {
            bail!("captured Responses SSE frame exceeded the frame limit");
        }
    }
    if !carry.is_empty() {
        bail!("captured Responses SSE stream ended inside a frame");
    }
    Ok(())
}

impl CaptureState {
    fn observe(&mut self, value: &Value) {
        if let Some(sequence) = value.get("sequence_number").and_then(Value::as_u64) {
            self.next_sequence = self.next_sequence.max(sequence.saturating_add(1));
        }
        match value.get("type").and_then(Value::as_str) {
            Some("response.created") => {
                self.created_response = value.get("response").cloned();
            }
            Some("response.completed" | "response.incomplete") => {
                self.terminal = Some(CapturedResponsesOutcome::Completed);
            }
            Some("response.failed") => {
                self.terminal = Some(CapturedResponsesOutcome::Failed);
            }
            _ => {}
        }
    }
}

async fn finish_capture_failure(
    sender: &mpsc::Sender<Value>,
    state: &mut CaptureState,
    error: &anyhow::Error,
) -> Result<CapturedResponsesOutcome> {
    if let Some(outcome) = state.terminal {
        return Ok(outcome);
    }
    let Some(mut response) = state.created_response.take() else {
        return Err(anyhow::anyhow!(error.to_string()));
    };
    let code = "mesh_response_capture_failed";
    let message = error.to_string();
    let error_body = json!({
        "type": "server_error",
        "code": code,
        "message": message,
        "param": null
    });
    sender
        .send(json!({
            "type": "error",
            "sequence_number": state.next_sequence,
            "error": error_body
        }))
        .await
        .map_err(|_| anyhow::anyhow!("Responses WebSocket writer stopped"))?;
    state.next_sequence = state.next_sequence.saturating_add(1);
    response["status"] = Value::String("failed".into());
    response["output"] = json!([]);
    response["error"] = json!({"code": code, "message": message});
    response["store"] = Value::Bool(false);
    sender
        .send(json!({
            "type": "response.failed",
            "sequence_number": state.next_sequence,
            "response": response
        }))
        .await
        .map_err(|_| anyhow::anyhow!("Responses WebSocket writer stopped"))?;
    Ok(CapturedResponsesOutcome::Failed)
}

async fn captured_http_error<R: AsyncRead + Unpin>(body: &mut BodyReader<R>, status: u16) -> Value {
    let mut bytes = Vec::new();
    while let Ok(Some(chunk)) = body.next().await {
        if bytes.len().saturating_add(chunk.len()) > MAX_ERROR_BODY_BYTES {
            break;
        }
        bytes.extend_from_slice(&chunk);
    }
    let parsed = serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null);
    let error = parsed.get("error").cloned().unwrap_or_else(|| {
        json!({
            "message": "Mesh Responses turn failed before streaming",
            "type": "server_error",
            "param": null,
            "code": "mesh_responses_turn_failed"
        })
    });
    json!({"type":"error", "status":status, "error":error})
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;

    #[tokio::test]
    async fn eof_after_created_is_repaired_with_a_failed_terminal() {
        let created = json!({
            "type": "response.created",
            "sequence_number": 0,
            "response": {
                "id": "resp_capture",
                "object": "response",
                "created_at": 1,
                "model": "m",
                "status": "in_progress",
                "output": [],
                "error": null,
                "store": true
            }
        });
        let body = format!("data: {created}\n\n");
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let (mut writer, reader) = tokio::io::duplex(response.len());
        writer.write_all(response.as_bytes()).await.unwrap();
        writer.shutdown().await.unwrap();
        let (sender, mut events) = mpsc::channel(4);

        let outcome = capture_responses_events(reader, sender).await.unwrap();
        assert_eq!(outcome, CapturedResponsesOutcome::Failed);
        let mut captured = Vec::new();
        while let Some(event) = events.recv().await {
            captured.push(event);
        }
        assert_eq!(captured.len(), 3);
        assert_eq!(captured[0]["type"], "response.created");
        assert_eq!(captured[1]["type"], "error");
        assert_eq!(captured[1]["sequence_number"], 1);
        assert_eq!(captured[2]["type"], "response.failed");
        assert_eq!(captured[2]["sequence_number"], 2);
        assert_eq!(captured[2]["response"]["id"], "resp_capture");
        assert_eq!(captured[2]["response"]["status"], "failed");
        assert_eq!(captured[2]["response"]["store"], false);
    }
}
