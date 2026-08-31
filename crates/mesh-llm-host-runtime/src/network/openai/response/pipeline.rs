use crate::mesh;
use crate::network::openai::client_stream::ClientStream;
use tokio::io::AsyncWriteExt;

use crate::network::openai::request_parse::pipeline_request_supported;
use crate::network::openai::response::common::parse_token_usage_from_json_body;
use mesh_llm_events::logging::events::TokenUsage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineProxyResult {
    Responded(u16),
    RespondedWithUsage { status_code: u16, usage: TokenUsage },
    Dropped,
    FallbackToDirect,
}

/// Pipeline-aware HTTP proxy for local targets.
///
/// Instead of TCP tunneling, this:
/// 1. Parses the HTTP request body
/// 2. Calls the planner model for a pre-plan
/// 3. Injects the plan into the request
/// 4. Forwards to the strong model via HTTP
/// 5. Streams the response back to the client
pub async fn pipeline_proxy_local(
    client_stream: &mut ClientStream,
    request_path: &str,
    mut body: serde_json::Value,
    planner_port: u16,
    planner_model: &str,
    strong_port: u16,
    node: &mesh::Node,
) -> PipelineProxyResult {
    if !pipeline_request_supported(request_path, &body) {
        tracing::debug!("pipeline: request path/body not eligible, falling back to direct proxy");
        return PipelineProxyResult::FallbackToDirect;
    }

    let http_client = reqwest::Client::new();
    let planner_url = format!("http://127.0.0.1:{planner_port}");
    if !pipeline_preplan_request(&http_client, &planner_url, planner_model, &mut body).await {
        return PipelineProxyResult::FallbackToDirect;
    }

    let strong_url = format!("http://127.0.0.1:{strong_port}/v1/chat/completions");
    let _inflight = node.begin_inflight_request();
    let is_streaming = pipeline_streaming_requested(&body);
    if is_streaming {
        pipeline_proxy_streaming(client_stream, &http_client, &strong_url, &body).await
    } else {
        pipeline_proxy_non_streaming(client_stream, &http_client, &strong_url, &body).await
    }
}

fn pipeline_streaming_requested(body: &serde_json::Value) -> bool {
    body.get("stream")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

async fn pipeline_preplan_request(
    http_client: &reqwest::Client,
    planner_url: &str,
    planner_model: &str,
    body: &mut serde_json::Value,
) -> bool {
    let messages = body
        .get("messages")
        .and_then(|messages| messages.as_array())
        .cloned()
        .unwrap_or_default();
    match crate::inference::pipeline::pre_plan(http_client, planner_url, planner_model, &messages)
        .await
    {
        Ok(plan) => {
            tracing::info!(
                "pipeline: pre-plan by {} in {}ms — {}",
                plan.model_used,
                plan.elapsed_ms,
                plan.plan_text.chars().take(200).collect::<String>()
            );
            crate::inference::pipeline::inject_plan(body, &plan);
            true
        }
        Err(err) => {
            tracing::warn!("pipeline: pre-plan failed ({err}), falling back to direct proxy");
            false
        }
    }
}

async fn pipeline_proxy_streaming(
    client_stream: &mut ClientStream,
    http_client: &reqwest::Client,
    strong_url: &str,
    body: &serde_json::Value,
) -> PipelineProxyResult {
    match http_client.post(strong_url).json(body).send().await {
        Ok(resp) => relay_pipeline_streaming_response(client_stream, resp).await,
        Err(err) => {
            tracing::warn!(
                "pipeline: strong model request failed: {err}, falling back to direct proxy"
            );
            PipelineProxyResult::FallbackToDirect
        }
    }
}

fn completed_pipeline_response(
    status: reqwest::StatusCode,
    usage: Option<TokenUsage>,
) -> PipelineProxyResult {
    if status.is_success() {
        usage.map_or(PipelineProxyResult::Responded(status.as_u16()), |usage| {
            PipelineProxyResult::RespondedWithUsage {
                status_code: status.as_u16(),
                usage,
            }
        })
    } else {
        PipelineProxyResult::Responded(status.as_u16())
    }
}

async fn relay_pipeline_streaming_response(
    client_stream: &mut ClientStream,
    resp: reqwest::Response,
) -> PipelineProxyResult {
    let status = resp.status();
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("text/event-stream")
        .to_string();
    let header = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nTransfer-Encoding: chunked\r\nCache-Control: no-cache\r\n\r\n",
    );
    if client_stream.write_all(header.as_bytes()).await.is_err() {
        return PipelineProxyResult::Dropped;
    }

    use tokio_stream::StreamExt;
    let mut stream = resp.bytes_stream();
    let mut usage_parser = SseUsageParser::default();
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(bytes) if write_pipeline_chunk(client_stream, &bytes).await.is_err() => {
                return PipelineProxyResult::Dropped;
            }
            Ok(bytes) => usage_parser.push(&bytes),
            Err(err) => {
                tracing::debug!("pipeline: stream error: {err}");
                return PipelineProxyResult::Dropped;
            }
        }
    }
    if client_stream.write_all(b"0\r\n\r\n").await.is_err()
        || client_stream.shutdown().await.is_err()
    {
        return PipelineProxyResult::Dropped;
    }
    completed_pipeline_response(status, usage_parser.usage)
}

#[derive(Default)]
struct SseUsageParser {
    carry: Vec<u8>,
    usage: Option<TokenUsage>,
}

impl SseUsageParser {
    fn push(&mut self, bytes: &[u8]) {
        self.carry.extend_from_slice(bytes);
        while let Some(end) = self
            .carry
            .windows(2)
            .position(|window| matches!(window, b"\n\n" | b"\r\n"))
        {
            let frame = self.carry.drain(..end + 2).collect::<Vec<_>>();
            for line in frame.split(|byte| *byte == b'\n') {
                let line = line.strip_suffix(b"\r").unwrap_or(line);
                let Some(data) = line.strip_prefix(b"data:") else {
                    continue;
                };
                let data = data.strip_prefix(b" ").unwrap_or(data);
                if let Some(usage) = parse_token_usage_from_json_body(data) {
                    self.usage = Some(usage);
                }
            }
        }
        if self.carry.len() > 64 * 1024 {
            self.carry.clear();
        }
    }
}

async fn write_pipeline_chunk(
    client_stream: &mut ClientStream,
    bytes: &[u8],
) -> std::io::Result<()> {
    let chunk_header = format!("{:x}\r\n", bytes.len());
    client_stream.write_all(chunk_header.as_bytes()).await?;
    client_stream.write_all(bytes).await?;
    client_stream.write_all(b"\r\n").await
}

async fn pipeline_proxy_non_streaming(
    client_stream: &mut ClientStream,
    http_client: &reqwest::Client,
    strong_url: &str,
    body: &serde_json::Value,
) -> PipelineProxyResult {
    match http_client.post(strong_url).json(body).send().await {
        Ok(resp) => relay_pipeline_non_streaming_response(client_stream, resp).await,
        Err(err) => {
            tracing::warn!(
                "pipeline: strong model request failed: {err}, falling back to direct proxy"
            );
            PipelineProxyResult::FallbackToDirect
        }
    }
}

async fn relay_pipeline_non_streaming_response(
    client_stream: &mut ClientStream,
    resp: reqwest::Response,
) -> PipelineProxyResult {
    let status = resp.status();
    match resp.bytes().await {
        Ok(resp_bytes) => {
            let usage = parse_token_usage_from_json_body(&resp_bytes);
            let header = format!(
                "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                resp_bytes.len()
            );
            if client_stream.write_all(header.as_bytes()).await.is_err()
                || client_stream.write_all(&resp_bytes).await.is_err()
                || client_stream.shutdown().await.is_err()
            {
                PipelineProxyResult::Dropped
            } else {
                completed_pipeline_response(status, usage)
            }
        }
        Err(err) => {
            tracing::warn!("pipeline: response read failed: {err}, falling back to direct proxy");
            PipelineProxyResult::FallbackToDirect
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_sse_usage_parser_keeps_last_valid_full_usage() {
        let mut parser = SseUsageParser::default();
        parser.push(b"data: {\"usage\":{\"prompt_tokens\":2,");
        parser.push(b"\"completion_tokens\":3,\"total_tokens\":5}}\n\n");
        parser.push(
            b"data: {\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":5,\"total_tokens\":8}}\n\n",
        );
        parser.push(
            b"data: {\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":6,\"total_tokens\":10}}\n\n",
        );

        assert_eq!(
            parser.usage,
            Some(TokenUsage {
                prompt_tokens: Some(4),
                cached_prompt_tokens: None,
                completion_tokens: Some(6),
                total_tokens: Some(10),
            })
        );
    }

    #[test]
    fn pipeline_sse_usage_parser_does_not_estimate_partial_usage() {
        let mut parser = SseUsageParser::default();
        parser.push(b"data: {\"usage\":{\"completion_tokens\":3}}\n\n");
        assert_eq!(parser.usage, None);
    }

    #[test]
    fn pipeline_error_statuses_discard_usage_from_error_bodies() {
        let usage = Some(TokenUsage {
            prompt_tokens: Some(2),
            cached_prompt_tokens: None,
            completion_tokens: Some(3),
            total_tokens: Some(5),
        });

        assert_eq!(
            completed_pipeline_response(reqwest::StatusCode::BAD_REQUEST, usage),
            PipelineProxyResult::Responded(400)
        );
        assert_eq!(
            completed_pipeline_response(reqwest::StatusCode::BAD_GATEWAY, usage),
            PipelineProxyResult::Responded(502)
        );
    }
}
