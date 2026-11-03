//! Authenticate and acknowledge exact request bytes without retaining prompt text.
use anyhow::{Context, Result, ensure};
use mesh_llm_plugin::{PluginError, PluginResult, bind_side_stream, proto};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};

const LIMIT: u64 = 64 * 1024;

#[derive(Clone)]
pub struct RequestStreams(Arc<tokio::sync::Semaphore>);
impl Default for RequestStreams {
    fn default() -> Self {
        Self(Arc::new(tokio::sync::Semaphore::new(32)))
    }
}
impl RequestStreams {
    pub async fn open(
        &self,
        request: proto::OpenStreamRequest,
    ) -> PluginResult<Option<proto::OpenStreamResponse>> {
        validate(&request).map_err(PluginError::from)?;
        let permit =
            self.0.clone().try_acquire_owned().map_err(|_| {
                PluginError::invalid_request("request side-stream capacity exhausted")
            })?;
        let socket_id = uuid::Uuid::parse_str(&request.stream_id)
            .map_err(|error| PluginError::invalid_request(error.to_string()))?
            .simple()
            .to_string();
        let listener = bind_side_stream("scb", &socket_id[..16])
            .await
            .map_err(PluginError::from)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(listener.endpoint(), std::fs::Permissions::from_mode(0o600))
                .map_err(anyhow::Error::from)?;
        }
        let mut response = listener.open_stream_response(&request);
        let token = uuid::Uuid::new_v4().to_string();
        response.token = Some(token.clone());
        tokio::spawn(async move {
            let _permit = permit;
            let result = tokio::time::timeout(std::time::Duration::from_secs(30), async {
                let stream = listener.accept().await?;
                let (mut read, mut write) = stream.into_split();
                let mut authentication = [0u8; 36];
                read.read_exact(&mut authentication).await?;
                ensure!(
                    authentication == token.as_bytes(),
                    "request side-stream token mismatch"
                );
                let (sha256, byte_count) = hash_body(
                    &mut read,
                    request.expected_bytes.context("missing byte length")?,
                )
                .await?;
                let receipt =
                    serde_json::json!({"sha256":sha256,"byte_count":byte_count}).to_string() + "\n";
                write.write_all(receipt.as_bytes()).await?;
                write.shutdown().await?;
                Ok::<_, anyhow::Error>(())
            })
            .await;
            if !matches!(result, Ok(Ok(()))) {
                tracing::warn!("stablecoin request side-stream receipt unavailable");
            }
        });
        Ok(Some(response))
    }
}
fn validate(request: &proto::OpenStreamRequest) -> Result<()> {
    ensure!(
        request.mode == proto::StreamMode::RawBytes as i32 && request.bidirectional,
        "expected bidirectional raw request side stream"
    );
    ensure!(
        request.expected_bytes.is_some_and(|count| count <= LIMIT),
        "request side stream exceeds 64 KiB or omits length"
    );
    let metadata: serde_json::Value =
        serde_json::from_str(request.metadata_json.as_deref().unwrap_or("{}"))?;
    ensure!(
        metadata["receipt_protocol"] == "sha256-v1",
        "unsupported request receipt protocol"
    );
    ensure!(
        matches!(
            metadata["kind"].as_str(),
            Some(
                "openai_exchange_request"
                    | "openai_exchange_original"
                    | "openai_exchange_effective"
            )
        ),
        "unsupported request side-stream kind"
    );
    let correlation = metadata["exchange_id"]
        .as_str()
        .or(request.correlation_id.as_deref())
        .context("request side stream lacks correlation")?;
    ensure!(
        !correlation.is_empty() && correlation.len() <= 256,
        "invalid request side-stream correlation"
    );
    Ok(())
}
async fn hash_body(read: &mut (impl AsyncRead + Unpin), expected: u64) -> Result<(String, u64)> {
    ensure!(expected <= LIMIT, "request body exceeds 64 KiB");
    let mut digest = Sha256::new();
    let mut count = 0u64;
    let mut buffer = [0u8; 8192];
    loop {
        let size = read.read(&mut buffer).await?;
        if size == 0 {
            break;
        }
        count += size as u64;
        ensure!(
            count <= LIMIT && count <= expected,
            "request body exceeded authorized byte length"
        );
        digest.update(&buffer[..size]);
    }
    ensure!(
        count == expected,
        "request side stream ended before expected byte length"
    );
    Ok((hex::encode(digest.finalize()), count))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn exact_receipt_preserves_json_whitespace_and_rejects_partial_or_extra_bytes() {
        let bytes = b"{ \"max_tokens\" : 4 }\n";
        let (hash, count) = hash_body(&mut &bytes[..], bytes.len() as u64)
            .await
            .unwrap();
        assert_eq!(hash, hex::encode(Sha256::digest(bytes)));
        assert_eq!(count, bytes.len() as u64);
        assert!(
            hash_body(&mut &bytes[..], bytes.len() as u64 + 1)
                .await
                .is_err()
        );
        assert!(
            hash_body(&mut &bytes[..], bytes.len() as u64 - 1)
                .await
                .is_err()
        );
        assert!(hash_body(&mut &bytes[..], LIMIT + 1).await.is_err());
    }
    #[test]
    fn rejects_ungranted_response_streams_and_unbounded_request_streams() {
        let mut request=proto::OpenStreamRequest{stream_id:uuid::Uuid::new_v4().to_string(),mode:proto::StreamMode::RawBytes as i32,bidirectional:true,
            expected_bytes:Some(1),metadata_json:Some(serde_json::json!({"kind":"openai_exchange_original","receipt_protocol":"sha256-v1","exchange_id":"id"}).to_string()),..Default::default()};
        validate(&request).unwrap();
        request.expected_bytes = Some(LIMIT + 1);
        assert!(validate(&request).is_err());
        request.expected_bytes = Some(1);
        request.metadata_json=Some(serde_json::json!({"kind":"openai_exchange_response","receipt_protocol":"sha256-v1","exchange_id":"id"}).to_string());
        assert!(validate(&request).is_err());
    }
}
