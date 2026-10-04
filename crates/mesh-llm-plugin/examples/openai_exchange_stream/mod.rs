use mesh_llm_plugin::{PluginError, PluginResult, bind_side_stream, proto};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, sync::Arc};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    sync::Mutex,
};

// Preserve exact permissioned request bytes even when core JSON parsing fails.
// A malformed client entity is evidence, not an observer transport failure.
type RetainedRequests = BTreeMap<(String, String), (Option<serde_json::Value>, Vec<u8>)>;
type Evidence = BTreeMap<(String, String), (String, u64)>;
#[derive(Clone, Default)]
pub struct StreamEvidence(Arc<Mutex<Evidence>>, Arc<Mutex<RetainedRequests>>);

impl StreamEvidence {
    pub async fn open(
        &self,
        request: proto::OpenStreamRequest,
    ) -> PluginResult<Option<proto::OpenStreamResponse>> {
        if request.mode != proto::StreamMode::RawBytes as i32 || !request.bidirectional {
            return Err(PluginError::invalid_request(
                "expected read-only raw entity bytes",
            ));
        }
        let metadata: serde_json::Value =
            serde_json::from_str(request.metadata_json.as_deref().unwrap_or("{}"))
                .map_err(|e| PluginError::invalid_request(e.to_string()))?;
        if metadata["receipt_protocol"] != "sha256-v1" {
            return Err(PluginError::invalid_request(
                "required sha256-v1 receipt protocol",
            ));
        }
        let kind = metadata["kind"]
            .as_str()
            .ok_or_else(|| PluginError::invalid_request("missing lifecycle stream kind"))?
            .to_owned();
        if !matches!(
            kind.as_str(),
            "openai_exchange_request"
                | "openai_exchange_original"
                | "openai_exchange_effective"
                | "openai_exchange_response"
        ) {
            return Err(PluginError::invalid_request(
                "unsupported lifecycle stream kind",
            ));
        }
        let exchange_id = metadata["exchange_id"]
            .as_str()
            .or(request.correlation_id.as_deref())
            .ok_or_else(|| PluginError::invalid_request("missing exchange correlation"))?
            .to_owned();
        // macOS temp directories are long; keep the negotiated socket pathname
        // below sockaddr_un limits. Correlation still uses the full UUID.
        let socket_id = uuid::Uuid::parse_str(&request.stream_id)
            .map_err(|e| PluginError::invalid_request(e.to_string()))?
            .simple()
            .to_string();
        let listener = bind_side_stream("oe", &socket_id[..16])
            .await
            .map_err(|e| PluginError::invalid_request(e.to_string()))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(listener.endpoint(), std::fs::Permissions::from_mode(0o600))
                .map_err(|e| PluginError::invalid_request(e.to_string()))?;
        }
        let mut response = listener.open_stream_response(&request);
        let token = uuid::Uuid::new_v4().to_string();
        response.token = Some(token.clone());
        let evidence = self.clone();
        tokio::spawn(async move {
            // Response bytes are hashed without retention. Request retention is
            // bounded separately and parsing never changes the byte receipt.
            let result = tokio::time::timeout(std::time::Duration::from_secs(30), async {
                let stream = listener.accept().await?;
                let (mut read, mut write) = stream.into_split();
                let mut auth = [0u8; 36];
                read.read_exact(&mut auth).await?;
                anyhow::ensure!(auth == token.as_bytes(), "side stream token mismatch");
                let mut digest = Sha256::new();
                let mut count = 0u64;
                let mut buffer = [0u8; 8192];
                let mut request_body = Vec::new();
                loop {
                    let length = read.read(&mut buffer).await?;
                    if length == 0 {
                        break;
                    }
                    count += length as u64;
                    anyhow::ensure!(count <= 16_777_216, "exemplar stream limit exceeded");
                    digest.update(&buffer[..length]);
                    if kind == "openai_exchange_response" && count == length as u64 {
                        log_receipt(
                            &metadata,
                            &serde_json::json!({"byte_count":count}),
                            None,
                            false,
                        )
                        .await?;
                    }
                    if kind != "openai_exchange_response" {
                        request_body.extend_from_slice(&buffer[..length]);
                    }
                }
                if let Some(expected) = request.expected_bytes {
                    anyhow::ensure!(expected == count, "incomplete side stream");
                }
                if kind != "openai_exchange_response" {
                    let parsed = serde_json::from_slice(&request_body).ok();
                    let mut bodies = evidence.1.lock().await;
                    let total: u64 = bodies.values().map(|(_, bytes)| bytes.len() as u64).sum();
                    anyhow::ensure!(
                        total + count <= 16_777_216,
                        "exemplar request retention limit exceeded"
                    );
                    bodies.insert((exchange_id.clone(), kind.clone()), (parsed, request_body));
                }
                let hash = hex::encode(digest.finalize());
                let mut receipts = evidence.0.lock().await;
                if receipts.len() >= 1024 {
                    receipts.pop_first();
                }
                receipts.insert((exchange_id.clone(), kind.clone()), (hash.clone(), count));
                drop(receipts);
                let parsed = evidence
                    .1
                    .lock()
                    .await
                    .get(&(exchange_id, kind))
                    .and_then(|(parsed, _)| parsed.clone());
                log_receipt(
                    &metadata,
                    &serde_json::json!({"sha256":hash,"byte_count":count}),
                    parsed,
                    true,
                )
                .await?;
                let receipt =
                    serde_json::json!({"sha256":hash,"byte_count":count}).to_string() + "\n";
                write.write_all(receipt.as_bytes()).await?;
                write.shutdown().await?;
                anyhow::Ok(())
            })
            .await;
            if !matches!(result, Ok(Ok(()))) {
                eprintln!("lifecycle side-stream evidence unavailable");
            }
        });
        Ok(Some(response))
    }

    pub async fn request_body(&self, event: &serde_json::Value) -> Option<serde_json::Value> {
        let id = event["exchange_id"].as_str()?;
        let kind = if event["phase"] == "request_received" {
            "openai_exchange_request"
        } else {
            "openai_exchange_effective"
        };
        let mut bodies = self.1.lock().await;
        bodies
            .remove(&(id.into(), kind.into()))
            .or_else(|| bodies.remove(&(id.into(), "openai_exchange_original".into())))
            .and_then(|(value, _)| value)
    }

    pub async fn verify_terminal(&self, event: &serde_json::Value) -> bool {
        let Some(id) = event["exchange_id"].as_str() else {
            return false;
        };
        self.1
            .lock()
            .await
            .retain(|(exchange, _), _| exchange != id);
        let commitment = &event["response_wire_commitment"];
        if !commitment["incomplete"].is_null() || commitment["side_stream_complete"] != true {
            self.0
                .lock()
                .await
                .retain(|(exchange, _), _| exchange != id);
            return false;
        }
        let Some(digest) = commitment["sha256"].as_str() else {
            return false;
        };
        let Some(count) = commitment["byte_count"].as_u64() else {
            return false;
        };
        let mut receipts = self.0.lock().await;
        let verified = receipts
            .get(&(id.to_owned(), "openai_exchange_response".into()))
            .is_some_and(|value| value == &(digest.into(), count));
        receipts.retain(|(exchange, _), _| exchange != id);
        verified
    }
}

async fn log_receipt(
    metadata: &serde_json::Value,
    receipt: &serde_json::Value,
    parsed_body: Option<serde_json::Value>,
    complete: bool,
) -> anyhow::Result<()> {
    let Ok(path) = std::env::var("MESH_LLM_EXEMPLAR_RECEIPT_LOG") else {
        return Ok(());
    };
    let mut options = tokio::fs::OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    options.mode(0o600);
    let mut file = options.open(path).await?;
    let mut line = serde_json::to_vec(
        &serde_json::json!({"metadata":metadata,"receipt":receipt,"parsed_body":parsed_body,"complete":complete}),
    )?;
    line.push(b'\n');
    file.write_all(&line).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn malformed_request_retains_exact_bytes_and_acknowledges_digest() {
        let evidence = StreamEvidence::default();
        let id = uuid::Uuid::new_v4().to_string();
        let bytes = b"{invalid json";
        let response = evidence.open(proto::OpenStreamRequest {
            stream_id: uuid::Uuid::new_v4().to_string(),
            purpose: proto::StreamPurpose::HttpRequestBody as i32,
            mode: proto::StreamMode::RawBytes as i32,
            bidirectional: true,
            metadata_json: Some(serde_json::json!({"kind":"openai_exchange_request","exchange_id":id,"receipt_protocol":"sha256-v1"}).to_string()),
            expected_bytes: Some(bytes.len() as u64),
            ..Default::default()
        }).await.unwrap().unwrap();
        let stream = mesh_llm_plugin::connect_side_stream(
            response.endpoint.as_deref().unwrap(),
            response.transport_kind,
        )
        .await
        .unwrap();
        let (mut read, mut write) = stream.into_split();
        write
            .write_all(response.token.as_ref().unwrap().as_bytes())
            .await
            .unwrap();
        write.write_all(bytes).await.unwrap();
        write.shutdown().await.unwrap();
        let mut ack = Vec::new();
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            read.read_to_end(&mut ack),
        )
        .await
        .unwrap()
        .unwrap();
        let ack: serde_json::Value = serde_json::from_slice(&ack).unwrap();
        assert_eq!(ack["sha256"], hex::encode(Sha256::digest(bytes)));
        assert_eq!(ack["byte_count"], bytes.len());
        let retained = evidence.1.lock().await;
        let (parsed, exact) = &retained[&(id.clone(), "openai_exchange_request".into())];
        assert!(parsed.is_none());
        assert_eq!(exact, bytes);
        drop(retained);
        let event = serde_json::json!({"exchange_id":id,"phase":"request_received","parse_status":"invalid_json"});
        assert!(evidence.request_body(&event).await.is_none());
    }

    #[tokio::test]
    async fn authenticated_live_side_stream_hashes_exact_bytes() {
        let evidence = StreamEvidence::default();
        let id = uuid::Uuid::new_v4().to_string();
        let bytes = b"data: {\"text\":\"a\\nb\"}\r\n\r\ndata: [DONE]\n\n";
        let request = proto::OpenStreamRequest {
            stream_id: uuid::Uuid::new_v4().to_string(),
            purpose: proto::StreamPurpose::HttpResponseBody as i32,
            mode: proto::StreamMode::RawBytes as i32,
            bidirectional: true,
            metadata_json: Some(
                serde_json::json!({"kind":"openai_exchange_response", "exchange_id":id, "receipt_protocol":"sha256-v1"})
                    .to_string(),
            ),
            expected_bytes: Some(bytes.len() as u64),
            ..Default::default()
        };
        let response = evidence.open(request).await.unwrap().unwrap();
        let stream = mesh_llm_plugin::connect_side_stream(
            response.endpoint.as_deref().unwrap(),
            response.transport_kind,
        )
        .await
        .unwrap();
        let (mut read, mut write) = stream.into_split();
        write
            .write_all(response.token.as_ref().unwrap().as_bytes())
            .await
            .unwrap();
        for chunk in bytes.chunks(3) {
            write.write_all(chunk).await.unwrap();
        }
        write.shutdown().await.unwrap();
        let mut ack = Vec::new();
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            read.read_to_end(&mut ack),
        )
        .await
        .unwrap()
        .unwrap();
        let ack: serde_json::Value = serde_json::from_slice(&ack).unwrap();
        assert_eq!(ack["sha256"], hex::encode(Sha256::digest(bytes)));
        assert_eq!(ack["byte_count"], bytes.len());
        let commitment = serde_json::json!({"sha256":hex::encode(Sha256::digest(bytes)), "byte_count":bytes.len(), "incomplete":null, "side_stream_complete":true});
        let event = serde_json::json!({"exchange_id":id, "response_wire_commitment":commitment});
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                if evidence
                    .0
                    .lock()
                    .await
                    .contains_key(&(id.clone(), "openai_exchange_response".into()))
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(evidence.verify_terminal(&event).await);
        assert!(evidence.0.lock().await.is_empty());
    }
}
