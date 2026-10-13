use anyhow::{Context, Result, ensure};
use serde_json::Value;

/// Output tokens a paid request may produce when the caller set no limit.
pub(crate) const DEFAULT_OUTPUT_ALLOWANCE: u32 = 16_384;

pub(crate) struct PaidRequest {
    pub model: String,
    /// The caller's output limit, or [`DEFAULT_OUTPUT_ALLOWANCE`].
    pub max_tokens: u32,
    pub path: String,
    pub body: Value,
    pub intent: Option<mesh_llm_payments_types::intent::PaymentIntent>,
}

impl PaidRequest {
    pub fn parse(raw: &[u8]) -> Result<Self> {
        ensure!(raw.len() <= 1024 * 1024, "paid request exceeds 1 MiB limit");
        let mut headers = [httparse::EMPTY_HEADER; 64];
        let mut request = httparse::Request::new(&mut headers);
        let httparse::Status::Complete(offset) = request.parse(raw)? else {
            anyhow::bail!("incomplete paid request");
        };
        ensure!(
            request.method == Some("POST"),
            "paid inference requires POST"
        );
        let path = request.path.context("missing request path")?;
        ensure!(
            matches!(path, "/v1/chat/completions" | "/v1/completions"),
            "paid endpoint unsupported"
        );
        let mut body: Value = serde_json::from_slice(&raw[offset..])?;
        let intent = body
            .as_object_mut()
            .and_then(|body| body.remove("mesh_payment"))
            .map(serde_json::from_value::<mesh_llm_payments_types::intent::PaymentIntent>)
            .transpose()?;
        if let Some(intent) = &intent {
            intent.validate()?;
        }
        let model = body
            .get("model")
            .and_then(Value::as_str)
            .context("exact model required")?
            .to_owned();
        ensure!(!model.is_empty() && model.len() <= 1024, "invalid model");
        ensure!(
            body.get("n").is_none_or(|n| n.as_u64() == Some(1)),
            "paid fan-out is unsupported"
        );
        let read_limit = |key| -> Result<Option<u32>> {
            body.get(key)
                .filter(|value| !value.is_null())
                .map(|value| {
                    let limit = value.as_u64().context("invalid output token limit")?;
                    ensure!(limit > 0, "output token limit must be positive");
                    u32::try_from(limit).context("output token limit exceeds u32")
                })
                .transpose()
        };
        let max_tokens = if path == "/v1/chat/completions" {
            read_limit("max_completion_tokens")?.or(read_limit("max_tokens")?)
        } else {
            read_limit("max_tokens")?
        };
        // A metered request is always bounded: the buyer reserves budget for
        // the output allowance, so a caller that set no limit gets the
        // default one, forwarded to the backend as its limit.
        let max_tokens = match max_tokens {
            Some(limit) => limit,
            None => {
                let key = if path == "/v1/chat/completions" {
                    "max_completion_tokens"
                } else {
                    "max_tokens"
                };
                body.as_object_mut()
                    .context("request must be an object")?
                    .insert(key.into(), Value::from(DEFAULT_OUTPUT_ALLOWANCE));
                DEFAULT_OUTPUT_ALLOWANCE
            }
        };
        if body.get("stream").and_then(Value::as_bool) == Some(true) {
            let options = body
                .as_object_mut()
                .context("request must be an object")?
                .entry("stream_options")
                .or_insert_with(|| serde_json::json!({}));
            if options.is_null() {
                *options = serde_json::json!({});
            }
            let options = options.as_object_mut().context("invalid stream options")?;
            options.insert("include_usage".into(), Value::Bool(true));
        }
        Ok(Self {
            model,
            max_tokens,
            path: path.into(),
            body,
            intent,
        })
    }

    pub fn use_backend_model(&mut self, backend_model: &str) {
        if backend_model != self.model {
            self.body["model"] = Value::String(backend_model.to_owned());
        }
    }

    pub fn backend_http(&self, request_id: &str) -> Result<Vec<u8>> {
        let bytes = serde_json::to_vec(&self.body)?;
        let mut raw = format!("POST {} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nAccept-Encoding: identity\r\nConnection: close\r\nx-request-id: {}\r\nContent-Length: {}\r\n\r\n", self.path, request_id, bytes.len()).into_bytes();
        raw.extend(bytes);
        Ok(raw)
    }
}

/// Remove trusted-local policy metadata without altering any other request field.
pub(crate) fn strip_intent(raw: &[u8]) -> Result<Vec<u8>> {
    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut request = httparse::Request::new(&mut headers);
    let httparse::Status::Complete(offset) = request.parse(raw)? else {
        anyhow::bail!("incomplete request");
    };
    let Ok(mut body) = serde_json::from_slice::<Value>(&raw[offset..]) else {
        return Ok(raw.to_vec());
    };
    let Some(value) = body
        .as_object_mut()
        .and_then(|body| body.remove("mesh_payment"))
    else {
        return Ok(raw.to_vec());
    };
    let intent: mesh_llm_payments_types::intent::PaymentIntent = serde_json::from_value(value)?;
    intent.validate()?;
    let bytes = serde_json::to_vec(&body)?;
    let headers = std::str::from_utf8(&raw[..offset - 4])?;
    let mut rebuilt = String::new();
    for line in headers.split("\r\n") {
        if !line.to_ascii_lowercase().starts_with("content-length:") {
            rebuilt.push_str(line);
            rebuilt.push_str("\r\n");
        }
    }
    rebuilt.push_str(&format!("Content-Length: {}\r\n\r\n", bytes.len()));
    let mut result = rebuilt.into_bytes();
    result.extend(bytes);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_intent_is_validated_and_not_forwarded() {
        let raw = b"POST /v1/completions HTTP/1.1\r\nx-request-id: kept\r\n\r\n{\"model\":\"m\",\"mesh_payment\":{\"mode\":\"free_only\"}}";
        let request = PaidRequest::parse(raw).unwrap();
        assert!(request.intent.is_some());
        assert!(request.body.get("mesh_payment").is_none());
        let stripped = String::from_utf8(strip_intent(raw).unwrap()).unwrap();
        assert!(!stripped.contains("mesh_payment"));
        assert!(stripped.contains("x-request-id: kept"));
        assert!(PaidRequest::parse(b"POST /v1/completions HTTP/1.1\r\n\r\n{\"model\":\"m\",\"mesh_payment\":{\"mode\":\"bad\"}}").is_err());
    }

    #[test]
    fn payments_keep_explicit_limits_and_bound_unlimited_requests() {
        let parsed = PaidRequest::parse(b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"model\":\"m\",\"max_completion_tokens\":65536,\"max_tokens\":8}").unwrap();
        assert_eq!(parsed.max_tokens, 65_536);
        assert_eq!(parsed.body["max_tokens"], 8);
        assert_eq!(parsed.body["max_completion_tokens"], 65_536);
        let parsed = PaidRequest::parse(b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"model\":\"m\",\"max_completion_tokens\":null,\"max_tokens\":8192}").unwrap();
        assert_eq!(parsed.max_tokens, 8192);
        // No caller limit: the default allowance bounds the backend too.
        let parsed = PaidRequest::parse(
            b"POST /v1/completions HTTP/1.1\r\n\r\n{\"model\":\"m\",\"max_tokens\":null}",
        )
        .unwrap();
        assert_eq!(parsed.max_tokens, DEFAULT_OUTPUT_ALLOWANCE);
        assert_eq!(parsed.body["max_tokens"], DEFAULT_OUTPUT_ALLOWANCE);
        let parsed =
            PaidRequest::parse(b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"model\":\"m\"}")
                .unwrap();
        assert_eq!(
            parsed.body["max_completion_tokens"],
            DEFAULT_OUTPUT_ALLOWANCE
        );
        assert!(parsed.body.get("max_tokens").is_none());
    }

    #[test]
    fn payments_strip_caller_identity_headers() {
        let parsed = PaidRequest::parse(b"POST /v1/chat/completions HTTP/1.1\r\nx-request-id: attacker\r\nAuthorization: secret\r\n\r\n{\"model\":\"model\",\"stream\":true}").unwrap();
        assert_eq!(parsed.max_tokens, DEFAULT_OUTPUT_ALLOWANCE);
        assert!(parsed.body.get("max_tokens").is_none());
        let forwarded = String::from_utf8(parsed.backend_http("trusted-id").unwrap()).unwrap();
        assert!(!forwarded.contains("attacker"));
        assert!(!forwarded.contains("secret"));
        assert!(forwarded.contains("trusted-id"));
        assert_eq!(parsed.body["stream_options"]["include_usage"], true);
        for body in [
            r#"{"model":"m","max_tokens":0}"#,
            r#"{"model":"m","max_tokens":4294967296}"#,
            r#"{"model":"m","n":2}"#,
        ] {
            assert!(
                PaidRequest::parse(
                    format!("POST /v1/completions HTTP/1.1\r\n\r\n{body}").as_bytes()
                )
                .is_err()
            );
        }
    }

    #[test]
    fn backend_model_rewrites_only_the_forwarded_body() {
        let mut parsed = PaidRequest::parse(b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"model\":\"org/repo:Q4_K_M\",\"max_tokens\":8}").unwrap();
        parsed.use_backend_model("local-gguf/sha256-abc");
        assert_eq!(parsed.model, "org/repo:Q4_K_M");
        let raw = parsed.backend_http("id").unwrap();
        let body = &raw[raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap() + 4..];
        let body: Value = serde_json::from_slice(body).unwrap();
        assert_eq!(body["model"], "local-gguf/sha256-abc");
    }
    #[test]
    fn backend_stream_requests_include_usage_without_losing_options() {
        let parsed = PaidRequest::parse(b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"model\":\"m\",\"stream\":true,\"stream_options\":{\"include_usage\":false,\"other\":true}}").unwrap();
        assert_eq!(parsed.body["stream_options"]["include_usage"], true);
        assert_eq!(parsed.body["stream_options"]["other"], true);
        for options in [
            "",
            ",\"stream_options\":null",
            ",\"stream_options\":{\"include_usage\":true}",
        ] {
            let raw = format!(
                "POST /v1/chat/completions HTTP/1.1\r\n\r\n{{\"model\":\"m\",\"stream\":true{options}}}"
            );
            let parsed = PaidRequest::parse(raw.as_bytes()).unwrap();
            assert_eq!(parsed.body["stream_options"]["include_usage"], true);
        }
        let raw = parsed.backend_http("id").unwrap();
        let raw = String::from_utf8(raw).unwrap();
        assert!(raw.contains("Accept-Encoding: identity\r\n"));
        assert!(raw.contains("Connection: close\r\n"));
    }
    #[tokio::test]
    async fn backend_stream_opt_in_reaches_usage_collector_over_http() {
        use mesh_llm_payments_types::metering::MeteredUsage;
        use mesh_llm_payments_types::response_usage::{ResponseEvidence, ResponseUsage};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let backend = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut received = Vec::new();
            let mut byte = [0_u8; 1];
            loop {
                socket.read_exact(&mut byte).await.unwrap();
                received.extend(byte);
                let mut headers = [httparse::EMPTY_HEADER; 64];
                let mut request = httparse::Request::new(&mut headers);
                if let httparse::Status::Complete(offset) = request.parse(&received).unwrap() {
                    let length: usize = request
                        .headers
                        .iter()
                        .find(|header| header.name.eq_ignore_ascii_case("content-length"))
                        .map(|header| std::str::from_utf8(header.value).unwrap().parse().unwrap())
                        .unwrap();
                    if received.len() == offset + length {
                        let body: Value = serde_json::from_slice(&received[offset..]).unwrap();
                        assert_eq!(body["stream_options"]["include_usage"], true);
                        assert_eq!(body["stream"], true);
                        break;
                    }
                }
            }
            socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: {\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n").await.unwrap();
        });
        let request = PaidRequest::parse(
            b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"model\":\"m\",\"stream\":true}",
        )
        .unwrap();
        let mut socket = tokio::net::TcpStream::connect(address).await.unwrap();
        socket
            .write_all(&request.backend_http("id").unwrap())
            .await
            .unwrap();
        let mut collector = ResponseUsage::default();
        let mut byte = [0_u8; 1];
        while socket.read(&mut byte).await.unwrap() != 0 {
            collector.observe(&byte);
        }
        collector.finish();
        assert_eq!(
            collector.evidence(),
            ResponseEvidence::Reported(MeteredUsage {
                prompt_tokens: 5,
                completion_tokens: 2
            })
        );
        backend.await.unwrap();
    }
}
