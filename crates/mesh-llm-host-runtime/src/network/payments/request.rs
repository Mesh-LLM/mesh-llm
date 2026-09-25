use anyhow::{Context, Result, ensure};
use serde_json::Value;

pub(crate) struct PaidRequest {
    pub model: String,
    pub max_tokens: Option<u32>,
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
        // Preserve normal inference defaults and caller limits. The backend
        // reports its resolved context allowance after tokenizing the prompt.
        if body.get("stream").and_then(Value::as_bool) == Some(true) {
            body["stream_options"] = serde_json::json!({"include_usage": true});
        }
        Ok(Self {
            model,
            max_tokens,
            path: path.into(),
            body,
            intent,
        })
    }

    pub fn validate_output_allowance(&self, tokens: u64) -> Result<()> {
        ensure!(
            tokens > 0 && tokens <= u64::from(self.max_tokens.unwrap_or(u32::MAX)),
            "backend output allowance exceeds caller limit"
        );
        Ok(())
    }

    /// Address the local backend by its internal model name. The public
    /// `model` stays unchanged for pricing, terms and invoices.
    pub fn use_backend_model(&mut self, backend_model: &str) {
        if backend_model != self.model {
            self.body["model"] = Value::String(backend_model.to_owned());
        }
    }

    pub fn backend_http(&self, request_id: &str) -> Result<Vec<u8>> {
        let bytes = serde_json::to_vec(&self.body)?;
        let mut raw = format!("POST {} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nConnection: close\r\nx-request-id: {}\r\nContent-Length: {}\r\n\r\n", self.path, request_id, bytes.len()).into_bytes();
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
    fn payments_allow_large_explicit_limits_and_context_clamping() {
        let parsed = PaidRequest::parse(b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"model\":\"m\",\"max_completion_tokens\":65536,\"max_tokens\":8}").unwrap();
        assert_eq!(parsed.max_tokens, Some(65_536));
        assert_eq!(parsed.body["max_tokens"], 8);
        assert_eq!(parsed.body["max_completion_tokens"], 65_536);
        parsed.validate_output_allowance(6000).unwrap();
        assert!(parsed.validate_output_allowance(65_537).is_err());
        assert!(parsed.validate_output_allowance(0).is_err());
        let parsed = PaidRequest::parse(b"POST /v1/chat/completions HTTP/1.1\r\n\r\n{\"model\":\"m\",\"max_completion_tokens\":null,\"max_tokens\":8192}").unwrap();
        assert_eq!(parsed.max_tokens, Some(8192));
        let parsed = PaidRequest::parse(
            b"POST /v1/completions HTTP/1.1\r\n\r\n{\"model\":\"m\",\"max_tokens\":null}",
        )
        .unwrap();
        assert_eq!(parsed.max_tokens, None);
        assert!(
            parsed
                .validate_output_allowance(u64::from(u32::MAX) + 1)
                .is_err()
        );
    }

    #[test]
    fn payments_preserve_backend_defaults_and_strip_caller_identity_headers() {
        let parsed = PaidRequest::parse(b"POST /v1/chat/completions HTTP/1.1\r\nx-request-id: attacker\r\nAuthorization: secret\r\n\r\n{\"model\":\"model\",\"stream\":true}").unwrap();
        assert_eq!(parsed.max_tokens, None);
        assert!(parsed.body.get("max_tokens").is_none());
        assert!(parsed.body.get("max_completion_tokens").is_none());
        parsed.validate_output_allowance(65_536).unwrap();
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
}
