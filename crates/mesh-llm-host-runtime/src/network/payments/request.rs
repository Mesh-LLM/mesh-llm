use anyhow::{Context, Result, ensure};
use serde_json::Value;

pub(crate) struct PaidRequest {
    pub model: String,
    pub max_tokens: Option<u32>,
    pub path: String,
    pub body: Value,
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
        })
    }

    pub fn validate_output_allowance(&self, tokens: u64) -> Result<()> {
        ensure!(
            tokens > 0 && tokens <= u64::from(self.max_tokens.unwrap_or(u32::MAX)),
            "backend output allowance exceeds caller limit"
        );
        Ok(())
    }

    pub fn backend_http(&self, request_id: &str) -> Result<Vec<u8>> {
        let bytes = serde_json::to_vec(&self.body)?;
        let mut raw = format!("POST {} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nConnection: close\r\nx-request-id: {}\r\nContent-Length: {}\r\n\r\n", self.path, request_id, bytes.len()).into_bytes();
        raw.extend(bytes);
        Ok(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
