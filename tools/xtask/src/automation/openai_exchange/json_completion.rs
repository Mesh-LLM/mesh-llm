//! Bounded nonstream OpenAI completion projection, reusing shared POST ownership.
use super::Decoder;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::time::Duration;
#[derive(Debug, Serialize)]
pub(in crate::automation) struct Evidence {
    pub elapsed_seconds: f64,
    pub prompt_tokens: u64,
    pub cached_tokens: Option<u64>,
    pub content_sha256: String,
}
struct JsonDecoder {
    bytes: Vec<u8>,
}
impl Decoder for JsonDecoder {
    type Evidence = Evidence;
    fn accepts(&self, status: hyper::StatusCode) -> bool {
        status.is_success()
    }
    fn consume(&mut self, bytes: &[u8], _: Duration) -> Result<(), String> {
        if bytes.len() > (1024 * 1024_usize).saturating_sub(self.bytes.len()) {
            return Err("nonstream OpenAI body exceeds1MiB".into());
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    fn terminal(&self) -> bool {
        false
    }
    fn finish(self, elapsed: Duration) -> Result<Evidence, String> {
        let value: Value =
            serde_json::from_slice(&self.bytes).map_err(|_| "invalid nonstream OpenAI JSON")?;
        if value.get("error").is_some() {
            return Err("nonstream OpenAI server error".into());
        }
        let prompt = value["usage"]["prompt_tokens"]
            .as_u64()
            .filter(|n| *n > 0)
            .ok_or("missing/invalid prompt usage")?;
        let cached = value["usage"]["prompt_tokens_details"]
            .get("cached_tokens")
            .map(|value| value.as_u64().ok_or("invalid cache usage"))
            .transpose()?;
        if cached.is_some_and(|n| n > prompt) {
            return Err("cache usage exceeds prompt usage".into());
        }
        let content = value["choices"][0]["message"]["content"]
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or("missing generated nonstream content")?;
        Ok(Evidence {
            elapsed_seconds: elapsed.as_secs_f64(),
            prompt_tokens: prompt,
            cached_tokens: cached,
            content_sha256: hex::encode(Sha256::digest(content.as_bytes())),
        })
    }
}
/// Reuse the same bounded decoder for an externally supervised transport receipt.
pub(in crate::automation) fn decode_body(
    bytes: &[u8],
    elapsed: Duration,
) -> Result<Evidence, String> {
    let mut decoder = JsonDecoder { bytes: Vec::new() };
    decoder.consume(bytes, elapsed)?;
    decoder.finish(elapsed)
}
