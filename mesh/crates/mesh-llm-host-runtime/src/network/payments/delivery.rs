//! Read token watermarks from bytes already accepted by the peer transport.
//! Generation may run ahead of the socket; queued tokens are not yet billable.

use anyhow::{Result, ensure};

#[derive(Default)]
pub(super) struct DeliveryUsage {
    headers_done: bool,
    chunked: bool,
    streaming: bool,
    buffer: Vec<u8>,
    body: Vec<u8>,
    tokens: u64,
}

impl DeliveryUsage {
    pub fn observe(&mut self, bytes: &[u8]) -> Result<u64> {
        self.buffer.extend_from_slice(bytes);
        ensure!(
            self.buffer.len() <= 2 * 1024 * 1024,
            "response framing exceeded limit"
        );
        if !self.headers_done {
            let mut headers = [httparse::EMPTY_HEADER; 64];
            let mut response = httparse::Response::new(&mut headers);
            let httparse::Status::Complete(offset) = response.parse(&self.buffer)? else {
                return Ok(self.tokens);
            };
            self.chunked = response.headers.iter().any(|h| {
                h.name.eq_ignore_ascii_case("transfer-encoding")
                    && h.value.eq_ignore_ascii_case(b"chunked")
            });
            self.streaming = response.headers.iter().any(|h| {
                h.name.eq_ignore_ascii_case("content-type")
                    && h.value.starts_with(b"text/event-stream")
            });
            self.buffer.drain(..offset);
            self.headers_done = true;
        }
        if self.chunked {
            loop {
                let httparse::Status::Complete((offset, length)) =
                    httparse::parse_chunk_size(&self.buffer)
                        .map_err(|_| anyhow::anyhow!("invalid HTTP chunk size"))?
                else {
                    break;
                };
                ensure!(length <= 2 * 1024 * 1024, "response chunk exceeded limit");
                let length = length as usize;
                if length == 0 {
                    self.buffer.clear();
                    break;
                }
                if self.buffer.len() < offset + length + 2 {
                    break;
                }
                ensure!(
                    &self.buffer[offset + length..offset + length + 2] == b"\r\n",
                    "invalid response chunk"
                );
                let bytes = self.buffer[offset..offset + length].to_vec();
                self.buffer.drain(..offset + length + 2);
                self.observe_body(&bytes)?;
            }
        } else {
            let bytes = std::mem::take(&mut self.buffer);
            self.observe_body(&bytes)?;
        }
        Ok(self.tokens)
    }

    fn observe_body(&mut self, bytes: &[u8]) -> Result<()> {
        self.body.extend_from_slice(bytes);
        ensure!(
            self.body.len() <= 2 * 1024 * 1024,
            "response usage buffer exceeded limit"
        );
        if self.streaming {
            while let Some(end) = self.body.iter().position(|byte| *byte == b'\n') {
                let line = self.body.drain(..=end).collect::<Vec<_>>();
                if let Some(data) = line.strip_prefix(b"data: ") {
                    self.observe_json(data);
                }
            }
        } else {
            let body = self.body.clone();
            self.observe_json(&body);
        }
        Ok(())
    }

    fn observe_json(&mut self, bytes: &[u8]) {
        if let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes)
            && let Some(tokens) = value
                .get("usage")
                .and_then(|u| u.get("completion_tokens"))
                .and_then(serde_json::Value::as_u64)
        {
            self.tokens = self.tokens.max(tokens);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payments_usage_handles_every_byte_boundary_and_partial_delivery() {
        let mut usage = DeliveryUsage::default();
        let prefix = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
        for byte in prefix {
            assert_eq!(usage.observe(&[*byte]).unwrap(), 0);
        }
        let data = b"data: {\"usage\":{\"completion_tokens\":3}}\n\n";
        let chunk = format!(
            "{:x}\r\n{}\r\n",
            data.len(),
            std::str::from_utf8(data).unwrap()
        );
        for (index, byte) in chunk.bytes().enumerate() {
            let count = usage.observe(&[byte]).unwrap();
            if index < chunk.len() - 1 {
                assert_eq!(count, 0);
            } else {
                assert_eq!(count, 3);
            }
        }
        // A partly sent next watermark never bills its claimed token count.
        usage
            .observe(b"3f\r\ndata: {\"usage\":{\"completion_tokens\":100")
            .unwrap();
        assert_eq!(usage.tokens, 3);
    }
}
