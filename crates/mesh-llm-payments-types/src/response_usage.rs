//! Bounded HTTP/JSON/SSE usage evidence. Response bytes remain unchanged and
//! stream chunks are never counted as model tokens.

use crate::metering::MeteredUsage;
use anyhow::{Result, ensure};

const LIMIT: usize = 2 * 1024 * 1024;

#[derive(Default)]
pub struct ResponseUsage {
    headers_done: bool,
    chunked: bool,
    streaming: bool,
    buffer: Vec<u8>,
    body: Vec<u8>,
    data: Vec<u8>,
    usage: Option<MeteredUsage>,
    ended: bool,
    finished: bool,
}

impl ResponseUsage {
    pub fn observe(&mut self, bytes: &[u8]) -> Result<()> {
        self.buffer.extend_from_slice(bytes);
        ensure!(
            self.buffer.len() <= LIMIT,
            "response framing exceeded limit"
        );
        if !self.headers_done {
            let mut headers = [httparse::EMPTY_HEADER; 64];
            let mut response = httparse::Response::new(&mut headers);
            let httparse::Status::Complete(offset) = response.parse(&self.buffer)? else {
                return Ok(());
            };
            ensure!(
                response.code.is_some_and(|code| (200..300).contains(&code)),
                "backend error response"
            );
            self.chunked = response.headers.iter().any(|header| {
                header.name.eq_ignore_ascii_case("transfer-encoding")
                    && header.value.eq_ignore_ascii_case(b"chunked")
            });
            self.streaming = response.headers.iter().any(|header| {
                header.name.eq_ignore_ascii_case("content-type")
                    && header.value.starts_with(b"text/event-stream")
            });
            self.buffer.drain(..offset);
            self.headers_done = true;
        }
        if self.chunked {
            self.observe_chunks()?;
        } else {
            let body = std::mem::take(&mut self.buffer);
            self.observe_body(&body)?;
        }
        Ok(())
    }

    fn observe_chunks(&mut self) -> Result<()> {
        while !self.ended {
            let httparse::Status::Complete((offset, length)) =
                httparse::parse_chunk_size(&self.buffer)
                    .map_err(|_| anyhow::anyhow!("invalid HTTP chunk size"))?
            else {
                break;
            };
            ensure!(length <= LIMIT as u64, "response chunk exceeded limit");
            let length = length as usize;
            if length == 0 {
                self.ended = true;
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
        Ok(())
    }

    fn observe_body(&mut self, bytes: &[u8]) -> Result<()> {
        self.body.extend_from_slice(bytes);
        ensure!(self.body.len() <= LIMIT, "usage buffer exceeded limit");
        if self.streaming {
            while let Some(end) = self.body.iter().position(|byte| *byte == b'\n') {
                let line = self.body.drain(..=end).collect::<Vec<_>>();
                self.observe_line(&line)?;
            }
        }
        Ok(())
    }

    fn observe_line(&mut self, line: &[u8]) -> Result<()> {
        let line = line.strip_suffix(b"\n").unwrap_or(line);
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.is_empty() {
            if let Ok(value) = serde_json::from_slice(&self.data)
                && let Some(usage) = MeteredUsage::from_response(&value)
            {
                if let Some(previous) = self.usage {
                    ensure!(
                        usage.prompt_tokens == previous.prompt_tokens
                            && usage.completion_tokens >= previous.completion_tokens,
                        "backend usage changed inconsistently"
                    );
                }
                self.usage = Some(usage);
            }
            self.data.clear();
        } else if let Some(data) = line.strip_prefix(b"data:") {
            if !self.data.is_empty() {
                self.data.push(b'\n');
            }
            self.data
                .extend_from_slice(data.strip_prefix(b" ").unwrap_or(data));
            ensure!(self.data.len() <= LIMIT, "SSE event exceeded limit");
        }
        Ok(())
    }

    /// Mark successful upstream EOF. Truncated chunk framing must not promote
    /// a non-streaming JSON prefix into a final bill.
    pub fn finish(&mut self) -> Result<()> {
        ensure!(self.headers_done, "incomplete backend response");
        ensure!(!self.chunked || self.ended, "incomplete chunked response");
        self.finished = true;
        Ok(())
    }

    /// Evidence received so far, usable after disconnect. No report is None;
    /// a partially received usage event cannot supply a final bill. Ordinary
    /// JSON is authoritative only after successful response completion.
    pub fn usage(&self) -> Option<MeteredUsage> {
        if self.streaming {
            self.usage
        } else if self.finished {
            serde_json::from_slice(&self.body)
                .ok()
                .and_then(|value| MeteredUsage::from_response(&value))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_and_chunked_sse_handle_every_byte_boundary() {
        let body = b"{\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7}}";
        let mut json = ResponseUsage::default();
        for byte in b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n"
            .iter()
            .chain(body)
        {
            json.observe(&[*byte]).unwrap();
        }
        let expected = Some(MeteredUsage {
            prompt_tokens: 12,
            completion_tokens: 7,
        });
        assert_eq!(json.usage(), None);
        json.finish().unwrap();
        assert_eq!(json.usage(), expected);
        let mut sse = ResponseUsage::default();
        let event =
            b"data:{\"usage\":\r\ndata: {\"prompt_tokens\":12,\"completion_tokens\":7}}\r\n\r\n";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n{:x}\r\n{}\r\n0\r\n\r\n",
            event.len(),
            std::str::from_utf8(event).unwrap()
        );
        for byte in response.bytes() {
            sse.observe(&[byte]).unwrap();
        }
        assert_eq!(sse.usage(), expected);
    }

    #[test]
    fn missing_partial_and_reasoning_chunks_are_not_token_counts() {
        let mut usage = ResponseUsage::default();
        usage.observe(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\ndata: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: {\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":9}}").unwrap();
        assert_eq!(usage.usage(), None);
        usage.observe(b"\n\n").unwrap();
        assert_eq!(
            usage.usage(),
            Some(MeteredUsage {
                prompt_tokens: 4,
                completion_tokens: 9
            })
        );
        assert!(
            usage
                .observe(b"data: {\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":8}}\n\n")
                .is_err()
        );
    }
}
