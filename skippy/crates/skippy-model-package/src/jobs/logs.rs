//! Bounded byte-oriented SSE lines. Payloads remain explicit job log output.
use super::{LogEntry, TransportLimits};
use anyhow::{Result, anyhow, bail};
use futures::{Stream, StreamExt as _};
use std::{collections::VecDeque, pin::Pin, time::Instant};
type BytesStream =
    Pin<Box<dyn Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>> + Send>>;
struct State {
    source: BytesStream,
    partial: Vec<u8>,
    ready: VecDeque<String>,
    seen: usize,
    until: Instant,
    limits: TransportLimits,
    ended: bool,
}
impl State {
    fn line(&mut self) -> Result<()> {
        let text = std::str::from_utf8(&self.partial)
            .map_err(|_| anyhow!("HF Jobs log line UTF-8 invalid"))?
            .trim();
        if let Some(payload) = text.strip_prefix("data:") {
            let payload = payload.strip_prefix(' ').unwrap_or(payload);
            // Preserve the existing plain-data fallback; error diagnostics never contain it.
            let data = serde_json::from_str::<LogEntry>(payload)
                .map_or_else(|_| payload.to_owned(), |entry| entry.data);
            self.ready.push_back(data);
        }
        self.partial.clear();
        Ok(())
    }
    fn chunk(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.len() > self.limits.log_bytes.saturating_sub(self.seen) {
            bail!("HF Jobs log total byte bound exceeded");
        }
        self.seen += bytes.len();
        for &byte in bytes {
            if byte == b'\n' {
                self.line()?;
            } else {
                if self.partial.len() >= self.limits.log_line_bytes {
                    bail!("HF Jobs log line byte bound exceeded");
                }
                self.partial.push(byte);
            }
        }
        Ok(())
    }
}
pub(super) fn stream(
    response: reqwest::Response,
    until: Instant,
    limits: TransportLimits,
) -> impl Stream<Item = Result<String>> + use<> {
    let state = State {
        source: Box::pin(response.bytes_stream()),
        partial: Vec::new(),
        ready: VecDeque::new(),
        seen: 0,
        until,
        limits,
        ended: false,
    };
    futures::stream::unfold(state, |mut state| async move {
        loop {
            // Expiration covers queued lines as well as pending network chunks.
            if state.ended {
                return None;
            }
            if Instant::now() >= state.until {
                state.ended = true;
                return Some((Err(anyhow!("HF Jobs log deadline expired")), state));
            }
            if let Some(line) = state.ready.pop_front() {
                return Some((Ok(line), state));
            }
            let next = tokio::time::timeout_at(state.until.into(), state.source.next()).await;
            let result = match next {
                Err(_) => Err(anyhow!("HF Jobs log deadline expired")),
                Ok(Some(Err(_))) => Err(anyhow!("HF Jobs log read failed")),
                Ok(Some(Ok(bytes))) => state.chunk(&bytes),
                Ok(None) if state.partial.is_empty() => {
                    return None;
                }
                Ok(None) => Err(anyhow!("HF Jobs log ended with incomplete SSE line")),
            };
            if let Err(error) = result {
                state.ended = true;
                return Some((Err(error), state));
            }
        }
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    fn state() -> State {
        State {
            source: Box::pin(futures::stream::empty()),
            partial: Vec::new(),
            ready: VecDeque::new(),
            seen: 0,
            until: Instant::now() + std::time::Duration::from_secs(1),
            limits: TransportLimits::default(),
            ended: false,
        }
    }
    #[test]
    fn jobs_sse_split_utf8_and_json_reassemble_exactly_before_decoding() {
        let mut state = state();
        let line = "data: {\"data\":\"café\",\"timestamp\":null}\n".as_bytes();
        for byte in line {
            state.chunk(std::slice::from_ref(byte)).unwrap();
        }
        assert!(state.partial.is_empty());
        assert_eq!(state.ready.pop_front().unwrap(), "café");
        assert!(state.ready.is_empty());
    }
    #[test]
    fn jobs_sse_invalid_utf8_and_line_bound_errors_do_not_echo_payload() {
        let mut invalid = state();
        let error = invalid.chunk(b"data: private\xff\n").unwrap_err();
        assert!(!error.to_string().contains("private"));
        let mut capped = state();
        capped.limits.log_line_bytes = 4;
        assert!(capped.chunk(b"data: private\n").is_err());
        assert!(capped.partial.len() <= 4);
    }
}
