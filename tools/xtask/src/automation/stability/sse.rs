use serde_json::Value;

const LINE_LIMIT: usize = 1_048_576;
const EVENT_LIMIT: usize = 8192;

#[derive(Default)]
pub(super) struct Decoder {
    pending: Vec<u8>,
    data: Vec<String>,
    data_bytes: usize,
    pub events: Vec<Value>,
    pub first_event_ms: Option<u64>,
    pub done: bool,
}

impl Decoder {
    pub fn push(&mut self, bytes: &[u8], elapsed_ms: u64) -> Result<(), String> {
        for part in bytes.split_inclusive(|byte| *byte == b'\n') {
            if self.done {
                break;
            }
            if part.len() > LINE_LIMIT.saturating_sub(self.pending.len()) {
                return Err("stream line exceeds 1 MiB".into());
            }
            self.pending.extend_from_slice(part);
            if self.pending.last() == Some(&b'\n') {
                self.pending.pop();
                self.flush_line(elapsed_ms)?;
            }
        }
        Ok(())
    }

    fn flush_line(&mut self, elapsed_ms: u64) -> Result<(), String> {
        let line = std::str::from_utf8(&self.pending)
            .map_err(|_| "stream event is not UTF-8")?
            .trim_end_matches('\r')
            .to_owned();
        self.pending.clear();
        if line.is_empty() {
            return self.dispatch(elapsed_ms);
        }
        if let Some(data) = line.strip_prefix("data:") {
            let data = data.strip_prefix(' ').unwrap_or(data);
            if self.data.len() >= EVENT_LIMIT
                || data.len().saturating_add(1) > LINE_LIMIT.saturating_sub(self.data_bytes)
            {
                return Err("stream event exceeds 1 MiB".into());
            }
            self.data_bytes += data.len() + 1;
            self.data.push(data.into());
        }
        Ok(())
    }

    fn dispatch(&mut self, elapsed_ms: u64) -> Result<(), String> {
        if self.data.is_empty() {
            return Ok(());
        }
        let data = std::mem::take(&mut self.data).join("\n");
        self.data_bytes = 0;
        if data.trim() == "[DONE]" {
            self.done = true;
            return Ok(());
        }
        let event: Value = serde_json::from_str(&data).map_err(|_| "stream event was not JSON")?;
        if !event.is_object() {
            return Err("stream event JSON was not an object".into());
        }
        if self.events.len() >= EVENT_LIMIT {
            return Err("stream event count exceeds 8192".into());
        }
        self.first_event_ms.get_or_insert(elapsed_ms);
        self.events.push(event);
        Ok(())
    }

    pub fn finish(mut self, elapsed_ms: u64) -> Result<Self, String> {
        if !self.done {
            if !self.pending.is_empty() {
                self.flush_line(elapsed_ms)?;
            }
            self.dispatch(elapsed_ms)?;
        }
        if self.events.is_empty() {
            return Err("stream returned no events".into());
        }
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fragmented_utf8_multiline_events_comments_and_done_preserve_objects() {
        let text = ": heartbeat\r\ndata: {\"choices\":\r\ndata: [{\"delta\":{\"content\":\"é\"}}]}\r\n\r\ndata: [DONE]\r\n\r\ndata: invalid\r\n\r\n";
        let mut decoder = Decoder::default();
        for byte in text.as_bytes() {
            decoder.push(&[*byte], 17).unwrap();
        }
        let result = decoder.finish(20).unwrap();
        assert_eq!(result.events.len(), 1);
        assert_eq!(result.events[0]["choices"][0]["delta"]["content"], "é");
        assert_eq!(result.first_event_ms, Some(17));
        assert!(result.done);
    }

    #[test]
    fn eof_can_complete_an_event_but_empty_or_malformed_streams_fail() {
        let mut decoder = Decoder::default();
        decoder
            .push(b"data: {\"usage\":{\"completion_tokens\":3}}", 1)
            .unwrap();
        assert_eq!(decoder.finish(9).unwrap().first_event_ms, Some(9));
        for body in [
            b"data: [DONE]\n\n".as_slice(),
            b": comment\n\n",
            b"data: []\n\n",
            b"data: invalid\n\n",
            b"data: \xff\n\n",
        ] {
            let mut decoder = Decoder::default();
            let result = decoder.push(body, 1).and_then(|()| decoder.finish(1));
            assert!(result.is_err());
        }
    }
}
