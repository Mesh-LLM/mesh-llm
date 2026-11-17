//! Bounded incremental SSE framing for the Responses relay and HTTP capture.

const MAX_FRAME_BYTES: usize = 1024 * 1024;

#[derive(Default)]
pub(super) struct SseFrames {
    bytes: Vec<u8>,
    consumed: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum FrameError {
    TooLarge,
    InvalidUtf8,
}

impl FrameError {
    pub(super) fn code(&self) -> &'static str {
        match self {
            Self::TooLarge => "upstream_stream_frame_too_large",
            Self::InvalidUtf8 => "upstream_stream_invalid",
        }
    }

    pub(super) fn message(&self) -> &'static str {
        match self {
            Self::TooLarge => "upstream stream event exceeded the frame limit",
            Self::InvalidUtf8 => "upstream emitted a non-UTF-8 stream event",
        }
    }
}

impl SseFrames {
    pub(super) fn push(&mut self, bytes: &[u8]) {
        self.bytes.drain(..self.consumed);
        self.consumed = 0;
        self.bytes.extend_from_slice(bytes);
        normalize_crlf(&mut self.bytes);
    }

    pub(super) fn is_empty(&self) -> bool {
        self.consumed == self.bytes.len()
    }

    pub(super) fn next_data(&mut self) -> Result<Option<String>, FrameError> {
        let pending = &self.bytes[self.consumed..];
        let end = pending.windows(2).position(|window| window == b"\n\n");
        // A delimiter split across feeds may occupy two bytes beyond a full frame.
        if end.is_some_and(|end| end > MAX_FRAME_BYTES)
            || (end.is_none()
                && pending.len() > MAX_FRAME_BYTES
                && !matches!(&pending[MAX_FRAME_BYTES..], b"\n" | b"\r" | b"\n\r"))
        {
            return Err(FrameError::TooLarge);
        }
        let Some(end) = end else { return Ok(None) };
        let frame = std::str::from_utf8(&pending[..end]).map_err(|_| FrameError::InvalidUtf8)?;
        let data = frame
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(str::trim_start)
            .collect::<Vec<_>>()
            .join("\n");
        self.consumed += end + 2;
        Ok(Some(data))
    }
}

fn normalize_crlf(bytes: &mut Vec<u8>) {
    if !bytes.contains(&b'\r') {
        return;
    }
    let mut read = 0;
    let mut write = 0;
    while read < bytes.len() {
        if bytes[read] == b'\r' && bytes.get(read + 1) == Some(&b'\n') {
            bytes[write] = b'\n';
            read += 2;
        } else {
            bytes[write] = bytes[read];
            read += 1;
        }
        write += 1;
    }
    bytes.truncate(write);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_split_preserves_crlf_utf8_and_multiline_data() {
        let wire = "event: ignored\r\ndata: café\r\ndata: second\r\n\r\ndata: last\r\n\r\n";
        for split in 0..=wire.len() {
            let mut frames = SseFrames::default();
            let mut data = Vec::new();
            for bytes in [&wire.as_bytes()[..split], &wire.as_bytes()[split..]] {
                frames.push(bytes);
                while let Some(frame) = frames.next_data().unwrap() {
                    data.push(frame);
                }
            }
            assert_eq!(data, ["café\nsecond", "last"], "split {split}");
            assert!(frames.is_empty());
        }
    }

    #[test]
    fn bounds_each_complete_or_unterminated_frame_not_the_whole_batch() {
        for terminal in ["", "\n\n"] {
            let mut frames = SseFrames::default();
            frames.push(format!("{}{}", "x".repeat(MAX_FRAME_BYTES + 1), terminal).as_bytes());
            assert_eq!(frames.next_data(), Err(FrameError::TooLarge));
        }
        let mut frames = SseFrames::default();
        let frame = format!("data: {}\n\n", "x".repeat(MAX_FRAME_BYTES - 6));
        frames.push(format!("{frame}{frame}").as_bytes());
        assert!(frames.next_data().unwrap().is_some());
        assert!(frames.next_data().unwrap().is_some());
        assert!(frames.is_empty());
    }

    #[test]
    fn exact_limit_accepts_split_lf_and_crlf_delimiters_only() {
        let frame = format!("data: {}", "x".repeat(MAX_FRAME_BYTES - 6));
        for delimiter in ["\n\n", "\r\n\r\n"] {
            for split in 0..delimiter.len() {
                let mut frames = SseFrames::default();
                frames.push(format!("{frame}{}", &delimiter[..split]).as_bytes());
                assert_eq!(frames.next_data(), Ok(None), "{delimiter:?} split {split}");
                frames.push(&delimiter.as_bytes()[split..]);
                assert_eq!(
                    frames.next_data().unwrap().unwrap().len(),
                    MAX_FRAME_BYTES - 6
                );
                assert!(frames.is_empty());
            }
        }
        let mut frames = SseFrames::default();
        frames.push(format!("{frame}\nx").as_bytes());
        assert_eq!(frames.next_data(), Err(FrameError::TooLarge));
    }

    #[test]
    fn incomplete_frames_wait_and_invalid_utf8_fails_without_lossy_decoding() {
        let mut frames = SseFrames::default();
        frames.push(b"data: \xff");
        assert_eq!(frames.next_data(), Ok(None));
        assert!(!frames.is_empty());
        frames.push(b"\n\n");
        assert_eq!(frames.next_data(), Err(FrameError::InvalidUtf8));
        assert_eq!(FrameError::InvalidUtf8.code(), "upstream_stream_invalid");
        assert!(FrameError::InvalidUtf8.message().contains("UTF-8"));
        assert_eq!(
            FrameError::TooLarge.code(),
            "upstream_stream_frame_too_large"
        );
        assert!(FrameError::TooLarge.message().contains("frame limit"));
    }
}
