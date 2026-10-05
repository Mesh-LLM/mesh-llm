//! Incremental, bounded decoding of a growing HTTP chunked wire buffer.

use anyhow::{Context, Result, bail};

#[derive(Clone, Copy)]
enum Phase {
    Size,
    Data(usize),
    DataTerminator,
    Trailers,
    Complete,
}

/// Retain parsing progress so each payload byte is copied at most once.
pub(super) struct ChunkedDecoder {
    phase: Phase,
    cursor: usize,
    line_scan: usize,
    decoded: Vec<u8>,
    limit: usize,
}

impl ChunkedDecoder {
    /// Create a decoder with a cumulative decoded-payload limit.
    pub(super) fn new(limit: usize) -> Self {
        Self {
            phase: Phase::Size,
            cursor: 0,
            line_scan: 0,
            decoded: Vec::new(),
            limit,
        }
    }

    /// Accept the same wire prefix plus newly read bytes; return its consumed length.
    pub(super) fn decode(&mut self, wire: &[u8]) -> Result<Option<usize>> {
        loop {
            match self.phase {
                Phase::Size => {
                    let Some(line) = self.line(wire) else {
                        return Ok(None);
                    };
                    let header = std::str::from_utf8(line).context("invalid chunk header")?;
                    let text = header.split(';').next().unwrap_or("").trim();
                    let size = usize::from_str_radix(text, 16)
                        .with_context(|| format!("invalid chunk size: {text}"))?;
                    if size > self.limit - self.decoded.len() {
                        bail!("HTTP chunked body exceeds {} bytes", self.limit);
                    }
                    self.phase = if size == 0 {
                        Phase::Trailers
                    } else {
                        Phase::Data(size)
                    };
                }
                Phase::Data(remaining) => {
                    let available = remaining.min(wire.len() - self.cursor);
                    if available == 0 {
                        return Ok(None);
                    }
                    self.decoded
                        .extend_from_slice(&wire[self.cursor..self.cursor + available]);
                    self.cursor += available;
                    self.phase = if available == remaining {
                        Phase::DataTerminator
                    } else {
                        Phase::Data(remaining - available)
                    };
                }
                Phase::DataTerminator => {
                    let Some(terminator) = wire.get(self.cursor..self.cursor + 2) else {
                        return Ok(None);
                    };
                    if terminator != b"\r\n" {
                        bail!("invalid chunk terminator")
                    }
                    self.cursor += 2;
                    self.line_scan = self.cursor;
                    self.phase = Phase::Size;
                }
                Phase::Trailers => {
                    let Some(line) = self.line(wire) else {
                        return Ok(None);
                    };
                    if line.is_empty() {
                        self.phase = Phase::Complete;
                    }
                }
                Phase::Complete => return Ok(Some(self.cursor)),
            }
        }
    }

    /// Search only newly available framing bytes, retaining a split CR/LF boundary.
    fn line<'a>(&mut self, wire: &'a [u8]) -> Option<&'a [u8]> {
        let Some(relative) = wire[self.line_scan..]
            .windows(2)
            .position(|pair| pair == b"\r\n")
        else {
            self.line_scan = wire.len().saturating_sub(1).max(self.cursor);
            return None;
        };
        let end = self.line_scan + relative;
        let line = &wire[self.cursor..end];
        self.cursor = end + 2;
        self.line_scan = self.cursor;
        Some(line)
    }

    /// Transfer the accumulated payload without an additional copy.
    pub(super) fn into_body(self) -> Vec<u8> {
        self.decoded
    }
}

/// Decode a complete buffered request for callers that do not read incrementally.
pub(super) fn try_decode_chunked_body(
    wire: &[u8],
    limit: usize,
) -> Result<Option<(usize, Vec<u8>)>> {
    let mut decoder = ChunkedDecoder::new(limit);
    Ok(decoder
        .decode(wire)?
        .map(|consumed| (consumed, decoder.into_body())))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every byte boundary, including extensions/trailers, preserves state and suffixes.
    #[test]
    fn fragmented_chunks_keep_progress_and_stop_before_the_next_request() {
        let wire = b"3;foo=bar\r\nabc\r\n2\r\nde\r\n0\r\nX-Trailer: value\r\n\r\nNEXT";
        let consumed = wire.len() - 4;
        let mut decoder = ChunkedDecoder::new(5);
        let mut previous_cursor = 0;
        for end in 0..=wire.len() {
            let result = decoder.decode(&wire[..end]).unwrap();
            assert!(decoder.cursor >= previous_cursor);
            previous_cursor = decoder.cursor;
            assert_eq!(result, (end >= consumed).then_some(consumed));
        }
        assert_eq!(decoder.into_body(), b"abcde");
    }

    /// An incomplete large chunk copies each new fragment without re-decoding its prefix.
    #[test]
    fn large_chunk_is_consumed_incrementally() {
        let payload = vec![42; 1024 * 1024];
        let mut wire = format!("{:x}\r\n", payload.len()).into_bytes();
        let prefix = wire.len();
        wire.extend_from_slice(&payload);
        wire.extend_from_slice(b"\r\n0\r\n\r\n");
        let mut decoder = ChunkedDecoder::new(payload.len());
        for end in (prefix..prefix + payload.len()).step_by(8192) {
            assert_eq!(decoder.decode(&wire[..end]).unwrap(), None);
            assert_eq!(decoder.decoded.len(), end - prefix);
            assert_eq!(decoder.cursor, end);
        }
        assert_eq!(decoder.decode(&wire).unwrap(), Some(wire.len()));
        assert_eq!(decoder.into_body(), payload);
    }

    /// Reject a declared over-limit size before buffering its payload or overflowing offsets.
    #[test]
    fn declared_sizes_are_checked_before_payload_allocation() {
        for wire in [
            b"6\r\n".to_vec(),
            format!("{:x}\r\n", usize::MAX).into_bytes(),
            b"3\r\nabc\r\n3\r\n".to_vec(),
        ] {
            let mut decoder = ChunkedDecoder::new(5);
            assert!(
                decoder
                    .decode(&wire)
                    .unwrap_err()
                    .to_string()
                    .contains("exceeds")
            );
            assert!(decoder.decoded.len() <= 3);
        }
    }

    /// Framing validation remains strict while incomplete input stays retryable.
    #[test]
    fn malformed_and_incomplete_framing_are_distinct() {
        assert!(try_decode_chunked_body(b"1\r\nx!!", 1).is_err());
        assert!(try_decode_chunked_body(b"xyz\r\n", 1).is_err());
        assert_eq!(try_decode_chunked_body(b"1\r\nx\r", 1).unwrap(), None);
        assert_eq!(
            try_decode_chunked_body(b"0\r\n\r\n", 0).unwrap(),
            Some((5, vec![]))
        );
    }
}
