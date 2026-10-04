//! Bounded extraction of final HTTP response entity bytes for raw ingress.

use std::sync::Arc;

use openai_frontend::wire_bytes::{WireBytesIncomplete, WireBytesObserver, WireBytesTap};

const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_CHUNK_LINE_BYTES: usize = 1024;

enum Framing {
    Headers(Vec<u8>),
    Fixed(u64),
    UntilClose,
    ChunkSize(Vec<u8>),
    ChunkData(u64),
    ChunkEnd(u8),
    Trailers { line: Vec<u8>, total: usize },
    Done,
    Invalid,
}

/// Parses only framing, never retains a response body or waits for an observer.
pub(crate) struct HttpResponseByteTap {
    framing: Framing,
    tap: WireBytesTap,
    observer: Arc<dyn WireBytesObserver>,
}

impl HttpResponseByteTap {
    pub(super) fn execution_outcome(&self, outcome: &str) {
        self.observer.execution_outcome(outcome);
    }
    pub(super) fn new(observer: Arc<dyn WireBytesObserver>) -> Self {
        Self {
            framing: Framing::Headers(Vec::new()),
            tap: WireBytesTap::new(Some(observer.clone())),
            observer,
        }
    }

    /// Called only for the prefix accepted by the downstream AsyncWrite.
    pub(super) fn update(&mut self, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            match &mut self.framing {
                Framing::Headers(header) => {
                    let next = advance_headers(header, bytes[0], &*self.observer);
                    bytes = &bytes[1..];
                    self.advance(next);
                }
                Framing::Fixed(remaining) | Framing::ChunkData(remaining) => {
                    let count = usize::try_from(*remaining)
                        .unwrap_or(usize::MAX)
                        .min(bytes.len());
                    self.tap.update(&bytes[..count]);
                    bytes = &bytes[count..];
                    *remaining -= count as u64;
                    if *remaining == 0 {
                        self.framing = if matches!(self.framing, Framing::Fixed(_)) {
                            Framing::Done
                        } else {
                            Framing::ChunkEnd(0)
                        };
                    }
                }
                Framing::UntilClose => {
                    self.tap.update(bytes);
                    return;
                }
                Framing::ChunkSize(line) => {
                    let next = advance_chunk_size(line, bytes[0]);
                    bytes = &bytes[1..];
                    self.advance(next);
                }
                Framing::ChunkEnd(position) => {
                    let next = advance_chunk_end(position, bytes[0]);
                    bytes = &bytes[1..];
                    self.advance(next);
                }
                Framing::Trailers { line, total } => {
                    let next = advance_trailers(line, total, bytes[0]);
                    bytes = &bytes[1..];
                    self.advance(next);
                }
                Framing::Done | Framing::Invalid => return,
            }
        }
    }

    fn advance(&mut self, next: Option<Framing>) {
        if let Some(next) = next {
            self.framing = next;
            if matches!(self.framing, Framing::Invalid) {
                self.tap.finish(Some(WireBytesIncomplete::InvalidFraming));
            }
        }
    }

    pub(super) fn finish(&mut self, failure: Option<WireBytesIncomplete>) {
        let incomplete = failure.or(match self.framing {
            Framing::Done | Framing::UntilClose => None,
            Framing::Invalid => Some(WireBytesIncomplete::InvalidFraming),
            _ => Some(WireBytesIncomplete::TransportError),
        });
        self.tap.finish(incomplete);
    }
}

fn advance_headers(
    header: &mut Vec<u8>,
    byte: u8,
    observer: &dyn WireBytesObserver,
) -> Option<Framing> {
    header.push(byte);
    if header.len() > MAX_HEADER_BYTES {
        return Some(Framing::Invalid);
    }
    if !header.ends_with(b"\r\n\r\n") {
        return None;
    }
    if let Some(status) = response_status(header).filter(|status| *status >= 200) {
        observer.response_status(status);
    }
    Some(response_framing(header).unwrap_or(Framing::Invalid))
}

fn advance_chunk_size(line: &mut Vec<u8>, byte: u8) -> Option<Framing> {
    line.push(byte);
    if line.len() > MAX_CHUNK_LINE_BYTES {
        return Some(Framing::Invalid);
    }
    if !line.ends_with(b"\r\n") {
        return None;
    }
    Some(match parse_chunk_size(line) {
        Some(0) => Framing::Trailers {
            line: Vec::new(),
            total: 0,
        },
        Some(size) => Framing::ChunkData(size),
        None => Framing::Invalid,
    })
}

fn advance_chunk_end(position: &mut u8, byte: u8) -> Option<Framing> {
    if byte != b"\r\n"[*position as usize] {
        return Some(Framing::Invalid);
    }
    *position += 1;
    (*position == 2).then(|| Framing::ChunkSize(Vec::new()))
}

fn advance_trailers(line: &mut Vec<u8>, total: &mut usize, byte: u8) -> Option<Framing> {
    line.push(byte);
    *total += 1;
    if *total > MAX_HEADER_BYTES {
        return Some(Framing::Invalid);
    }
    if !line.ends_with(b"\r\n") {
        return None;
    }
    if line.len() == 2 {
        return Some(Framing::Done);
    }
    line.clear();
    None
}

fn parse_chunk_size(line: &[u8]) -> Option<u64> {
    let line = std::str::from_utf8(line.strip_suffix(b"\r\n")?).ok()?;
    let size = line.split(';').next()?;
    if size.is_empty() || !size.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(size, 16).ok()
}

fn response_framing(header: &[u8]) -> Option<Framing> {
    let header = std::str::from_utf8(header).ok()?;
    let mut lines = header.split("\r\n");
    let status = lines
        .next()?
        .split_whitespace()
        .nth(1)?
        .parse::<u16>()
        .ok()?;
    let mut length = None;
    let mut transfer_encoding = None;
    for line in lines.filter(|line| !line.is_empty()) {
        let (name, value) = line.split_once(':')?;
        if name.eq_ignore_ascii_case("content-length") {
            let parsed = value.trim().parse::<u64>().ok()?;
            if length.replace(parsed).is_some() {
                return None;
            }
        } else if name.eq_ignore_ascii_case("transfer-encoding")
            && transfer_encoding.replace(value.trim()).is_some()
        {
            return None;
        }
    }
    if (100..200).contains(&status) {
        return Some(Framing::Headers(Vec::new()));
    }
    if matches!(status, 204 | 304) {
        return Some(Framing::Done);
    }
    if let Some(encoding) = transfer_encoding {
        if length.is_some() || !encoding.eq_ignore_ascii_case("chunked") {
            return None;
        }
        return Some(Framing::ChunkSize(Vec::new()));
    }
    Some(match length {
        Some(0) => Framing::Done,
        Some(length) => Framing::Fixed(length),
        None => Framing::UntilClose,
    })
}

fn response_status(header: &[u8]) -> Option<u16> {
    std::str::from_utf8(header)
        .ok()?
        .split("\r\n")
        .next()?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use openai_frontend::wire_bytes::{WireBytesCommitment, commit_wire_bytes};
    use std::sync::Mutex;

    #[derive(Default)]
    struct Recorder(Mutex<Vec<WireBytesCommitment>>, Mutex<Vec<String>>);
    impl WireBytesObserver for Recorder {
        fn execution_outcome(&self, outcome: &str) {
            self.1.lock().unwrap().push(outcome.into());
        }
        fn try_chunk(&self, _offset: u64, _bytes: &[u8]) -> bool {
            true
        }
        fn finish(&self, commitment: WireBytesCommitment) {
            self.0.lock().unwrap().push(commitment);
        }
    }

    #[test]
    fn encoded_sse_error_has_complete_bytes_and_separate_backend_failure() {
        let observer = Arc::new(Recorder::default());
        let mut tap = HttpResponseByteTap::new(observer.clone());
        let entity = b"data: {\"error\":{\"message\":\"failed\"}}\n\ndata: [DONE]\n\n";
        tap.update(
            format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                entity.len()
            )
            .as_bytes(),
        );
        tap.update(entity);
        tap.execution_outcome("backend_error");
        tap.finish(None);
        assert_eq!(observer.0.lock().unwrap()[0], commit_wire_bytes(entity));
        assert_eq!(*observer.1.lock().unwrap(), ["backend_error"]);
    }

    #[test]
    fn chunked_sse_hash_excludes_transfer_framing_for_every_fragment_size() {
        let wire = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n8;v=1\r\ndata: hi\r\n2\r\n\n\n\r\n0\r\nX-Trailer: value\r\n\r\n";
        for size in 1..=wire.len() {
            let observer = Arc::new(Recorder::default());
            let mut tap = HttpResponseByteTap::new(observer.clone());
            for bytes in wire.chunks(size) {
                tap.update(bytes);
            }
            tap.finish(None);
            assert_eq!(
                observer.0.lock().unwrap()[0],
                commit_wire_bytes(b"data: hi\n\n")
            );
        }
    }

    #[test]
    fn fixed_length_excludes_overread_and_sensitive_headers() {
        let observer = Arc::new(Recorder::default());
        let mut tap = HttpResponseByteTap::new(observer.clone());
        tap.update(b"HTTP/1.1 200 OK\r\nAuthorization: secret\r\nContent-Length: 3\r\n\r\nabcNEXT");
        tap.finish(None);
        assert_eq!(observer.0.lock().unwrap()[0], commit_wire_bytes(b"abc"));
    }

    #[test]
    fn truncated_entity_and_conflicting_framing_never_claim_complete() {
        for wire in [
            &b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nabc"[..],
            &b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\nTransfer-Encoding: chunked\r\n\r\nabc"[..],
        ] {
            let observer = Arc::new(Recorder::default());
            let mut tap = HttpResponseByteTap::new(observer.clone());
            tap.update(wire);
            tap.finish(None);
            drop(tap);
            let recorded = observer.0.lock().unwrap();
            assert_eq!(recorded.len(), 1);
            assert!(recorded[0].incomplete.is_some());
        }
    }
}
