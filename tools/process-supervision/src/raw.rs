use super::{Failure, Stream};

#[derive(Default)]
pub struct RawCaptureOptions {
    pub stdout: Option<std::num::NonZeroUsize>,
    pub stderr: Option<std::num::NonZeroUsize>,
}

#[derive(Debug)]
pub struct RawProcessReport {
    pub process: super::ProcessReport,
    pub stdout: Option<RawBytes>,
    pub stderr: Option<RawBytes>,
}

pub struct RawBytes(Vec<u8>);

impl RawBytes {
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl std::fmt::Debug for RawBytes {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RawBytes")
            .field("length", &self.0.len())
            .finish()
    }
}

pub(super) struct RawCapture {
    bytes: Vec<u8>,
    limit: usize,
    overflowed: bool,
}

impl RawCapture {
    pub(super) fn new(limit: std::num::NonZeroUsize) -> Self {
        Self {
            bytes: Vec::new(),
            limit: limit.get(),
            overflowed: false,
        }
    }

    pub(super) fn consume(&mut self, bytes: &[u8], stream: Stream) -> Result<(), Failure> {
        if self.overflowed {
            return Ok(());
        }
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            self.bytes = Vec::new();
            self.overflowed = true;
            return Err(Failure::RawCaptureOverflow {
                stream,
                limit: self.limit,
            });
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    pub(super) fn finish(self, stream: Stream, eof: bool) -> Result<RawBytes, Failure> {
        if self.overflowed {
            Err(Failure::RawCaptureOverflow {
                stream,
                limit: self.limit,
            })
        } else if !eof {
            Err(Failure::RawCaptureIncomplete { stream })
        } else {
            Ok(RawBytes(self.bytes))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_omits_raw_payload() {
        let payload = RawBytes(b"credential-private-payload".to_vec());
        let rendered = format!("{payload:?}");
        assert_eq!(rendered, "RawBytes { length: 26 }");
        assert!(!rendered.contains("credential"));
    }

    #[test]
    fn exact_bytes_when_payload_contains_diagnostic_triggers() {
        let mut bytes = b"token\0\xff\r\npassword secret authorization invite\n".to_vec();
        bytes.extend(vec![b'x'; 9000]);
        let mut capture = RawCapture::new(std::num::NonZeroUsize::new(bytes.len()).unwrap());
        for chunk in bytes.chunks(17) {
            capture.consume(chunk, Stream::Stdout).unwrap();
        }
        assert_eq!(
            capture.finish(Stream::Stdout, true).unwrap().as_bytes(),
            bytes
        );
    }

    #[test]
    fn typed_overflow_without_partial_payload_when_bound_is_exceeded() {
        let mut capture = RawCapture::new(std::num::NonZeroUsize::new(3).unwrap());
        capture.consume(b"abc", Stream::Stderr).unwrap();
        assert!(matches!(
            capture.consume(b"d", Stream::Stderr),
            Err(Failure::RawCaptureOverflow {
                stream: Stream::Stderr,
                limit: 3
            })
        ));
        capture.consume(b"drain", Stream::Stderr).unwrap();
        assert!(matches!(
            capture.finish(Stream::Stderr, true),
            Err(Failure::RawCaptureOverflow { .. })
        ));
    }

    #[test]
    fn incomplete_capture_when_writer_has_not_reached_eof() {
        let capture = RawCapture::new(std::num::NonZeroUsize::new(1).unwrap());
        assert!(matches!(
            capture.finish(Stream::Stdout, false),
            Err(Failure::RawCaptureIncomplete {
                stream: Stream::Stdout
            })
        ));
    }

    #[test]
    fn diagnostic_ceiling_does_not_limit_raw_payload() {
        let bytes = vec![b'x'; 16 * 1024 * 1024 + 1];
        let mut capture = RawCapture::new(std::num::NonZeroUsize::new(bytes.len()).unwrap());
        capture.consume(&bytes, Stream::Stdout).unwrap();
        assert_eq!(
            capture.finish(Stream::Stdout, true).unwrap().as_bytes(),
            bytes
        );
    }
}
