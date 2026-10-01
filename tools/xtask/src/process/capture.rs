use super::{Failure, LineEnding, ObservedLine, Readiness, Stream, StreamReport, platform};
use std::fs::File;
use std::io::{self, Read, Write};

pub(super) const LINE_LIMIT: usize = 8192;
const CHUNK: usize = 4096;
const READS_PER_TICK: usize = 8;

#[cfg(all(test, unix))]
#[path = "capture_tests.rs"]
mod tests;

pub(super) struct Capture<R> {
    pipe: R,
    stream: Stream,
    file: Option<File>,
    report: StreamReport,
    pending: Vec<u8>,
    oversized: bool,
    eof: bool,
    limit: usize,
    secrets: Vec<Vec<u8>>,
    raw: Option<super::raw::RawCapture>,
}

impl<R: Read + platform::Pipe> Capture<R> {
    pub(super) fn new(
        pipe: R,
        stream: Stream,
        file: Option<File>,
        limit: usize,
        secrets: Vec<Vec<u8>>,
    ) -> Result<Self, Failure> {
        platform::prepare_pipe(&pipe).map_err(|error| Failure::io("prepare pipe", error))?;
        Ok(Self {
            pipe,
            stream,
            file,
            report: StreamReport::default(),
            pending: Vec::with_capacity(LINE_LIMIT),
            oversized: false,
            eof: false,
            limit,
            secrets,
            raw: None,
        })
    }

    pub(super) fn poll(&mut self, readiness: &Readiness) -> Result<bool, Failure> {
        self.poll_bytes(readiness, CHUNK * READS_PER_TICK)
    }

    pub(super) fn poll_probe(
        &mut self,
        callback: &mut dyn FnMut(ObservedLine<'_>),
    ) -> Result<(), Failure> {
        self.poll_with(
            &mut |line| {
                callback(line);
                false
            },
            CHUNK * READS_PER_TICK,
        )?;
        Ok(())
    }

    pub(super) fn pending_bytes(&self) -> Result<usize, Failure> {
        platform::pending_bytes(&self.pipe).map_err(|error| Failure::io("snapshot output", error))
    }

    pub(super) fn poll_snapshot(
        &mut self,
        readiness: &Readiness,
        bytes: usize,
    ) -> Result<bool, Failure> {
        let mut ready = self.poll_bytes(readiness, bytes)?;
        if !self.eof {
            let mut probe = [0];
            match platform::read_pipe(&mut self.pipe, &mut probe) {
                Ok(0) => {
                    self.eof = true;
                    ready |= self.flush_line(readiness)?;
                }
                Ok(length) => {
                    self.consume(&probe[..length], &Readiness::None)?;
                }
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock | io::ErrorKind::Interrupted
                    ) => {}
                Err(error) => return Err(Failure::io("observe output EOF", error)),
            }
        }
        Ok(ready)
    }

    pub(super) fn poll_bytes(
        &mut self,
        readiness: &Readiness,
        remaining: usize,
    ) -> Result<bool, Failure> {
        self.poll_with(&mut |line| matches_readiness(readiness, line), remaining)
    }

    fn poll_with(
        &mut self,
        callback: &mut impl FnMut(ObservedLine<'_>) -> bool,
        mut remaining: usize,
    ) -> Result<bool, Failure> {
        let mut ready = false;
        let mut buffer = [0; CHUNK];
        while remaining > 0 {
            if self.eof {
                break;
            }
            let length = remaining.min(CHUNK);
            match platform::read_pipe(&mut self.pipe, &mut buffer[..length]) {
                Ok(0) => {
                    self.eof = true;
                    ready |= self.flush_with(callback)?;
                }
                Ok(length) => {
                    remaining -= length;
                    ready |= self.consume_with(&buffer[..length], callback)?;
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => break,
                Err(error) => return Err(Failure::io("read output", error)),
            }
        }
        Ok(ready)
    }

    fn consume(&mut self, bytes: &[u8], readiness: &Readiness) -> Result<bool, Failure> {
        self.consume_with(bytes, &mut |line| matches_readiness(readiness, line))
    }

    fn consume_with(
        &mut self,
        bytes: &[u8],
        callback: &mut impl FnMut(ObservedLine<'_>) -> bool,
    ) -> Result<bool, Failure> {
        let raw_result = match &mut self.raw {
            Some(raw) => raw.consume(bytes, self.stream),
            None => Ok(()),
        };
        self.report.bytes_seen = self
            .report
            .bytes_seen
            .saturating_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
        let mut ready = false;
        for byte in bytes {
            if *byte == b'\n' {
                if !self.oversized {
                    self.pending.push(*byte);
                }
                ready |= self.flush_with(callback)?;
            } else if self.pending.len() < LINE_LIMIT {
                self.pending.push(*byte);
            } else {
                self.oversized = true;
            }
        }
        raw_result?;
        Ok(ready)
    }

    fn flush_line(&mut self, readiness: &Readiness) -> Result<bool, Failure> {
        self.flush_with(&mut |line| matches_readiness(readiness, line))
    }

    fn flush_with(
        &mut self,
        callback: &mut impl FnMut(ObservedLine<'_>) -> bool,
    ) -> Result<bool, Failure> {
        let line = self.pending.strip_suffix(b"\n").unwrap_or(&self.pending);
        let ready = !self.oversized
            && !self.pending.is_empty()
            && callback(ObservedLine {
                stream: self.stream,
                bytes: line,
                ending: if self.pending.ends_with(b"\n") {
                    LineEnding::Lf
                } else {
                    LineEnding::Eof
                },
            });
        let sensitive = sensitive_line(&self.pending);
        if self.oversized || sensitive {
            self.report.suppressed_lines = self.report.suppressed_lines.saturating_add(1);
            self.pending.clear();
            self.pending
                .extend_from_slice(b"[output line suppressed]\n");
        } else {
            redact(&mut self.pending, &self.secrets);
        }
        let remaining = self.limit.saturating_sub(self.report.bytes_retained.len());
        let retained = self.pending.len().min(remaining);
        self.report.truncated |= retained < self.pending.len() || self.oversized;
        self.report
            .bytes_retained
            .extend_from_slice(&self.pending[..retained]);
        if let Some(file) = &mut self.file {
            file.write_all(&self.pending[..retained])
                .map_err(|error| Failure::io("write output", error))?;
        }
        self.pending.clear();
        self.oversized = false;
        Ok(ready)
    }

    pub(super) fn finish(mut self) -> (StreamReport, Option<Failure>) {
        let failure = self.flush_line(&Readiness::None).err();
        (self.report, failure)
    }

    pub(super) fn eof(&self) -> bool {
        self.eof
    }

    pub(super) fn enable_raw(&mut self, limit: Option<std::num::NonZeroUsize>) {
        self.raw = limit.map(super::raw::RawCapture::new);
    }

    pub(super) fn take_raw(&mut self) -> Result<Option<super::RawBytes>, Failure> {
        self.raw
            .take()
            .map(|raw| raw.finish(self.stream, self.eof))
            .transpose()
    }
}

#[cfg(all(test, unix))]
#[path = "line_tests.rs"]
mod line_tests;

#[cfg(all(test, unix))]
#[path = "probe_capture_tests.rs"]
mod probe_tests;

fn matches_readiness(readiness: &Readiness, line: ObservedLine<'_>) -> bool {
    match readiness {
        Readiness::None => false,
        Readiness::Line { stream, bytes, .. } => {
            *stream == line.stream && line.bytes.strip_suffix(b"\r").unwrap_or(line.bytes) == bytes
        }
        Readiness::ObservedLines { matcher, .. } => matcher(line),
    }
}

fn sensitive_line(bytes: &[u8]) -> bool {
    [
        b"token".as_slice(),
        b"password",
        b"secret",
        b"authorization",
        b"invite",
    ]
    .iter()
    .any(|needle| {
        bytes
            .windows(needle.len())
            .any(|window| window.eq_ignore_ascii_case(needle))
    })
}

fn redact(bytes: &mut [u8], secrets: &[Vec<u8>]) {
    let mut masked = vec![false; bytes.len()];
    for secret in secrets {
        for (offset, window) in bytes.windows(secret.len()).enumerate() {
            if window == secret {
                masked[offset..offset + secret.len()].fill(true);
            }
        }
    }
    for (byte, mask) in bytes.iter_mut().zip(masked) {
        if mask {
            *byte = b'*';
        }
    }
}
