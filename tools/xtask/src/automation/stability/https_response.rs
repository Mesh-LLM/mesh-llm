use super::super::{BODY_LIMIT, Decoder, http::decode_reply, millis};
use super::{Failure, Reply, Request, failure};
use crate::process::{Cancellation, ObservedLine, ProbeContext, ProbeDecision, ReadinessProbe};
use std::{
    fs::File,
    io::Read,
    sync::mpsc::{self, Receiver, SyncSender, TryRecvError},
    thread::{self, JoinHandle},
    time::Duration,
};

pub(super) enum Observation {
    Done,
    Rejected(Failure),
}
pub(super) struct Reader {
    pub observations: Receiver<Observation>,
    finished: Cancellation,
    handle: Option<JoinHandle<Result<Reply, Failure>>>,
}

impl Reader {
    pub(super) fn start(headers: File, body: File, request: &Request) -> Result<Self, Failure> {
        let finished = Cancellation::default();
        let stopped = finished.clone();
        let (sent, observations) = mpsc::sync_channel(1);
        let stream = request.stream;
        let started = request.started;
        let handle = thread::Builder::new()
            .name("stability-https-reader".into())
            .spawn(move || {
                let result = read(headers, body, stream, started, &stopped, &sent);
                if let Err(error) = &result {
                    let _ = sent.try_send(Observation::Rejected(error.clone()));
                }
                result
            })
            .map_err(|_| failure("HTTPS response reader unavailable", None))?;
        Ok(Self {
            observations,
            finished,
            handle: Some(handle),
        })
    }
    pub(super) fn finish(&mut self) -> Result<Reply, Failure> {
        self.finished.cancel();
        self.handle
            .take()
            .ok_or_else(|| failure("HTTPS reader ownership lost", None))?
            .join()
            .map_err(|_| failure("HTTPS response reader failed", None))?
    }
}
impl Drop for Reader {
    fn drop(&mut self) {
        self.finished.cancel();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub(super) struct Observer<'a> {
    pub observations: &'a Receiver<Observation>,
    pub cancellation: &'a Cancellation,
}
impl ReadinessProbe for Observer<'_> {
    type Rejection = Failure;
    fn line(&mut self, _: ObservedLine<'_>) -> ProbeDecision<Failure> {
        ProbeDecision::Pending
    }
    fn tick(&mut self, _: ProbeContext) -> ProbeDecision<Failure> {
        if self.cancellation.is_cancelled() {
            return ProbeDecision::Rejected(failure("stability operation cancelled", None));
        }
        match self.observations.try_recv() {
            Ok(Observation::Done) => ProbeDecision::Candidate,
            Ok(Observation::Rejected(error)) => ProbeDecision::Rejected(error),
            Err(TryRecvError::Empty | TryRecvError::Disconnected) => ProbeDecision::Pending,
        }
    }
}

fn read(
    mut headers: File,
    mut body_file: File,
    stream: bool,
    started: std::time::Instant,
    finished: &Cancellation,
    sent: &SyncSender<Observation>,
) -> Result<Reply, Failure> {
    let mut head = Headers::default();
    let mut decoder = Decoder::default();
    let mut body = Vec::new();
    let mut size = 0usize;
    let mut buffer = [0u8; 8192];
    loop {
        // Observe completion before draining: the final scan includes bytes
        // written immediately before the transfer leader exited.
        let final_scan = finished.is_cancelled();
        while head.status.is_none() {
            let count = headers
                .read(&mut buffer)
                .map_err(|_| failure("HTTPS headers unreadable", None))?;
            if count == 0 {
                break;
            }
            head.push(&buffer[..count])?;
        }
        if let Some(status) = head.status {
            validate_status(status)?;
            loop {
                let count = body_file
                    .read(&mut buffer)
                    .map_err(|_| failure("HTTPS body unreadable", Some(status)))?;
                if count == 0 {
                    break;
                }
                if count > BODY_LIMIT.saturating_sub(size) {
                    return Err(failure("stability response exceeds 16 MiB", Some(status)));
                }
                size += count;
                if stream {
                    decoder
                        .push(&buffer[..count], millis(started))
                        .map_err(|detail| failure(&detail, Some(status)))?;
                    if decoder.done {
                        let reply = decode_reply(status, body, decoder, stream, started)?;
                        let _ = sent.try_send(Observation::Done);
                        return Ok(reply);
                    }
                } else {
                    body.extend_from_slice(&buffer[..count]);
                }
            }
        }
        if final_scan {
            let status = head
                .status
                .ok_or_else(|| failure("HTTPS response headers incomplete", None))?;
            return decode_reply(status, body, decoder, stream, started);
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn validate_status(status: u16) -> Result<(), Failure> {
    if (300..400).contains(&status) {
        Err(failure("stability endpoint redirected", Some(status)))
    } else if !(200..300).contains(&status) {
        Err(failure(&format!("HTTP {status}"), Some(status)))
    } else {
        Ok(())
    }
}

#[derive(Default)]
struct Headers {
    bytes: Vec<u8>,
    offset: usize,
    status: Option<u16>,
}
impl Headers {
    fn push(&mut self, bytes: &[u8]) -> Result<(), Failure> {
        if bytes.len() > 65536usize.saturating_sub(self.bytes.len()) {
            return Err(failure("HTTPS headers exceed 64 KiB", None));
        }
        self.bytes.extend_from_slice(bytes);
        while let Some(end) = self.bytes[self.offset..]
            .windows(4)
            .position(|part| part == b"\r\n\r\n")
        {
            let block = &self.bytes[self.offset..self.offset + end];
            let line = block
                .split(|byte| *byte == b'\n')
                .next()
                .unwrap_or_default();
            let text = std::str::from_utf8(line)
                .map_err(|_| failure("invalid HTTPS status line", None))?;
            let mut tokens = text.split_whitespace();
            if !matches!(
                tokens.next(),
                Some("HTTP/1.0" | "HTTP/1.1" | "HTTP/2" | "HTTP/3")
            ) {
                return Err(failure("invalid HTTPS status line", None));
            }
            let status = tokens
                .next()
                .filter(|value| value.len() == 3)
                .and_then(|value| value.parse::<u16>().ok())
                .filter(|value| (100..600).contains(value))
                .ok_or_else(|| failure("invalid HTTPS status code", None))?;
            self.offset += end + 4;
            if status >= 200 {
                self.status = Some(status);
                break;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn stability_https_headers_accept_fragmented_interim_status_and_reject_invalid_or_oversized_headers()
     {
        let mut headers = Headers::default();
        for byte in b"HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n" { headers.push(&[*byte]).unwrap(); }
        assert_eq!(headers.status, Some(200));
        for bytes in [
            b"HTTP/1.1 xyz Wrong\r\n\r\n".as_slice(),
            b"bad 200 OK\r\n\r\n",
            b"HTTP/1.1 999 Bad\r\n\r\n",
        ] {
            assert!(Headers::default().push(bytes).is_err());
        }
        assert!(Headers::default().push(&vec![b'x'; 65537]).is_err());
    }
}
