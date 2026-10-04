//! Insert grant-checked plugin metadata before the final HTTP entity is emitted.
//!
//! Only the bounded header prefix is buffered. All entity bytes pass directly
//! through to the wrapped writer, preserving their order and chunk boundaries.

use super::ClientStream;
use std::{
    io,
    pin::Pin,
    task::{Context, Poll},
};
use tokio::io::AsyncWrite;

const MAX_HEADER_BYTES: usize = 64 * 1024;

pub(crate) struct ResponseMetadata {
    headers: Vec<(String, String)>,
    prefix: Vec<u8>,
    emitted: usize,
    header_complete: bool,
    invalid: bool,
}

impl ResponseMetadata {
    pub(super) fn new(headers: Vec<(String, String)>) -> Self {
        let invalid = headers.len() > 16
            || headers
                .iter()
                .any(|(name, value)| !safe_metadata_header(name, value));
        Self {
            headers,
            prefix: Vec::new(),
            emitted: 0,
            header_complete: false,
            invalid,
        }
    }
    pub(super) fn add(&mut self, headers: Vec<(String, String)>) -> io::Result<()> {
        if self.header_complete {
            return Err(io::Error::other(
                "response headers have already been committed",
            ));
        }
        if headers
            .iter()
            .any(|(name, value)| !safe_metadata_header(name, value))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "response metadata exceeds limits",
            ));
        }
        let mut merged: std::collections::BTreeMap<_, _> = self
            .headers
            .drain(..)
            .chain(headers)
            .map(|(name, value)| (name.to_ascii_lowercase(), value))
            .collect();
        while merged.len() > 16 {
            merged.pop_last();
        }
        self.headers.extend(merged);
        Ok(())
    }

    pub(super) fn poll_write(
        &mut self,
        stream: &mut ClientStream,
        cx: &mut Context<'_>,
        bytes: &[u8],
    ) -> Poll<io::Result<usize>> {
        if self.invalid {
            return self.invalid_framing(stream);
        }
        if self.header_complete {
            std::task::ready!(self.poll_emit_prefix(stream, cx))?;
            return Pin::new(stream).poll_write(cx, bytes);
        }
        let mut consumed = 0;
        for byte in bytes {
            self.prefix.push(*byte);
            consumed += 1;
            if self.prefix.len() > MAX_HEADER_BYTES {
                return self.invalid_framing(stream);
            }
            if self.prefix.ends_with(b"\r\n\r\n") {
                self.prefix.truncate(self.prefix.len() - 2);
                for (name, value) in &self.headers {
                    if !safe_metadata_header(name, value) {
                        return self.invalid_framing(stream);
                    }
                    self.prefix.extend_from_slice(name.as_bytes());
                    self.prefix.extend_from_slice(b": ");
                    self.prefix.extend_from_slice(value.as_bytes());
                    self.prefix.extend_from_slice(b"\r\n");
                }
                self.prefix.extend_from_slice(b"\r\n");
                if self.prefix.len() > MAX_HEADER_BYTES {
                    return self.invalid_framing(stream);
                }
                self.header_complete = true;
                break;
            }
        }
        Poll::Ready(Ok(consumed))
    }

    pub(super) fn poll_emit_prefix(
        &mut self,
        stream: &mut ClientStream,
        cx: &mut Context<'_>,
    ) -> Poll<io::Result<()>> {
        if self.invalid {
            return self.invalid_framing(stream);
        }
        if !self.header_complete {
            if self.prefix.is_empty() {
                return Poll::Ready(Ok(()));
            }
            return self.invalid_framing(stream);
        }
        while self.emitted < self.prefix.len() {
            let count = std::task::ready!(
                Pin::new(&mut *stream).poll_write(cx, &self.prefix[self.emitted..])
            )?;
            if count == 0 {
                return Poll::Ready(Err(io::ErrorKind::WriteZero.into()));
            }
            self.emitted += count;
        }
        self.prefix.clear();
        self.emitted = 0;
        Poll::Ready(Ok(()))
    }

    pub(super) fn poll_shutdown(
        &mut self,
        stream: &mut ClientStream,
        cx: &mut Context<'_>,
    ) -> Poll<io::Result<()>> {
        if !self.header_complete && !self.prefix.is_empty() {
            return self.invalid_framing(stream);
        }
        std::task::ready!(self.poll_emit_prefix(stream, cx))?;
        Pin::new(stream).poll_shutdown(cx)
    }

    fn invalid_framing<T>(&mut self, stream: &mut ClientStream) -> Poll<io::Result<T>> {
        self.invalid = true;
        stream.finish_wire_bytes(openai_frontend::wire_bytes::WireBytesIncomplete::InvalidFraming);
        Poll::Ready(Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid or oversized response metadata header",
        )))
    }
}

fn safe_metadata_header(name: &str, value: &str) -> bool {
    name.len() <= 128
        && value.len() <= 1024
        && name.to_ascii_lowercase().starts_with("x-plugin-")
        && mesh_llm_config::safe_exchange_header(name)
        && value
            .bytes()
            .all(|byte| byte == b'\t' || (32..127).contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;
    use openai_frontend::wire_bytes::{
        WireBytesCommitment, WireBytesIncomplete, WireBytesObserver, commit_wire_bytes,
    };
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[derive(Default)]
    struct Recorder(Mutex<Vec<WireBytesCommitment>>);
    impl WireBytesObserver for Recorder {
        fn try_chunk(&self, _offset: u64, _bytes: &[u8]) -> bool {
            true
        }
        fn finish(&self, commitment: WireBytesCommitment) {
            self.0.lock().unwrap().push(commitment);
        }
    }

    #[tokio::test]
    async fn fragmented_header_insertion_preserves_chunked_entity_and_wire_commitment() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let mut client = tokio::net::TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        let (server, _) = listener.accept().await.unwrap();
        let recorder = Arc::new(Recorder::default());
        let mut stream = ClientStream::from(server)
            .with_wire_bytes_observer(recorder.clone())
            .with_response_metadata(vec![("x-plugin-observer-first".into(), "one".into())]);
        let header = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n";
        for byte in &header[..10] {
            assert_eq!(stream.write(&[*byte]).await.unwrap(), 1);
        }
        stream
            .add_response_metadata(vec![("x-plugin-observer-second".into(), "two".into())])
            .unwrap();
        stream.write_all(&header[10..]).await.unwrap();
        stream.flush().await.unwrap();
        let entity = b"d\r\ndata: first\n\n\r\ne\r\ndata: [DONE]\n\n\r\n0\r\n\r\n";
        for bytes in entity.chunks(3) {
            stream.write_all(bytes).await.unwrap();
        }
        stream.shutdown().await.unwrap();
        let mut delivered = Vec::new();
        client.read_to_end(&mut delivered).await.unwrap();
        let boundary = delivered.windows(4).position(|v| v == b"\r\n\r\n").unwrap() + 4;
        let headers = std::str::from_utf8(&delivered[..boundary]).unwrap();
        assert_eq!(
            headers.matches("x-plugin-observer-first: one\r\n").count(),
            1
        );
        assert_eq!(
            headers.matches("x-plugin-observer-second: two\r\n").count(),
            1
        );
        assert_eq!(&delivered[boundary..], entity);
        assert_eq!(
            *recorder.0.lock().unwrap(),
            vec![commit_wire_bytes(b"data: first\n\ndata: [DONE]\n\n")]
        );
        assert!(
            stream
                .add_response_metadata(vec![("x-plugin-observer-late".into(), "late".into())])
                .is_err()
        );
    }

    #[tokio::test]
    async fn shutdown_flushes_bodyless_header_and_rejects_truncated_header() {
        for truncated in [false, true] {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let client = tokio::net::TcpStream::connect(listener.local_addr().unwrap())
                .await
                .unwrap();
            let (server, _) = listener.accept().await.unwrap();
            let recorder = Arc::new(Recorder::default());
            let mut stream = ClientStream::from(server)
                .with_wire_bytes_observer(recorder.clone())
                .with_response_metadata(vec![("x-plugin-observer-note".into(), "safe".into())]);
            let bytes = if truncated {
                &b"HTTP/1.1 200 OK\r\n"[..]
            } else {
                &b"HTTP/1.1 204 No Content\r\n\r\n"[..]
            };
            stream.write_all(bytes).await.unwrap();
            assert_eq!(stream.shutdown().await.is_err(), truncated);
            let records = recorder.0.lock().unwrap();
            assert_eq!(records.len(), 1);
            assert_eq!(
                records[0].incomplete,
                truncated.then_some(WireBytesIncomplete::InvalidFraming)
            );
            drop(client);
        }
    }
}
