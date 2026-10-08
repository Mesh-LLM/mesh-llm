//! Commitments over HTTP entity bytes, including the complete encoded SSE frames.
//!
//! HTTP headers and transfer framing are excluded. JSON whitespace, field order,
//! SSE delimiters, and terminal events are preserved. These commitments are
//! distinct from canonical JSON or reconstructed assistant-content digests.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
};

use axum::body::{Body, Bytes};
use http_body::{Body as HttpBody, Frame, SizeHint};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Why an exact byte observation did not reach its natural end.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WireBytesIncomplete {
    Cancelled,
    TransportError,
    Timeout,
    InvalidFraming,
    ObserverUnavailable,
}

/// A SHA-256 commitment to the observed byte prefix, never a semantic digest.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct WireBytesCommitment {
    pub sha256: String,
    pub byte_count: u64,
    /// `None` means the observed entity reached its natural end.
    pub incomplete: Option<WireBytesIncomplete>,
    /// False when the optional ordered byte side stream overflowed or disconnected.
    /// The hash can still be complete even when the observer missed chunks.
    pub side_stream_complete: bool,
}

/// An ordered, nonblocking byte side stream owned by an explicitly granted observer.
pub trait WireBytesObserver: Send + Sync + 'static {
    /// Grant-checked metadata accumulated before backend dispatch.
    fn response_headers(&self) -> Vec<(String, String)> {
        Vec::new()
    }
    /// Response status at the final emission point, before any entity bytes.
    fn response_status(&self, _status_code: u16) {}
    /// A host-observed execution outcome, independent of HTTP status or hashing.
    fn execution_outcome(&self, _outcome: &str) {}
    /// Enqueue bytes without waiting. Return false on overflow or disconnect.
    /// Offsets count entity bytes, independent of HTTP/SSE chunk boundaries.
    fn try_chunk(&self, offset: u64, bytes: &[u8]) -> bool;
    /// One terminal attempt, including cancellation when the tap is dropped.
    fn finish(&self, commitment: WireBytesCommitment);
}

/// Constant-space incremental byte hashing with an optional bounded side stream.
pub struct WireBytesTap {
    hash: Sha256,
    byte_count: u64,
    observer: Option<Arc<dyn WireBytesObserver>>,
    side_stream_complete: bool,
    terminal: Option<WireBytesCommitment>,
}

impl WireBytesTap {
    pub fn new(observer: Option<Arc<dyn WireBytesObserver>>) -> Self {
        Self {
            hash: Sha256::new(),
            byte_count: 0,
            observer,
            side_stream_complete: true,
            terminal: None,
        }
    }

    pub fn update(&mut self, bytes: &[u8]) {
        if self.terminal.is_some() {
            return;
        }
        self.hash.update(bytes);
        if let Some(observer) = &self.observer {
            // Each independently granted recipient must continue even if another
            // recipient overflowed. Keep the aggregate evidence flag sticky.
            self.side_stream_complete &= observer.try_chunk(self.byte_count, bytes);
        }
        self.byte_count = self.byte_count.saturating_add(bytes.len() as u64);
    }

    pub fn finish(&mut self, incomplete: Option<WireBytesIncomplete>) -> WireBytesCommitment {
        if let Some(terminal) = &self.terminal {
            return terminal.clone();
        }
        let commitment = self.snapshot(incomplete);
        self.terminal = Some(commitment.clone());
        if let Some(observer) = &self.observer {
            observer.finish(commitment.clone());
        }
        commitment
    }

    fn snapshot(&self, incomplete: Option<WireBytesIncomplete>) -> WireBytesCommitment {
        WireBytesCommitment {
            sha256: format!("{:x}", self.hash.clone().finalize()),
            byte_count: self.byte_count,
            incomplete,
            side_stream_complete: self.side_stream_complete,
        }
    }
}

impl Drop for WireBytesTap {
    fn drop(&mut self) {
        if self.terminal.is_none() {
            self.finish(Some(WireBytesIncomplete::Cancelled));
        }
    }
}

/// Hash one complete request body exactly as received or dispatched.
pub fn commit_wire_bytes(bytes: &[u8]) -> WireBytesCommitment {
    let mut tap = WireBytesTap::new(None);
    tap.update(bytes);
    tap.finish(None)
}

/// Tap the encoded response body immediately before handing bytes to HTTP.
///
/// This does not buffer, reserialize, reorder, or modify any bytes. Completion
/// means the HTTP transport polled the full entity, not peer acknowledgement.
pub fn observe_response_body(body: Body, observer: Arc<dyn WireBytesObserver>) -> Body {
    let mut tap = WireBytesTap::new(Some(observer));
    if body.is_end_stream() {
        tap.finish(None);
    }
    Body::new(ObservedBody { inner: body, tap })
}

struct ObservedBody {
    inner: Body,
    tap: WireBytesTap,
}

impl HttpBody for ObservedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        let this = self.get_mut();
        let item = Pin::new(&mut this.inner).poll_frame(cx);
        match &item {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(bytes) = frame.data_ref() {
                    this.tap.update(bytes);
                }
                if this.inner.is_end_stream() {
                    this.tap.finish(None);
                }
            }
            Poll::Ready(Some(Err(_))) => {
                this.tap.finish(Some(WireBytesIncomplete::TransportError));
            }
            Poll::Ready(None) => {
                this.tap.finish(None);
            }
            Poll::Pending => {}
        }
        item
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    const SSE: &[u8] =
        b"data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n";
    const SSE_SHA: &str = "2774745745fc204699d02f6f2d032c205fd17caf18070316bcfbbb16956f66fd";

    #[derive(Default)]
    struct Recorder {
        chunks: Mutex<Vec<(u64, Vec<u8>)>>,
        terminals: Mutex<Vec<WireBytesCommitment>>,
        overflow: bool,
    }
    impl WireBytesObserver for Recorder {
        fn try_chunk(&self, offset: u64, bytes: &[u8]) -> bool {
            if self.overflow {
                return false;
            }
            self.chunks.lock().unwrap().push((offset, bytes.to_vec()));
            true
        }
        fn finish(&self, commitment: WireBytesCommitment) {
            self.terminals.lock().unwrap().push(commitment);
        }
    }

    #[test]
    fn full_sse_transcript_matches_independent_sha256_for_every_chunk_size() {
        for size in 1..=SSE.len() {
            let mut tap = WireBytesTap::new(None);
            for bytes in SSE.chunks(size) {
                tap.update(bytes);
            }
            let commitment = tap.finish(None);
            assert_eq!(commitment.sha256, SSE_SHA);
            assert_eq!(commitment.byte_count, SSE.len() as u64);
            assert_eq!(commitment.incomplete, None);
        }
        assert_ne!(commit_wire_bytes(b"hi").sha256, SSE_SHA);
    }

    #[test]
    fn request_whitespace_is_committed_without_canonicalization() {
        let body = b"{ \"model\": \"tiny\", \"stream\":true }\n";
        assert_eq!(
            commit_wire_bytes(body).sha256,
            "61b54b4afd933a3dad0f1854e1ee723d92e10e0289a84d14079fefe34b99921e"
        );
        assert_ne!(
            commit_wire_bytes(body).sha256,
            commit_wire_bytes(br#"{"model":"tiny","stream":true}"#).sha256
        );
    }

    #[test]
    fn overflow_does_not_interrupt_hashing_and_drop_attempts_one_terminal() {
        let observer = Arc::new(Recorder {
            overflow: true,
            ..Recorder::default()
        });
        {
            let mut tap = WireBytesTap::new(Some(observer.clone()));
            for chunk in SSE.chunks(3) {
                tap.update(chunk);
            }
            let commitment = tap.finish(None);
            assert_eq!(commitment.sha256, SSE_SHA);
            assert!(!commitment.side_stream_complete);
        }
        assert_eq!(observer.terminals.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn response_tap_preserves_encoded_frames_and_order() {
        let observer = Arc::new(Recorder::default());
        let stream = futures_util::stream::iter(
            SSE.chunks(7)
                .map(|bytes| Ok::<_, std::io::Error>(Bytes::copy_from_slice(bytes)))
                .collect::<Vec<_>>(),
        );
        let body = observe_response_body(Body::from_stream(stream), observer.clone());
        assert_eq!(
            axum::body::to_bytes(body, SSE.len())
                .await
                .unwrap()
                .as_ref(),
            SSE
        );
        let chunks = observer.chunks.lock().unwrap();
        assert_eq!(
            chunks
                .iter()
                .flat_map(|(_, bytes)| bytes.clone())
                .collect::<Vec<_>>(),
            SSE
        );
        for pair in chunks.windows(2) {
            assert_eq!(pair[1].0, pair[0].0 + pair[0].1.len() as u64);
        }
        assert_eq!(observer.terminals.lock().unwrap()[0].sha256, SSE_SHA);
    }

    #[tokio::test]
    async fn dropping_unpolled_response_is_incomplete_not_empty_success() {
        let observer = Arc::new(Recorder::default());
        drop(observe_response_body(Body::from(SSE), observer.clone()));
        let terminals = observer.terminals.lock().unwrap();
        assert_eq!(terminals.len(), 1);
        assert_eq!(
            terminals[0].incomplete,
            Some(WireBytesIncomplete::Cancelled)
        );
    }

    #[tokio::test]
    async fn final_frame_commitment_describes_body_poll_even_if_transport_drops_it() {
        use http_body_util::BodyExt;
        let observer = Arc::new(Recorder::default());
        let mut body = observe_response_body(Body::from("abc"), observer.clone());
        let handed_to_transport = body.frame().await.unwrap().unwrap();
        // No socket write or peer receipt occurs in this test. The typed
        // boundary promises only the exact entity returned by poll_frame.
        drop(handed_to_transport);
        drop(body);
        let terminals = observer.terminals.lock().unwrap();
        assert_eq!(terminals.len(), 1);
        assert_eq!(terminals[0], commit_wire_bytes(b"abc"));
    }

    #[tokio::test]
    async fn response_tap_preserves_trailers_and_size_hint() {
        use http_body_util::{BodyExt, StreamBody};
        let observer = Arc::new(Recorder::default());
        let mut trailers = axum::http::HeaderMap::new();
        trailers.insert("x-checksum", "original".parse().unwrap());
        let stream = futures_util::stream::iter(vec![
            Ok::<_, std::io::Error>(Frame::data(Bytes::from_static(b"abc"))),
            Ok(Frame::trailers(trailers.clone())),
        ]);
        let mut body = observe_response_body(Body::new(StreamBody::new(stream)), observer.clone());
        assert_eq!(
            body.frame().await.unwrap().unwrap().into_data().unwrap(),
            b"abc"[..]
        );
        assert_eq!(
            body.frame()
                .await
                .unwrap()
                .unwrap()
                .into_trailers()
                .unwrap(),
            trailers
        );
        assert!(body.frame().await.is_none());
        assert_eq!(
            observer.terminals.lock().unwrap()[0],
            commit_wire_bytes(b"abc")
        );
        let original = Body::from("abc");
        let hint = original.size_hint();
        let wrapped = observe_response_body(original, Arc::new(Recorder::default()));
        assert_eq!(wrapped.size_hint().exact(), hint.exact());
    }

    #[tokio::test]
    async fn response_transport_error_preserves_only_emitted_prefix() {
        let observer = Arc::new(Recorder::default());
        let stream = futures_util::stream::iter(vec![
            Ok(Bytes::from_static(b"abc")),
            Err(std::io::Error::other("upstream failed")),
        ]);
        let body = observe_response_body(Body::from_stream(stream), observer.clone());
        assert!(axum::body::to_bytes(body, 100).await.is_err());
        let terminals = observer.terminals.lock().unwrap();
        assert_eq!(terminals.len(), 1);
        assert_eq!(
            terminals[0].sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(terminals[0].byte_count, 3);
        assert_eq!(
            terminals[0].incomplete,
            Some(WireBytesIncomplete::TransportError)
        );
    }

    #[test]
    fn timeout_is_distinct_and_terminal_is_idempotent() {
        let mut tap = WireBytesTap::new(None);
        tap.update(b"abc");
        let terminal = tap.finish(Some(WireBytesIncomplete::Timeout));
        tap.update(b"must not change a terminal commitment");
        assert_eq!(tap.finish(None), terminal);
        assert_eq!(terminal.incomplete, Some(WireBytesIncomplete::Timeout));
        assert_eq!(terminal.byte_count, 3);
    }
}
