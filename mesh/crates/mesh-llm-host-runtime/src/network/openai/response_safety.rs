//! Bounded quarantine of image-capable responses before the first socket write.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::task::Poll;

use tokio::io::AsyncWriteExt;

use super::ClientStream;
use crate::network::openai::transport::RouteDispatchOutcome;

const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

/// Local safety plugins whose image checks must all approve this response.
#[derive(Clone, Debug)]
pub(crate) struct ImageResponseSafetyPlan {
    pub(crate) guard_plugins: Vec<String>,
}

pub(crate) struct ResponseSafetyState {
    pub(crate) stream: Mutex<Option<ClientStream>>,
    buffered: Mutex<BufferedResponse>,
    armed: Arc<AtomicBool>,
    started: AtomicBool,
}

#[derive(Default)]
struct BufferedResponse {
    plan: Option<ImageResponseSafetyPlan>,
    bytes: Vec<u8>,
    shutdown: bool,
    failed: bool,
}

/// Kept by ingress after dispatch consumes its stream, so a response cannot
/// bypass screening by returning through a different dispatch branch.
pub(crate) struct ResponseSafetyBuffer(Arc<ResponseSafetyState>);

impl ResponseSafetyBuffer {
    pub(crate) fn wrap(stream: ClientStream) -> (ClientStream, Self) {
        let state = Arc::new(ResponseSafetyState {
            stream: Mutex::new(Some(stream)),
            buffered: Mutex::new(BufferedResponse::default()),
            armed: Arc::new(AtomicBool::new(false)),
            started: AtomicBool::new(false),
        });
        (
            ClientStream::Safety {
                state: state.clone(),
            },
            Self(state),
        )
    }

    pub(crate) fn capture_guard(&self) -> Arc<AtomicBool> {
        self.0.armed.clone()
    }

    pub(crate) async fn finish(
        &self,
        node: &crate::mesh::Node,
        outcome: RouteDispatchOutcome,
    ) -> RouteDispatchOutcome {
        let buffered = std::mem::take(&mut *self.0.buffered.lock().unwrap());
        let Some(plan) = buffered.plan.as_ref() else {
            return outcome;
        };
        let rejection = if buffered.failed
            || (!buffered.shutdown && !complete_content_length_response(&buffered.bytes))
        {
            tracing::debug!(
                stage = "incomplete_buffer",
                failed = buffered.failed,
                shutdown = buffered.shutdown,
                "image safety response failed"
            );
            Some(503)
        } else {
            crate::inference::image_safety::screen_response(node, plan, &buffered.bytes)
                .await
                .err()
                .map(|error| match error {
                    crate::inference::image_safety::ImageSafetyResponseError::Denied => 403,
                    crate::inference::image_safety::ImageSafetyResponseError::Unavailable => 503,
                })
        };
        self.release(buffered, outcome, rejection).await
    }

    async fn release(
        &self,
        buffered: BufferedResponse,
        outcome: RouteDispatchOutcome,
        rejection: Option<u16>,
    ) -> RouteDispatchOutcome {
        let Some(mut stream) = self.0.stream.lock().unwrap().take() else {
            return RouteDispatchOutcome::Dropped("image_safety_stream_unavailable");
        };
        let (bytes, outcome) = if let Some(status) = rejection {
            (
                denial_response(status),
                if status == 403 {
                    RouteDispatchOutcome::admission_rejected(status)
                } else {
                    RouteDispatchOutcome::Responded(status)
                },
            )
        } else {
            (buffered.bytes, outcome)
        };
        if stream.write_all(&bytes).await.is_err() || stream.shutdown().await.is_err() {
            return RouteDispatchOutcome::Dropped("image_safety_response_write_failed");
        }
        outcome
    }
}

/// Some terminal admission writers finish by dropping their stream rather
/// than calling shutdown. Their length-delimited response is still complete;
/// a connection-delimited stream needs an explicit successful shutdown.
fn complete_content_length_response(wire: &[u8]) -> bool {
    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut response = httparse::Response::new(&mut headers);
    let Ok(httparse::Status::Complete(body_start)) = response.parse(wire) else {
        return false;
    };
    let mut length = None;
    for header in response.headers {
        if header.name.eq_ignore_ascii_case("transfer-encoding") {
            return false;
        }
        if header.name.eq_ignore_ascii_case("content-length") {
            if length.is_some() {
                return false;
            }
            length = std::str::from_utf8(header.value)
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok());
            if length.is_none() {
                return false;
            }
        }
    }
    length.is_some_and(|length| wire.len() - body_start == length)
}

fn denial_response(status: u16) -> Vec<u8> {
    let (reason, code, message) = if status == 403 {
        (
            "Forbidden",
            "content_policy_violation",
            "The image response was blocked by the local safety policy.",
        )
    } else {
        (
            "Service Unavailable",
            "image_safety_unavailable",
            "The image response could not be verified safely.",
        )
    };
    let body =
        serde_json::json!({"error": {"message": message, "type": code, "code": code}}).to_string();
    format!("HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).into_bytes()
}

impl ResponseSafetyState {
    pub(crate) fn require(&self, plan: ImageResponseSafetyPlan) -> std::io::Result<()> {
        if self.started.load(Ordering::Acquire) {
            return Err(std::io::Error::other(
                "image safety must be armed before the response starts",
            ));
        }
        let mut buffered = self.buffered.lock().unwrap();
        buffered.plan = Some(plan);
        self.armed.store(true, Ordering::Release);
        Ok(())
    }

    pub(crate) fn is_armed(&self) -> bool {
        self.armed.load(Ordering::Acquire)
    }

    pub(crate) fn mark_started(&self) {
        self.started.store(true, Ordering::Release);
    }

    pub(crate) fn poll_buffer_write(&self, bytes: &[u8]) -> Poll<std::io::Result<usize>> {
        let mut buffered = self.buffered.lock().unwrap();
        if buffered.failed || buffered.shutdown {
            return Poll::Ready(Err(std::io::Error::other(
                "image safety response is closed",
            )));
        }
        if bytes.len() > MAX_RESPONSE_BYTES.saturating_sub(buffered.bytes.len()) {
            buffered.failed = true;
            buffered.bytes.clear();
            return Poll::Ready(Err(std::io::Error::other(
                "image safety response exceeds size limit",
            )));
        }
        buffered.bytes.extend_from_slice(bytes);
        Poll::Ready(Ok(bytes.len()))
    }

    pub(crate) fn shutdown_buffer(&self) {
        self.buffered.lock().unwrap().shutdown = true;
    }

    /// A cancelled disconnect watcher restores the socket before relay begins.
    /// The lease owns the socket rather than holding a mutex across an await.
    pub(crate) fn lease_stream(self: &Arc<Self>) -> Option<ResponseSafetyStreamLease> {
        let stream = self.stream.lock().unwrap().take()?;
        Some(ResponseSafetyStreamLease {
            stream: Some(stream),
            state: self.clone(),
        })
    }
}

pub(crate) struct ResponseSafetyStreamLease {
    pub(crate) stream: Option<ClientStream>,
    state: Arc<ResponseSafetyState>,
}

impl Drop for ResponseSafetyStreamLease {
    fn drop(&mut self) {
        if let Some(stream) = self.stream.take() {
            *self.state.stream.lock().unwrap() = Some(stream);
        }
    }
}

pub(crate) fn unavailable_stream<T>() -> Poll<std::io::Result<T>> {
    Poll::Ready(Err(std::io::Error::other(
        "image safety stream is unavailable",
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use skippy_inference_api::wire_bytes::{WireBytesCommitment, WireBytesObserver};
    use std::future::Future;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpStream;

    #[derive(Default)]
    struct Recorder(Mutex<Vec<u8>>);

    impl WireBytesObserver for Recorder {
        fn response_status(&self, _: u16) {}
        fn try_chunk(&self, _: u64, bytes: &[u8]) -> bool {
            self.0.lock().unwrap().extend_from_slice(bytes);
            true
        }
        fn finish(&self, _: WireBytesCommitment) {}
    }

    async fn socket_pair() -> (TcpStream, ClientStream) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let client = TcpStream::connect(listener.local_addr().unwrap())
            .await
            .unwrap();
        let (server, _) = listener.accept().await.unwrap();
        (client, ClientStream::from(server))
    }

    fn plan() -> ImageResponseSafetyPlan {
        ImageResponseSafetyPlan {
            guard_plugins: Vec::new(),
        }
    }

    #[test]
    fn terminal_length_delimited_errors_are_complete_without_shutdown() {
        assert!(complete_content_length_response(
            b"HTTP/1.1 403 Forbidden\r\nContent-Length: 2\r\n\r\n{}"
        ));
        assert!(!complete_content_length_response(
            b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\n{}"
        ));
        assert!(!complete_content_length_response(
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nContent-Length: 2\r\n\r\n{}"
        ));
        assert!(!complete_content_length_response(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 2\r\n\r\n{}"
        ));
    }

    #[tokio::test]
    async fn armed_response_keeps_every_frame_from_socket_and_observer() {
        let (mut client, server) = socket_pair().await;
        let recorder = Arc::new(Recorder::default());
        let server = server.with_wire_bytes_observer(recorder.clone());
        let (mut server, held) = ResponseSafetyBuffer::wrap(server);
        server.require_image_safety(plan()).unwrap();
        let response = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: 22\r\n\r\ndata: image-fragment\n\n";
        server.write_all(response).await.unwrap();
        server.flush().await.unwrap();
        server.shutdown().await.unwrap();
        assert!(recorder.0.lock().unwrap().is_empty());
        assert_eq!(held.0.buffered.lock().unwrap().bytes, response);
        let mut byte = [0];
        assert!(
            matches!(client.try_read(&mut byte), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock)
        );

        let buffered = std::mem::take(&mut *held.0.buffered.lock().unwrap());
        let outcome = held
            .release(buffered, RouteDispatchOutcome::Responded(200), None)
            .await;
        assert_eq!(outcome, RouteDispatchOutcome::Responded(200));
        let mut received = Vec::new();
        client.read_to_end(&mut received).await.unwrap();
        assert_eq!(received, response);
        assert_eq!(
            recorder.0.lock().unwrap().as_slice(),
            b"data: image-fragment\n\n"
        );
    }

    #[tokio::test]
    async fn denied_response_releases_only_the_policy_error() {
        let (mut client, server) = socket_pair().await;
        let recorder = Arc::new(Recorder::default());
        let (mut server, held) = ResponseSafetyBuffer::wrap(server);
        // Observers added after quarantine must remain behind the guard too.
        server = server.with_wire_bytes_observer(recorder.clone());
        server.require_image_safety(plan()).unwrap();
        server
            .write_all(b"HTTP/1.1 200 OK\r\n\r\nUNAPPROVED_IMAGE")
            .await
            .unwrap();
        server.shutdown().await.unwrap();
        let buffered = std::mem::take(&mut *held.0.buffered.lock().unwrap());
        let outcome = held
            .release(buffered, RouteDispatchOutcome::Responded(200), Some(403))
            .await;
        assert_eq!(outcome, RouteDispatchOutcome::admission_rejected(403));
        let mut received = Vec::new();
        client.read_to_end(&mut received).await.unwrap();
        assert!(received.starts_with(b"HTTP/1.1 403 Forbidden\r\n"));
        assert!(!String::from_utf8_lossy(&received).contains("UNAPPROVED_IMAGE"));
        assert!(!String::from_utf8_lossy(&recorder.0.lock().unwrap()).contains("UNAPPROVED_IMAGE"));
    }

    #[tokio::test]
    async fn ordinary_text_streams_before_shutdown_and_cannot_be_armed_late() {
        let (mut client, server) = socket_pair().await;
        let (mut server, _held) = ResponseSafetyBuffer::wrap(server);
        server.write_all(b"first text frame").await.unwrap();
        server.flush().await.unwrap();
        let mut received = [0; 16];
        client.read_exact(&mut received).await.unwrap();
        assert_eq!(&received, b"first text frame");
        assert!(server.require_image_safety(plan()).is_err());
    }

    #[tokio::test]
    async fn cancelling_disconnect_watch_restores_stream_for_relay() {
        let (mut client, server) = socket_pair().await;
        let (mut server, held) = ResponseSafetyBuffer::wrap(server);
        let mut watch = Box::pin(server.wait_for_response_disconnect());
        std::future::poll_fn(|cx| {
            assert!(watch.as_mut().poll(cx).is_pending());
            Poll::Ready(())
        })
        .await;
        assert!(held.0.stream.lock().unwrap().is_none());
        drop(watch);
        assert!(held.0.stream.lock().unwrap().is_some());
        server.write_all(b"restored").await.unwrap();
        let mut received = [0; 8];
        client.read_exact(&mut received).await.unwrap();
        assert_eq!(&received, b"restored");
    }

    #[tokio::test]
    async fn oversized_response_is_discarded_before_release() {
        let (client, server) = socket_pair().await;
        let (mut server, held) = ResponseSafetyBuffer::wrap(server);
        server.require_image_safety(plan()).unwrap();
        assert!(
            server
                .write_all(&vec![0; MAX_RESPONSE_BYTES + 1])
                .await
                .is_err()
        );
        let buffered = held.0.buffered.lock().unwrap();
        assert!(buffered.failed);
        assert!(buffered.bytes.is_empty());
        drop(buffered);
        let mut byte = [0];
        assert!(
            matches!(client.try_read(&mut byte), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock)
        );
    }
}
