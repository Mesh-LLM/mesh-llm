use std::net::SocketAddr;
use std::pin::Pin;
use std::task::{Context, Poll};

use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::TcpStream;

type QuicBiStream = tokio::io::Join<iroh::endpoint::RecvStream, iroh::endpoint::SendStream>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TcpDisconnectWatch {
    Disconnected,
    PipelinedBytes,
}

/// A read-ready downstream socket can be checked without consuming request
/// bytes. EOF keeps the watcher pending so clients may legally half-close their
/// write side after sending a request and still receive the response.
async fn wait_for_tcp_disconnect(stream: &TcpStream) -> TcpDisconnectWatch {
    if stream.readable().await.is_err() {
        return TcpDisconnectWatch::Disconnected;
    }

    let mut peeked = [0u8; 1];
    match stream.peek(&mut peeked).await {
        Ok(0) => std::future::pending::<TcpDisconnectWatch>().await,
        Err(_) => TcpDisconnectWatch::Disconnected,
        Ok(_) => TcpDisconnectWatch::PipelinedBytes,
    }
}

/// Client-facing byte stream accepted by the OpenAI ingress.
///
/// Local callers arrive over TCP. Remote mesh callers already have an
/// authenticated QUIC bi-stream, which enters the same request path without
/// opening a second plaintext loopback connection.
pub(crate) enum ClientStream {
    Tcp {
        stream: TcpStream,
        anthropic: bool,
    },
    Quic {
        stream: QuicBiStream,
        anthropic: bool,
        prefix: std::io::Cursor<Vec<u8>>,
    },
    /// Discards every byte written and reports immediate EOF on read, with no
    /// real downstream socket behind it.
    ///
    /// Used where the relay runs with no client waiting on its output: the
    /// paid seller replays the response it already delivered through the
    /// relay to learn the usage and digests the payer's relay computes (see
    /// `network::openai::response::replay`). The relay streams response bytes
    /// to a `ClientStream` by design, so it needs somewhere to write that
    /// isn't a client's socket.
    #[cfg_attr(not(feature = "payments"), allow(dead_code))]
    Null,
}

impl From<TcpStream> for ClientStream {
    fn from(stream: TcpStream) -> Self {
        Self::Tcp {
            stream,
            anthropic: false,
        }
    }
}

impl ClientStream {
    pub(crate) fn set_client_path(&mut self, path: &str) {
        let messages = matches!(
            path.split('?').next(),
            Some("/v1/messages" | "/v1/messages/count_tokens")
        );
        match self {
            Self::Tcp { anthropic, .. } | Self::Quic { anthropic, .. } => *anthropic = messages,
            // The discard sink answers no client, so it has no response
            // dialect to switch.
            Self::Null => {}
        }
    }

    pub(crate) fn is_anthropic(&self) -> bool {
        match self {
            Self::Tcp { anthropic, .. } | Self::Quic { anthropic, .. } => *anthropic,
            Self::Null => false,
        }
    }

    pub(crate) fn from_quic_with_prefix(
        recv: iroh::endpoint::RecvStream,
        send: iroh::endpoint::SendStream,
        prefix: Vec<u8>,
    ) -> Self {
        Self::Quic {
            stream: tokio::io::join(recv, send),
            anthropic: false,
            prefix: std::io::Cursor::new(prefix),
        }
    }

    pub(crate) async fn connect<A: tokio::net::ToSocketAddrs>(addr: A) -> std::io::Result<Self> {
        TcpStream::connect(addr).await.map(Self::from)
    }

    /// A discard sink with no real downstream socket — see [`Self::Null`].
    #[cfg_attr(not(feature = "payments"), allow(dead_code))]
    pub(crate) fn null() -> Self {
        Self::Null
    }

    pub(crate) fn set_nodelay(&self, nodelay: bool) -> std::io::Result<()> {
        match self {
            Self::Tcp { stream, .. } => stream.set_nodelay(nodelay),
            Self::Quic { .. } => Ok(()),
            Self::Null => Ok(()),
        }
    }

    /// Wait until the downstream can no longer receive a response.
    ///
    /// TCP resets are detected without consuming pipelined request bytes. For
    /// QUIC, the peer dropping or resetting its receive half completes the
    /// response send stream's `stopped` future. A clean acknowledgement after
    /// a locally finished response is not a disconnect signal.
    pub(crate) async fn wait_for_response_disconnect(&self) -> bool {
        match self {
            Self::Tcp { stream, .. } => match wait_for_tcp_disconnect(stream).await {
                TcpDisconnectWatch::Disconnected => true,
                TcpDisconnectWatch::PipelinedBytes => std::future::pending::<bool>().await,
            },
            Self::Quic { stream, .. } => match stream.writer().stopped().await {
                Ok(Some(_)) | Err(_) => true,
                Ok(None) => std::future::pending::<bool>().await,
            },
            // No real client to disconnect -- never report one, so a relay
            // into the discard sink always runs to its own natural completion
            // instead of racing a phantom cancellation.
            Self::Null => std::future::pending::<bool>().await,
        }
    }

    pub(crate) fn peer_addr(&self) -> std::io::Result<SocketAddr> {
        match self {
            Self::Tcp { stream, .. } => stream.peer_addr(),
            Self::Quic { .. } => Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "QUIC ingress does not expose a socket address",
            )),
            Self::Null => Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "the discard sink has no socket address",
            )),
        }
    }
}

impl AsyncRead for ClientStream {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Self::Tcp { stream, .. } => Pin::new(stream).poll_read(cx, buf),
            Self::Quic { stream, prefix, .. } => {
                let position = prefix.position() as usize;
                let bytes = prefix.get_ref();
                if position < bytes.len() {
                    let count = buf.remaining().min(bytes.len() - position);
                    buf.put_slice(&bytes[position..position + count]);
                    prefix.set_position((position + count) as u64);
                    Poll::Ready(Ok(()))
                } else {
                    Pin::new(stream).poll_read(cx, buf)
                }
            }
            // Immediate EOF -- 0 bytes filled, buf untouched -- rather than
            // pending forever, so nothing that unexpectedly tries to read a
            // request body back off this sink can hang.
            Self::Null => Poll::Ready(Ok(())),
        }
    }
}

impl AsyncWrite for ClientStream {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        match self.get_mut() {
            Self::Tcp { stream, .. } => Pin::new(stream).poll_write(cx, buf),
            Self::Quic { stream, .. } => Pin::new(stream).poll_write(cx, buf),
            // Discard -- claim the whole buffer was written, same as writing
            // to `/dev/null`, so callers see no backpressure and no error.
            Self::Null => Poll::Ready(Ok(buf.len())),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Self::Tcp { stream, .. } => Pin::new(stream).poll_flush(cx),
            Self::Quic { stream, .. } => Pin::new(stream).poll_flush(cx),
            Self::Null => Poll::Ready(Ok(())),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Self::Tcp { stream, .. } => Pin::new(stream).poll_shutdown(cx),
            Self::Quic { stream, .. } => Pin::new(stream).poll_shutdown(cx),
            Self::Null => Poll::Ready(Ok(())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::{Endpoint, SecretKey};
    use tokio::time::{Duration, timeout};

    const TEST_ALPN: &[u8] = b"mesh-llm/client-stream-test/1";

    #[tokio::test]
    async fn null_stream_discards_writes_without_error() {
        use tokio::io::AsyncWriteExt;

        let mut sink = ClientStream::null();
        let written = sink.write(b"replayed response bytes").await.unwrap();
        assert_eq!(written, "replayed response bytes".len());
        sink.flush().await.unwrap();
        sink.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn null_stream_read_reports_immediate_eof() {
        use tokio::io::AsyncReadExt;

        let mut sink = ClientStream::null();
        let mut buf = [0u8; 8];
        let n = sink.read(&mut buf).await.unwrap();
        assert_eq!(n, 0, "the discard sink must report EOF, never pend forever");
    }

    #[tokio::test]
    async fn null_stream_never_reports_a_response_disconnect() {
        let sink = ClientStream::null();
        // No real client to disconnect; a relay into the sink must run to its
        // own natural completion rather than racing a phantom cancellation.
        assert!(
            timeout(
                Duration::from_millis(50),
                sink.wait_for_response_disconnect()
            )
            .await
            .is_err(),
            "the discard sink must never resolve a disconnect signal"
        );
    }

    #[test]
    fn null_stream_set_nodelay_and_peer_addr_are_inert() {
        let sink = ClientStream::null();
        assert!(sink.set_nodelay(true).is_ok());
        assert!(sink.peer_addr().is_err());
    }

    #[tokio::test]
    async fn quic_stop_sending_reports_response_disconnect() {
        let server = Endpoint::builder(iroh::endpoint::presets::Minimal)
            .secret_key(SecretKey::generate())
            .alpns(vec![TEST_ALPN.to_vec()])
            .relay_mode(iroh::endpoint::RelayMode::Disabled)
            .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
            .unwrap()
            .bind()
            .await
            .unwrap();
        let server_endpoint = server.clone();
        let accepted = tokio::spawn(async move {
            let incoming = server_endpoint.accept().await.expect("connection arrives");
            let connection = incoming.await.expect("connection negotiates");
            let (send, recv) = connection.accept_bi().await.expect("stream arrives");
            ClientStream::from_quic_with_prefix(recv, send, Vec::new())
        });

        let client = Endpoint::builder(iroh::endpoint::presets::Minimal)
            .secret_key(SecretKey::generate())
            .relay_mode(iroh::endpoint::RelayMode::Disabled)
            .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))
            .unwrap()
            .bind()
            .await
            .unwrap();
        let connection = client.connect(server.addr(), TEST_ALPN).await.unwrap();
        let (mut request_send, mut response_recv) = connection.open_bi().await.unwrap();
        request_send.write_all(b"request").await.unwrap();
        let downstream = accepted.await.unwrap();

        response_recv.stop(42u32.into()).unwrap();
        assert!(
            timeout(
                Duration::from_secs(2),
                downstream.wait_for_response_disconnect()
            )
            .await
            .expect("STOP_SENDING must reach the response sender")
        );

        client.close().await;
        server.close().await;
    }
}
