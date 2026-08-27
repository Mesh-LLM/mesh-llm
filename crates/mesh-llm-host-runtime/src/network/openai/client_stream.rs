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
    Tcp(TcpStream),
    Quic {
        stream: QuicBiStream,
        prefix: std::io::Cursor<Vec<u8>>,
    },
}

impl From<TcpStream> for ClientStream {
    fn from(stream: TcpStream) -> Self {
        Self::Tcp(stream)
    }
}

impl ClientStream {
    pub(crate) fn from_quic_with_prefix(
        recv: iroh::endpoint::RecvStream,
        send: iroh::endpoint::SendStream,
        prefix: Vec<u8>,
    ) -> Self {
        Self::Quic {
            stream: tokio::io::join(recv, send),
            prefix: std::io::Cursor::new(prefix),
        }
    }

    pub(crate) async fn connect<A: tokio::net::ToSocketAddrs>(addr: A) -> std::io::Result<Self> {
        TcpStream::connect(addr).await.map(Self::Tcp)
    }

    pub(crate) fn set_nodelay(&self, nodelay: bool) -> std::io::Result<()> {
        match self {
            Self::Tcp(stream) => stream.set_nodelay(nodelay),
            Self::Quic { .. } => Ok(()),
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
            Self::Tcp(stream) => match wait_for_tcp_disconnect(stream).await {
                TcpDisconnectWatch::Disconnected => true,
                TcpDisconnectWatch::PipelinedBytes => std::future::pending::<bool>().await,
            },
            Self::Quic { stream, .. } => match stream.writer().stopped().await {
                Ok(Some(_)) | Err(_) => true,
                Ok(None) => std::future::pending::<bool>().await,
            },
        }
    }

    pub(crate) fn peer_addr(&self) -> std::io::Result<SocketAddr> {
        match self {
            Self::Tcp(stream) => stream.peer_addr(),
            Self::Quic { .. } => Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "QUIC ingress does not expose a socket address",
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
            Self::Tcp(stream) => Pin::new(stream).poll_read(cx, buf),
            Self::Quic { stream, prefix } => {
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
            Self::Tcp(stream) => Pin::new(stream).poll_write(cx, buf),
            Self::Quic { stream, .. } => Pin::new(stream).poll_write(cx, buf),
        }
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Self::Tcp(stream) => Pin::new(stream).poll_flush(cx),
            Self::Quic { stream, .. } => Pin::new(stream).poll_flush(cx),
        }
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        match self.get_mut() {
            Self::Tcp(stream) => Pin::new(stream).poll_shutdown(cx),
            Self::Quic { stream, .. } => Pin::new(stream).poll_shutdown(cx),
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
