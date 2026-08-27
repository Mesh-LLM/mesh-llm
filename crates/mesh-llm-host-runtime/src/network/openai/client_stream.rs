use std::net::SocketAddr;
use std::pin::Pin;
use std::task::{Context, Poll};

use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::TcpStream;

type QuicBiStream = tokio::io::Join<iroh::endpoint::RecvStream, iroh::endpoint::SendStream>;

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
