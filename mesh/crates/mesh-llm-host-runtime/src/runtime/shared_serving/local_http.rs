//! Local HTTP connections are owned by their attachment, including slow bodies.

use std::future::Future;
use std::io;
use std::net::SocketAddr;
use std::pin::Pin;
use std::task::{Context, Poll};

use axum::serve::Listener;
use skippy_inference_api::CancellationToken;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};

pub(super) struct AttachmentListener {
    listener: TcpListener,
    shutdown: CancellationToken,
}

impl AttachmentListener {
    pub(super) fn new(listener: TcpListener, shutdown: CancellationToken) -> Self {
        Self { listener, shutdown }
    }
}

impl Listener for AttachmentListener {
    type Io = AttachmentConnection;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        let (stream, address) = Listener::accept(&mut self.listener).await;
        let shutdown = self.shutdown.clone();
        (
            AttachmentConnection {
                stream,
                cancelled: Box::pin(async move { shutdown.cancelled().await }),
                closed: false,
            },
            address,
        )
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.listener.local_addr()
    }
}

pub(super) struct AttachmentConnection {
    stream: TcpStream,
    cancelled: Pin<Box<dyn Future<Output = ()> + Send>>,
    closed: bool,
}

impl AttachmentConnection {
    fn check_open(&mut self, cx: &mut Context<'_>) -> io::Result<()> {
        if self.closed || self.cancelled.as_mut().poll(cx).is_ready() {
            self.closed = true;
            Err(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "membership detached",
            ))
        } else {
            Ok(())
        }
    }
}

impl AsyncRead for AttachmentConnection {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        self.check_open(cx)?;
        Pin::new(&mut self.stream).poll_read(cx, buffer)
    }
}

impl AsyncWrite for AttachmentConnection {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.check_open(cx)?;
        Pin::new(&mut self.stream).poll_write(cx, buffer)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.check_open(cx)?;
        Pin::new(&mut self.stream).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    #[tokio::test]
    async fn detach_wakes_pending_connection_io_without_waiting_for_the_client() {
        let shutdown = CancellationToken::new();
        let tcp = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = tcp.local_addr().unwrap();
        let mut listener = AttachmentListener::new(tcp, shutdown.clone());
        let _client = TcpStream::connect(address).await.unwrap();
        let (mut connection, _) = listener.accept().await;
        let read = tokio::spawn(async move {
            let result = connection.read_u8().await;
            (connection, result)
        });
        tokio::task::yield_now().await;
        shutdown.cancel();
        let (mut connection, result) =
            tokio::time::timeout(std::time::Duration::from_secs(5), read)
                .await
                .unwrap()
                .unwrap();
        assert_eq!(result.unwrap_err().kind(), io::ErrorKind::ConnectionAborted);
        assert_eq!(
            connection
                .write_all(b"stale response")
                .await
                .unwrap_err()
                .kind(),
            io::ErrorKind::ConnectionAborted
        );
    }
}
