use iroh::EndpointId;
use mesh_llm_transport::protocol::ProtocolStream;
use std::{
    future::Future,
    io,
    pin::Pin,
    task::{Context, Poll},
};
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    sync::watch,
    task::JoinHandle,
};

use super::super::Node;

type Cancellation = Pin<Box<dyn Future<Output = ()> + Send>>;

/// Keep policy and admission enforcement alive after execution takes the stream.
pub(super) struct GuardedStageStream {
    stream: ProtocolStream,
    read_cancelled: Cancellation,
    write_cancelled: Cancellation,
    monitor: JoinHandle<()>,
}

impl GuardedStageStream {
    pub(super) fn new(
        node: Node,
        peer: EndpointId,
        stream: ProtocolStream,
        mut stage: watch::Receiver<bool>,
    ) -> Self {
        let (cancelled, read_cancelled) = watch::channel(false);
        let write_cancelled = cancelled.subscribe();
        let monitor = tokio::spawn(async move {
            let mut peers = node.application_stage_peers();
            let mut admission = node.peer_change_rx.clone();
            let mut stopped = node.application_stage_cancellation();
            loop {
                if *stage.borrow() || *stopped.borrow() {
                    break;
                }
                if peer != node.id()
                    && (!node.stage_stream_admitted(peer).await
                        || !node.stage_required_peer_allowed(peer, false).await)
                {
                    break;
                }
                tokio::select! {
                    result = peers.changed() => { if result.is_err() { break; } },
                    result = admission.changed() => { if result.is_err() { break; } },
                    result = stopped.changed() => { if result.is_err() { break; } },
                    result = stage.changed() => { if result.is_err() { break; } },
                }
            }
            cancelled.send_replace(true);
        });
        Self {
            stream,
            read_cancelled: cancellation(read_cancelled),
            write_cancelled: cancellation(write_cancelled),
            monitor,
        }
    }
}

fn cancellation(mut receiver: watch::Receiver<bool>) -> Cancellation {
    Box::pin(async move {
        loop {
            if *receiver.borrow() {
                return;
            }
            if receiver.changed().await.is_err() {
                return;
            }
        }
    })
}

impl Drop for GuardedStageStream {
    fn drop(&mut self) {
        self.monitor.abort();
    }
}

impl AsyncRead for GuardedStageStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if self.read_cancelled.as_mut().poll(cx).is_ready() {
            return Poll::Ready(Err(io::ErrorKind::ConnectionAborted.into()));
        }
        Pin::new(&mut self.stream).poll_read(cx, buffer)
    }
}

impl AsyncWrite for GuardedStageStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<io::Result<usize>> {
        if self.write_cancelled.as_mut().poll(cx).is_ready() {
            return Poll::Ready(Err(io::ErrorKind::ConnectionAborted.into()));
        }
        Pin::new(&mut self.stream).poll_write(cx, buffer)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        if self.write_cancelled.as_mut().poll(cx).is_ready() {
            return Poll::Ready(Err(io::ErrorKind::ConnectionAborted.into()));
        }
        Pin::new(&mut self.stream).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_shutdown(cx)
    }
}
