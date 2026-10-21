use mesh_llm_transport::protocol::ProtocolStream;
use std::{
    io,
    pin::Pin,
    task::{Context, Poll},
};
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    sync::OwnedSemaphorePermit,
};

pub(super) fn hold_permit(stream: ProtocolStream, permit: OwnedSemaphorePermit) -> ProtocolStream {
    ProtocolStream::new(AdmittedStream {
        stream,
        _permit: permit,
    })
}

/// The handler can hand its stream to execution and return before I/O ends.
struct AdmittedStream {
    stream: ProtocolStream,
    _permit: OwnedSemaphorePermit,
}

impl AsyncRead for AdmittedStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_read(context, buffer)
    }
}

impl AsyncWrite for AdmittedStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.stream).poll_write(context, buffer)
    }

    fn poll_flush(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_flush(context)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_shutdown(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn incoming_limit_stays_held_after_handoff_until_execution_drops_the_stream() {
        let pool = std::sync::Arc::new(tokio::sync::Semaphore::new(1));
        let permit = pool.clone().try_acquire_owned().unwrap();
        let (incoming, _peer) = tokio::io::duplex(64);
        let execution = skippy_serving::stage_transport::StageConnection::from_async(
            hold_permit(ProtocolStream::new(incoming), permit),
            tokio::runtime::Handle::current(),
        );
        assert!(pool.clone().try_acquire_owned().is_err());
        drop(execution);
        assert!(pool.try_acquire_owned().is_ok());
    }
}
