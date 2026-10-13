//! Deterministic transport failures retain only terminal backend evidence.
use super::*;
use mesh_llm_payments_types::metering::MeteredUsage;
use std::{
    io,
    pin::Pin,
    task::{Context as TaskContext, Poll},
};
use tokio::io::ReadBuf;

const HEADERS: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n";
const REPORT: &[u8] = b"data: {\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7}}\n\n";

#[derive(Clone, Copy)]
enum Abort {
    Cancel,
    Write,
    Read,
}

struct Backend {
    bytes: Option<Vec<u8>>,
    read_error: bool,
}

impl AsyncRead for Backend {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _: &mut TaskContext<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if let Some(bytes) = self.bytes.take() {
            buffer.put_slice(&bytes);
            Poll::Ready(Ok(()))
        } else if self.read_error {
            Poll::Ready(Err(io::Error::new(
                io::ErrorKind::ConnectionReset,
                "backend reset",
            )))
        } else {
            Poll::Pending
        }
    }
}

struct Writer {
    failed: bool,
    cancel: Option<tokio::sync::oneshot::Sender<()>>,
}

impl AsyncWrite for Writer {
    fn poll_write(
        self: Pin<&mut Self>,
        _: &mut TaskContext<'_>,
        bytes: &[u8],
    ) -> Poll<io::Result<usize>> {
        if self.failed {
            Poll::Ready(Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "buyer disconnected",
            )))
        } else {
            Poll::Ready(Ok(bytes.len()))
        }
    }

    fn poll_flush(mut self: Pin<&mut Self>, _: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
        if let Some(cancel) = self.cancel.take() {
            let _ = cancel.send(());
        }
        Poll::Ready(Ok(()))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

async fn interrupted(response: Vec<u8>, abort: Abort) -> Result<MeteredOutcome> {
    let backend = Backend {
        bytes: Some(response),
        read_error: matches!(abort, Abort::Read),
    };
    let (signal, ready) = tokio::sync::oneshot::channel();
    let (mut sender, mut reader) = tokio::io::duplex(256);
    let cancellation = tokio::spawn(async move {
        if ready.await.is_ok() {
            wire::write(&mut sender, &Frame::Cancel {}).await?;
        }
        Ok::<_, anyhow::Error>(())
    });
    let mut writer = Writer {
        failed: matches!(abort, Abort::Write),
        cancel: matches!(abort, Abort::Cancel).then_some(signal),
    };
    let result = tokio::time::timeout(
        Duration::from_secs(5),
        forward_backend(backend, &mut reader, &mut writer),
    )
    .await?;
    cancellation.await??;
    Ok(result)
}

#[tokio::test]
async fn terminal_usage_survives_cancel_write_and_read_failure() -> Result<()> {
    for abort in [Abort::Cancel, Abort::Write, Abort::Read] {
        let response = [HEADERS, REPORT, b"data: [DONE]\n\n"].concat();
        assert_eq!(
            interrupted(response, abort).await?,
            MeteredOutcome::Usage(MeteredUsage {
                prompt_tokens: 12,
                completion_tokens: 7,
            })
        );
    }
    Ok(())
}

#[tokio::test]
async fn unknown_partial_and_provisional_usage_never_become_abort_bills() -> Result<()> {
    let responses = [
        // Non-2xx headers are not a terminal SSE report, even with a DONE body.
        b"HTTP/1.1 500 Error\r\nContent-Type: text/event-stream\r\n\r\ndata: [DONE]\n\n".to_vec(),
        [HEADERS, b"data: {\"choices\":[{\"text\":\"hello\"}]}\n\n"].concat(),
        [HEADERS, REPORT].concat(),
        [HEADERS, REPORT, b"data: [DONE]"].concat(),
        [HEADERS, b"data: {\"usage\":{\"prompt_tokens\":12}}\n\ndata: [DONE]\n\n"].concat(),
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7}}".to_vec(),
    ];
    for response in responses {
        for abort in [Abort::Cancel, Abort::Write, Abort::Read] {
            assert_eq!(
                interrupted(response.clone(), abort).await?,
                MeteredOutcome::Abandoned
            );
        }
    }
    Ok(())
}
