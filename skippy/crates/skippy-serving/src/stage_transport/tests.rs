use super::*;
use skippy_protocol::binary::{recv_ready, recv_reply, send_ready, send_reply_ack};

fn pair() -> (StageConnection, StageConnection) {
    let (a, b) = tokio::io::duplex(64);
    let runtime = tokio::runtime::Handle::current();
    (
        StageConnection::from_async(a, runtime.clone()),
        StageConnection::from_async(b, runtime),
    )
}

#[tokio::test]
async fn direct_execution_stream_preserves_framing_backpressure_and_half_close() {
    let (mut source, mut destination) = pair();
    let sender = tokio::task::spawn_blocking(move || {
        recv_ready(&mut source).unwrap();
        // Exceeds the entire transport buffer; completion requires receiver progress.
        source.write_all(&vec![0x5a; 256 * 1024]).unwrap();
        source.shutdown(Shutdown::Write).unwrap();
        let reply = recv_reply(&mut source).unwrap();
        assert_eq!(reply.kind, skippy_protocol::binary::WireReplyKind::Ack);
        let mut end = [0];
        assert_eq!(source.read(&mut end).unwrap(), 0);
    });
    let receiver = tokio::task::spawn_blocking(move || {
        send_ready(&mut destination).unwrap();
        let mut bytes = Vec::new();
        destination.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, vec![0x5a; 256 * 1024]);
        send_reply_ack(&mut destination).unwrap();
        destination.shutdown(Shutdown::Write).unwrap();
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        sender.await.unwrap();
        receiver.await.unwrap();
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn peeking_and_timeouts_preserve_bytes_and_shutdown_interrupts_blocked_io() {
    let (mut source, mut destination) = pair();
    tokio::task::spawn_blocking(move || {
        source
            .set_read_timeout(Some(Duration::from_millis(10)))
            .unwrap();
        assert_eq!(
            source.peek(&mut [0]).unwrap_err().kind(),
            io::ErrorKind::TimedOut
        );
        destination.write_all(b"hello").unwrap();
        let mut bytes = [0; 5];
        assert_eq!(source.peek(&mut bytes).unwrap(), 5);
        assert_eq!(&bytes, b"hello");
        source.read_exact(&mut bytes).unwrap();
        assert_eq!(&bytes, b"hello");
        source.set_read_timeout(None).unwrap();
        let mut reader = source.try_clone().unwrap();
        let (started, waiting) = std::sync::mpsc::channel();
        let task = std::thread::spawn(move || {
            started.send(()).unwrap();
            reader.read(&mut [0]).unwrap_err().kind()
        });
        waiting.recv().unwrap();
        source.shutdown(Shutdown::Read).unwrap();
        assert_eq!(task.join().unwrap(), io::ErrorKind::ConnectionAborted);
        // Read cancellation must not close the opposite direction.
        source.write_all(b"x").unwrap();
        destination.read_exact(&mut [0]).unwrap();
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn shutdown_interrupts_a_backpressured_writer() {
    let (source, _destination) = pair();
    tokio::task::spawn_blocking(move || {
        let mut writer = source.try_clone().unwrap();
        let task = std::thread::spawn(move || writer.write_all(&vec![1; 256 * 1024]));
        source.shutdown(Shutdown::Both).unwrap();
        assert_eq!(
            task.join().unwrap().unwrap_err().kind(),
            io::ErrorKind::ConnectionAborted
        );
    })
    .await
    .unwrap();
}

struct PendingClose {
    stream: tokio::io::DuplexStream,
    closing: std::sync::mpsc::Sender<()>,
}

impl AsyncRead for PendingClose {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buffer: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<io::Result<()>> {
        std::pin::Pin::new(&mut self.stream).poll_read(cx, buffer)
    }
}

impl AsyncWrite for PendingClose {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buffer: &[u8],
    ) -> std::task::Poll<io::Result<usize>> {
        std::pin::Pin::new(&mut self.stream).poll_write(cx, buffer)
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<io::Result<()>> {
        std::pin::Pin::new(&mut self.stream).poll_flush(cx)
    }

    fn poll_shutdown(
        self: std::pin::Pin<&mut Self>,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<io::Result<()>> {
        let _ = self.closing.send(());
        std::task::Poll::Pending
    }
}

#[tokio::test]
async fn teardown_interrupts_a_provider_waiting_for_graceful_close() {
    let (stream, _peer) = tokio::io::duplex(64);
    let (closing, waiting) = std::sync::mpsc::channel();
    let source = StageConnection::from_async(
        PendingClose { stream, closing },
        tokio::runtime::Handle::current(),
    );
    tokio::task::spawn_blocking(move || {
        let half_close = source.try_clone().unwrap();
        let task = std::thread::spawn(move || half_close.shutdown(Shutdown::Write));
        waiting.recv_timeout(Duration::from_secs(1)).unwrap();
        source.shutdown(Shutdown::Both).unwrap();
        assert_eq!(
            task.join().unwrap().unwrap_err().kind(),
            io::ErrorKind::ConnectionAborted
        );
    })
    .await
    .unwrap();
}
