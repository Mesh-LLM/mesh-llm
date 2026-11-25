//! Reads a peer's HTTP answer to a challenge from a tunnel stream.
//!
//! The peer decides what it sends, so the answer is read as it arrives and
//! never past what the caller allows: the head by [`MAX_HEAD_BYTES`], the
//! body by `max_body`, whichever way it is framed. The body is decoded
//! into one buffer, with no copy of the whole answer beside it. An answer
//! with an error status is read by [`MAX_REFUSAL_BYTES`] only: its body is
//! shown to the operator, not checked.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use skippy_receipts::turn;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, BufReader};

use super::answer_limits::MAX_REFUSAL_BYTES;

/// The longest head of an answer.
const MAX_HEAD_BYTES: usize = 16 << 10;
/// The longest line of chunked framing: a size, extensions, a trailer.
const MAX_FRAMING_LINE: u64 = 1024;

/// A peer's answer: its status, its signature header, when it asks to be
/// asked again (`Retry-After` in whole seconds), and its body as sent.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Answer {
    pub(super) status: u16,
    pub(super) signature: Option<String>,
    pub(super) retry_after: Option<Duration>,
    pub(super) body: Vec<u8>,
}

enum Framing {
    Chunked,
    Sized(usize),
    UntilClose,
}

/// Reads one answer from `stream`. A body of an answer with status 200
/// that is larger than `max_body` is an error, before the rest is read.
pub(super) async fn read_answer<R: AsyncRead + Unpin>(
    stream: R,
    max_body: usize,
) -> Result<Answer> {
    let mut reader = BufReader::new(stream);
    let head = read_head(&mut reader).await?;
    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut response = httparse::Response::new(&mut headers);
    let httparse::Status::Complete(_) = response.parse(&head).context("the answer's head")? else {
        bail!("the answer ended inside its headers");
    };
    let header = |name: &str| {
        response
            .headers
            .iter()
            .find(|h| h.name.eq_ignore_ascii_case(name))
            .and_then(|h| std::str::from_utf8(h.value).ok())
    };
    let chunked = header("transfer-encoding").is_some_and(|v| v.eq_ignore_ascii_case("chunked"));
    let length = header("content-length").and_then(|v| v.trim().parse::<usize>().ok());
    let framing = match (chunked, length) {
        (true, _) => Framing::Chunked,
        (false, Some(length)) => Framing::Sized(length),
        (false, None) => Framing::UntilClose,
    };
    let signature = header(turn::OPENING_SIGNATURE_HEADER).map(str::to_owned);
    let retry_after = header("retry-after")
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(Duration::from_secs);
    let status = response.code.unwrap_or(0);
    let body = if status == 200 {
        read_body(&mut reader, framing, max_body).await?
    } else {
        read_refusal(&mut reader, framing, max_body.min(MAX_REFUSAL_BYTES)).await
    };
    Ok(Answer {
        status,
        signature,
        retry_after,
        body,
    })
}

/// The answer's head, up to and with its empty line.
async fn read_head<R: AsyncRead + Unpin>(reader: &mut BufReader<R>) -> Result<Vec<u8>> {
    let mut head = Vec::new();
    loop {
        let available = reader.fill_buf().await.context("read the answer's head")?;
        if available.is_empty() {
            bail!("the answer ended inside its headers");
        }
        let mut used = 0;
        let mut ended = false;
        for &byte in available {
            head.push(byte);
            used += 1;
            if head.ends_with(b"\r\n\r\n") || head.ends_with(b"\n\n") {
                ended = true;
                break;
            }
            if head.len() >= MAX_HEAD_BYTES {
                bail!("the answer's head is longer than {MAX_HEAD_BYTES} bytes");
            }
        }
        reader.consume(used);
        if ended {
            return Ok(head);
        }
    }
}

/// The body of an answer with an error status, or its first `max_body`
/// bytes: what a refusal says is for the operator to read, not checked.
async fn read_refusal<R: AsyncRead + Unpin>(
    reader: &mut BufReader<R>,
    framing: Framing,
    max_body: usize,
) -> Vec<u8> {
    match framing {
        Framing::Chunked => read_body(reader, framing, max_body)
            .await
            .unwrap_or_default(),
        Framing::Sized(_) | Framing::UntilClose => {
            let mut body = Vec::new();
            let _ = (&mut *reader)
                .take(max_body as u64)
                .read_to_end(&mut body)
                .await;
            body
        }
    }
}

fn too_large(max_body: usize) -> anyhow::Error {
    anyhow::anyhow!("the answer is larger than the {max_body} bytes this node reads of it")
}

async fn read_body<R: AsyncRead + Unpin>(
    reader: &mut BufReader<R>,
    framing: Framing,
    max_body: usize,
) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    match framing {
        Framing::Sized(length) => {
            if length > max_body {
                return Err(too_large(max_body));
            }
            read_exactly(reader, length, &mut body).await?;
        }
        Framing::UntilClose => {
            let limit = max_body as u64 + 1;
            (&mut *reader)
                .take(limit)
                .read_to_end(&mut body)
                .await
                .context("read the answer's body")?;
            if body.len() > max_body {
                return Err(too_large(max_body));
            }
        }
        Framing::Chunked => loop {
            let size = chunk_size(reader).await?;
            if size == 0 {
                // trailers, to the empty line
                while !framing_line(reader).await?.is_empty() {}
                break;
            }
            if size > max_body - body.len() {
                return Err(too_large(max_body));
            }
            read_exactly(reader, size, &mut body).await?;
            if !framing_line(reader).await?.is_empty() {
                bail!("a chunk of the answer is longer than it says");
            }
        },
    }
    Ok(body)
}

/// Appends exactly `length` bytes to `body`, which grows as they arrive:
/// a peer that says a size and sends nothing makes this node hold nothing.
async fn read_exactly<R: AsyncRead + Unpin>(
    reader: &mut BufReader<R>,
    length: usize,
    body: &mut Vec<u8>,
) -> Result<()> {
    let read = (&mut *reader)
        .take(length as u64)
        .read_to_end(body)
        .await
        .context("read the answer's body")?;
    if read < length {
        bail!("the answer is shorter than it says");
    }
    Ok(())
}

/// One line of chunked framing, without its line ending.
async fn framing_line<R: AsyncRead + Unpin>(reader: &mut BufReader<R>) -> Result<Vec<u8>> {
    let mut line = Vec::new();
    (&mut *reader)
        .take(MAX_FRAMING_LINE)
        .read_until(b'\n', &mut line)
        .await
        .context("read the answer's body")?;
    if line.pop() != Some(b'\n') {
        bail!("the answer ended inside its body");
    }
    if line.last() == Some(&b'\r') {
        line.pop();
    }
    Ok(line)
}

async fn chunk_size<R: AsyncRead + Unpin>(reader: &mut BufReader<R>) -> Result<usize> {
    let line = framing_line(reader).await?;
    let digits = line.split(|&b| b == b';').next().unwrap_or_default();
    std::str::from_utf8(digits)
        .ok()
        .and_then(|digits| usize::from_str_radix(digits.trim(), 16).ok())
        .context("a chunk of the answer has no size")
}

#[cfg(test)]
mod tests {
    use std::pin::Pin;
    use std::task::{Context as TaskContext, Poll};

    use tokio::io::ReadBuf;

    use super::*;

    /// A stream that gives its bytes one at a time, and counts them.
    struct Trickle<'a> {
        bytes: &'a [u8],
        given: usize,
    }

    impl AsyncRead for Trickle<'_> {
        fn poll_read(
            mut self: Pin<&mut Self>,
            _: &mut TaskContext<'_>,
            buf: &mut ReadBuf<'_>,
        ) -> Poll<std::io::Result<()>> {
            if let Some(&byte) = self.bytes.get(self.given) {
                buf.put_slice(&[byte]);
                self.given += 1;
            }
            Poll::Ready(Ok(()))
        }
    }

    async fn read(raw: &[u8], max_body: usize) -> Result<Answer> {
        let whole = read_answer(raw, max_body).await;
        let mut trickle = Trickle {
            bytes: raw,
            given: 0,
        };
        let trickled = read_answer(&mut trickle, max_body).await;
        // however the bytes arrive, the answer is the same
        match (&whole, &trickled) {
            (Ok(a), Ok(b)) => assert_eq!(a, b),
            (Err(a), Err(b)) => assert_eq!(format!("{a:#}"), format!("{b:#}")),
            _ => panic!("the answer depends on how its bytes arrive"),
        }
        whole
    }

    fn error(result: Result<Answer>) -> String {
        format!("{:#}", result.expect_err("an answer this node refuses"))
    }

    #[tokio::test]
    async fn a_chunked_answer_is_read_with_its_signature() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nx-receipt-opening-signature: ab\r\n\r\n3\r\n{\"a\r\n4;x=y\r\n\":1}\r\n0\r\nx-trailer: 1\r\n\r\n";
        let answer = read(raw, 64).await.unwrap();
        assert_eq!(answer.status, 200);
        assert_eq!(answer.signature.as_deref(), Some("ab"));
        assert_eq!(answer.body, b"{\"a\":1}");
    }

    #[tokio::test]
    async fn a_busy_answer_says_when_to_ask_again() {
        let raw =
            b"HTTP/1.1 503 Service Unavailable\r\nRetry-After: 2\r\nContent-Length: 2\r\n\r\n{}";
        let answer = read(raw, 64).await.unwrap();
        assert_eq!(answer.status, 503);
        assert_eq!(answer.retry_after, Some(Duration::from_secs(2)));
        // a date, which no stage sends, is no time this node waits for
        let dated = b"HTTP/1.1 503 Service Unavailable\r\nRetry-After: Wed, 11 Nov 2026 07:28:00 GMT\r\n\r\n";
        assert_eq!(read(dated, 64).await.unwrap().retry_after, None);
    }

    #[tokio::test]
    async fn a_sized_answer_is_read_to_its_length() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}trailing";
        assert_eq!(read(raw, 64).await.unwrap().body, b"{}");
        let short = b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\n{}";
        assert!(error(read(short, 64).await).contains("shorter than it says"));
        let unframed = b"HTTP/1.1 200 OK\r\n\r\n{\"b\":2}";
        assert_eq!(read(unframed, 64).await.unwrap().body, b"{\"b\":2}");
    }

    #[tokio::test]
    async fn an_answer_past_the_bound_is_refused_however_it_is_framed() {
        let body = "x".repeat(100);
        let sized = format!("HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{body}");
        let unframed = format!("HTTP/1.1 200 OK\r\n\r\n{body}");
        let chunked = format!(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n32\r\n{}\r\n32\r\n{}\r\n0\r\n\r\n",
            &body[..50],
            &body[50..]
        );
        for raw in [&sized, &unframed, &chunked] {
            assert_eq!(read(raw.as_bytes(), 100).await.unwrap().body.len(), 100);
            let refused = error(read(raw.as_bytes(), 99).await);
            assert!(refused.contains("larger than the 99 bytes"), "{refused}");
        }
    }

    #[tokio::test]
    async fn a_size_past_the_bound_is_refused_before_its_body_is_read() {
        // the peer says 100 GiB and need not send a byte of it
        for raw in [
            &b"HTTP/1.1 200 OK\r\nContent-Length: 107374182400\r\n\r\n"[..],
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n1900000000\r\n",
        ] {
            let mut stream = Trickle {
                bytes: raw,
                given: 0,
            };
            let refused = error(read_answer(&mut stream, 1 << 20).await);
            assert!(refused.contains("larger than"), "{refused}");
        }
    }

    #[tokio::test]
    async fn an_error_status_keeps_a_bounded_body() {
        let raw = b"HTTP/1.1 404 Not Found\r\nContent-Length: 4\r\n\r\ngone";
        let answer = read(raw, 1 << 20).await.unwrap();
        assert_eq!((answer.status, &answer.body[..]), (404, &b"gone"[..]));
        // a megabyte of refusal is not read into memory
        let long = format!("HTTP/1.1 503 Busy\r\n\r\n{}", "y".repeat(1 << 20));
        let answer = read_answer(long.as_bytes(), 128 << 20).await.unwrap();
        assert_eq!(answer.status, 503);
        assert_eq!(answer.body.len(), MAX_REFUSAL_BYTES);
    }

    #[tokio::test]
    async fn a_head_that_does_not_end_is_refused() {
        let endless = format!("HTTP/1.1 200 OK\r\nx: {}", "h".repeat(1 << 20));
        let refused = error(read_answer(endless.as_bytes(), 64).await);
        assert!(refused.contains("head is longer"), "{refused}");
        let cut = b"HTTP/1.1 200 OK\r\nContent-Le";
        assert!(error(read(cut, 64).await).contains("ended inside its headers"));
        let no_size = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n";
        assert!(error(read(no_size, 64).await).contains("no size"));
    }
}
