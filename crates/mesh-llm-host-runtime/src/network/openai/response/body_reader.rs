//! Decode HTTP body framing before interpreting protocol JSON or SSE.
use anyhow::{Result, anyhow, bail};
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncReadExt, BufReader, Chain};

pub(super) struct BodyReader<R> {
    reader: BufReader<Chain<std::io::Cursor<Vec<u8>>, R>>,
    chunked: bool,
    remaining: Option<usize>,
    chunk_remaining: usize,
    chunk_tail: bool,
    ended: bool,
}

impl<R: AsyncRead + Unpin> BodyReader<R> {
    pub(super) fn new(reader: R, buffered: Vec<u8>, chunked: bool, length: Option<usize>) -> Self {
        Self {
            reader: BufReader::new(std::io::Cursor::new(buffered).chain(reader)),
            chunked,
            remaining: length,
            chunk_remaining: 0,
            chunk_tail: false,
            ended: false,
        }
    }

    pub(super) async fn next(&mut self) -> Result<Option<Vec<u8>>> {
        if self.ended {
            return Ok(None);
        }
        if self.chunked && self.chunk_remaining == 0 {
            if self.chunk_tail {
                let mut tail = [0; 2];
                self.reader.read_exact(&mut tail).await?;
                if tail != *b"\r\n" {
                    bail!("invalid HTTP chunk terminator");
                }
            }
            let mut line = Vec::new();
            (&mut self.reader)
                .take(8193)
                .read_until(b'\n', &mut line)
                .await?;
            if line.len() > 8192 || !line.ends_with(b"\r\n") {
                bail!("invalid HTTP chunk size line");
            }
            let size = std::str::from_utf8(&line)?
                .split(';')
                .next()
                .unwrap_or_default()
                .trim();
            self.chunk_remaining =
                usize::from_str_radix(size, 16).map_err(|_| anyhow!("invalid HTTP chunk size"))?;
            if self.chunk_remaining == 0 {
                self.ended = true;
                return Ok(None);
            }
            self.chunk_tail = true;
        }
        let limit = if self.chunked {
            self.chunk_remaining.min(8192)
        } else {
            self.remaining.unwrap_or(8192).min(8192)
        };
        if limit == 0 {
            self.ended = true;
            return Ok(None);
        }
        let mut bytes = vec![0; limit];
        let count = self.reader.read(&mut bytes).await?;
        if count == 0 {
            if self.chunked || self.remaining.is_some_and(|remaining| remaining != 0) {
                bail!("truncated HTTP body");
            }
            self.ended = true;
            return Ok(None);
        }
        bytes.truncate(count);
        if self.chunked {
            self.chunk_remaining -= count;
        } else if let Some(remaining) = self.remaining.as_mut() {
            *remaining -= count;
        }
        Ok(Some(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn chunk_boundaries_do_not_enter_protocol_bytes() {
        let mut reader = BodyReader::new(
            std::io::Cursor::new(b"\xa9\r\n3\r\n!\n\n\r\n0\r\n\r\n"),
            b"1\r\n\xc3\r\n1;ext=x\r\n".to_vec(),
            true,
            None,
        );
        let mut body = Vec::new();
        while let Some(bytes) = reader.next().await.unwrap() {
            body.extend(bytes);
        }
        assert_eq!(String::from_utf8(body).unwrap(), "é!\n\n");
    }
}
