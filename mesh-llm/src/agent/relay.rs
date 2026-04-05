//! Length-prefixed JSON-RPC framing for ACP messages over QUIC streams.
//!
//! Each message is framed as `[4-byte LE length][UTF-8 JSON-RPC payload]`.
//! This is used on STREAM_ACP (0x0A) bidirectional QUIC streams.

use anyhow::{bail, Context, Result};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Maximum single ACP message size (16 MiB — generous for large tool outputs).
const MAX_MESSAGE_SIZE: u32 = 16 * 1024 * 1024;

/// Read one length-prefixed JSON-RPC message from a stream.
///
/// Returns `Ok(None)` on clean EOF (remote closed the stream).
pub async fn read_message<R: AsyncRead + Unpin>(reader: &mut R) -> Result<Option<String>> {
    let mut len_buf = [0u8; 4];
    match reader.read_exact(&mut len_buf).await {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e).context("read ACP message length"),
    }
    let len = u32::from_le_bytes(len_buf);
    if len > MAX_MESSAGE_SIZE {
        bail!("ACP message too large: {len} bytes (max {MAX_MESSAGE_SIZE})");
    }
    let mut buf = vec![0u8; len as usize];
    reader
        .read_exact(&mut buf)
        .await
        .context("read ACP message body")?;
    String::from_utf8(buf)
        .context("ACP message is not valid UTF-8")
        .map(Some)
}

/// Write one length-prefixed JSON-RPC message to a stream.
pub async fn write_message<W: AsyncWrite + Unpin>(writer: &mut W, msg: &str) -> Result<()> {
    let len = msg.len() as u32;
    writer
        .write_all(&len.to_le_bytes())
        .await
        .context("write ACP message length")?;
    writer
        .write_all(msg.as_bytes())
        .await
        .context("write ACP message body")?;
    writer.flush().await.context("flush ACP message")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::duplex;

    #[tokio::test]
    async fn roundtrip_message() {
        let (mut client, mut server) = duplex(1024);
        let msg = r#"{"jsonrpc":"2.0","method":"initialize","id":1}"#;
        write_message(&mut client, msg).await.unwrap();
        drop(client); // close write side
        let received = read_message(&mut server).await.unwrap();
        assert_eq!(received.as_deref(), Some(msg));
    }

    #[tokio::test]
    async fn eof_returns_none() {
        let (client, mut server) = duplex(1024);
        drop(client);
        let received = read_message(&mut server).await.unwrap();
        assert!(received.is_none());
    }

    #[tokio::test]
    async fn rejects_oversized_message() {
        let (mut client, mut server) = duplex(64);
        let bad_len = (MAX_MESSAGE_SIZE + 1).to_le_bytes();
        client.write_all(&bad_len).await.unwrap();
        let err = read_message(&mut server).await;
        assert!(err.is_err());
    }

    #[tokio::test]
    async fn multiple_messages() {
        let (mut client, mut server) = duplex(4096);
        let msgs = vec![
            r#"{"jsonrpc":"2.0","method":"a","id":1}"#,
            r#"{"jsonrpc":"2.0","method":"b","id":2}"#,
            r#"{"jsonrpc":"2.0","result":{},"id":1}"#,
        ];
        for msg in &msgs {
            write_message(&mut client, msg).await.unwrap();
        }
        drop(client);
        for expected in &msgs {
            let received = read_message(&mut server).await.unwrap();
            assert_eq!(received.as_deref(), Some(*expected));
        }
        assert!(read_message(&mut server).await.unwrap().is_none());
    }
}
