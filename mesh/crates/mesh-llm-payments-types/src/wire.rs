//! Bounded payment framing inside an authenticated mesh QUIC stream.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use mesh_llm_wallet::invoice::Invoice;

use crate::{pricing::Pricing, terms::RequestTerms};

pub const HTTP_UPGRADE: &[u8] =
    b"POST /mesh/payment/v1 HTTP/1.1\r\nHost: mesh\r\nContent-Length: 0\r\n\r\n";
pub const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Frame {
    Request {
        id: String,
        model: String,
        pricing: Pricing,
        http: Vec<u8>,
    },
    InputInvoice {
        terms: RequestTerms,
        invoice: Invoice,
    },
    Output {
        bytes: Vec<u8>,
    },
    OutputInvoice {
        request_id: String,
        tokens: u64,
        invoice: Invoice,
    },
    Cancel,
    Complete,
    Pending,
    Recover {
        id: String,
    },
    Error {
        message: String,
    },
}

pub async fn write(writer: &mut (impl AsyncWrite + Unpin), frame: &Frame) -> Result<()> {
    let bytes = serde_json::to_vec(frame)?;
    ensure!(bytes.len() <= MAX_FRAME_BYTES, "payment frame too large");
    writer.write_u32(bytes.len() as u32).await?;
    writer.write_all(&bytes).await?;
    writer.flush().await?;
    Ok(())
}

pub async fn read(reader: &mut (impl AsyncRead + Unpin)) -> Result<Frame> {
    let length = reader.read_u32().await? as usize;
    ensure!(length <= MAX_FRAME_BYTES, "payment frame too large");
    let mut bytes = vec![0; length];
    reader.read_exact(&mut bytes).await?;
    Ok(serde_json::from_slice(&bytes)?)
}
