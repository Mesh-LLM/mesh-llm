//! Explicitly negotiated token-metered billing: a deposit against the
//! estimated input, then one final invoice for the backend-reported usage.
//! Neither side may downgrade to a retired payment protocol.

use anyhow::{Result, ensure};
use mesh_llm_wallet::invoice::Invoice;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::{metering::MeteredUsage, pricing::Pricing};

/// Maximum encoded metered frame length, checked before allocation.
pub const MAX_FRAME_BYTES: usize = 4 * 1024 * 1024;

/// Metered-stream upgrade. `/mesh/payment/v2` ([`crate::wire::HTTP_UPGRADE`])
/// is the fixed-allowance input/output protocol; its frames are not
/// interchangeable with these, so the path is bumped.
pub const HTTP_UPGRADE: &[u8] =
    b"POST /mesh/payment/v3 HTTP/1.1\r\nHost: mesh\r\nContent-Length: 0\r\n\r\n";

/// Admission terms are agreed before any backend work. The deposit is a
/// share of the *estimated* input cost; it is applied against the final bill,
/// never an extra fee, a reservation or an exact input-token claim. Output is
/// always post-paid from the backend's reported usage.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdmissionTerms {
    pub id: String,
    pub model: String,
    pub pricing: Pricing,
    pub prompt_estimate: u64,
    pub deposit_msat: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Frame {
    Request {
        id: String,
        model: String,
        pricing: Pricing,
        http: Vec<u8>,
    },
    Admission {
        terms: AdmissionTerms,
        invoice: Option<Invoice>,
    },
    /// Payer has durably authorized the terms. A deposit, if required, is
    /// independently verified by the provider wallet before backend dispatch.
    Accepted {},
    Output {
        bytes: Vec<u8>,
    },
    /// `usage: None` means the request was abandoned: no bill, no invoice.
    /// It is never a zero-token bill.
    Settlement {
        request_id: String,
        usage: Option<MeteredUsage>,
        bill_msat: Option<u64>,
        /// Received deposit applied against this bill. A deposit exceeding
        /// the bill is kept; nothing is refunded or carried forward.
        deposit_applied_msat: u64,
        invoice: Option<Invoice>,
    },
    Complete {},
    /// Replay the settlement of a finalized request after a buyer restart.
    Recover {
        id: String,
    },
    Cancel {},
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metered_frames_reject_retired_frames_and_unknown_fields() {
        assert!(HTTP_UPGRADE.starts_with(b"POST /mesh/payment/v3 HTTP/1.1\r\n"));
        assert_ne!(HTTP_UPGRADE, crate::wire::HTTP_UPGRADE);
        assert!(serde_json::from_slice::<Frame>(br#"{"type":"input_invoice"}"#).is_err());
        assert!(serde_json::from_slice::<Frame>(br#"{"type":"accepted","hidden":1}"#).is_err());
    }

    #[test]
    fn missing_usage_is_explicit_not_a_zero_bill() {
        let frame = Frame::Settlement {
            request_id: "r".into(),
            usage: None,
            bill_msat: None,
            deposit_applied_msat: 0,
            invoice: None,
        };
        let value = serde_json::to_value(frame).unwrap();
        assert!(value["usage"].is_null());
        assert!(value["bill_msat"].is_null());
        assert!(serde_json::from_value::<Frame>(value).is_ok());
    }
}
