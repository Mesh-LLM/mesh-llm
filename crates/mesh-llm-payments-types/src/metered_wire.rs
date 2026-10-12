//! Explicitly negotiated request-in/response-out billing. Payment v1 retains
//! its old semantics; neither side may downgrade after accepting these terms.

use anyhow::{Result, ensure};
use mesh_llm_wallet::invoice::Invoice;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use crate::{metering::MeteredUsage, pricing::Pricing, wire::MAX_FRAME_BYTES};

pub const HTTP_UPGRADE: &[u8] =
    b"POST /mesh/payment/v2 HTTP/1.1\r\nHost: mesh\r\nContent-Length: 0\r\n\r\n";

/// Admission terms are agreed before any backend work. `max_total_msat`
/// includes wallet routing fees, not just the backend bill. The deposit is
/// credited, never an extra fee or an exact input-token claim.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AdmissionTerms {
    pub id: String,
    pub model: String,
    pub pricing: Pricing,
    pub prompt_estimate: u64,
    pub max_output_tokens: u64,
    pub max_total_msat: u64,
    pub deposit_msat: u64,
    pub credit_msat: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Frame {
    Request {
        id: String,
        model: String,
        pricing: Pricing,
        max_total_msat: u64,
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
    Settlement {
        request_id: String,
        usage: Option<MeteredUsage>,
        bill_msat: Option<u64>,
        credit_applied_msat: u64,
        credit_remaining_msat: u64,
        invoice: Option<Invoice>,
    },
    Complete {},
    Pending {},
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
    fn version_two_is_not_a_silent_version_one_reinterpretation() {
        assert_ne!(HTTP_UPGRADE, crate::wire::HTTP_UPGRADE);
        let bytes = serde_json::to_vec(&Frame::Accepted {}).unwrap();
        assert!(serde_json::from_slice::<crate::wire::Frame>(&bytes).is_err());
        assert!(serde_json::from_slice::<Frame>(br#"{"type":"accepted","hidden":1}"#).is_err());
    }

    #[test]
    fn missing_usage_is_explicit_not_a_zero_bill() {
        let frame = Frame::Settlement {
            request_id: "r".into(),
            usage: None,
            bill_msat: None,
            credit_applied_msat: 0,
            credit_remaining_msat: 10,
            invoice: None,
        };
        let value = serde_json::to_value(frame).unwrap();
        assert!(value["usage"].is_null());
        assert!(value["bill_msat"].is_null());
        assert!(serde_json::from_value::<Frame>(value).is_ok());
    }
}
