//! Metered paid-stream protocol (`/mesh/payment/v3`): a prompt-estimate
//! deposit before the backend is invoked, and one final bill from actual
//! input + output usage that credits the deposit.
//!
//! It is a separate upgrade path with its own frame type, so the strict v2
//! [`crate::wire::Frame`] keeps its shape for older peers, and a
//! prompt-inclusive charge is never carried in a v2 output-only invoice.

use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

use mesh_llm_wallet::invoice::Invoice;

use crate::{metering::MeteredUsage, pricing::Pricing, settlement::FinalBill};

pub const HTTP_UPGRADE: &[u8] =
    b"POST /mesh/payment/v3 HTTP/1.1\r\nHost: mesh\r\nContent-Length: 0\r\n\r\n";
pub const UPGRADE_PREFIX: &[u8] = b"POST /mesh/payment/v3 HTTP/1.1\r\n";

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum MeteredFrame {
    Request {
        id: String,
        model: String,
        pricing: Pricing,
        max_output_tokens: u64,
        http: Vec<u8>,
    },
    /// Cold buyer: pay before the backend is invoked. Priced from the
    /// prompt estimate at the input rate.
    Deposit {
        estimated_input_tokens: u64,
        invoice: Invoice,
    },
    /// Repeat buyer admitted without a deposit.
    Admitted,
    Output {
        bytes: Vec<u8>,
    },
    /// Actual usage, priced, less the confirmed deposit. `invoice` is absent
    /// when the deposit covers the bill (no refund is implied).
    FinalBill {
        request_id: String,
        bill: FinalBill,
        invoice: Option<Invoice>,
    },
    Cancel,
    Complete,
    Error {
        message: String,
    },
}

impl MeteredFrame {
    pub async fn write(&self, writer: &mut (impl tokio::io::AsyncWrite + Unpin)) -> Result<()> {
        write_json(writer, self).await
    }

    pub async fn read(reader: &mut (impl tokio::io::AsyncRead + Unpin)) -> Result<Self> {
        use tokio::io::AsyncReadExt;
        let length = reader.read_u32().await? as usize;
        ensure!(
            length <= crate::wire::MAX_FRAME_BYTES,
            "payment frame too large"
        );
        let mut bytes = vec![0; length];
        reader.read_exact(&mut bytes).await?;
        Ok(serde_json::from_slice(&bytes)?)
    }
}

async fn write_json(
    writer: &mut (impl tokio::io::AsyncWrite + Unpin),
    frame: &MeteredFrame,
) -> Result<()> {
    use tokio::io::AsyncWriteExt;
    let bytes = serde_json::to_vec(frame)?;
    ensure!(
        bytes.len() <= crate::wire::MAX_FRAME_BYTES,
        "payment frame too large"
    );
    writer.write_u32(bytes.len() as u32).await?;
    writer.write_all(&bytes).await?;
    writer.flush().await?;
    Ok(())
}

/// Payer check of a deposit: at the agreed input rate for the stated
/// estimate, and within the payer's own estimate tolerance.
pub fn validate_deposit(
    pricing: &Pricing,
    estimated_input_tokens: u64,
    payer_estimate: u64,
    invoice: &Invoice,
) -> Result<u64> {
    // The provider may estimate differently (e.g. media); allow it to ask for
    // at most twice the payer's own estimate plus a small floor.
    let ceiling = payer_estimate.saturating_mul(2).saturating_add(64);
    ensure!(
        estimated_input_tokens <= ceiling,
        "deposit estimate exceeds the payer's ceiling"
    );
    let amount = pricing.input_charge(estimated_input_tokens)?;
    ensure!(
        invoice.amount_msat == Some(amount),
        "deposit invoice does not match the estimate"
    );
    Ok(amount)
}

/// Payer check of a final bill: arithmetic re-derived from the agreed
/// pricing, deposit equal to what this payer actually paid, output within
/// the allowance, and the invoice exactly the remaining amount.
pub fn validate_final_bill(
    pricing: &Pricing,
    max_output_tokens: u64,
    deposit_paid_msat: u64,
    bill: &FinalBill,
    invoice: Option<&Invoice>,
) -> Result<()> {
    ensure!(
        bill.usage.completion_tokens <= max_output_tokens,
        "billed output exceeds the allowance"
    );
    ensure!(
        bill.deposit_msat == deposit_paid_msat,
        "final bill credits a different deposit"
    );
    let expected = FinalBill::reconcile(pricing, bill.usage, deposit_paid_msat)?;
    ensure!(expected == *bill, "final bill arithmetic mismatch");
    match invoice {
        None => ensure!(bill.remaining_msat == 0, "final bill without an invoice"),
        Some(invoice) => {
            ensure!(bill.remaining_msat > 0, "invoice for a covered bill");
            ensure!(
                invoice.amount_msat == Some(bill.remaining_msat),
                "final invoice does not match the remaining amount"
            );
        }
    }
    Ok(())
}

/// Usage priced by the provider: the inclusive counters it reports.
pub fn usage(prompt_tokens: u64, completion_tokens: u64) -> MeteredUsage {
    MeteredUsage {
        prompt_tokens,
        completion_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pricing() -> Pricing {
        Pricing {
            input_msat_per_million: 1_000_000,
            output_msat_per_million: 2_000_000,
        }
    }

    fn invoice(amount: u64) -> Invoice {
        Invoice {
            bolt11: String::new(),
            payment_hash: "h".into(),
            payee: "p".into(),
            amount_msat: Some(amount),
            expires_at_ms: 0,
        }
    }

    #[test]
    fn v3_upgrade_is_distinct_from_v2() {
        assert!(HTTP_UPGRADE.starts_with(UPGRADE_PREFIX));
        assert!(!HTTP_UPGRADE.starts_with(b"POST /mesh/payment/v2"));
    }

    #[test]
    fn strict_v2_decoder_rejects_v3_frames() {
        let bytes = serde_json::to_vec(&MeteredFrame::Admitted).unwrap();
        assert!(serde_json::from_slice::<crate::wire::Frame>(&bytes).is_err());
    }

    #[test]
    fn deposit_must_match_estimate_and_stay_within_payer_ceiling() {
        assert_eq!(
            validate_deposit(&pricing(), 40, 40, &invoice(40)).unwrap(),
            40
        );
        assert!(validate_deposit(&pricing(), 40, 40, &invoice(41)).is_err());
        assert!(validate_deposit(&pricing(), 1_000, 40, &invoice(1_000)).is_err());
    }

    #[test]
    fn final_bill_is_rederived_and_matches_the_invoice() {
        let bill = FinalBill::reconcile(&pricing(), usage(33, 10), 7).unwrap();
        validate_final_bill(&pricing(), 16, 7, &bill, Some(&invoice(46))).unwrap();
        // Wrong invoice amount, wrong deposit, inflated output, missing invoice.
        assert!(validate_final_bill(&pricing(), 16, 7, &bill, Some(&invoice(47))).is_err());
        assert!(validate_final_bill(&pricing(), 16, 0, &bill, Some(&invoice(46))).is_err());
        assert!(validate_final_bill(&pricing(), 9, 7, &bill, Some(&invoice(46))).is_err());
        assert!(validate_final_bill(&pricing(), 16, 7, &bill, None).is_err());
        let mut forged = bill;
        forged.total_msat += 1;
        forged.remaining_msat += 1;
        assert!(validate_final_bill(&pricing(), 16, 7, &forged, Some(&invoice(47))).is_err());
    }

    #[test]
    fn covered_bill_has_no_invoice() {
        let bill = FinalBill::reconcile(&pricing(), usage(3, 1), 10).unwrap();
        validate_final_bill(&pricing(), 16, 10, &bill, None).unwrap();
        assert!(validate_final_bill(&pricing(), 16, 10, &bill, Some(&invoice(0))).is_err());
    }
}
