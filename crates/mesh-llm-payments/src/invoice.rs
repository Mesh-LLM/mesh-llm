use anyhow::{Context, Result, ensure};
use lightning_invoice::{Bolt11Invoice, Currency};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Serializable invoice information. Always reparse `bolt11` at a trust boundary;
/// peer-supplied hashes, amounts, and expiry fields are not authoritative.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Invoice {
    pub bolt11: String,
    pub payment_hash: String,
    pub payee: String,
    pub amount_msat: Option<u64>,
    pub expires_at_ms: u64,
}

impl Invoice {
    pub fn parse(bolt11: &str) -> Result<Self> {
        let parsed: Bolt11Invoice = bolt11.parse().context("invalid BOLT11 invoice")?;
        ensure!(
            parsed.currency() == Currency::Bitcoin,
            "mainnet invoice required"
        );
        let expires_at_ms = parsed
            .expires_at()
            .context("invoice expiry overflow")?
            .as_millis()
            .try_into()
            .context("invoice expiry overflow")?;
        Ok(Self {
            bolt11: bolt11.to_owned(),
            payment_hash: parsed.payment_hash().to_string(),
            payee: parsed.recover_payee_pub_key().to_string(),
            amount_msat: parsed.amount_milli_satoshis(),
            expires_at_ms,
        })
    }

    pub fn validate_payment(&self, amount_msat: u64, now_ms: u64) -> Result<()> {
        ensure!(
            *self == Self::parse(&self.bolt11)?,
            "invoice metadata mismatch"
        );
        ensure!(self.expires_at_ms > now_ms, "invoice has expired");
        ensure!(amount_msat > 0, "payment amount must be positive");
        ensure!(
            self.amount_msat.is_none_or(|amount| amount == amount_msat),
            "invoice amount mismatch"
        );
        Ok(())
    }

    pub fn verifies_preimage(&self, preimage: &[u8; 32]) -> bool {
        hex::encode(Sha256::digest(preimage)) == self.payment_hash
    }
}
