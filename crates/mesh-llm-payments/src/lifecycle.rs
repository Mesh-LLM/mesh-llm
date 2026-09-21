//! Best-effort observations of durable payment state, not cryptographic receipts.
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::ledger::RequestTerms;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PaymentPhase {
    TermsAccepted,
    InvoiceIssued,
    InvoiceSettled,
    FinalAmount,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PaymentRole {
    Payer,
    Provider,
}

/// Deliberate allowlist: never serialize RequestTerms, Invoice or Transaction here.
#[derive(Clone, Debug, Serialize)]
pub struct PaymentEvent {
    pub version: u8,
    pub event_id: String,
    pub exchange_id: String,
    pub role: PaymentRole,
    pub phase: PaymentPhase,
    pub evidence: &'static str,
    pub terms_digest: String,
    pub segment: Option<u32>,
    pub invoice_ref: Option<String>,
    pub amount_msat: Option<u64>,
}

impl PaymentEvent {
    pub(crate) fn new(
        terms: &RequestTerms,
        role: PaymentRole,
        phase: PaymentPhase,
        segment: Option<u32>,
        invoice_ref: Option<String>,
        amount_msat: Option<u64>,
    ) -> Option<Self> {
        let exchange_id = terms.exchange_id.clone()?;
        // Fixed-order JSON tuple, domain-separated. Excludes local peer identity,
        // which has opposite meanings on buyer and seller, and the bearer ID.
        let value = serde_json::to_vec(&(
            "mesh.payment.terms.v1",
            &exchange_id,
            &terms.payee,
            &terms.model,
            terms.pricing.input_msat_per_million,
            terms.pricing.output_msat_per_million,
            terms.pricing.minimum_invoice_msat,
            terms.input_tokens,
            terms.max_output_tokens,
            terms.max_total_msat,
            terms.expires_at_ms,
        ))
        .ok()?;
        let terms_digest = hex::encode(Sha256::digest(value));
        let evidence = match (&phase, role) {
            (PaymentPhase::TermsAccepted, _) => "local_authorization",
            (PaymentPhase::InvoiceIssued, _) => "provider_claim",
            _ => "wallet_reported",
        };
        let identity = serde_json::to_vec(&(
            "mesh.payment.event.v1",
            &exchange_id,
            role,
            &phase,
            &terms_digest,
            segment,
            &invoice_ref,
            amount_msat,
        ))
        .ok()?;
        Some(Self {
            version: 1,
            event_id: hex::encode(Sha256::digest(identity)),
            exchange_id,
            role,
            phase,
            evidence,
            terms_digest,
            segment,
            invoice_ref,
            amount_msat,
        })
    }
}
