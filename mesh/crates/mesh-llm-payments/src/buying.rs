//! Buyer-side coarse operations behind `payments.v1`: token-metered billing.
//! The host keeps the transport and the response; these own every ledger and
//! wallet step of accepting an admission, paying its deposit, checking the
//! seller's settlement and paying the final invoice. Amounts are derived from
//! the stored admission, never taken from the caller.

use anyhow::{Context, Result, ensure};
use mesh_llm_payments_types::contract::{
    AcceptAdmissionRequest, PayDepositRequest, SettleFinalRequest, SettleFinalResponse,
};
use mesh_llm_payments_types::metered_wire::AdmissionTerms;
use mesh_llm_payments_types::metering::{DEPOSIT_MSAT, final_invoice_msat};

use crate::invoice::Invoice;
use crate::ledger::Charge;
use crate::pricing::payment_cap_msat;
use crate::service::PaymentService;
use crate::wallet::{PaymentStatus, Transaction};

fn admission_key(id: &str) -> String {
    format!("metered-payer/{id}")
}

fn payee_key(id: &str) -> String {
    format!("metered-payee/{id}")
}

impl PaymentService {
    /// Durably approve a seller's admission. `terms` carry the buyer's local
    /// cap and are approved against policy and the daily budget; `admission` is pinned so every later payment of
    /// the request is checked against what the seller stated here.
    pub async fn accept_admission(&self, request: AcceptAdmissionRequest) -> Result<()> {
        let AcceptAdmissionRequest { terms, admission } = request;
        ensure!(
            admission.id == terms.id
                && admission.model == terms.model
                && admission.pricing == terms.pricing
                && admission.prompt_estimate == terms.input_tokens,
            "admission does not match the proposed terms"
        );
        ensure!(
            admission.deposit_msat == 0 || admission.deposit_msat == DEPOSIT_MSAT,
            "deposit mismatch"
        );
        self.authorize_terms(&terms).await?;
        self.save_admission(&admission)
    }

    /// Pay the deposit of an accepted admission. The amount is the admission's
    /// deposit; the receiving wallet is pinned to this invoice's payee.
    pub async fn pay_deposit(&self, request: PayDepositRequest) -> Result<Transaction> {
        let PayDepositRequest {
            request_id,
            invoice,
        } = request;
        let admission = self
            .admission(&request_id)?
            .context("unknown metered admission")?;
        ensure!(admission.deposit_msat > 0, "admission required no deposit");
        self.pay_metered(&request_id, 0, admission.deposit_msat, invoice)
            .await
    }

    /// Check a seller's settlement against the admission and pay the final
    /// invoice. An abandoned settlement (no usage) creates no charge, and a
    /// bill the deposit covered needs no invoice.
    pub async fn settle_final(&self, request: SettleFinalRequest) -> Result<SettleFinalResponse> {
        let SettleFinalRequest {
            request_id,
            usage,
            bill_msat,
            deposit_applied_msat,
            invoice,
        } = request;
        let admission = self
            .admission(&request_id)?
            .context("unknown metered admission")?;
        let Some(usage) = usage else {
            ensure!(
                bill_msat.is_none() && invoice.is_none() && deposit_applied_msat == 0,
                "abandoned request cannot create a charge"
            );
            return Ok(SettleFinalResponse::default());
        };
        let terms = self
            .ledger
            .request_terms(&request_id)?
            .context("unknown metered request")?;
        let bill = usage.bill_msat(&admission.pricing)?;
        ensure!(
            bill_msat == Some(bill),
            "bill does not match the admitted rates"
        );
        // The buyer's own cap, approved against policy and balance; the
        // seller never saw it.
        ensure!(
            bill <= terms.max_total_msat,
            "bill exceeds the approved cap"
        );
        ensure!(
            deposit_applied_msat <= admission.deposit_msat && deposit_applied_msat <= bill,
            "invalid deposit application"
        );
        // A remainder below the payment floor is waived by the seller.
        let remaining = final_invoice_msat(bill, deposit_applied_msat);
        let payment = match invoice {
            Some(invoice) => {
                ensure!(remaining > 0, "final invoice for a covered bill");
                Some(self.pay_metered(&request_id, 1, remaining, invoice).await?)
            }
            None => {
                ensure!(remaining == 0, "final invoice missing");
                None
            }
        };
        Ok(SettleFinalResponse {
            bill_msat: bill,
            payment,
        })
    }

    /// Close a metered request on the seller's authoritative Complete: a
    /// deposit that was never submitted is retired, then the request finishes.
    pub async fn finish_metered(&self, id: &str) -> Result<()> {
        let _guard = self.payment_lock.lock().await;
        if self.admission(id)?.is_some() {
            self.ledger.retire_unsubmitted_metered_deposit(id)?;
        }
        self.ledger.finish(id)
    }

    /// Whether `id` is a metered buyer request, whose charges recovery only
    /// observes: a deposit or final that was never submitted is never
    /// resubmitted without the seller's settlement in hand.
    pub(crate) fn is_metered_buyer_request(&self, id: &str) -> Result<bool> {
        Ok(self.admission(id)?.is_some())
    }

    /// Observe wallet facts for one metered charge; never resubmit, and never
    /// infer failure from invoice expiry.
    pub(crate) async fn observe_metered_charge(&self, charge: &Charge) -> Result<()> {
        let _guard = self.payment_lock.lock().await;
        if self
            .ledger
            .charge_state(&charge.invoice.payment_hash)?
            .as_deref()
            == Some("prepared")
        {
            return Ok(());
        }
        if let Some(payment) = self
            .wallet()
            .await?
            .lookup(&charge.invoice.payment_hash)
            .await?
        {
            crate::service::validate_payment_update(&payment, &charge.invoice.payment_hash, false)?;
            self.ledger.reconcile(&payment, crate::now_ms())?;
        }
        Ok(())
    }

    async fn pay_metered(
        &self,
        id: &str,
        segment: u32,
        amount: u64,
        invoice: Invoice,
    ) -> Result<Transaction> {
        ensure!(
            invoice.amount_msat == Some(amount),
            "invoice amount does not match the admission"
        );
        invoice.validate_payment(amount, crate::now_ms())?;
        self.pin_payee(id, &invoice.payee).await?;
        let payment = self
            .pay_charge(&Charge {
                request_id: id.to_owned(),
                segment,
                invoice,
                amount_msat: amount,
                max_total_msat: payment_cap_msat(amount)?,
            })
            .await?;
        ensure!(
            payment.status == PaymentStatus::Succeeded,
            "payment unresolved or failed"
        );
        Ok(payment)
    }

    /// Pin the receiving wallet to the first invoice's payee, so a final
    /// invoice cannot redirect the remainder to another wallet.
    async fn pin_payee(&self, id: &str, payee: &str) -> Result<()> {
        let _guard = self.payment_lock.lock().await;
        let key = payee_key(id);
        match self.ledger.get_setting(&key)? {
            Some(existing) => ensure!(existing == payee, "receiving wallet changed"),
            None => self.ledger.set_setting(&key, payee)?,
        }
        Ok(())
    }

    fn save_admission(&self, admission: &AdmissionTerms) -> Result<()> {
        let key = admission_key(&admission.id);
        let value = serde_json::to_string(admission)?;
        match self.ledger.get_setting(&key)? {
            Some(existing) => ensure!(existing == value, "admission changed"),
            None => self.ledger.set_setting(&key, &value)?,
        }
        Ok(())
    }

    fn admission(&self, id: &str) -> Result<Option<AdmissionTerms>> {
        self.ledger
            .get_setting(&admission_key(id))?
            .map(|value| serde_json::from_str(&value).map_err(Into::into))
            .transpose()
    }
}
