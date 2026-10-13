//! Two-stage metered invoice orchestration; no inference or delivery gates.
use anyhow::{Context, Result, ensure};
use mesh_llm_payments_types::metered_wire::AdmissionTerms;

use super::PaymentService;
use crate::{
    invoice::Invoice,
    ledger::metered_invoices::{MeteredInvoice, MeteredInvoiceKind},
    wallet::PaymentStatus,
};

impl PaymentService {
    /// Submit new v2 charges once; an uncertain existing submission is only
    /// observed, never resubmitted or failed based on expiry.
    pub async fn metered_pay_charge(
        &self,
        charge: &crate::ledger::Charge,
    ) -> Result<crate::wallet::Transaction> {
        let _guard = self.payment_lock.lock().await;
        ensure!(
            self.metered_authorization(&charge.request_id)?.is_some(),
            "unknown metered authority"
        );
        self.ledger.prepare_charge(charge)?;
        let state = self
            .ledger
            .charge_state(&charge.invoice.payment_hash)?
            .context("charge missing")?;
        ensure!(
            matches!(state.as_str(), "prepared" | "pending" | "succeeded"),
            "charge is terminal"
        );
        let wallet = self.wallet().await?;
        let mut payment = if state == "prepared" {
            self.ledger.begin_submission(&charge.invoice.payment_hash)?;
            match self.submit(wallet, charge).await {
                Err(crate::wallet::PayError::NotSubmitted(error)) => {
                    self.ledger.fail_unsubmitted(&charge.invoice.payment_hash)?;
                    return Err(error);
                }
                result => result?,
            }
        } else {
            wallet
                .lookup(&charge.invoice.payment_hash)
                .await?
                .context("submitted payment unresolved")?
        };
        super::validate_payment_update(&payment, &charge.invoice.payment_hash, false)?;
        drop(_guard);
        if payment.status == PaymentStatus::Pending {
            payment = wallet
                .wait_for_payment(&charge.invoice.payment_hash)
                .await?;
        }
        super::validate_payment_update(&payment, &charge.invoice.payment_hash, false)?;
        self.ledger.reconcile(&payment, crate::now_ms())?;
        ensure!(
            payment.status == PaymentStatus::Succeeded,
            "payment unresolved or failed"
        );
        Ok(payment)
    }

    /// Pin the first invoice wallet, including a trusted no-deposit request.
    pub async fn metered_pin_payee(&self, id: &str, payee: &str) -> Result<()> {
        let _guard = self.payment_lock.lock().await;
        ensure!(
            self.metered_authorization(id)?.is_some(),
            "unknown metered authority"
        );
        let key = format!("metered-payee/{id}");
        if let Some(existing) = self.ledger.get_setting(&key)? {
            ensure!(existing == payee, "receiving wallet changed");
        } else {
            self.ledger.set_setting(&key, payee)?;
        }
        Ok(())
    }

    /// Serialize authoritative closure with the deposit submission boundary.
    pub async fn metered_complete(&self, id: &str) -> Result<()> {
        let _guard = self.payment_lock.lock().await;
        self.ledger.retire_unsubmitted_metered_deposit(id)?;
        self.ledger.finish(id)
    }

    /// Recovery observes v2 wallet facts; never resubmit or infer failure from expiry.
    pub(super) async fn metered_reconcile_charge(
        &self,
        charge: &crate::ledger::Charge,
    ) -> Result<()> {
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
            super::validate_payment_update(&payment, &charge.invoice.payment_hash, false)?;
            self.ledger.reconcile(&payment, crate::now_ms())?;
        }
        Ok(())
    }

    /// Provider-owned, disposable trust. Money and exposure remain durable.
    pub async fn metered_trusted(&self, peer: &str) -> Result<bool> {
        Ok(self.metered_standing.lock().await.contains(peer)
            && !self
                .ledger
                .metered_invoices(None)?
                .iter()
                .any(|r| r.peer == peer && !r.paid && r.invoice.expires_at_ms <= crate::now_ms()))
    }

    pub async fn metered_note_settled(&self, id: &str, peer: &str) -> Result<()> {
        if self.ledger.metered_settlement(id)?.status
            == crate::ledger::metered::MeteredStatus::Settled
        {
            let mut standing = self.metered_standing.lock().await;
            // Trust is an optimization, not a durable registry or unlimited cache.
            if standing.len() >= 1024 {
                standing.clear();
            }
            standing.insert(peer.to_owned());
        }
        Ok(())
    }

    /// Serialize invoice creation and reuse the exact persisted invoice on retry.
    pub async fn metered_invoice(
        &self,
        id: &str,
        peer: &str,
        kind: MeteredInvoiceKind,
        amount: u64,
    ) -> Result<Option<Invoice>> {
        if amount == 0 {
            return Ok(None);
        }
        let _guard = self.receivable_lock.lock().await;
        if let Some(receipt) = self
            .ledger
            .metered_invoices(Some(id))?
            .into_iter()
            .find(|r| r.kind == kind)
        {
            ensure!(
                receipt.peer == peer && receipt.invoice.amount_msat == Some(amount),
                "invoice terms changed"
            );
            return Ok(Some(receipt.invoice));
        }
        let expiry = if kind == MeteredInvoiceKind::Deposit {
            crate::lifetimes::INPUT_INVOICE_EXPIRY_SECS
        } else {
            crate::lifetimes::OUTPUT_INVOICE_EXPIRY_SECS
        };
        let invoice = self
            .wallet()
            .await?
            .create_invoice(Some(amount), expiry)
            .await?;
        invoice.validate_payment(amount, crate::now_ms())?;
        ensure!(
            invoice.amount_msat == Some(amount),
            "wallet changed invoice amount"
        );
        self.ledger.record_metered_invoice(&MeteredInvoice {
            request_id: id.into(),
            peer: peer.into(),
            kind,
            invoice: invoice.clone(),
            paid: false,
        })?;
        Ok(Some(invoice))
    }

    /// Reuse the persisted final invoice, or create one for the billed
    /// remainder. The remainder is fixed at finalize, so retries never change
    /// the amount or create a second invoice.
    pub async fn metered_final_invoice(&self, id: &str, peer: &str) -> Result<Option<Invoice>> {
        let owner = self
            .ledger
            .metered_owner(id)?
            .context("unknown metered request")?;
        ensure!(owner == peer, "metered invoice peer mismatch");
        if let Some(receipt) = self
            .ledger
            .metered_invoices(Some(id))?
            .into_iter()
            .find(|r| r.kind == MeteredInvoiceKind::Final)
        {
            return Ok(Some(receipt.invoice));
        }
        let settlement = self.ledger.metered_settlement(id)?;
        if settlement.status != crate::ledger::metered::MeteredStatus::Billed {
            return Ok(None);
        }
        self.metered_invoice(
            id,
            peer,
            MeteredInvoiceKind::Final,
            settlement.remaining_msat,
        )
        .await
    }

    /// Refresh incoming facts only; never invoke legacy input/output recovery.
    pub async fn metered_refresh(&self, id: &str) -> Result<()> {
        for receipt in self.ledger.metered_invoices(Some(id))? {
            if !receipt.paid
                && let Some(payment) = self
                    .wallet()
                    .await?
                    .lookup(&receipt.invoice.payment_hash)
                    .await?
            {
                super::validate_payment_update(&payment, &receipt.invoice.payment_hash, true)?;
                if payment.status == PaymentStatus::Succeeded {
                    ensure!(
                        payment.amount_msat
                            >= receipt.invoice.amount_msat.context("amountless invoice")?,
                        "incoming underpayment"
                    );
                    self.ledger
                        .mark_metered_received(&receipt.invoice.payment_hash)?;
                }
            }
        }
        Ok(())
    }

    /// Bind recovery to the original v2 admission, separately from v1 terms.
    pub fn save_metered_authorization(&self, terms: &AdmissionTerms) -> Result<()> {
        let key = format!("metered-payer/{}", terms.id);
        let value = serde_json::to_string(terms)?;
        if let Some(existing) = self.ledger.get_setting(&key)? {
            ensure!(existing == value, "metered authorization changed");
        } else {
            self.ledger.set_setting(&key, &value)?;
        }
        Ok(())
    }

    pub fn metered_authorization(&self, id: &str) -> Result<Option<AdmissionTerms>> {
        self.ledger
            .get_setting(&format!("metered-payer/{id}"))?
            .map(|value| serde_json::from_str(&value).map_err(Into::into))
            .transpose()
    }
}
