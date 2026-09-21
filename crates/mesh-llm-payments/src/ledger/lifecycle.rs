//! Project durable state onto a bounded, non-blocking observation channel.
use anyhow::Result;
use rusqlite::OptionalExtension;

use super::{Ledger, RequestTerms, read_amount};
use crate::lifecycle::{PaymentEvent, PaymentPhase, PaymentRole};

impl Ledger {
    pub fn subscribe_events(&self) -> tokio::sync::broadcast::Receiver<PaymentEvent> {
        self.events.subscribe()
    }

    /// Persist provider terms before issuing its input invoice. No bearer ID is
    /// exposed by the event projection. Older records without correlation emit none.
    pub fn record_serving_terms(&self, terms: &RequestTerms) -> Result<()> {
        self.lock()?.execute(
            "INSERT INTO serving_terms(id,terms) VALUES (?1,?2)",
            rusqlite::params![terms.id, serde_json::to_string(terms)?],
        )?;
        Ok(())
    }

    pub(super) fn observe_payment(&self, id: &str) {
        if self.events.receiver_count() == 0 {
            return;
        }
        // Financial state already committed. Observation failures cannot roll it back.
        if let Ok(events) = self.payment_observations(id) {
            for event in events {
                let _ = self.events.send(event);
            }
        }
    }

    /// Snapshot projection permits duplicate observations with stable event IDs.
    /// This is not an outbox and does not promise replay of every transition.
    pub fn payment_observations(&self, id: &str) -> Result<Vec<PaymentEvent>> {
        let connection = self.lock()?;
        let payer: Option<(String, String, u64)> = connection
            .query_row(
                "SELECT terms,state,spent FROM requests WHERE id=?",
                [id],
                |r| Ok((r.get(0)?, r.get(1)?, read_amount(r, 2)?)),
            )
            .optional()?;
        let mut events = Vec::new();
        if let Some((json, state, spent)) = payer {
            let terms: RequestTerms = serde_json::from_str(&json)?;
            if matches!(state.as_str(), "approved" | "completed") {
                events.extend(PaymentEvent::new(
                    &terms,
                    PaymentRole::Payer,
                    PaymentPhase::TermsAccepted,
                    None,
                    None,
                    None,
                ));
            }
            let mut statement = connection.prepare(
                "SELECT segment,hash,amount,state,total FROM charges WHERE request_id=? ORDER BY segment"
            )?;
            let rows = statement.query_map([id], |r| {
                Ok((
                    r.get::<_, u32>(0)?,
                    r.get::<_, String>(1)?,
                    read_amount(r, 2)?,
                    r.get::<_, String>(3)?,
                    read_amount(r, 4)?,
                ))
            })?;
            for row in rows {
                let (segment, hash, amount, status, total) = row?;
                events.extend(PaymentEvent::new(
                    &terms,
                    PaymentRole::Payer,
                    PaymentPhase::InvoiceIssued,
                    Some(segment),
                    Some(hash.clone()),
                    Some(amount),
                ));
                if status == "succeeded" {
                    events.extend(PaymentEvent::new(
                        &terms,
                        PaymentRole::Payer,
                        PaymentPhase::InvoiceSettled,
                        Some(segment),
                        Some(hash),
                        Some(total),
                    ));
                }
            }
            if state == "completed" {
                events.extend(PaymentEvent::new(
                    &terms,
                    PaymentRole::Payer,
                    PaymentPhase::FinalAmount,
                    None,
                    None,
                    Some(spent),
                ));
            }
        }
        drop(connection);
        events.extend(self.provider_observations(id)?);
        Ok(events)
    }

    fn provider_observations(&self, id: &str) -> Result<Vec<PaymentEvent>> {
        let connection = self.lock()?;
        let mut events = Vec::new();
        let json: Option<String> = connection
            .query_row("SELECT terms FROM serving_terms WHERE id=?", [id], |r| {
                r.get(0)
            })
            .optional()?;
        drop(connection);
        if let Some(json) = json {
            let terms: RequestTerms = serde_json::from_str(&json)?;
            let receipts = self.receivables(Some(id))?;
            for receipt in &receipts {
                events.extend(PaymentEvent::new(
                    &terms,
                    PaymentRole::Provider,
                    PaymentPhase::InvoiceIssued,
                    Some(receipt.segment),
                    Some(receipt.invoice.payment_hash.clone()),
                    receipt.invoice.amount_msat,
                ));
                if receipt.paid {
                    events.extend(PaymentEvent::new(
                        &terms,
                        PaymentRole::Provider,
                        PaymentPhase::InvoiceSettled,
                        Some(receipt.segment),
                        Some(receipt.invoice.payment_hash.clone()),
                        receipt.invoice.amount_msat,
                    ));
                }
            }
            let (_, _, tokens, finished) = self.serving_account(id)?;
            if finished
                && receipts.iter().any(|r| r.segment == 0 && r.paid)
                && (tokens == 0 || receipts.iter().any(|r| r.segment == 1 && r.paid))
            {
                let total = receipts
                    .iter()
                    .filter_map(|r| r.invoice.amount_msat)
                    .try_fold(0u64, u64::checked_add);
                events.extend(PaymentEvent::new(
                    &terms,
                    PaymentRole::Provider,
                    PaymentPhase::FinalAmount,
                    None,
                    None,
                    total,
                ));
            }
        }
        Ok(events)
    }
}
