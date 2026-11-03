//! Recovery reconciles existing obligations; only an operator can authorize resubmission.
use super::*;

impl Engine {
    pub async fn retry(&self) -> Result<()> {
        for exchange in self.ledger.active()? {
            if exchange.seller {
                self.retry_seller(exchange).await?;
            } else {
                self.retry_buyer(exchange).await?;
            }
        }
        Ok(())
    }
    async fn retry_seller(&self, exchange: Exchange) -> Result<()> {
        if let Some((input, output)) = exchange.usage {
            return self
                .send(
                    &exchange.peer,
                    &Message::Bill {
                        id: exchange.id,
                        input,
                        output,
                        payment: exchange.usage_payment,
                    },
                )
                .await;
        }
        if exchange
            .quote
            .as_ref()
            .is_some_and(|q| q.expires_at <= now())
            && matches!(exchange.state.as_str(), "quoted" | "ready")
        {
            return self.ledger.update(&exchange.id, |x| {
                if matches!(x.state.as_str(), "quoted" | "ready")
                    && x.exchange_id.is_none()
                    && x.quote.as_ref().is_some_and(|q| q.expires_at <= now())
                {
                    x.state = "cancelled".into();
                }
                Ok(())
            });
        }
        if exchange.state == "ready" {
            return self
                .send(&exchange.peer, &Message::Ready { id: exchange.id })
                .await;
        }
        if exchange.state == "quoted"
            && let Some(quote) = exchange.quote
        {
            return self.send(&exchange.peer, &Message::Quote { quote }).await;
        }
        Ok(())
    }
    async fn retry_buyer(&self, exchange: Exchange) -> Result<()> {
        if let Some(receipt) = exchange.usage_receipt {
            return self
                .send(
                    &exchange.peer,
                    &Message::UsageProof {
                        id: exchange.id,
                        tx_hash: receipt.tx_hash,
                    },
                )
                .await;
        }
        if let Some(payment) = exchange.usage_payment.as_ref() {
            if self.ledger.budget_reserved(&payment.obligation.id)? {
                let receipt = self.recover_payment(payment).await?;
                return self.store_proof(&exchange, payment, receipt).await;
            }
            return Ok(());
        }
        if let Some(receipt) = exchange.admission_receipt {
            return self
                .send(
                    &exchange.peer,
                    &Message::AdmissionProof {
                        id: exchange.id,
                        tx_hash: receipt.tx_hash,
                    },
                )
                .await;
        }
        let payment = exchange.quote.as_ref().and_then(|q| q.admission.as_ref());
        if let Some(payment) = payment
            && self.ledger.budget_reserved(&payment.obligation.id)?
        {
            let receipt = self.recover_payment(payment).await?;
            return self.store_proof(&exchange, payment, receipt).await;
        }
        if exchange.state == "negotiating" && exchange.quote.is_none() {
            return self
                .send(
                    &exchange.peer,
                    &Message::Hello {
                        id: exchange.id,
                        model: exchange.model,
                        payer: exchange.payer,
                    },
                )
                .await;
        }
        Ok(())
    }
    async fn store_proof(
        &self,
        exchange: &Exchange,
        payment: &PaymentRequest,
        receipt: Receipt,
    ) -> Result<()> {
        let usage = payment.obligation.purpose == Purpose::Usage;
        self.ledger.update(&exchange.id, |x| {
            if usage {
                x.usage_receipt = Some(receipt.clone());
                if x.state != "settled" {
                    x.state = "paid".into();
                }
            } else {
                x.admission_receipt = Some(receipt.clone());
            }
            Ok(())
        })?;
        let message = if usage {
            Message::UsageProof {
                id: exchange.id.clone(),
                tx_hash: receipt.tx_hash,
            }
        } else {
            Message::AdmissionProof {
                id: exchange.id.clone(),
                tx_hash: receipt.tx_hash,
            }
        };
        self.send(&exchange.peer, &message).await
    }
    /// Explicit local operator action; automatic recovery never reaches PAY.
    pub async fn resubmit(&self, id: &str) -> Result<()> {
        let exchange = self.ledger.get(id)?;
        ensure!(!exchange.seller, "only the payer can resubmit a transfer");
        let payment = exchange
            .usage_payment
            .as_ref()
            .or_else(|| exchange.quote.as_ref().and_then(|q| q.admission.as_ref()))
            .context("request has no payable obligation")?;
        let authorized = self.authorize_payment(id, payment)?;
        let lookup = self
            .wallet
            .call::<_, PaymentAttempt>(
                ops::LOOKUP,
                LookupRequest {
                    payment_id: payment.obligation.id.clone(),
                },
            )
            .await;
        let attempt = match lookup {
            Ok(attempt) => {
                ensure!(
                    attempt.request == *payment,
                    "wallet attempt changed immutable payment terms"
                );
                attempt
            }
            Err(error) => {
                tracing::warn!(
                    "operator authorized same-intent resubmission after lookup failed: {error:#}"
                );
                self.wallet.call(ops::PAY, authorized).await?
            }
        };
        let receipt = self.wait_payment(attempt).await?;
        self.store_proof(&exchange, payment, receipt).await
    }
}
