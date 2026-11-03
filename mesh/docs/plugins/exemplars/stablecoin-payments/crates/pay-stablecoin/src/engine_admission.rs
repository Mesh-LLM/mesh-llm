//! Long finality preserves a pending payment; only an explicit HTTP retry starts inference.
use super::*;

/// Drops on a normal return, hook deadline, or task cancellation. It cannot detach
/// running work or an observation already attached by another hook.
struct Attachment<'a> {
    engine: &'a Engine,
    id: String,
    observation: String,
}
impl Drop for Attachment<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.engine.ledger.update(&self.id, |exchange| {
            if !exchange.seller
                && matches!(exchange.state.as_str(), "negotiating" | "ready")
                && exchange.exchange_id.as_deref() == Some(self.observation.as_str())
            {
                exchange.exchange_id = None;
            }
            Ok(())
        }) {
            tracing::error!("cannot detach pending admission observation: {error:#}");
        }
    }
}

impl Engine {
    pub(super) async fn buyer_hook(&self, event: &Value) -> Result<OpenAiExchangeDecision> {
        self.buyer_admission(event, std::time::Duration::from_secs(25))
            .await
    }

    pub(super) async fn buyer_admission(
        &self,
        event: &Value,
        deadline: std::time::Duration,
    ) -> Result<OpenAiExchangeDecision> {
        let Some(peer) = event["target_peer_id"].as_str() else {
            return Ok(Default::default());
        };
        if !self.config.buying.enabled || !self.config.buying.peers.iter().any(|p| p == peer) {
            return Ok(Default::default());
        }
        bounded_request(event)?;
        let model = event["model"].as_str().context("selected model missing")?;
        let observation = serving::correlation(event)?;
        let exchange = match self.attach_admission(peer, model, observation) {
            Ok(exchange) => exchange,
            Err(error) => return Ok(deny(&error.to_string())),
        };
        if exchange.state == "running" {
            return Ok(allow());
        }
        let _attachment = Attachment {
            engine: self,
            id: exchange.id.clone(),
            observation: observation.into(),
        };
        if exchange.quote.is_none() {
            self.send(
                peer,
                &Message::Hello {
                    id: exchange.id.clone(),
                    model: model.into(),
                    payer: self.wallet.info.address.clone(),
                },
            )
            .await?;
        }
        match tokio::time::timeout(deadline, self.wait_admission(&exchange.id, observation)).await {
            Ok(Ok(())) => Ok(allow()),
            Ok(Err(error)) => Ok(deny(&error.to_string())),
            Err(_) => Ok(deny(
                "payment admission is pending; explicitly retry this peer/model after settlement",
            )),
        }
    }

    fn attach_admission(&self, peer: &str, model: &str, observation: &str) -> Result<Exchange> {
        if let Some(exchange) = self.ledger.active()?.into_iter().find(|x| x.peer == peer) {
            ensure!(
                !exchange.seller && exchange.model == model,
                "another model or serving request is outstanding for this peer"
            );
            if exchange.state == "running" && exchange.exchange_id.as_deref() == Some(observation) {
                return Ok(exchange);
            }
            self.ledger.update(&exchange.id, |current| {
                ensure!(!current.seller && current.model == model
                    && matches!(current.state.as_str(), "negotiating" | "ready")
                    && current.exchange_id.is_none(),
                    "another request is in flight or prior inference is unpaid");
                ensure!(current.quote.as_ref().is_none_or(|quote| quote.expires_at > now()),
                    "pending admission quote expired; reconcile its original payment before new work");
                current.exchange_id = Some(observation.into());
                Ok(())
            })?;
            return self.ledger.get(&exchange.id);
        }
        let exchange = Exchange {
            id: uuid::Uuid::new_v4().to_string(),
            peer: peer.into(),
            model: model.into(),
            payer: self.wallet.info.address.clone(),
            seller: false,
            exchange_id: Some(observation.into()),
            state: "negotiating".into(),
            quote: None,
            usage: None,
            usage_payment: None,
            admission_receipt: None,
            usage_receipt: None,
            error: None,
        };
        self.ledger.insert(&exchange)?;
        Ok(exchange)
    }

    async fn wait_admission(&self, id: &str, observation: &str) -> Result<()> {
        loop {
            let notified = self.changed.notified();
            let exchange = self.ledger.get(id)?;
            ensure!(
                exchange.exchange_id.as_deref() == Some(observation),
                "payment admission observation changed"
            );
            if let Some(error) = exchange.error {
                bail!("seller admission failed: {error}");
            }
            if exchange.state == "ready" {
                self.ledger.update(id, |current| {
                    ensure!(
                        current.state == "ready"
                            && current.exchange_id.as_deref() == Some(observation),
                        "payment admission was consumed concurrently"
                    );
                    ensure!(
                        current
                            .quote
                            .as_ref()
                            .is_some_and(|quote| quote.expires_at > now()),
                        "payment admission expired"
                    );
                    current.state = "running".into();
                    Ok(())
                })?;
                return Ok(());
            }
            ensure!(
                exchange.state == "negotiating",
                "payment admission is no longer pending"
            );
            tokio::select! {
                _ = notified => {},
                _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {},
            }
        }
    }
}
