use crate::{
    binding::{Wallet, validate_identifier},
    config::{Config, now, validate_peer},
    ledger::Ledger,
    protocol::{CHANNEL, Exchange, Message, Quote},
};
use anyhow::{Context, Result, bail, ensure};
use mesh_llm_plugin::{
    PluginContext,
    openai_exchange::{OpenAiAdmissionDecision, OpenAiExchangeDecision},
};
use serde_json::Value;
use std::sync::Arc;
use wallet_contract::{
    LookupRequest, Obligation, PayRequest, PaymentAttempt, PaymentRequest, PaymentState, Purpose,
    Receipt, VerifyRequest, VerifyResult, ops,
};
#[path = "engine_admission.rs"]
mod admission;
#[path = "engine_recovery.rs"]
mod recovery;
#[path = "engine_serving.rs"]
mod serving;

// Direct settlement may require several epochs of finality. This bounds how
// long an unconsumed admission offer remains usable without holding HTTP open.
const ADMISSION_QUOTE_TTL: u64 = 3600;

pub struct Engine {
    pub config: Config,
    pub ledger: Ledger,
    pub wallet: Wallet,
    pub node: String,
    context: Option<PluginContext<'static>>,
    changed: tokio::sync::Notify,
    peers: std::sync::Mutex<std::collections::BTreeMap<String, Arc<tokio::sync::Mutex<()>>>>,
    ingress: std::sync::Mutex<std::collections::BTreeMap<String, serving::RemoteIngress>>,
    #[cfg(test)]
    outbox: std::sync::Mutex<Vec<(String, Message)>>,
}
impl Engine {
    pub fn new(
        config: Config,
        wallet: Wallet,
        node: String,
        context: PluginContext<'static>,
    ) -> Result<Arc<Self>> {
        validate_peer(&node)?;
        let ledger = Ledger::open(&config.data_dir)?;
        ledger.pin_node(&node)?;
        ledger.pin(&wallet.info)?;
        ledger.detach_buyer_admissions()?;
        Ok(Arc::new(Self {
            config,
            ledger,
            wallet,
            node,
            context: Some(context),
            changed: Default::default(),
            peers: Default::default(),
            ingress: Default::default(),
            #[cfg(test)]
            outbox: Default::default(),
        }))
    }
    pub async fn send(&self, peer: &str, message: &Message) -> Result<()> {
        if let Some(context) = &self.context {
            return context
                .clone()
                .send_json_channel(CHANNEL, peer, "payment", message)
                .await;
        }
        #[cfg(test)]
        {
            self.outbox
                .lock()
                .unwrap()
                .push((peer.into(), message.clone()));
            Ok(())
        }
        #[cfg(not(test))]
        bail!("peer transport is not initialized")
    }
    pub async fn message(&self, peer: &str, message: Message) -> Result<()> {
        validate_peer(peer)?;
        let serial = {
            let mut peers = self.peers.lock().unwrap();
            ensure!(
                peers.contains_key(peer) || peers.len() < 4096,
                "peer coordination capacity exhausted"
            );
            peers.entry(peer.into()).or_default().clone()
        };
        let _guard = serial.lock().await;
        match message {
            Message::Hello { id, model, payer } => self.hello(peer, id, model, payer).await,
            Message::Quote { quote } => self.quote(peer, *quote).await,
            Message::AdmissionProof { id, tx_hash } => {
                self.admission_proof(peer, &id, &tx_hash).await
            }
            Message::Ready { id } => {
                self.ledger.update(&id, |x| {
                    ensure!(
                        !x.seller && x.peer == peer && x.quote.is_some(),
                        "unexpected admission ready"
                    );
                    if matches!(x.state.as_str(), "negotiating" | "ready") {
                        ensure!(
                            x.quote.as_ref().is_some_and(|q| q.admission.is_none())
                                || x.admission_receipt.is_some(),
                            "ready arrived before local admission payment"
                        );
                        x.state = "ready".into();
                        x.error = None;
                    }
                    Ok(())
                })?;
                self.changed.notify_waiters();
                Ok(())
            }
            Message::Bill {
                id,
                input,
                output,
                payment,
            } => {
                self.bill(peer, &id, input, output, payment.map(|value| *value))
                    .await
            }
            Message::UsageProof { id, tx_hash } => self.usage_proof(peer, &id, &tx_hash).await,
            Message::Settled { id } => {
                self.ledger.update(&id, |x| {
                    ensure!(!x.seller && x.peer == peer, "unexpected settlement");
                    ensure!(
                        matches!(x.state.as_str(), "paid" | "settled") || x.usage == Some((0, 0)),
                        "settlement lacks a verified local payment"
                    );
                    x.state = "settled".into();
                    Ok(())
                })?;
                self.changed.notify_waiters();
                Ok(())
            }
            Message::Cancel { id } => self.cancel(peer, &id),
            Message::Error { id, reason } => {
                self.ledger.update(&id, |x| {
                    ensure!(x.peer == peer, "unexpected error peer");
                    if x.state == "negotiating" {
                        x.error = Some(reason);
                    }
                    Ok(())
                })?;
                self.changed.notify_waiters();
                Ok(())
            }
        }
    }
    async fn hello(&self, peer: &str, id: String, model: String, payer: String) -> Result<()> {
        ensure!(
            id.len() <= 128 && uuid::Uuid::parse_str(&id).is_ok(),
            "invalid request ID"
        );
        validate_identifier(&payer)?;
        if let Ok(existing) = self.ledger.get(&id) {
            ensure!(
                existing.seller
                    && existing.peer == peer
                    && existing.model == model
                    && existing.payer == payer,
                "request ID reused with different terms"
            );
            if let Some(quote) = existing.quote {
                self.send(
                    peer,
                    &Message::Quote {
                        quote: Box::new(quote),
                    },
                )
                .await?;
            }
            if existing.state == "ready" {
                self.send(peer, &Message::Ready { id }).await?;
            }
            return Ok(());
        }
        let price = self
            .config
            .prices
            .get(&model)
            .context("model is not priced")?
            .clone();
        let first = !self.ledger.earned(peer)?;
        let fee = if first { self.config.admission_fee } else { 0 };
        let offer_id = uuid::Uuid::new_v4().to_string();
        let expires_at = now() + ADMISSION_QUOTE_TTL;
        let mut exchange = Exchange {
            id: id.clone(),
            peer: peer.into(),
            model: model.clone(),
            payer,
            seller: true,
            exchange_id: None,
            state: "quoted".into(),
            quote: None,
            usage: None,
            usage_payment: None,
            admission_receipt: None,
            usage_receipt: None,
            error: None,
        };
        let admission = if fee > 0 {
            Some(PaymentRequest::new(self.obligation(
                &exchange,
                &offer_id,
                Purpose::Admission,
                fee,
                expires_at,
            ))?)
        } else {
            None
        };
        let quote = Quote {
            request_id: id.clone(),
            offer_id,
            model,
            price,
            wallet: self.wallet.info.clone(),
            admission,
            first_credit: fee,
            expires_at,
        };
        exchange.quote = Some(quote.clone());
        if fee == 0 {
            exchange.state = "ready".into();
        }
        self.ledger.insert(&exchange)?;
        self.send(
            peer,
            &Message::Quote {
                quote: Box::new(quote),
            },
        )
        .await?;
        if fee == 0 {
            self.send(peer, &Message::Ready { id }).await?;
        }
        Ok(())
    }
    fn obligation(
        &self,
        exchange: &Exchange,
        offer_id: &str,
        purpose: Purpose,
        amount: u64,
        expires_at: u64,
    ) -> Obligation {
        let suffix = if purpose == Purpose::Admission {
            "admission"
        } else {
            "usage"
        };
        Obligation {
            version: 1,
            id: format!("{}/{suffix}", exchange.id),
            purpose,
            buyer_peer: exchange.peer.clone(),
            seller_peer: self.node.clone(),
            request_id: exchange.id.clone(),
            model: exchange.model.clone(),
            offer_id: offer_id.into(),
            asset: self.wallet.info.asset.clone(),
            payer: exchange.payer.clone(),
            recipient: self.wallet.info.address.clone(),
            amount,
            expires_at,
        }
    }
    async fn quote(&self, peer: &str, quote: Quote) -> Result<()> {
        let exchange = self.ledger.get(&quote.request_id)?;
        ensure!(
            !exchange.seller && exchange.peer == peer && exchange.model == quote.model,
            "unexpected quote"
        );
        ensure!(
            quote.expires_at > now() && quote.expires_at <= now() + ADMISSION_QUOTE_TTL,
            "quote expiry out of range"
        );
        ensure!(
            quote.wallet.asset == self.wallet.info.asset,
            "seller asset/network differs from selected wallet"
        );
        ensure!(
            quote.wallet.method == self.wallet.info.method,
            "unsupported payment method"
        );
        validate_identifier(&quote.wallet.address)?;
        ensure!(
            quote.price.within(&self.config.buying.maximum_price),
            "seller prices exceed policy"
        );
        ensure!(
            quote.first_credit == quote.admission.as_ref().map_or(0, |p| p.obligation.amount),
            "invalid first-request credit"
        );
        if let Some(payment) = &quote.admission {
            self.validate_payment(
                &exchange,
                &quote,
                payment,
                Purpose::Admission,
                payment.obligation.amount,
            )?;
        }
        self.ledger.update(&exchange.id, |x| {
            if let Some(old) = &x.quote {
                ensure!(
                    serde_json::to_value(old)? == serde_json::to_value(&quote)?,
                    "quote changed"
                );
            }
            x.quote = Some(quote.clone());
            Ok(())
        })?;
        if let Some(payment) = quote.admission {
            let receipt = self.pay(&exchange.id, &payment).await?;
            self.ledger.update(&exchange.id, |x| {
                x.admission_receipt = Some(receipt.clone());
                Ok(())
            })?;
            self.send(
                peer,
                &Message::AdmissionProof {
                    id: exchange.id,
                    tx_hash: receipt.tx_hash,
                },
            )
            .await?;
        }
        Ok(())
    }
    fn validate_payment(
        &self,
        exchange: &Exchange,
        quote: &Quote,
        payment: &PaymentRequest,
        purpose: Purpose,
        amount: u64,
    ) -> Result<()> {
        payment.validate()?;
        let p = &payment.obligation;
        ensure!(
            p.buyer_peer == self.node
                && p.seller_peer == exchange.peer
                && p.request_id == exchange.id
                && p.model == exchange.model
                && p.offer_id == quote.offer_id
                && p.asset == quote.wallet.asset
                && p.payer == self.wallet.info.address
                && p.recipient == quote.wallet.address
                && p.purpose == purpose
                && p.amount == amount,
            "payment does not match accepted request and seller"
        );
        let suffix = if purpose == Purpose::Admission {
            "admission"
        } else {
            "usage"
        };
        ensure!(
            p.id == format!("{}/{suffix}", exchange.id),
            "unexpected obligation ID"
        );
        ensure!(p.expires_at > now(), "payment obligation expired");
        ensure!(
            p.expires_at <= now() + 86400 + 60,
            "payment obligation validity window exceeds policy"
        );
        if purpose == Purpose::Admission {
            ensure!(
                p.expires_at == quote.expires_at,
                "admission expiry differs from quote"
            );
        }
        Ok(())
    }
    async fn pay(&self, id: &str, request: &PaymentRequest) -> Result<Receipt> {
        let was_reserved = self.ledger.budget_reserved(&request.obligation.id)?;
        let authorized = self.authorize_payment(id, request)?;
        if was_reserved {
            return self.recover_payment(request).await;
        }
        let attempt: PaymentAttempt = self.wallet.call(ops::PAY, authorized).await?;
        self.wait_payment(attempt).await
    }
    fn authorize_payment(&self, id: &str, request: &PaymentRequest) -> Result<PayRequest> {
        let policy = &self.config.buying;
        let exchange = self.ledger.get(id)?;
        ensure!(
            policy.enabled && policy.peers.contains(&exchange.peer) && self.wallet.info.can_send,
            "spending is disabled or seller is not allowed"
        );
        ensure!(exchange.state != "cancelled", "exchange was cancelled");
        ensure!(
            request.obligation.amount <= policy.max_payment,
            "payment exceeds configured limit"
        );
        let total = if self.wallet.info.effective_fee_asset() == &self.wallet.info.asset {
            request
                .obligation
                .amount
                .checked_add(policy.max_fee)
                .context("maximum total overflow")?
        } else {
            request.obligation.amount
        };
        self.ledger
            .reserve_budget(&request.obligation.id, total, policy.daily_budget)?;
        Ok(PayRequest {
            payment_id: request.obligation.id.clone(),
            request: request.clone(),
            max_fee: policy.max_fee,
            max_total: total,
        })
    }
    async fn recover_payment(&self, request: &PaymentRequest) -> Result<Receipt> {
        let attempt = self.lookup_payment(request).await?;
        self.wait_payment(attempt).await
    }
    async fn lookup_payment(&self, request: &PaymentRequest) -> Result<PaymentAttempt> {
        let attempt: PaymentAttempt = self
            .wallet
            .call(
                ops::LOOKUP,
                LookupRequest {
                    payment_id: request.obligation.id.clone(),
                },
            )
            .await?;
        ensure!(
            attempt.request == *request,
            "wallet attempt changed immutable payment terms"
        );
        Ok(attempt)
    }
    fn finalized_receipt(attempt: PaymentAttempt) -> Result<Option<Receipt>> {
        if attempt.state == PaymentState::Finalized {
            return Ok(Some(
                attempt
                    .receipt
                    .filter(|r| r.finalized)
                    .context("finalized payment omitted receipt")?,
            ));
        }
        ensure!(
            !matches!(
                attempt.state,
                PaymentState::Reverted | PaymentState::Held | PaymentState::Expired
            ),
            "payment failed: {:?} {:?}",
            attempt.state,
            attempt.error
        );
        Ok(None)
    }
    async fn wait_payment(&self, mut attempt: PaymentAttempt) -> Result<Receipt> {
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(120);
        loop {
            if attempt.state == PaymentState::Finalized {
                return attempt
                    .receipt
                    .filter(|r| r.finalized)
                    .context("finalized payment omitted receipt");
            }
            ensure!(
                !matches!(
                    attempt.state,
                    PaymentState::Reverted | PaymentState::Held | PaymentState::Expired
                ),
                "payment failed: {:?} {:?}",
                attempt.state,
                attempt.error
            );
            ensure!(
                tokio::time::Instant::now() < deadline,
                "payment remains pending; inspect and reconcile it"
            );
            tokio::time::sleep(std::time::Duration::from_millis(300)).await;
            attempt = self
                .wallet
                .call(
                    ops::LOOKUP,
                    LookupRequest {
                        payment_id: attempt.payment_id,
                    },
                )
                .await?;
        }
    }
    async fn verify(&self, payment: &PaymentRequest, hash: &str) -> Result<Receipt> {
        let verification: VerifyResult = self
            .wallet
            .call(
                ops::VERIFY,
                VerifyRequest {
                    request: payment.clone(),
                    tx_hash: hash.into(),
                },
            )
            .await?;
        ensure!(
            verification.state == PaymentState::Finalized,
            "payment is not finalized: {:?}",
            verification.reason
        );
        verification
            .receipt
            .filter(|r| r.finalized)
            .context("verification omitted finalized delivery receipt")
    }
    async fn admission_proof(&self, peer: &str, id: &str, hash: &str) -> Result<()> {
        let exchange = self.ledger.get(id)?;
        ensure!(
            exchange.seller && exchange.peer == peer,
            "unexpected admission payer"
        );
        let payment = exchange
            .quote
            .as_ref()
            .and_then(|q| q.admission.as_ref())
            .context("request has no admission payment")?;
        let receipt = self.verify(payment, hash).await?;
        self.ledger
            .receipt(&payment.obligation.id, peer, &receipt, true)?;
        self.ledger.update(id, |x| {
            x.admission_receipt = Some(receipt);
            if x.state == "quoted" {
                x.state = "ready".into();
            }
            Ok(())
        })?;
        if self.ledger.get(id)?.state == "ready" {
            self.send(peer, &Message::Ready { id: id.into() }).await?;
        }
        self.changed.notify_waiters();
        Ok(())
    }
    async fn bill(
        &self,
        peer: &str,
        id: &str,
        input: u64,
        output: u64,
        payment: Option<PaymentRequest>,
    ) -> Result<()> {
        let exchange = self.ledger.get(id)?;
        ensure!(
            !exchange.seller && exchange.peer == peer,
            "unexpected bill seller"
        );
        ensure!(
            matches!(
                exchange.state.as_str(),
                "running" | "billed" | "paid" | "settled"
            ),
            "usage bill arrived before inference admission"
        );
        let quote = exchange
            .quote
            .as_ref()
            .context("bill arrived without quote")?;
        let amount = quote
            .price
            .cost(input, output)?
            .saturating_sub(quote.first_credit);
        ensure!(
            payment.as_ref().map_or(0, |p| p.obligation.amount) == amount,
            "bill amount does not match agreed prices and usage"
        );
        if let Some(p) = &payment {
            self.validate_payment(&exchange, quote, p, Purpose::Usage, amount)?;
        }
        if let Some(old) = exchange.usage {
            ensure!(
                old == (input, output) && exchange.usage_payment == payment,
                "bill changed immutable terms"
            );
            if let Some(receipt) = exchange.usage_receipt {
                return self
                    .send(
                        peer,
                        &Message::UsageProof {
                            id: id.into(),
                            tx_hash: receipt.tx_hash,
                        },
                    )
                    .await;
            }
            if exchange.state == "settled" {
                return Ok(());
            }
        }
        self.ledger.update(id, |x| {
            if let Some(old) = x.usage {
                ensure!(old == (input, output), "bill usage changed");
            }
            x.usage = Some((input, output));
            x.usage_payment = payment.clone();
            x.state = if amount == 0 { "settled" } else { "billed" }.into();
            Ok(())
        })?;
        if let Some(payment) = payment {
            let receipt = self.pay(id, &payment).await?;
            self.ledger.update(id, |x| {
                x.usage_receipt = Some(receipt.clone());
                x.state = "paid".into();
                Ok(())
            })?;
            self.send(
                peer,
                &Message::UsageProof {
                    id: id.into(),
                    tx_hash: receipt.tx_hash,
                },
            )
            .await?;
        }
        self.changed.notify_waiters();
        Ok(())
    }
    async fn usage_proof(&self, peer: &str, id: &str, hash: &str) -> Result<()> {
        let exchange = self.ledger.get(id)?;
        ensure!(
            exchange.seller && exchange.peer == peer,
            "unexpected payment peer"
        );
        let payment = exchange.usage_payment.context("usage bill is missing")?;
        let receipt = self.verify(&payment, hash).await?;
        self.ledger
            .receipt(&payment.obligation.id, peer, &receipt, false)?;
        self.ledger.update(id, |x| {
            x.usage_receipt = Some(receipt);
            x.state = "settled".into();
            Ok(())
        })?;
        self.send(peer, &Message::Settled { id: id.into() }).await?;
        self.changed.notify_waiters();
        Ok(())
    }
    fn cancel(&self, peer: &str, id: &str) -> Result<()> {
        self.ledger.update(id, |x| {
            ensure!(x.peer == peer, "unexpected cancellation peer");
            ensure!(x.seller,
                "a peer cannot cancel a buyer's pending payment; reconcile the original obligation locally");
            ensure!(
                x.exchange_id.is_none(),
                "consumed work cannot be cancelled to erase debt"
            );
            x.state = "cancelled".into();
            Ok(())
        })
    }
    pub async fn hook(&self, mut event: Value) -> Result<OpenAiExchangeDecision> {
        if let Err(error) = self.remember_ingress(&event) {
            return Ok(deny(&error.to_string()));
        }
        self.restore_ingress(&mut event);
        if event["endpoint"]
            .as_str()
            .is_some_and(|endpoint| endpoint != "chat_completions")
        {
            let paid_ingress = matches!(
                event["phase"].as_str(),
                Some("request_received" | "backend_selected")
            ) && event["requested_by_node_id"].as_str().is_some()
                && event["model"]
                    .as_str()
                    .is_some_and(|model| self.config.prices.contains_key(model));
            let paid_route = event["phase"] == "backend_selected"
                && self.config.buying.enabled
                && event["target_peer_id"].as_str().is_some_and(|peer| {
                    self.config
                        .buying
                        .peers
                        .iter()
                        .any(|allowed| allowed == peer)
                });
            if paid_ingress || paid_route {
                return Ok(deny(
                    "stablecoin demo supports paid chat_completions only; completions and responses are not supported",
                ));
            }
        }
        match event["phase"].as_str() {
            Some("backend_selected") if self.is_remote_paid_selection(&event) => {
                self.seller_hook(&event)
            }
            Some("backend_selected") => self.buyer_hook(&event).await,
            Some("request_received") => self.seller_hook(&event),
            Some("exchange_finished") => {
                self.forget_ingress(&event);
                self.terminal(&event).await?;
                Ok(Default::default())
            }
            _ => Ok(Default::default()),
        }
    }
    fn seller_hook(&self, event: &Value) -> Result<OpenAiExchangeDecision> {
        let Some(peer) = event["requested_by_node_id"].as_str() else {
            return Ok(Default::default());
        };
        let Some(model) = event["model"].as_str() else {
            return Ok(Default::default());
        };
        if !self.config.prices.contains_key(model) {
            return Ok(Default::default());
        }
        if let Err(error) = bounded_request(event) {
            return Ok(deny(&error.to_string()));
        }
        let Some(exchange) = self
            .ledger
            .active()?
            .into_iter()
            .find(|x| x.seller && x.peer == peer && x.model == model)
        else {
            return Ok(deny("negotiate stablecoin admission before inference"));
        };
        let host_id = serving::correlation(event)?;
        if exchange.state == "running" && exchange.exchange_id.as_deref() == Some(host_id) {
            return Ok(allow());
        }
        if exchange.state != "ready" {
            return Ok(deny("prior inference is unpaid or admission is incomplete"));
        }
        ensure!(
            exchange
                .quote
                .as_ref()
                .is_some_and(|q| q.expires_at > now()),
            "admission expired"
        );
        self.ledger.update(&exchange.id, |x| {
            ensure!(x.state == "ready", "admission was consumed concurrently");
            x.state = "running".into();
            x.exchange_id = Some(host_id.into());
            Ok(())
        })?;
        Ok(allow())
    }
    async fn terminal(&self, event: &Value) -> Result<()> {
        let Ok(host_id) = serving::correlation(event) else {
            return Ok(());
        };
        let Some(exchange) = self
            .ledger
            .active()?
            .into_iter()
            .find(|x| x.seller && x.exchange_id.as_deref() == Some(host_id))
        else {
            return Ok(());
        };
        let input = event["usage"]["prompt_tokens"].as_u64();
        let output = event["usage"]["completion_tokens"].as_u64();
        let (Some(input), Some(output)) = (input, output) else {
            self.ledger.update(&exchange.id, |x| {
                x.state = "usage_unknown".into();
                x.error = Some("terminal usage missing; operator reconciliation required".into());
                Ok(())
            })?;
            return Ok(());
        };
        if let Some(old) = exchange.usage {
            ensure!(
                old == (input, output),
                "duplicate terminal changed reported usage"
            );
            return self
                .send(
                    &exchange.peer,
                    &Message::Bill {
                        id: exchange.id,
                        input,
                        output,
                        payment: exchange.usage_payment.map(Box::new),
                    },
                )
                .await;
        }
        let quote = exchange.quote.as_ref().context("admission quote missing")?;
        let amount = quote
            .price
            .cost(input, output)?
            .saturating_sub(quote.first_credit);
        let payment = if amount > 0 {
            Some(PaymentRequest::new(self.obligation(
                &exchange,
                &quote.offer_id,
                Purpose::Usage,
                amount,
                now() + 86400,
            ))?)
        } else {
            None
        };
        self.ledger.update(&exchange.id, |x| {
            ensure!(
                x.usage.is_none() || x.usage == Some((input, output)),
                "terminal usage changed"
            );
            x.usage = Some((input, output));
            x.usage_payment = payment.clone();
            x.state = if amount > 0 { "billed" } else { "settled" }.into();
            Ok(())
        })?;
        self.send(
            &exchange.peer,
            &Message::Bill {
                id: exchange.id,
                input,
                output,
                payment: payment.map(Box::new),
            },
        )
        .await
    }
}
fn bounded_request(event: &Value) -> Result<()> {
    let body = event
        .get("body")
        .context("bounded request body permission is required")?;
    let maximum = body
        .get("max_completion_tokens")
        .or_else(|| body.get("max_tokens"))
        .or_else(|| body.get("max_output_tokens"))
        .and_then(Value::as_u64);
    ensure!(
        maximum.is_some_and(|n| n > 0 && n <= 4096),
        "demo requests require a finite output limit of 1..4096 tokens"
    );
    ensure!(
        serde_json::to_vec(body)?.len() <= 64 * 1024,
        "demo request body exceeds 64 KiB"
    );
    Ok(())
}
fn allow() -> OpenAiExchangeDecision {
    OpenAiExchangeDecision {
        decision: OpenAiAdmissionDecision::Allow,
        ..Default::default()
    }
}
fn deny(reason: &str) -> OpenAiExchangeDecision {
    OpenAiExchangeDecision {
        decision: OpenAiAdmissionDecision::Deny,
        status: Some(402),
        reason: Some(reason.into()),
        ..Default::default()
    }
}

#[cfg(test)]
#[path = "engine_tests.rs"]
mod tests;
