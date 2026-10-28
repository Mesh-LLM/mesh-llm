//! Credit-first plugin protocol. A prepared admission is single-use and owner-bound.
mod connections;
pub mod diagnostics;
mod receipt_schedule;
mod runtime;
mod tickets;
pub mod wallet_binding;
/// Subprocess `wallet.v1` adapter kept for the offline two-wallet fixture only;
/// the installed utility reaches wallets through the host binding.
#[cfg(all(unix, feature = "offline-mock"))]
pub mod wallet_child;
#[cfg(all(unix, feature = "offline-mock"))]
mod wallet_transport;
/// Side-effect-free permission declaration for packaging.
pub fn manifest_json(name: &str) -> anyhow::Result<String> {
    let namespace: String = name.as_bytes().iter().map(|b| format!("{b:02x}")).collect();
    mesh_llm_plugin::package_manifest_json(&runtime::manifest(&format!(
        "x-plugin-{namespace}-pay-terms"
    )))
}
mod config;
use anyhow::{Context, Result, ensure};
pub use config::Config;
use mesh_llm_payments::service::PaymentService;
use mesh_llm_payments_types::{
    contract::*, metered_wire::AdmissionTerms, pricing::Pricing, terms::RequestTerms,
};
use mesh_llm_plugin::openai_exchange::{
    OpenAiAdmissionDecision as Decision, OpenAiExchangeDecision,
};
use mesh_llm_wallet::invoice::Invoice;
pub use runtime::{run, run_bound};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

pub const CHANNEL: &str = "pay.ln.v1";
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Message {
    Hello {
        version: u32,
        models: Vec<String>,
    },
    Quote,
    Admission {
        terms: AdmissionTerms,
        invoice: Option<Invoice>,
    },
    Paid {
        id: String,
    },
    Ready {
        id: String,
    },
    Settlement {
        id: String,
        bill: FinalBill,
    },
    Recover {
        id: String,
    },
    Abandoned {
        id: String,
    },
}

impl Message {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Hello { .. } => "hello",
            Self::Quote => "quote",
            Self::Admission { .. } => "admission",
            Self::Paid { .. } => "paid",
            Self::Ready { .. } => "ready",
            Self::Settlement { .. } => "settlement",
            Self::Recover { .. } => "recover",
            Self::Abandoned { .. } => "abandoned",
        }
    }
}

pub struct Engine {
    known: std::sync::Mutex<BTreeSet<String>>,
    quotes: tokio::sync::Mutex<()>,
    connected: connections::Connections,
    vouched: std::sync::Mutex<BTreeSet<String>>,
    ready_seen: std::sync::Mutex<BTreeSet<String>>,
    buyer_ready: std::sync::Mutex<BTreeSet<(String, String)>>,
    changed: tokio::sync::Notify,
    receipt_schedule: std::sync::Mutex<receipt_schedule::ReceiptSchedule>,
    pub service: Arc<PaymentService>,
    tickets: tickets::Tickets,
    pub header: String,
    pub cap_msat: u64,
    pub buying: BTreeMap<String, Pricing>,
    pub buy_peers: BTreeSet<String>,
}
impl Engine {
    pub fn new(
        service: Arc<PaymentService>,
        namespace: &str,
        cap_msat: u64,
        directory: &std::path::Path,
    ) -> Result<Self> {
        Ok(Self {
            known: Default::default(),
            quotes: Default::default(),
            connected: Default::default(),
            vouched: Default::default(),
            ready_seen: Default::default(),
            buyer_ready: Default::default(),
            changed: Default::default(),
            receipt_schedule: Default::default(),
            service,
            tickets: tickets::Tickets::open(directory)?,
            header: format!("x-plugin-{namespace}-pay-terms"),
            cap_msat,
            buying: BTreeMap::new(),
            buy_peers: BTreeSet::new(),
        })
    }
    /// Open the configured wallet during plugin startup, for either role. The
    /// service caches the provider, so concurrent admission uses the same open.
    pub async fn start(&self) -> Result<()> {
        self.service
            .wallet()
            .await
            .context("opening payment wallet at startup")?
            .balance()
            .await
            .context("warming payment wallet at startup")?;
        Ok(())
    }
    /// Observe seller wallet receipts independently of buyer notifications.
    /// A partial sweep can record receipts before another lookup fails, so wake
    /// admission even on error. Durable invoice facts make repeated sweeps safe.
    pub(crate) async fn refresh_receipts(&self) -> Result<()> {
        let receipts = self.service.ledger.unpaid_metered_invoices()?;
        let due = self.receipt_schedule.lock().unwrap().reserve(
            receipts
                .iter()
                .map(|receipt| receipt.invoice.payment_hash.clone()),
            std::time::Instant::now(),
        );
        let mut observations = tokio::task::JoinSet::new();
        for hash in due {
            let receipt = receipts
                .iter()
                .find(|receipt| receipt.invoice.payment_hash == hash)
                .expect("scheduled invoice came from ledger")
                .clone();
            let service = self.service.clone();
            observations.spawn(async move {
                tokio::time::timeout(
                    std::time::Duration::from_secs(1),
                    service.refresh_invoice(&receipt),
                )
                .await
                .map_err(anyhow::Error::new)
                .and_then(|result| result)
            });
        }
        let mut first_error = None;
        while let Some(result) = observations.join_next().await {
            self.changed.notify_waiters();
            if let Err(error) = result.map_err(anyhow::Error::new).and_then(|result| result) {
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    pub async fn channel(&self, peer: &str, message: Message) -> Result<Vec<Message>> {
        self.known.lock().unwrap().insert(peer.into());
        match message {
            Message::Hello { version, .. } => {
                ensure!(version == 1, "unsupported channel version");
                Ok(vec![])
            }
            Message::Quote => self.quote(peer).await,
            Message::Admission { terms, invoice } => {
                ensure!(self.cap_msat > 0, "buyer payments disabled");
                ensure!(self.buy_peers.contains(peer), "seller not authorized");
                let ceiling = self
                    .buying
                    .get(&terms.model)
                    .context("model not authorized for purchases")?;
                ensure!(
                    terms.pricing.input_msat_per_million <= ceiling.input_msat_per_million
                        && terms.pricing.output_msat_per_million <= ceiling.output_msat_per_million
                        && terms.pricing.cache_discount_percent >= ceiling.cache_discount_percent,
                    "seller rates exceed authorized ceiling"
                );
                let stored = self.service.ledger.request_terms(&terms.id)?;
                let proposed = RequestTerms {
                    exchange_id: None,
                    id: terms.id.clone(),
                    peer: peer.into(),
                    payee: None,
                    model: terms.model.clone(),
                    pricing: terms.pricing.clone(),
                    max_output_tokens: 1,
                    max_total_msat: self.cap_msat,
                    expires_at_ms: mesh_llm_wallet::now_ms() + 60_000,
                };
                let proposed = if let Some(stored) = stored {
                    ensure!(
                        stored.peer == peer
                            && stored.model == terms.model
                            && stored.pricing == terms.pricing,
                        "admission owner or terms changed"
                    );
                    stored
                } else {
                    proposed
                };

                self.service
                    .accept_admission(AcceptAdmissionRequest {
                        billing_revision: BillingRevision::V3,
                        terms: proposed,
                        admission: terms.clone(),
                    })
                    .await?;
                if let Some(invoice) = invoice {
                    self.service
                        .pay_deposit(PayDepositRequest {
                            request_id: terms.id.clone(),
                            invoice,
                        })
                        .await?;
                }
                Ok(vec![Message::Paid { id: terms.id }])
            }
            Message::Paid { id } => {
                self.owner(peer, &id)?;
                if self
                    .service
                    .ledger
                    .metered_invoice(
                        &id,
                        mesh_llm_payments::ledger::metered_invoices::MeteredInvoiceKind::Deposit,
                    )?
                    .is_some()
                    && !self.service.ledger.metered_past_admission(&id)?
                {
                    self.service.serve_deposit_confirmed(&id).await?;
                }
                let settlement = self.service.serve_recover(&id).await?;
                // A paid final may clear the strike limit while a prepared
                // ticket is waiting at ingress, not only when a ticket readies.
                self.changed.notify_waiters();
                if matches!(settlement, ServeRecoverResponse::Pending)
                    && !self.tickets.consumed(&id, peer)?
                {
                    self.tickets.ready(&id, peer)?;
                    self.changed.notify_waiters();
                }
                let mut replies = vec![Message::Ready { id }];
                if matches!(settlement, ServeRecoverResponse::Billed { .. }) {
                    replies.extend(self.quote(peer).await?);
                }
                Ok(replies)
            }
            Message::Settlement { id, bill } => {
                let terms = self
                    .service
                    .ledger
                    .request_terms(&id)?
                    .context("unknown buyer admission")?;
                ensure!(terms.peer == peer, "settlement from wrong seller");
                self.service
                    .settle_final(SettleFinalRequest {
                        request_id: id.clone(),
                        usage: Some(bill.usage),
                        bill_msat: Some(bill.bill_msat),
                        deposit_applied_msat: bill.deposit_msat.min(bill.bill_msat),
                        invoice: bill.invoice,
                    })
                    .await?;
                self.service.finish_metered(&id).await?;
                Ok(vec![Message::Paid { id }])
            }
            Message::Recover { id } => {
                self.owner(peer, &id)?;
                Ok(match self.service.serve_recover(&id).await? {
                    ServeRecoverResponse::Billed { bill } => vec![Message::Settlement { id, bill }],
                    ServeRecoverResponse::Abandoned => vec![Message::Abandoned { id }],
                    ServeRecoverResponse::Pending => vec![],
                })
            }
            Message::Abandoned { id } => {
                let terms = self
                    .service
                    .ledger
                    .request_terms(&id)?
                    .context("unknown buyer admission")?;
                ensure!(terms.peer == peer, "abandonment from wrong seller");
                self.service.finish_metered(&id).await?;
                Ok(vec![Message::Quote])
            }
            Message::Ready { id } => self.receive_ready(peer, id),
        }
    }
    fn receive_ready(&self, peer: &str, id: String) -> Result<Vec<Message>> {
        let terms = self
            .service
            .ledger
            .request_terms(&id)?
            .context("unknown ready ticket")?;
        ensure!(terms.peer == peer, "ready from wrong seller");
        if self.ready_seen.lock().unwrap().insert(id) {
            self.buyer_ready
                .lock()
                .unwrap()
                .insert((peer.into(), terms.model));
            self.changed.notify_waiters();
        }
        Ok(vec![])
    }
    pub async fn peer_up(&self, peer: &str) -> Result<Vec<Message>> {
        if !self.buy_peers.contains(peer) || self.cap_msat == 0 {
            return Ok(vec![]);
        }
        // Only discovery/reconnect reconciles uncertain submissions. Holding this
        // async lock makes a concurrent first request wait for startup recovery,
        // not start a duplicate payment or send another hello/recover sequence.
        let _handshake = self.connected.handshake.lock().await;
        let Some(epoch) = self.connected.begin(peer) else {
            return Ok(vec![]);
        };
        let mut messages = Vec::new();
        // Observe uncertain payments before asking the authenticated seller for stored outcomes.
        for terms in self.service.reconcile().await?.approved {
            if terms.peer == peer {
                messages.push(Message::Recover { id: terms.id });
            }
        }
        messages.insert(
            0,
            Message::Hello {
                version: 1,
                models: self.buying.keys().cloned().collect(),
            },
        );
        messages.push(Message::Quote);
        ensure!(
            self.connected.complete(peer, epoch),
            "peer disconnected during recovery"
        );
        Ok(messages)
    }
    pub fn peer_down(&self, peer: &str) {
        self.connected.disconnect(peer);
    }
    fn owner(&self, peer: &str, id: &str) -> Result<()> {
        ensure!(
            self.service.ledger.metered_owner(id)?.as_deref() == Some(peer),
            "wrong admission owner"
        );
        Ok(())
    }
    async fn quote(&self, peer: &str) -> Result<Vec<Message>> {
        let _quote = self.quotes.lock().await;
        let mut replies = vec![];
        for (model, pricing) in self.service.ledger.pricing()? {
            if self.tickets.outstanding(peer, &model)? {
                if let Some(admission) = self.tickets.admission(peer, &model)? {
                    replies.push(serde_json::from_str(&admission)?);
                }
                continue;
            }

            let id = uuid::Uuid::new_v4().to_string();
            let admission = self
                .service
                .serve_admit(ServeAdmitRequest {
                    billing_revision: BillingRevision::V3,
                    request_id: id.clone(),
                    buyer: peer.into(),
                    model: model.clone(),
                    pricing: pricing.clone(),
                })
                .await?;
            let (deposit_msat, min_payment_msat, invoice) = match admission {
                ServeAdmitResponse::Deposit {
                    invoice,
                    min_payment_msat,
                    ..
                } => (
                    invoice.amount_msat.context("amountless deposit")?,
                    min_payment_msat,
                    Some(invoice),
                ),
                ServeAdmitResponse::Admitted {
                    min_payment_msat, ..
                } => (0, min_payment_msat, None),
                _ => continue,
            };
            self.tickets.insert(&id, peer, &model)?;
            let message = Message::Admission {
                terms: AdmissionTerms {
                    id: id.clone(),
                    model,
                    pricing,
                    deposit_msat,
                    min_payment_msat,
                },
                invoice,
            };
            self.tickets
                .save_admission(&id, &serde_json::to_string(&message)?)?;
            replies.push(message);
        }
        Ok(replies)
    }
    /// Replenish a consumed ticket without waiting for its final payment. Cold
    /// peers still establish standing through settlement before receiving credit.
    pub async fn replenish(&self, event: &Value) -> Result<Option<(String, Vec<Message>)>> {
        if event["observation_point"] != "serving_host_ingress"
            || event["phase"] != "request_received"
        {
            return Ok(None);
        }
        let exchange = event["exchange_id"]
            .as_str()
            .context("exchange id missing")?;
        let Some((peer, _)) = self.tickets.exchange(exchange)? else {
            return Ok(None);
        };
        if !self.service.metered_trusted(&peer).await? {
            return Ok(None);
        }
        let replies = self.quote(&peer).await?;
        Ok(Some((peer, replies)))
    }

    pub fn vouch_once(&self, peer: &str, id: &str) -> Result<bool> {
        let terms = self
            .service
            .ledger
            .request_terms(id)?
            .context("unknown ready ticket")?;
        ensure!(terms.peer == peer, "ready from wrong seller");
        Ok(self.vouched.lock().unwrap().insert(id.into()))
    }
    /// Only an authenticated seller's Ready permits dispatch, never Hello timing.
    pub async fn buyer_wait(
        &self,
        peer: &str,
        model: &str,
        deadline: std::time::Duration,
    ) -> Result<OpenAiExchangeDecision> {
        ensure!(self.buy_peers.contains(peer), "seller not authorized");
        ensure!(self.buying.contains_key(model), "model not authorized");
        let key = (peer.to_owned(), model.to_owned());
        let until = tokio::time::Instant::now() + deadline;
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.buyer_ready.lock().unwrap().remove(&key) {
                return Ok(OpenAiExchangeDecision::default());
            }
            if tokio::time::timeout_at(until, notified).await.is_err() {
                return Ok(OpenAiExchangeDecision {
                    decision: Decision::Deny,
                    status: Some(402),
                    ..Default::default()
                });
            }
        }
    }
    pub async fn gate_wait(
        &self,
        event: &Value,
        deadline: std::time::Duration,
    ) -> Result<OpenAiExchangeDecision> {
        let until = tokio::time::Instant::now() + deadline;
        loop {
            let notified = self.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let decision = self.gate(event)?;
            let peer = event["requested_by_node_id"].as_str();
            if decision.status != Some(402)
                || !peer.is_some_and(|peer| self.known.lock().unwrap().contains(peer))
            {
                return Ok(decision);
            }
            if tokio::time::timeout_at(until, notified).await.is_err() {
                return Ok(decision);
            }
        }
    }
    pub fn gate(&self, event: &Value) -> Result<OpenAiExchangeDecision> {
        if event["observation_point"] != "serving_host_ingress"
            || event["phase"] != "request_received"
        {
            return Ok(OpenAiExchangeDecision::default());
        }
        let model = event["model"].as_str().context("model missing")?;
        if !self.service.ledger.pricing()?.contains_key(model) {
            return Ok(OpenAiExchangeDecision::default());
        }
        let peer = event["requested_by_node_id"].as_str();
        let exchange = event["exchange_id"]
            .as_str()
            .context("exchange id missing")?;
        if let Some(peer) = peer
            && self.service.ledger.metered_unpaid_finals(peer)?
                < mesh_llm_payments::ledger::metered::STRIKE_LIMIT
            && self.tickets.consume(peer, model, exchange)?
        {
            return Ok(OpenAiExchangeDecision {
                decision: Decision::Allow,
                ..Default::default()
            });
        }

        Ok(OpenAiExchangeDecision { decision: Decision::Deny, status: Some(402),
            response_headers: vec![(self.header.clone(), json!({"channel": CHANNEL, "model": model, "pricing": self.service.ledger.pricing()?[model]}).to_string())], ..Default::default() })
    }
    pub async fn terminal(&self, event: &Value) -> Result<Option<(String, Message)>> {
        if event["phase"] != "exchange_finished"
            || event["ingress_observation_point"] != "serving_host_ingress"
        {
            return Ok(None);
        }
        let exchange = event["exchange_id"]
            .as_str()
            .context("exchange id missing")?;
        let active = self.tickets.exchange(exchange)?;
        let Some((peer, id)) = active else {
            return Ok(None);
        };
        let usage = reported_usage(event);
        let message = if let Some(usage) = usage {
            Message::Settlement {
                id: id.clone(),
                bill: self
                    .service
                    .serve_settle(ServeSettleRequest {
                        request_id: id,
                        usage,
                    })
                    .await?,
            }
        } else {
            self.service.serve_abandon(&id)?;
            Message::Abandoned { id }
        };
        Ok(Some((peer, message)))
    }
}

/// The host may normalize cached counters; preserve both supported evidence shapes.
pub fn reported_usage(event: &Value) -> Option<mesh_llm_payments_types::metering::MeteredUsage> {
    let mut normalized = event.clone();
    if let Some(cached) = event["usage"]
        .get("cached_prompt_tokens")
        .filter(|v| !v.is_null())
    {
        normalized["usage"]["prompt_tokens_details"] = json!({"cached_tokens": cached});
    }
    mesh_llm_payments_types::metering::MeteredUsage::from_response(&normalized)
}

pub fn parse_prices(raw: &str) -> Result<BTreeMap<String, Pricing>> {
    let prices: BTreeMap<String, Pricing> = serde_json::from_str(raw)?;
    for price in prices.values() {
        price.validate()?;
    }
    Ok(prices)
}

#[cfg(test)]
mod tests;
