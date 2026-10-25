use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use bitcoin::secp256k1::{Secp256k1, SecretKey};
use lightning_invoice::{Currency, InvoiceBuilder, PaymentHash, PaymentSecret};
use mesh_llm_payments::{
    invoice::Invoice,
    ledger::{ApprovalMode, Policy},
    pricing::Pricing,
    service::PaymentService,
    wallet::{Balance, PayError, PaymentStatus, Transaction, WalletProvider},
};
use mesh_llm_payments_types::metered_wire::{self as wire, Frame};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::{
    inference::election::{InferenceTarget, ModelTargets},
    mesh::{Node, NodeRole},
};

mod compatibility;
mod dispatch_ownership;
mod forwarding;
mod handoff;
mod metered;
mod recovery;
mod strikes;

#[derive(Default)]
struct Network {
    entries: Mutex<HashMap<String, (u8, Transaction)>>,
    next: AtomicUsize,
    payments: AtomicUsize,
    invoice_expiry_seconds: AtomicUsize,
    updates: tokio::sync::Notify,
    hold_payments: AtomicBool,
    fail_next_payment: AtomicBool,
    payment_failed: tokio::sync::Notify,
    payment_release: tokio::sync::Notify,
    /// When set, each node also loads a plugin that records every
    /// `payment.lifecycle.v1` message it is sent.
    record_lifecycle: AtomicBool,
    provider_lifecycle: Arc<Mutex<Vec<serde_json::Value>>>,
    payer_lifecycle: Arc<Mutex<Vec<serde_json::Value>>>,
}

/// A trusted local plugin that declares `payment.lifecycle.v1` and keeps
/// every event body it receives.
fn lifecycle_recorder(
    name: &'static str,
    events: Arc<Mutex<Vec<serde_json::Value>>>,
) -> (String, crate::plugin::InProcessPluginRunner) {
    let runner: crate::plugin::InProcessPluginRunner = Arc::new(move |stream| {
        let events = events.clone();
        let plugin = mesh_llm_plugin::SimplePlugin::new(mesh_llm_plugin::PluginMetadata::new(
            name,
            "0.0.0",
            mesh_llm_plugin::plugin_server_info(
                name,
                "0.0.0",
                "Lifecycle recorder",
                "Records payment lifecycle events",
                None::<String>,
            ),
        ))
        .with_manifest(mesh_llm_plugin::plugin_manifest![
            mesh_llm_plugin::mesh_channel("payment.lifecycle.v1")
        ])
        .on_channel_message(move |message, _context| {
            let events = events.clone();
            Box::pin(async move {
                if let Ok(event) = serde_json::from_slice(&message.body) {
                    events.lock().unwrap().push(event);
                }
                Ok(())
            })
        });
        Box::pin(mesh_llm_plugin::PluginRuntime::run_with_stream(
            plugin, stream,
        ))
    });
    (name.to_owned(), runner)
}

impl Network {
    fn recorder(
        &self,
        name: &'static str,
        events: &Arc<Mutex<Vec<serde_json::Value>>>,
    ) -> Vec<(String, crate::plugin::InProcessPluginRunner)> {
        if self.record_lifecycle.load(Ordering::SeqCst) {
            vec![lifecycle_recorder(name, events.clone())]
        } else {
            Vec::new()
        }
    }
}

struct TestWallet {
    owner: u8,
    network: Arc<Network>,
}

#[async_trait]
impl WalletProvider for TestWallet {
    async fn balance(&self) -> Result<Balance> {
        Ok(Balance {
            spendable_msat: 100_000,
        })
    }
    async fn transactions(&self, _: usize) -> Result<Vec<Transaction>> {
        Ok(Vec::new())
    }
    async fn create_invoice(&self, amount: Option<u64>, expiry_secs: u32) -> Result<Invoice> {
        let number = self.network.next.fetch_add(1, Ordering::SeqCst) + 1;
        let key = SecretKey::from_slice(&[7; 32])?;
        // The simulated wallet honours the host's expiry unless a test forces
        // a shorter one to exercise the expiry paths quickly.
        let expiry = match self.network.invoice_expiry_seconds.load(Ordering::SeqCst) {
            0 => u64::from(expiry_secs),
            seconds => seconds as u64,
        };
        let invoice = InvoiceBuilder::new(Currency::Bitcoin)
            .description("simulated wallet".into())
            .payment_hash(PaymentHash([number as u8; 32]))
            .payment_secret(PaymentSecret([42; 32]))
            .current_timestamp()
            .expiry_time(Duration::from_secs(expiry))
            .min_final_cltv_expiry_delta(144)
            .amount_milli_satoshis(amount.unwrap_or(1))
            .build_signed(|hash| Secp256k1::new().sign_ecdsa_recoverable(hash, &key))?;
        let invoice = Invoice::parse(&invoice.to_string())?;
        self.network.entries.lock().unwrap().insert(
            invoice.payment_hash.clone(),
            (
                self.owner,
                Transaction {
                    id: invoice.payment_hash.clone(),
                    payment_hash: Some(invoice.payment_hash.clone()),
                    inbound: true,
                    amount_msat: amount.unwrap_or(1),
                    fee_msat: 10,
                    status: PaymentStatus::Pending,
                    claiming: false,
                    status_msg: None,
                    created_at_ms: mesh_llm_payments::now_ms(),
                    settled_at_ms: None,
                },
            ),
        );
        Ok(invoice)
    }
    async fn lookup(&self, hash: &str) -> Result<Option<Transaction>> {
        Ok(self
            .network
            .entries
            .lock()
            .unwrap()
            .get(hash)
            .and_then(|(owner, payment)| {
                if *owner != self.owner && payment.status == PaymentStatus::Pending {
                    return None;
                }
                let mut payment = payment.clone();
                payment.inbound = *owner == self.owner;
                Some(payment)
            }))
    }
    async fn pay(&self, invoice: &Invoice, amount: u64, cap: u64) -> Result<Transaction, PayError> {
        invoice.validate_payment(amount, mesh_llm_payments::now_ms())?;
        assert!(amount + 10 <= cap);
        if self.network.fail_next_payment.swap(false, Ordering::SeqCst) {
            self.network.payment_failed.notify_one();
            return Err(PayError::Uncertain(anyhow::anyhow!(
                "simulated lost submission"
            )));
        }
        while self.network.hold_payments.load(Ordering::SeqCst) {
            let released = self.network.payment_release.notified();
            tokio::pin!(released);
            if !self.network.hold_payments.load(Ordering::SeqCst) {
                break;
            }
            released.await;
        }
        {
            let mut entries = self.network.entries.lock().unwrap();
            let (owner, payment) = entries.get_mut(&invoice.payment_hash).unwrap();
            assert_ne!(*owner, self.owner);
            assert_eq!(payment.status, PaymentStatus::Pending);
            payment.status = PaymentStatus::Succeeded;
            payment.settled_at_ms = Some(mesh_llm_payments::now_ms());
        }
        self.network.payments.fetch_add(1, Ordering::SeqCst);
        self.network.updates.notify_waiters();
        Ok(self.lookup(&invoice.payment_hash).await?.unwrap())
    }

    async fn wait_for_payment(&self, hash: &str) -> Result<Transaction> {
        loop {
            let changed = self.network.updates.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if let Some(payment) = self.lookup(hash).await?
                && payment.status != PaymentStatus::Pending
            {
                return Ok(payment);
            }
            changed.await;
        }
    }
}

fn allow_paid(service: &PaymentService) -> Result<()> {
    service.ledger.set_policy(&Policy {
        mode: ApprovalMode::Automatic,
        daily_budget_msat: Some(100_000),
    })
}

fn price() -> Pricing {
    Pricing {
        input_msat_per_million: 100_000_000,
        output_msat_per_million: 100_000_000,
        cache_discount_percent: 0,
    }
}

/// A seller and a buyer node on one simulated Lightning network, each with
/// its own ledger and the in-process `payments.v1` plugin attached, with the
/// buyer holding a cached connection to the seller.
struct Pair {
    _seller_dir: tempfile::TempDir,
    _buyer_dir: tempfile::TempDir,
    seller: Arc<PaymentService>,
    buyer: Arc<PaymentService>,
    seller_node: Node,
    buyer_node: Node,
}

impl Pair {
    async fn new(network: Arc<Network>) -> Result<Self> {
        let seller_dir = tempfile::tempdir()?;
        let buyer_dir = tempfile::tempdir()?;
        let seller = Arc::new(PaymentService::with_provider(
            seller_dir.path(),
            Arc::new(TestWallet {
                owner: 1,
                network: network.clone(),
            }),
        )?);
        let buyer = Arc::new(PaymentService::with_provider(
            buyer_dir.path(),
            Arc::new(TestWallet {
                owner: 2,
                network: network.clone(),
            }),
        )?);
        allow_paid(&buyer)?;
        seller.ledger.set_pricing("test", Some(&price()))?;
        let seller_node = Node::new_for_tests(NodeRole::Host { http_port: 0 }).await?;
        seller_node
            .payments
            .set(seller.clone())
            .map_err(|_| anyhow::anyhow!("already set"))?;
        crate::network::payments::node_ext::attach_payments_plugin_with(
            &seller_node,
            network.recorder("provider-lifecycle-recorder", &network.provider_lifecycle),
        )
        .await?;
        let buyer_node = Node::new_for_tests(NodeRole::Client).await?;
        // Keep node-local payment state (the payee blocklist) in the fixture.
        *buyer_node.config_state.lock().await =
            crate::runtime::config_state::ConfigState::load(&buyer_dir.path().join("config.toml"))?;
        buyer_node
            .payments
            .set(buyer.clone())
            .map_err(|_| anyhow::anyhow!("already set"))?;
        crate::network::payments::node_ext::attach_payments_plugin_with(
            &buyer_node,
            network.recorder("payer-lifecycle-recorder", &network.payer_lifecycle),
        )
        .await?;
        let mut announcement = seller_node
            .build_local_announcement(seller_node.snapshot_local_announcement_data().await);
        announcement.lightning_offers.insert("test".into(), price());
        buyer_node
            .add_peer_after_direct_requirements_validated(
                seller_node.id(),
                seller_node.endpoint.addr(),
                &announcement,
                Some(1),
            )
            .await;
        Ok(Self {
            _seller_dir: seller_dir,
            _buyer_dir: buyer_dir,
            seller,
            buyer,
            seller_node,
            buyer_node,
        })
    }

    /// Accept `count` metered exchanges on the seller, served from `targets`.
    fn serve(&self, targets: ModelTargets, count: usize) -> ServingTask {
        let node = self.seller_node.clone();
        tokio::spawn(async move {
            let connection = node.endpoint.accept().await.unwrap().await?;
            for _ in 0..count {
                let (send, mut recv) = accept_tunnel(&connection).await?;
                let mut prefix = vec![0; wire::HTTP_UPGRADE.len()];
                recv.read_exact(&mut prefix).await?;
                anyhow::ensure!(prefix == wire::HTTP_UPGRADE, "not a metered upgrade");
                let _ = super::serve(
                    node.clone(),
                    connection.remote_id(),
                    recv,
                    send,
                    targets.clone(),
                )
                .await;
            }
            Ok(connection)
        })
    }

    async fn connect(&self) -> Result<()> {
        let connection = self
            .buyer_node
            .endpoint
            .connect(self.seller_node.endpoint.addr(), crate::protocol::ALPN_V1)
            .await?;
        self.buyer_node
            .state
            .lock()
            .await
            .connections
            .insert(self.seller_node.id(), connection);
        Ok(())
    }

    /// Start one paid request as the buyer and return its response body
    /// after the exchange's HTTP side ends.
    async fn request(&self, raw: &[u8], exchange_id: Option<&str>) -> Result<String> {
        let (mut pipe, ready, _cancel) = crate::network::openai::test_metered_start(
            &self.buyer_node,
            self.seller_node.id(),
            raw,
            price(),
            exchange_id,
        )
        .await?;
        ready.await?;
        let mut response = String::new();
        pipe.read_to_string(&mut response).await?;
        Ok(response)
    }

    async fn wait_completed(&self, count: usize) -> Result<()> {
        loop {
            let completed = self
                .buyer
                .ledger
                .requests()?
                .iter()
                .filter(|r| r.state == "completed")
                .count();
            if completed == count {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    async fn close(self) {
        let buyer = Arc::downgrade(&self.buyer);
        let seller = Arc::downgrade(&self.seller);
        for node in [&self.buyer_node, &self.seller_node] {
            node.shutdown_control_listener().await;
            if let Some(manager) = node.take_plugin_manager().await {
                manager.shutdown().await;
            }
            node.endpoint.close().await;
        }
        drop(self);
        tokio::time::timeout(Duration::from_secs(2), async {
            while buyer.upgrade().is_some() || seller.upgrade().is_some() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("payment fixture leaked its service after close");
    }
}

type ServingTask = tokio::task::JoinHandle<Result<iroh::endpoint::Connection>>;

async fn accept_tunnel(
    connection: &iroh::endpoint::Connection,
) -> Result<(iroh::endpoint::SendStream, iroh::endpoint::RecvStream)> {
    loop {
        let (send, mut recv) = connection.accept_bi().await?;
        if recv.read_u8().await? == crate::protocol::STREAM_TUNNEL_HTTP {
            return Ok((send, recv));
        }
        // Peer announcements may open a gossip stream before the HTTP tunnel.
        drop(send);
        drop(recv);
    }
}

fn local_targets(port: u16) -> ModelTargets {
    let mut targets = ModelTargets::default();
    targets
        .targets
        .insert("test".into(), vec![InferenceTarget::Local(port)]);
    targets
}

/// Read one HTTP request off `stream`, asserting it is a metered backend
/// request, and return its JSON body.
async fn read_backend_request(stream: &mut tokio::net::TcpStream) -> Result<serde_json::Value> {
    let mut raw = Vec::new();
    let mut byte = [0_u8; 1];
    loop {
        anyhow::ensure!(stream.read(&mut byte).await? == 1, "request truncated");
        raw.extend(byte);
        let mut headers = [httparse::EMPTY_HEADER; 64];
        let mut request = httparse::Request::new(&mut headers);
        if let httparse::Status::Complete(offset) = request.parse(&raw)? {
            let length = match request
                .headers
                .iter()
                .find(|header| header.name.eq_ignore_ascii_case("content-length"))
            {
                Some(header) => std::str::from_utf8(header.value)?.parse::<usize>()?,
                None => 0,
            };
            if raw.len() == offset + length {
                return Ok(serde_json::from_slice(&raw[offset..])?);
            }
        }
    }
}
