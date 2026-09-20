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
    wire::{self, Frame},
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use crate::{
    inference::election::{InferenceTarget, ModelTargets},
    mesh::{Node, NodeRole},
};

mod admission;
mod expiry;
mod forwarding;
mod review_regressions;

#[derive(Default)]
struct Network {
    entries: Mutex<HashMap<String, (u8, Transaction)>>,
    next: AtomicUsize,
    payments: AtomicUsize,
    invoice_expiry_seconds: AtomicUsize,
    updates: tokio::sync::Notify,
    hold_payments: AtomicBool,
    payment_release: tokio::sync::Notify,
    decoded_before_payment: AtomicBool,
    backend_output_before_payment: AtomicBool,
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
    async fn create_invoice(&self, amount: Option<u64>) -> Result<Invoice> {
        let number = self.network.next.fetch_add(1, Ordering::SeqCst) + 1;
        let key = SecretKey::from_slice(&[7; 32])?;
        let expiry = match self.network.invoice_expiry_seconds.load(Ordering::SeqCst) {
            0 => 3600,
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

async fn simulated_backend(
    listener: tokio::net::TcpListener,
    payments: Arc<Network>,
    cancel_after_output: bool,
    output_allowance: u32,
) -> Result<()> {
    let (mut stream, _) = listener.accept().await?;
    let mut raw = vec![0; 16 * 1024];
    let count = stream.read(&mut raw).await?;
    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut request = httparse::Request::new(&mut headers);
    request.parse(&raw[..count])?;
    let id = request
        .headers
        .iter()
        .find(|h| h.name.eq_ignore_ascii_case("x-request-id"))
        .unwrap();
    let id = uuid::Uuid::parse_str(std::str::from_utf8(id.value)?)?;
    let gate = skippy_server::frontend::generation_gate::registered_for_test(*id.as_bytes())
        .unwrap()
        .unwrap();
    let output_tokens = if output_allowance > 4096 { 5000 } else { 3 };
    let generation_payments = payments.clone();
    let generate = tokio::task::spawn_blocking(move || -> Result<()> {
        // Payment starts after prefill, while decode is allowed to continue.
        assert_eq!(generation_payments.payments.load(Ordering::SeqCst), 0);
        gate.after_prefill(40, output_allowance)
            .map_err(|_| anyhow::anyhow!("gate failed"))?;
        if generation_payments.payments.load(Ordering::SeqCst) == 0 {
            generation_payments
                .decoded_before_payment
                .store(true, Ordering::SeqCst);
        }
        for _ in 0..output_tokens {
            gate.before_token()
                .map_err(|_| anyhow::anyhow!("cancelled"))?;
            gate.committed_token()
                .map_err(|_| anyhow::anyhow!("commit failed"))?;
        }
        Ok(())
    });
    generate.await??;
    if cancel_after_output {
        stream.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {\"choices\":[{\"text\":\"test output\"}],\"usage\":{\"completion_tokens\":3}}\n\n").await?;
        // The provider must close generation when it receives payer cancellation.
        let mut closed = [0; 1];
        assert_eq!(stream.read(&mut closed).await?, 0);
        return Ok(());
    }
    let body = serde_json::to_vec(&serde_json::json!({
        "choices": [{"text": "test output"}],
        "usage": {"prompt_tokens": 40, "completion_tokens": output_tokens, "total_tokens": 40 + output_tokens}
    }))?;
    stream.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",body.len()).as_bytes()).await?;
    stream.write_all(&body).await?;
    stream.shutdown().await?;
    if payments.payments.load(Ordering::SeqCst) == 0 {
        payments
            .backend_output_before_payment
            .store(true, Ordering::SeqCst);
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn payments_two_quic_nodes_gate_decode_and_settle_actual_output() -> Result<()> {
    tokio::time::timeout(
        Duration::from_secs(20),
        paid_exchange(false, false, false, Some(8), 8),
    )
    .await??;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn payments_decode_runs_while_input_payment_is_pending_but_delivery_waits() -> Result<()> {
    tokio::time::timeout(
        Duration::from_secs(20),
        paid_exchange(false, false, true, Some(8), 8),
    )
    .await??;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn payments_manual_approval_gates_both_invoices() -> Result<()> {
    tokio::time::timeout(
        Duration::from_secs(20),
        paid_exchange(true, false, false, Some(8), 8),
    )
    .await??;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn payments_cancellation_settles_transmitted_output() -> Result<()> {
    tokio::time::timeout(
        Duration::from_secs(20),
        paid_exchange(false, true, false, Some(8), 8),
    )
    .await??;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn payments_omitted_limit_uses_backend_context_and_settles_long_output() -> Result<()> {
    tokio::time::timeout(
        Duration::from_secs(20),
        paid_exchange(false, false, false, None, 6000),
    )
    .await??;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn payments_explicit_large_limit_can_be_clamped_to_context() -> Result<()> {
    tokio::time::timeout(
        Duration::from_secs(20),
        paid_exchange(true, false, false, Some(65_536), 6000),
    )
    .await??;
    Ok(())
}

async fn paid_exchange(
    manual: bool,
    cancel_after_output: bool,
    hold_input_payment: bool,
    requested: Option<u32>,
    output_allowance: u32,
) -> Result<()> {
    let provider_dir = tempfile::tempdir()?;
    let payer_dir = tempfile::tempdir()?;
    let network = Arc::new(Network::default());
    network
        .hold_payments
        .store(hold_input_payment, Ordering::SeqCst);
    let provider_service = Arc::new(PaymentService::with_provider(
        provider_dir.path(),
        Arc::new(TestWallet {
            owner: 1,
            network: network.clone(),
        }),
    )?);
    let payer_service = Arc::new(PaymentService::with_provider(
        payer_dir.path(),
        Arc::new(TestWallet {
            owner: 2,
            network: network.clone(),
        }),
    )?);
    payer_service.ledger.set_policy(&Policy {
        mode: if manual {
            ApprovalMode::Manual
        } else {
            ApprovalMode::Automatic
        },
        daily_budget_msat: Some(10_000),
    })?;
    let price = Pricing {
        input_msat_per_million: 1_000_000,
        output_msat_per_million: 1_000_000,
        minimum_invoice_msat: 1,
    };
    provider_service.ledger.set_pricing("test", Some(&price))?;
    let provider = Node::new_for_tests(NodeRole::Host { http_port: 0 }).await?;
    provider
        .payments
        .set(provider_service.clone())
        .map_err(|_| anyhow::anyhow!("service already initialized"))?;
    let payer = Node::new_for_tests(NodeRole::Client).await?;
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
    let port = listener.local_addr()?.port();
    let backend = tokio::spawn(simulated_backend(
        listener,
        network.clone(),
        cancel_after_output,
        output_allowance,
    ));
    let mut targets = ModelTargets::default();
    targets
        .targets
        .insert("test".into(), vec![InferenceTarget::Local(port)]);
    let provider_addr = provider.endpoint.addr();
    let provider_id = provider_addr.id;
    let serving_node = provider.clone();
    let serving = tokio::spawn(async move {
        let connection = serving_node.endpoint.accept().await.unwrap().await?;
        let peer = connection.remote_id();
        let (send, recv) = connection.accept_bi().await?;
        let result = super::serve(serving_node, peer, recv, send, targets).await;
        Ok::<_, anyhow::Error>((connection, result))
    });
    let connection = payer
        .endpoint
        .connect(provider_addr, crate::protocol::ALPN_V1)
        .await?;
    let (mut send, recv) = connection.open_bi().await?;
    let id = uuid::Uuid::new_v4().to_string();
    let request = paid_request(requested)?;
    wire::write(
        &mut send,
        &Frame::Request {
            id: id.clone(),
            model: "test".into(),
            pricing: price.clone(),
            http: request.backend_http(&id)?,
        },
    )
    .await?;
    let (mut output, mut receiver) = tokio::io::duplex(4096);
    let (ready, _ready_receiver) = tokio::sync::oneshot::channel();
    let (cancel, cancellation) = tokio::sync::watch::channel(false);
    let payer_for_exchange = payer_service.clone();
    let exchange = tokio::spawn(async move {
        crate::network::openai::test_payment_exchange(
            payer_for_exchange,
            provider_id,
            id,
            request,
            price,
            send,
            recv,
            &mut output,
            ready,
            cancellation,
        )
        .await
    });
    if manual {
        approve_pending_request(&payer_service, &network).await?;
    }
    release_held_payment_after_backend_output(hold_input_payment, &network, &mut receiver).await?;
    let response = receive_and_cancel(&mut receiver, cancel_after_output, cancel).await?;
    exchange.await??;
    assert!(response.contains("test output"));
    let (_server_connection, result) = serving.await??;
    result?;
    backend.await??;
    let output_tokens = if output_allowance > 4096 { 5000 } else { 3 };
    assert_settlement(&network, &payer_service, &provider_service, output_tokens)?;
    assert_eq!(
        payer_service.ledger.requests()?[0].terms.max_output_tokens,
        u64::from(output_allowance)
    );
    payer.endpoint.close().await;
    provider.endpoint.close().await;
    Ok(())
}

fn paid_request(max_tokens: Option<u32>) -> Result<super::request::PaidRequest> {
    let mut body = serde_json::json!({"model": "test", "prompt": "Hi"});
    if let Some(limit) = max_tokens {
        body["max_tokens"] = limit.into();
    }
    super::request::PaidRequest::parse(
        format!("POST /v1/completions HTTP/1.1\r\n\r\n{body}").as_bytes(),
    )
}

async fn wait_for_backend_output_before_payment(network: &Network) -> Result<()> {
    while !network.backend_output_before_payment.load(Ordering::SeqCst) {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        network.decoded_before_payment.load(Ordering::SeqCst),
        "decode did not start while payment was pending"
    );
    assert_eq!(network.payments.load(Ordering::SeqCst), 0);
    Ok(())
}

async fn release_held_payment_after_backend_output(
    held: bool,
    network: &Network,
    receiver: &mut tokio::io::DuplexStream,
) -> Result<()> {
    if !held {
        return Ok(());
    }
    wait_for_backend_output_before_payment(network).await?;
    let mut byte = [0; 1];
    assert!(
        tokio::time::timeout(Duration::from_millis(100), receiver.read(&mut byte))
            .await
            .is_err(),
        "provider released output before receiving payment"
    );
    network.hold_payments.store(false, Ordering::SeqCst);
    network.payment_release.notify_waiters();
    Ok(())
}

async fn approve_pending_request(service: &PaymentService, network: &Network) -> Result<()> {
    loop {
        let pending = service.ledger.requests()?;
        if let Some(request) = pending.first() {
            assert_eq!(request.state, "pending");
            assert_eq!(network.payments.load(Ordering::SeqCst), 0);
            service.approve(&request.terms.id).await?;
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    Ok(())
}

async fn receive_and_cancel(
    receiver: &mut tokio::io::DuplexStream,
    cancel_after_output: bool,
    cancel: tokio::sync::watch::Sender<bool>,
) -> Result<String> {
    let mut response = String::new();
    if cancel_after_output {
        let mut bytes = [0; 4096];
        while !response.contains("completion_tokens\":3") {
            let count = receiver.read(&mut bytes).await?;
            anyhow::ensure!(count > 0, "output closed before cancellation point");
            response.push_str(std::str::from_utf8(&bytes[..count])?);
        }
        cancel.send(true)?;
    }
    receiver.read_to_string(&mut response).await?;
    Ok(response)
}

fn assert_settlement(
    network: &Network,
    payer_service: &PaymentService,
    provider_service: &PaymentService,
    output_tokens: u64,
) -> Result<()> {
    assert_eq!(network.payments.load(Ordering::SeqCst), 2);
    let requests = payer_service.ledger.requests()?;
    assert_eq!(requests[0].state, "completed");
    assert_eq!(requests[0].spent_msat, 40 + output_tokens + 20);
    assert!(
        provider_service
            .ledger
            .receivables(None)?
            .iter()
            .all(|r| r.paid)
    );
    Ok(())
}
