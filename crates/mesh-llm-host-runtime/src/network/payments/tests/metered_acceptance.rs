//! Provider-path acceptance: missing evidence abandons without a bill, and
//! recovery never re-dispatches.
use super::*;
use mesh_llm_payments::ledger::metered::{MeteredOutcome, MeteredRequest, MeteredStatus};
use mesh_llm_payments_types::{metered_wire as v2, metering::MeteredUsage};

struct Fixture {
    _dir: tempfile::TempDir,
    service: Arc<PaymentService>,
    node: Node,
    peer: iroh::EndpointId,
    price: Pricing,
}

impl Fixture {
    async fn trusted() -> Result<Self> {
        let dir = tempfile::tempdir()?;
        let service = Arc::new(PaymentService::with_provider(
            dir.path(),
            Arc::new(TestWallet {
                owner: 1,
                network: Arc::new(Network::default()),
            }),
        )?);
        let node = Node::new_for_tests(NodeRole::Host { http_port: 0 }).await?;
        let peer = node.id();
        let price = Pricing {
            input_msat_per_million: 1_000_000,
            output_msat_per_million: 8_000_000,
            minimum_invoice_msat: 1,
        };
        service.ledger.set_pricing("test", Some(&price))?;
        // Seed a genuinely settled row, then exercise the service's trust cache.
        service.ledger.metered_admit(&MeteredRequest {
            id: "seed",
            peer: &peer.to_string(),
            pricing: &price,
            estimated_input_tokens: 1,
            require_deposit: false,
        })?;
        service.ledger.metered_finalize(
            "seed",
            MeteredOutcome::Usage(MeteredUsage {
                prompt_tokens: 0,
                completion_tokens: 0,
            }),
        )?;
        service
            .metered_note_settled("seed", &peer.to_string())
            .await?;
        assert!(service.metered_trusted(&peer.to_string()).await?);
        node.payments
            .set(service.clone())
            .map_err(|_| anyhow::anyhow!("already set"))?;
        Ok(Self {
            _dir: dir,
            service,
            node,
            peer,
            price,
        })
    }

    fn start(&self, port: u16) -> (tokio::io::DuplexStream, tokio::task::JoinHandle<Result<()>>) {
        let (client, server) = tokio::io::duplex(64 * 1024);
        let (read, write) = tokio::io::split(server);
        let mut targets = ModelTargets::default();
        targets
            .targets
            .insert("test".into(), vec![InferenceTarget::Local(port)]);
        let node = self.node.clone();
        let peer = self.peer;
        let task = tokio::spawn(super::super::metered::serve(
            node, peer, read, write, targets,
        ));
        (client, task)
    }

    async fn admit(&self, client: &mut tokio::io::DuplexStream, id: &str) -> Result<()> {
        v2::write(client, &v2::Frame::Request {
            id: id.into(), model: "test".into(), pricing: self.price.clone(),
            http: b"POST /v1/completions HTTP/1.1\r\n\r\n{\"model\":\"test\",\"prompt\":\"Hi\",\"stream\":true,\"max_tokens\":100000}".to_vec(),
        }).await?;
        match v2::read(client).await? {
            v2::Frame::Admission { terms, invoice } => {
                assert_eq!(terms.deposit_msat, 0);
                assert!(invoice.is_none());
            }
            other => anyhow::bail!("unexpected admission: {other:?}"),
        }
        v2::write(client, &v2::Frame::Accepted {}).await
    }

    async fn assert_recover(&self, id: &str, port: u16) -> Result<()> {
        let (mut client, task) = self.start(port);
        v2::write(&mut client, &v2::Frame::Recover { id: id.into() }).await?;
        assert_abandoned(v2::read(&mut client).await?);
        assert!(matches!(
            v2::read(&mut client).await?,
            v2::Frame::Complete {}
        ));
        task.await??;
        assert!(
            !self
                .service
                .ledger
                .metered_claim_dispatch(id, &self.peer.to_string())?
        );
        Ok(())
    }
}

fn assert_abandoned(frame: v2::Frame) {
    match frame {
        v2::Frame::Settlement {
            usage,
            bill_msat,
            invoice,
            deposit_applied_msat,
            ..
        } => {
            assert!(usage.is_none());
            assert!(bill_msat.is_none());
            assert!(invoice.is_none());
            assert_eq!(deposit_applied_msat, 0);
        }
        other => panic!("unexpected settlement: {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn missing_usage_abandons_without_bill_and_is_not_a_strike() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(20), missing_usage()).await?
}

async fn missing_usage() -> Result<()> {
    let fixture = Fixture::trusted().await?;
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
    let port = listener.local_addr()?.port();
    let id = uuid::Uuid::new_v4().to_string();
    let (mut client, task) = fixture.start(port);
    fixture.admit(&mut client, &id).await?;
    let (mut backend, _) = listener.accept().await?;
    let mut request = [0; 8192];
    assert!(backend.read(&mut request).await? > 0);
    backend.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\ndata: {\"choices\":[{\"text\":\"hello\"}]}\n\ndata: [DONE]\n\n").await?;
    backend.shutdown().await?;
    read_abandoned_settlement(&mut client).await?;
    task.await??;
    assert_eq!(
        fixture.service.ledger.metered_settlement(&id)?.status,
        MeteredStatus::Abandoned
    );
    let peer = fixture.peer.to_string();
    assert_eq!(fixture.service.ledger.metered_unpaid_finals(&peer)?, 0);
    assert!(fixture.service.metered_trusted(&peer).await?);
    fixture.assert_recover(&id, port).await?;
    // Recovery never reaches the backend.
    assert!(
        tokio::time::timeout(Duration::from_millis(30), listener.accept())
            .await
            .is_err()
    );
    fixture.node.endpoint.close().await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancel_and_receiver_disconnect_abandon_and_recovery_never_dispatches() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(20), interrupted_exchanges()).await?
}

async fn interrupted_exchanges() -> Result<()> {
    for explicit_cancel in [true, false] {
        let fixture = Fixture::trusted().await?;
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
        let port = listener.local_addr()?.port();
        let id = uuid::Uuid::new_v4().to_string();
        let (mut client, task) = fixture.start(port);
        fixture.admit(&mut client, &id).await?;
        let (mut backend, _) = listener.accept().await?;
        let mut request = [0; 8192];
        assert!(backend.read(&mut request).await? > 0);
        backend.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\ndata: {\"choices\":[{\"text\":\"partial\"}]}\n\n").await?;
        assert!(matches!(
            v2::read(&mut client).await?,
            v2::Frame::Output { .. }
        ));
        interrupt_client(client, backend, task, explicit_cancel).await?;
        assert_interrupted(&fixture, &listener, &id, port).await?;
        fixture.node.endpoint.close().await;
    }
    Ok(())
}

async fn assert_interrupted(
    fixture: &Fixture,
    listener: &tokio::net::TcpListener,
    id: &str,
    port: u16,
) -> Result<()> {
    let settlement = fixture.service.ledger.metered_settlement(id)?;
    assert_eq!(settlement.status, MeteredStatus::Abandoned);
    assert_eq!(settlement.bill_msat, None);
    assert_eq!(
        fixture
            .service
            .ledger
            .metered_unpaid_finals(&fixture.peer.to_string())?,
        0
    );
    fixture.assert_recover(id, port).await?;
    assert!(
        tokio::time::timeout(Duration::from_millis(30), listener.accept())
            .await
            .is_err()
    );
    Ok(())
}

async fn read_abandoned_settlement(client: &mut tokio::io::DuplexStream) -> Result<()> {
    loop {
        let frame = v2::read(client).await?;
        if matches!(frame, v2::Frame::Output { .. }) {
            continue;
        }
        assert_abandoned(frame);
        break;
    }
    assert!(matches!(v2::read(client).await?, v2::Frame::Complete {}));
    Ok(())
}

async fn interrupt_client(
    mut client: tokio::io::DuplexStream,
    mut backend: tokio::net::TcpStream,
    task: tokio::task::JoinHandle<Result<()>>,
    explicit_cancel: bool,
) -> Result<()> {
    if explicit_cancel {
        v2::write(&mut client, &v2::Frame::Cancel {}).await?;
        assert_abandoned(v2::read(&mut client).await?);
        assert!(matches!(
            v2::read(&mut client).await?,
            v2::Frame::Complete {}
        ));
        task.await??;
    } else {
        drop(client);
        backend
            .write_all(b"data: {\"choices\":[{\"text\":\"more\"}]}\n\n")
            .await?;
        backend.shutdown().await?;
        assert!(task.await?.is_err());
    }
    Ok(())
}
