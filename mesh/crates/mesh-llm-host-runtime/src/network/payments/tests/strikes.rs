//! A seller that takes the deposit and delivers nothing is struck on the
//! buyer; a seller that abandons after delivering output is not.
use super::*;
use mesh_llm_payments_types::metered_wire::AdmissionTerms;
use mesh_llm_payments_types::metering::deposit_msat;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_seller_that_keeps_the_deposit_and_delivers_nothing_is_struck() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(20), fake_seller(false)).await?
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_seller_that_delivered_output_is_not_struck() -> Result<()> {
    tokio::time::timeout(Duration::from_secs(20), fake_seller(true)).await?
}

async fn fake_seller(deliver: bool) -> Result<()> {
    let network = Arc::new(Network::default());
    let pair = Pair::new(network.clone()).await?;
    let seller_node = pair.seller_node.clone();
    let seller_network = network.clone();
    let buyer = pair.buyer.clone();
    let seller = tokio::spawn(async move {
        let connection = seller_node.endpoint.accept().await.unwrap().await?;
        let (mut send, mut recv) = accept_tunnel(&connection).await?;
        let mut prefix = vec![0; wire::HTTP_UPGRADE.len()];
        recv.read_exact(&mut prefix).await?;
        anyhow::ensure!(prefix == wire::HTTP_UPGRADE);
        let Frame::Request {
            id, pricing, http, ..
        } = wire::read(&mut recv).await?
        else {
            anyhow::bail!("request expected");
        };
        let request = crate::network::payments::request::PaidRequest::parse(&http)?;
        let estimate =
            mesh_llm_payments_types::metering::PromptEstimate::from_request(&request.body)?;
        let deposit = deposit_msat(pricing.input_charge(estimate.input_tokens)?)?;
        let wallet = TestWallet {
            owner: 1,
            network: seller_network.clone(),
        };
        let invoice = wallet.create_invoice(Some(deposit), 3600).await?;
        let hash = invoice.payment_hash.clone();
        wire::write(
            &mut send,
            &Frame::Admission {
                terms: AdmissionTerms {
                    id: id.clone(),
                    model: "test".into(),
                    pricing,
                    prompt_estimate: estimate.input_tokens,
                    deposit_msat: deposit,
                },
                invoice: Some(invoice),
            },
        )
        .await?;
        anyhow::ensure!(matches!(wire::read(&mut recv).await?, Frame::Accepted {}));
        // Wait until the buyer's own ledger has settled the deposit, so the
        // strike decision does not ride on a scheduling race.
        while buyer.ledger.charge_state(&hash)?.as_deref() != Some("succeeded") {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        if deliver {
            wire::write(
                &mut send,
                &Frame::Output {
                    bytes: b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}".to_vec(),
                },
            )
            .await?;
        }
        wire::write(
            &mut send,
            &Frame::Settlement {
                request_id: id,
                usage: None,
                bill_msat: None,
                deposit_applied_msat: 0,
                invoice: None,
            },
        )
        .await?;
        wire::write(&mut send, &Frame::Complete {}).await?;
        send.finish()?;
        let _ = send.stopped().await;
        Ok::<_, anyhow::Error>(connection)
    });
    pair.connect().await?;
    let raw = b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n\r\n{\"model\":\"test\",\"prompt\":\"Hi\",\"max_tokens\":8}";
    let response = pair.request(raw, None).await?;
    assert_eq!(response.starts_with("HTTP/1.1 200"), deliver, "{response}");
    pair.wait_completed(1).await?;
    assert_eq!(network.payments.load(Ordering::SeqCst), 1);
    assert_eq!(
        strike_recorded(&pair.buyer_node, &pair.seller_node).await,
        !deliver
    );
    let _connection = seller.await??;
    pair.close().await;
    Ok(())
}

/// Whether a paid-but-undelivered strike against `seller` was recorded on
/// `buyer`: one real strike plus THRESHOLD-1 synthetic ones reaches a block
/// only if the real one is there.
async fn strike_recorded(buyer: &Node, seller: &Node) -> bool {
    use crate::network::payments::strikes::{PayeeStrikes, STRIKE_THRESHOLD};
    let directory = buyer.config_state.lock().await.payment_directory();
    let now = mesh_llm_wallet::now_ms();
    for _ in 0..100 {
        let mut probe = PayeeStrikes::load(&directory);
        for _ in 1..STRIKE_THRESHOLD {
            probe.record(&seller.id().to_string(), now);
        }
        if probe.is_blocked(&seller.id().to_string(), now) {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    false
}
