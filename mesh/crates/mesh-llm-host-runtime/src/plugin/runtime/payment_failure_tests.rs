use super::*;
use tempfile::TempDir;

#[cfg(feature = "payments")]
#[tokio::test]
async fn failed_payments_provider_must_not_make_a_priced_model_free() -> Result<()> {
    use crate::network::payments::client::Payments;
    use mesh_llm_payments::service::PaymentService;
    use mesh_llm_payments_types::pricing::Pricing;

    const MODEL: &str = "priced-model";
    let directory = TempDir::new()?;
    let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker).await?;
    let service = Arc::new(PaymentService::open(directory.path())?);
    service.ledger.set_pricing(
        MODEL,
        Some(&Pricing {
            input_msat_per_million: 1_000_000,
            output_msat_per_million: 1_000_000,
            minimum_invoice_msat: 1,
        }),
    )?;
    let _payments = Payments::attach_for_tests(&node, service).await?;
    let manager = node.plugin_manager().await.context("plugin manager")?;

    // Keep the failure window deterministic: disable automatic supervision,
    // while leaving the real provider running for the initial pricing call.
    manager.inner.shutting_down.store(true, Ordering::SeqCst);
    assert!(node.advertised_payment_offers().await?.contains_key(MODEL));

    // Exercise the real lifecycle transition used for a failed health check.
    manager
        .inner
        .plugins
        .get(crate::plugin::PAYMENTS_PLUGIN_ID)
        .context("payments provider")?
        .handle_runtime_failure(None, "simulated provider failure".into())
        .await;

    // Failure must be safe immediately, before the health loop refreshes.
    assert!(node.advertised_payment_offers().await.is_err());
    manager
        .refresh_plugin_endpoints(crate::plugin::PAYMENTS_PLUGIN_ID)
        .await?;
    assert!(node.advertised_payment_offers().await.is_err());
    assert!(
        manager
            .available_provider_for_capability("payments.v1")
            .await?
            .is_none()
    );
    let plugin = manager
        .inner
        .plugins
        .get(crate::plugin::PAYMENTS_PLUGIN_ID)
        .unwrap();
    assert!(plugin.manifest_snapshot().await.is_none());
    assert!(plugin.summary().await.tools.is_empty());

    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        remote_inference_is_rejected(&node),
    )
    .await??;

    // A healthy restart restores live pricing without replacing registration.
    plugin.ensure_running().await?;
    manager
        .refresh_plugin_endpoints(crate::plugin::PAYMENTS_PLUGIN_ID)
        .await?;
    assert!(node.advertised_payment_offers().await?.contains_key(MODEL));
    manager.shutdown().await;
    node.endpoint.close().await;
    Ok(())
}

/// Exercise ordinary HTTP over a real two-endpoint QUIC connection. The backend
/// listener must never be contacted while seller pricing is unavailable.
async fn remote_inference_is_rejected(node: &crate::mesh::Node) -> Result<()> {
    use crate::inference::election::{InferenceTarget, ModelTargets};
    use crate::network::openai::client_stream::ClientStream;
    use std::time::Duration;

    let caller = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Client).await?;
    let backend = tokio::net::TcpListener::bind(("127.0.0.1", 0)).await?;
    let mut targets = ModelTargets::default();
    targets.targets.insert(
        "priced-model".into(),
        vec![InferenceTarget::Local(backend.local_addr()?.port())],
    );
    let serving = node.clone();
    let server = tokio::spawn(async move {
        let connection = serving.endpoint.accept().await.unwrap().await?;
        let (send, recv) = connection.accept_bi().await?;
        crate::network::openai::ingress::handle_remote_http_stream(
            serving,
            ClientStream::from_quic_with_prefix(recv, send, Vec::new()),
            targets,
            crate::network::affinity::AffinityRouter::default(),
            connection.remote_id(),
        )
        .await;
        Ok::<_, anyhow::Error>(connection)
    });
    let connection = caller
        .endpoint
        .connect(node.endpoint.addr(), crate::protocol::ALPN_V1)
        .await?;
    let (mut send, mut recv) = connection.open_bi().await?;
    let body = r#"{"model":"priced-model","prompt":"hi"}"#;
    let request = format!(
        "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    send.write_all(request.as_bytes()).await?;
    send.finish()?;
    let response = tokio::time::timeout(Duration::from_secs(5), recv.read_to_end(8192)).await??;
    let response = String::from_utf8(response)?;
    let _server_connection = server.await??;
    assert!(response.starts_with("HTTP/1.1 402 "), "{response}");
    assert!(
        response.contains("seller payment state unavailable"),
        "{response}"
    );
    assert!(
        tokio::time::timeout(Duration::from_millis(50), backend.accept())
            .await
            .is_err()
    );
    caller.endpoint.close().await;
    Ok(())
}
