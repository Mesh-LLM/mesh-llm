//! Seller payment admission on automatic (composite) dispatch.
//!
//! The ordinary local route refuses a remote caller's request for a priced
//! model with 402 unless it enters the payment protocol. Pipeline and the MoA
//! `mesh` virtual model open their own loopback backend requests and used to
//! skip that gate entirely (Loupe #3061). These tests drive a genuine QUIC
//! `ClientStream` through `handle_remote_http_stream` against priced local
//! backends and require that no backend is ever contacted.

use super::*;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;

enum Observed {
    Response(String),
    Backend(String),
}

/// A loopback "inference backend" that reports every model it is asked for.
async fn backend(seen: mpsc::UnboundedSender<String>) -> (u16, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let task = tokio::spawn(async move {
        loop {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = proxy::read_http_request(&mut stream).await.unwrap();
            let model = request.model_name.unwrap_or_default();
            seen.send(model.clone()).unwrap();
            let body = serde_json::json!({
                "id": "paid-backend-response",
                "object": "chat.completion",
                "created": 0,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "The answer is 42."},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.shutdown().await;
        }
    });
    (port, task)
}

/// Send one request over a real QUIC stream from a separate caller node, so
/// the remote-caller boundary is exercised without forged headers or source
/// addresses. Resolves to the first of: a backend being contacted, or the
/// response the caller received.
async fn remote_request(
    node: &mesh::Node,
    targets: &election::ModelTargets,
    body: serde_json::Value,
    seen: &mut mpsc::UnboundedReceiver<String>,
) -> Observed {
    let caller = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .unwrap();
    let accepting_node = node.clone();
    let targets = targets.clone();
    let handler = tokio::spawn(async move {
        let connection = accepting_node
            .endpoint
            .accept()
            .await
            .unwrap()
            .await
            .unwrap();
        let remote = connection.remote_id();
        let (send, recv) = connection.accept_bi().await.unwrap();
        handle_remote_http_stream(
            accepting_node,
            ClientStream::from_quic_with_prefix(recv, send, Vec::new()),
            targets,
            affinity::AffinityRouter::new(),
            remote,
        )
        .await;
        // Keep the QUIC connection alive until the response is consumed.
        std::future::pending::<()>().await;
        drop(connection);
    });
    let connection = tokio::time::timeout(
        Duration::from_secs(15),
        caller
            .endpoint
            .connect(node.endpoint.addr(), crate::protocol::ALPN_V1),
    )
    .await
    .expect("QUIC connect timed out")
    .unwrap();
    let (mut send, mut recv) = connection.open_bi().await.unwrap();
    let body = body.to_string();
    send.write_all(
        format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )
        .as_bytes(),
    )
    .await
    .unwrap();
    let observed = tokio::time::timeout(Duration::from_secs(15), async {
        tokio::select! {
            model = seen.recv() => Observed::Backend(model.unwrap()),
            bytes = recv.read_to_end(64 * 1024) =>
                Observed::Response(String::from_utf8(bytes.unwrap()).unwrap()),
        }
    })
    .await
    .expect("remote ingress must reject or dispatch");
    // A response and a backend notification can become ready together; the
    // backend contact is the fact that matters.
    let observed = match seen.try_recv() {
        Ok(model) => Observed::Backend(model),
        Err(_) => observed,
    };
    handler.abort();
    let _ = handler.await;
    caller.close_endpoint().await;
    observed
}

fn status_line(response: &str) -> &str {
    response.lines().next().unwrap_or_default()
}

fn assert_refused_before_backend(observed: Observed, context: &str) {
    match observed {
        Observed::Backend(model) => {
            panic!("{context}: unpaid remote request reached priced backend {model}")
        }
        Observed::Response(response) => assert_eq!(
            status_line(&response),
            "HTTP/1.1 402 Payment Required",
            "{context}: an unpaid request against priced local backends must fail closed: {response}"
        ),
    }
}

/// A chat request that omits `model` and is shaped so the planner/strong-model
/// pipeline activates on a host serving two local models.
fn pipeline_request_body() -> serde_json::Value {
    let body = serde_json::json!({
        "max_tokens": 32,
        "messages": [{"role": "user", "content":
            "Refactor this function and explain every change. ".repeat(20)}],
        "tools": [{"type": "function", "function": {
            "name": "read_file", "parameters": {"type": "object", "properties": {}}
        }}]
    });
    assert!(pipeline::should_pipeline(&router::classify(&body)));
    body
}

fn mesh_gateway_request_body() -> serde_json::Value {
    serde_json::json!({
        "model": automatic::DIRECTIVE, "max_tokens": 32,
        "messages": [{"role": "user", "content": "What is six times seven?"}]
    })
}

struct PricedSeller {
    node: mesh::Node,
    service: Arc<mesh_llm_payments::service::PaymentService>,
    manager: crate::plugin::PluginManager,
    targets: election::ModelTargets,
    backends: Vec<tokio::task::JoinHandle<()>>,
    seen: mpsc::UnboundedReceiver<String>,
    _directory: tempfile::TempDir,
}

const MODELS: [&str; 2] = ["Qwen3-8B", "Qwen3-32B"];

/// A node pricing two local models, with the real payments engine and the
/// real built-in MoA plugin registered so `"model":"mesh"` takes the virtual
/// model path rather than falling back to single-model routing.
async fn priced_seller() -> PricedSeller {
    let directory = tempfile::tempdir().unwrap();
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    let service =
        Arc::new(mesh_llm_payments::service::PaymentService::open(directory.path()).unwrap());
    node.payments
        .set(service.clone())
        .map_err(|_| "payments already installed")
        .unwrap();
    let moa_runner: crate::plugin::InProcessPluginRunner =
        Arc::new(|stream| Box::pin(mesh_llm_moa_plugin::run(stream)));
    let manager = crate::network::payments::node_ext::attach_payments_plugin_with(
        &node,
        vec![(crate::plugin::MOA_PLUGIN_ID.to_owned(), moa_runner)],
    )
    .await
    .unwrap();
    assert!(
        manager
            .virtual_model_for_model(automatic::DIRECTIVE)
            .await
            .unwrap()
            .is_some(),
        "the MoA virtual model must be registered for this test to exercise it"
    );
    let price = mesh_llm_payments::pricing::Pricing {
        input_msat_per_million: 1_000_000,
        output_msat_per_million: 1_000_000,
        minimum_invoice_msat: 1,
    };
    let (seen_tx, seen) = mpsc::unbounded_channel();
    let mut targets = election::ModelTargets::default();
    let mut backends = Vec::new();
    for name in MODELS {
        service.ledger.set_pricing(name, Some(&price)).unwrap();
        let (port, task) = backend(seen_tx.clone()).await;
        targets
            .targets
            .insert(name.into(), vec![election::InferenceTarget::Local(port)]);
        backends.push(task);
    }
    node.set_hosted_models(MODELS.iter().map(|name| name.to_string()).collect())
        .await;
    assert_eq!(node.advertised_payment_offers().await.unwrap().len(), 2);
    PricedSeller {
        node,
        service,
        manager,
        targets,
        backends,
        seen,
        _directory: directory,
    }
}

async fn unpaid_automatic_request_is_rejected(automatic: serde_json::Value, context: &str) {
    let mut seller = priced_seller().await;

    // Sanity check: the very same caller transport and priced model are
    // denied on the ordinary route, before contacting either backend.
    let explicit = serde_json::json!({
        "model": MODELS[1], "max_tokens": 32,
        "messages": [{"role": "user", "content": "What is six times seven?"}]
    });
    let observed = remote_request(&seller.node, &seller.targets, explicit, &mut seller.seen).await;
    assert_refused_before_backend(observed, "ordinary priced route");

    let observed = remote_request(&seller.node, &seller.targets, automatic, &mut seller.seen).await;
    for task in seller.backends {
        task.abort();
        let _ = task.await;
    }
    assert!(seller.service.ledger.requests().unwrap().is_empty());
    seller.manager.shutdown().await;
    seller.node.close_endpoint().await;

    assert_refused_before_backend(observed, context);
}

#[tokio::test]
async fn remote_pipeline_cannot_bypass_seller_payment_admission() {
    unpaid_automatic_request_is_rejected(pipeline_request_body(), "pipeline").await;
}

#[tokio::test]
async fn remote_mesh_gateway_cannot_bypass_seller_payment_admission() {
    unpaid_automatic_request_is_rejected(mesh_gateway_request_body(), "mesh gateway").await;
}
