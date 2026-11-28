//! Provider eligibility over loopback gossip with a deterministic verifier.
use super::*;
use crate::mesh::NodeRole as LocalRole;
use crate::plugin::{BridgeFuture, PluginManager, PluginRpcBridge, RpcResult, proto};
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const MESH: &str = "evidence-mesh";

/// Accepts evidence that names the verified node and the local mesh.
struct Verifier(Arc<AtomicUsize>);

impl PluginRpcBridge for Verifier {
    fn handle_request(
        &self,
        _: String,
        _: String,
        params: String,
    ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Box::pin(async move {
            let input: Value = serde_json::from_str(&params).unwrap();
            let arguments = &input["arguments"];
            let accepted = arguments["evidence"]["host"] == arguments["node_id"]
                && arguments["evidence"]["mesh"] == arguments["mesh_id"];
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            let result = json!({"accepted": accepted, "valid_until": now.as_secs() + 3600});
            Ok(RpcResult {
                result_json: result.to_string(),
            })
        })
    }
    fn handle_notification(&self, _: String, _: String, _: String) -> BridgeFuture<()> {
        Box::pin(async {})
    }
}

/// A host that publishes evidence naming `host`.
async fn provider(host: Option<&str>) -> Result<Node> {
    let node = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    node.set_mesh_id(MESH.into()).await;
    let host = host.map_or_else(|| node.id().to_string(), str::to_owned);
    let evidence = json!({"host": host, "mesh": MESH});
    let manager =
        PluginManager::for_test_peer_evidence(Arc::new(Verifier(Arc::default())), Some(evidence));
    *node.plugin_manager.lock().await = Some(manager);
    node.start_accepting();
    Ok(node)
}

async fn observer(calls: &Arc<AtomicUsize>) -> Result<(Node, PluginManager)> {
    let node = make_test_node(LocalRole::Client).await?;
    node.set_mesh_id(MESH.into()).await;
    let manager = PluginManager::for_test_peer_evidence(Arc::new(Verifier(calls.clone())), None);
    *node.plugin_manager.lock().await = Some(manager.clone());
    node.start_accepting();
    Ok((node, manager))
}

/// Makes `provider`'s entry on `observer` serve `model` as a chat model.
async fn serve_model(observer: &Node, provider: &Node, model: &str) {
    provider.set_hosted_models(vec![model.into()]).await;
    let ann = provider.build_local_announcement(provider.snapshot_local_announcement_data().await);
    let mut state = observer.state.lock().await;
    state.test_peer_liveness.insert(provider.id());
    let peer = state.peers.get_mut(&provider.id()).unwrap();
    peer.hosted_models_known = true;
    peer.hosted_models = vec![model.into()];
    peer.serving_models = peer.hosted_models.clone();
    peer.served_model_descriptors = ann.served_model_descriptors;
    for descriptor in &mut peer.served_model_descriptors {
        descriptor
            .metadata
            .get_or_insert_with(Default::default)
            .workload_class = Some(mesh_llm_types::mesh::ModelWorkloadClass::CausalGeneration);
    }
}

/// Model IDs the chat-compatible `/v1/models` offers on `node`.
async fn listed_llms(node: &Node) -> Vec<String> {
    let models = crate::models::discovery::collect(node).await.unwrap();
    let (_, body) = crate::network::openai::models_response(&models, "/v1/models", false);
    body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|model| model["id"].as_str().unwrap().to_string())
        .collect()
}

async fn is_routable(node: &Node, target: iroh::EndpointId) -> bool {
    node.routable_peers()
        .await
        .iter()
        .any(|peer| peer.id == target)
}

async fn wait_for_routable(node: &Node, target: iroh::EndpointId) {
    tokio::time::timeout(Duration::from_secs(30), async {
        while !is_routable(node, target).await {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("provider did not become routable within 30 s");
}

#[tokio::test]
async fn peer_evidence_gates_relay_and_routing_to_providers() -> Result<()> {
    let calls = Arc::new(AtomicUsize::new(0));
    let (observer, manager) = observer(&calls).await?;
    let accepted = provider(None).await?;
    let rejected = provider(Some("someone-else")).await?;
    let maintain = || {
        let observer = observer.clone();
        tokio::spawn(async move { observer.maintain_peer_evidence().await })
    };
    let maintenance = maintain();
    for node in [&accepted, &rejected] {
        observer
            .connect_to_peer(node.endpoint_addr_for_advertisement())
            .await?;
    }
    wait_for_routable(&observer, accepted.id()).await;

    // The rejected provider is connected and rejected, so never routable.
    let evidence = rejected
        .plugin_manager()
        .await
        .unwrap()
        .peer_evidence()
        .publications();
    let rejected_node = rejected.id().to_string();
    tokio::time::timeout(Duration::from_secs(30), async {
        while manager
            .peer_evidence()
            .provider_decision(MESH, &rejected_node, &evidence)
            != Some(false)
        {
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("rejected provider was not verified within 30 s");
    assert!(
        observer
            .state
            .lock()
            .await
            .peers
            .contains_key(&rejected.id())
    );
    assert!(!is_routable(&observer, rejected.id()).await);

    // The model list follows routing: only the accepted provider's model is offered.
    serve_model(&observer, &accepted, "accepted-model").await;
    serve_model(&observer, &rejected, "rejected-model").await;
    let listed = listed_llms(&observer).await;
    assert!(listed.contains(&"accepted-model".to_string()), "{listed:?}");
    assert!(
        !listed.contains(&"rejected-model".to_string()),
        "{listed:?}"
    );

    // Gossip relays the accepted provider but not the rejected one.
    let (accepted_record, rejected_record) = {
        let state = observer.state.lock().await;
        let now = crate::mesh::requirements::current_time_unix_ms();
        let signature = |id| {
            state
                .node_records
                .fresh(&id, now)
                .unwrap()
                .signature()
                .to_vec()
        };
        (signature(accepted.id()), signature(rejected.id()))
    };
    let gossip = observer.collect_outbound_gossip().await;
    let relayed = |id| gossip.announcements.iter().any(|ann| ann.addr.id == id);
    let signed = |signature: &[u8]| {
        gossip
            .signed_records
            .iter()
            .any(|record| record.signature == signature)
    };
    assert!(relayed(accepted.id()) && signed(&accepted_record));
    assert!(!relayed(rejected.id()) && !signed(&rejected_record));

    // Its signed record still decides when its own entry claims a client.
    if let Some(peer) = observer.state.lock().await.peers.get_mut(&rejected.id()) {
        peer.role = LocalRole::Client;
        peer.serving_models.clear();
        peer.hosted_models.clear();
        peer.served_model_descriptors.clear();
    }
    let gossip = observer.collect_outbound_gossip().await;
    assert!(
        !gossip
            .signed_records
            .iter()
            .any(|r| r.signature == rejected_record)
    );
    assert!(
        !gossip
            .announcements
            .iter()
            .any(|a| a.addr.id == rejected.id())
    );
    assert!(!is_routable(&observer, rejected.id()).await);

    // Routing reads cached verdicts without calling the verifier again.
    let verified = calls.load(Ordering::SeqCst);
    for _ in 0..3 {
        assert!(is_routable(&observer, accepted.id()).await);
    }
    assert!(!observer.verify_undecided_providers(&manager).await);
    assert_eq!(calls.load(Ordering::SeqCst), verified);

    // A verifier restart withdraws eligibility until the provider is
    // verified again.
    maintenance.abort();
    let _ = maintenance.await;
    manager.peer_evidence().invalidate("evidence");
    assert!(!is_routable(&observer, accepted.id()).await);
    let maintenance = maintain();
    wait_for_routable(&observer, accepted.id()).await;
    assert!(calls.load(Ordering::SeqCst) > verified);

    maintenance.abort();
    for node in [&observer, &accepted, &rejected] {
        node.close_endpoint().await;
    }
    Ok(())
}

#[tokio::test]
async fn peer_evidence_ignores_providers_without_a_signed_record() -> Result<()> {
    let calls = Arc::new(AtomicUsize::new(0));
    let (observer, manager) = observer(&calls).await?;
    let id = make_test_endpoint_id(0x76);
    let mut peer = make_test_peer(id, None, 0);
    peer.role = LocalRole::Host { http_port: 9337 };
    peer.mesh_id = Some(MESH.into());
    peer.serving_models = vec!["forged".into()];
    observer.state.lock().await.peers.insert(id, peer);

    assert!(!observer.verify_undecided_providers(&manager).await);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(!is_routable(&observer, id).await);
    assert!(observer.hosts_for_model("forged").await.is_empty());
    observer.close_endpoint().await;
    Ok(())
}

#[tokio::test]
async fn a_verifier_takes_third_party_descriptions_only_from_signed_records() -> Result<()> {
    let describe = |seed: u8| {
        let id = make_test_endpoint_id(seed);
        let mut ann = super::peer_state_test_announcement(iroh::EndpointAddr {
            id,
            addrs: Default::default(),
        });
        ann.version = Some(crate::VERSION.to_string());
        ann.serving_models = vec!["unsigned-model".into()];
        (ann.addr.clone(), ann)
    };
    let relay = make_test_endpoint_id(0x62);
    let third_party = make_test_endpoint_id(0x61);
    let resolved_ids = |node: Node| async move {
        let gossip = crate::protocol::InboundGossip {
            announcements: vec![describe(0x62), describe(0x61)],
            signed_records: Vec::new(),
            signed_cache_affinity: Vec::new(),
        };
        let ids: Vec<_> = node
            .resolve_inbound_gossip(relay, gossip)
            .await
            .iter()
            .map(|(addr, _)| addr.id)
            .collect();
        node.close_endpoint().await;
        ids
    };

    // Without a verifier, a relay may still describe a third party unsigned.
    let plain = resolved_ids(make_test_node(LocalRole::Client).await?).await;
    assert!(plain.contains(&relay) && plain.contains(&third_party));

    // With one, only the relay's description of itself is taken.
    let (observer, _) = observer(&Arc::default()).await?;
    let verifying = resolved_ids(observer).await;
    assert!(verifying.contains(&relay));
    assert!(!verifying.contains(&third_party));
    Ok(())
}
