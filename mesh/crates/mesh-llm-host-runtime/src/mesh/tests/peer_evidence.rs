//! Origin-bound evidence over actual loopback gossip, with a deterministic verifier.
use super::*;
use crate::mesh::NodeRole as LocalRole;
use crate::plugin::{BridgeFuture, PluginManager, PluginRpcBridge, RpcResult, proto};
use serde_json::{Value, json};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Verifier(Arc<AtomicUsize>, bool);

#[tokio::test]
async fn peer_evidence_split_request_admission_checks_its_generation_without_reverification()
-> Result<()> {
    use crate::mesh::runtime_admission::RuntimeProviderDependencies;
    use crate::runtime::{InstanceAdmissionError, InstanceLifecycleRecord, InstanceLifecycleState};
    let first = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let second = make_test_node(LocalRole::Host { http_port: 9338 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    for node in [&first, &second, &observer] {
        node.set_mesh_id("split-admission".into()).await;
        node.start_accepting();
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let manager =
        PluginManager::for_test_peer_evidence(Arc::new(Verifier(calls.clone(), false)), None);
    *observer.plugin_manager.lock().await = Some(manager.clone());
    for node in [&first, &second] {
        observer
            .connect_to_peer(node.endpoint_addr_for_advertisement())
            .await?;
        wait_for_eligible_peer(&observer, node.id()).await;
    }
    drop(manager.peer_evidence().refresh.lock().await);
    let lifecycle = Arc::new(tokio::sync::Mutex::new(InstanceLifecycleRecord::new(
        InstanceLifecycleState::Serving,
        10,
    )));
    let first_dependencies = Arc::new(RuntimeProviderDependencies::new(
        observer.clone(),
        vec![first.id()],
    ));
    observer.register_runtime_instance_admission(9, lifecycle.clone(), Some(&first_dependencies));
    let running = observer.begin_runtime_instance_request(9).await?.unwrap();
    manager.expire_peer_evidence_for_test();
    let verified = calls.load(Ordering::SeqCst);
    assert!(matches!(
        observer.begin_runtime_instance_request(9).await,
        Err(InstanceAdmissionError::ProviderIneligible)
    ));
    let mut membership = observer.clone();
    membership.plugin_manager = Arc::new(tokio::sync::Mutex::new(None));
    assert!(matches!(
        membership.begin_runtime_instance_request(9).await,
        Err(InstanceAdmissionError::ProviderIneligible)
    ));
    assert_eq!(calls.load(Ordering::SeqCst), verified);
    assert_eq!(lifecycle.lock().await.in_flight_count(), 1);
    manager
        .verify_provider_evidence(
            "split-admission",
            &second.id().to_string(),
            &Default::default(),
        )
        .await;
    let second_dependencies = Arc::new(RuntimeProviderDependencies::new(
        observer.clone(),
        vec![second.id()],
    ));
    observer.register_runtime_instance_admission(10, lifecycle.clone(), Some(&second_dependencies));
    let verified = calls.load(Ordering::SeqCst);
    let replacement = observer.begin_runtime_instance_request(10).await?.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), verified);
    assert!(observer.begin_runtime_instance_request(9).await.is_err());
    manager
        .verify_provider_evidence(
            "split-admission",
            &first.id().to_string(),
            &Default::default(),
        )
        .await;
    let retired = observer.retire_runtime_instance_lifecycle(9, &lifecycle);
    assert!(matches!(
        observer.begin_runtime_instance_request(9).await,
        Err(InstanceAdmissionError::Retired)
    ));
    let other_lifecycle = Arc::new(tokio::sync::Mutex::new(InstanceLifecycleRecord::new(
        InstanceLifecycleState::Serving,
        10,
    )));
    observer.register_runtime_instance_admission(9, other_lifecycle, Some(&second_dependencies));
    assert!(
        observer
            .retire_runtime_instance_lifecycle(9, &lifecycle)
            .is_none()
    );
    observer.unregister_retired_runtime_instance(9, retired.as_ref());
    let reused = observer.begin_runtime_instance_request(9).await?.unwrap();
    drop(reused);
    drop(second_dependencies);
    assert!(matches!(
        membership.begin_runtime_instance_request(10).await,
        Err(InstanceAdmissionError::ProviderIneligible)
    ));
    observer.register_runtime_instance_admission(11, lifecycle.clone(), Some(&first_dependencies));
    assert!(observer.begin_runtime_instance_request(11).await?.is_some());
    *first.role.lock().await = LocalRole::Client;
    let gossip = first.collect_own_outbound_gossip().await;
    let announcements = observer
        .resolve_inbound_gossip(
            first.id(),
            crate::protocol::InboundGossip {
                announcements: gossip
                    .announcements
                    .into_iter()
                    .map(|ann| (ann.addr.clone(), ann))
                    .collect(),
                signed_records: gossip.signed_records,
                signed_cache_affinity: gossip.signed_cache_affinity,
            },
        )
        .await;
    observer
        .apply_announced_peers(first.id(), &announcements, None, None, true)
        .await?;
    assert!(manager.peer_evidence().provider_cached(
        "split-admission",
        &first.id().to_string(),
        &Default::default()
    ));
    assert!(matches!(
        observer.begin_runtime_instance_request(11).await,
        Err(InstanceAdmissionError::ProviderIneligible)
    ));
    assert!(observer.connection_to_peer(first.id()).await.is_err());
    assert!(
        observer
            .admitted_connection_to_peer(first.id())
            .await
            .is_ok()
    );
    assert_eq!(lifecycle.lock().await.in_flight_count(), 2);
    drop(replacement);
    drop(running);
    assert_eq!(lifecycle.lock().await.in_flight_count(), 0);
    assert!(observer.begin_runtime_instance_request(99).await?.is_none());
    for node in [&first, &second, &observer] {
        node.close_endpoint().await;
    }
    Ok(())
}

#[tokio::test]
async fn peer_evidence_unsigned_transitive_identity_cannot_verify_or_reuse_an_old_verdict()
-> Result<()> {
    let observer = make_test_node(LocalRole::Client).await?;
    observer.set_mesh_id("legacy-relay".into()).await;
    let calls = Arc::new(AtomicUsize::new(0));
    let manager =
        PluginManager::for_test_peer_evidence(Arc::new(Verifier(calls.clone(), false)), None);
    *observer.plugin_manager.lock().await = Some(manager.clone());
    let id = make_test_endpoint_id(0x76);
    let mut peer = make_test_peer(id, None, 0);
    peer.role = LocalRole::Host { http_port: 9337 };
    peer.mesh_id = Some("legacy-relay".into());
    peer.admitted = false;
    let ann = mesh_llm_membership::announcements::announcement_from_peer(&peer);
    observer.state.lock().await.peers.insert(id, peer.clone());
    observer.refresh_peer_evidence().await;
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(observer.evidence_announcement(id, &ann).await.is_none());
    peer.admitted = true;
    observer.state.lock().await.peers.insert(id, peer.clone());
    observer.refresh_peer_evidence().await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(observer.evidence_announcement(id, &ann).await.is_some());
    observer.state.lock().await.peers.remove(&id);
    peer.admitted = false;
    observer.state.lock().await.peers.insert(id, peer);
    observer.refresh_peer_evidence().await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(observer.evidence_announcement(id, &ann).await.is_none());
    assert!(
        observer
            .collect_outbound_gossip()
            .await
            .announcements
            .iter()
            .all(|ann| ann.addr.id != id)
    );
    {
        let mut state = observer.state.lock().await;
        let peer = state.peers.get_mut(&id).unwrap();
        peer.admitted = true;
        peer.last_seen = std::time::Instant::now() - std::time::Duration::from_secs(1000);
        peer.last_mentioned = std::time::Instant::now();
        peer.hosted_models = vec!["relay-forged".into()];
    }
    observer.refresh_peer_evidence().await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(observer.evidence_announcement(id, &ann).await.is_none());
    assert!(observer.hosts_for_model("relay-forged").await.is_empty());
    assert!(
        observer
            .collect_outbound_gossip()
            .await
            .announcements
            .iter()
            .all(|ann| ann.addr.id != id)
    );
    observer.close_endpoint().await;
    Ok(())
}

#[tokio::test]
async fn peer_evidence_cache_eviction_does_not_trigger_another_verification() -> Result<()> {
    use std::{future::Future, task::Poll};
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    for node in [&origin, &observer] {
        node.set_mesh_id("cache-eviction".into()).await;
        node.start_accepting();
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let manager =
        PluginManager::for_test_peer_evidence(Arc::new(Verifier(calls.clone(), false)), None);
    *observer.plugin_manager.lock().await = Some(manager.clone());
    observer
        .connect_to_peer(origin.endpoint_addr_for_advertisement())
        .await?;
    wait_for_eligible_peer(&observer, origin.id()).await;
    let verified = calls.load(Ordering::SeqCst);
    let mut maintenance = Box::pin(observer.maintain_peer_evidence());
    std::future::poll_fn(|cx| {
        assert!(maintenance.as_mut().poll(cx).is_pending());
        Poll::Ready(())
    })
    .await;
    drop(manager.peer_evidence().refresh.lock().await);
    manager.evict_peer_evidence_for_test();
    observer.peer_change_tx.send_modify(|_| {});
    tokio::select! {
        _ = &mut maintenance => anyhow::bail!("evidence maintenance stopped"),
        _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {}
    }
    assert_eq!(calls.load(Ordering::SeqCst), verified);
    assert!(observer.eligible_peer(origin.id()).await.is_none());
    drop(maintenance);
    observer.close_endpoint().await;
    origin.close_endpoint().await;
    Ok(())
}

#[tokio::test]
async fn peer_evidence_refresh_keeps_requests_arriving_during_a_batch() -> Result<()> {
    struct FirstPending {
        calls: Arc<AtomicUsize>,
        entered: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }
    impl PluginRpcBridge for FirstPending {
        fn handle_request(
            &self,
            _: String,
            _: String,
            _: String,
        ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
            let first = self.calls.fetch_add(1, Ordering::SeqCst) == 0;
            let entered = self.entered.clone();
            let release = self.release.clone();
            Box::pin(async move {
                if first {
                    entered.notify_one();
                    release.notified().await;
                }
                Ok(RpcResult {
                    result_json: json!({"accepted":true,"cache_until_restart":true}).to_string(),
                })
            })
        }
        fn handle_notification(&self, _: String, _: String, _: String) -> BridgeFuture<()> {
            Box::pin(async {})
        }
    }
    let first = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let second = make_test_node(LocalRole::Host { http_port: 9338 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    for node in [&first, &second, &observer] {
        node.set_mesh_id("queued-refresh".into()).await;
        node.start_accepting();
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let entered = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    *observer.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
        Arc::new(FirstPending {
            calls: calls.clone(),
            entered: entered.clone(),
            release: release.clone(),
        }),
        None,
    ));
    observer
        .connect_to_peer(first.endpoint_addr_for_advertisement())
        .await?;
    tokio::time::timeout(std::time::Duration::from_secs(5), entered.notified()).await?;
    observer
        .connect_to_peer(second.endpoint_addr_for_advertisement())
        .await?;
    release.notify_one();
    wait_for_eligible_peer(&observer, first.id()).await;
    wait_for_eligible_peer(&observer, second.id()).await;
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    for node in [&first, &second, &observer] {
        node.close_endpoint().await;
    }
    Ok(())
}

#[tokio::test]
async fn peer_evidence_renews_at_its_deadline_without_waiting_for_gossip() -> Result<()> {
    struct ShortLease(Arc<AtomicUsize>);
    impl PluginRpcBridge for ShortLease {
        fn handle_request(
            &self,
            _: String,
            _: String,
            _: String,
        ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
            let first = self.0.fetch_add(1, Ordering::SeqCst) == 0;
            Box::pin(async move {
                Ok(RpcResult {
                    result_json: json!({
                        "accepted":true,
                        "valid_until":now_secs()+1,
                        "cache_until_restart":!first
                    })
                    .to_string(),
                })
            })
        }
        fn handle_notification(&self, _: String, _: String, _: String) -> BridgeFuture<()> {
            Box::pin(async {})
        }
    }
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    for node in [&origin, &observer] {
        node.set_mesh_id("expiry-refresh".into()).await;
        node.start_accepting();
    }
    let calls = Arc::new(AtomicUsize::new(0));
    *observer.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
        Arc::new(ShortLease(calls.clone())),
        None,
    ));
    observer.start_heartbeat();
    observer
        .connect_to_peer(origin.endpoint_addr_for_advertisement())
        .await?;
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while calls.load(Ordering::SeqCst) < 2 {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await?;
    wait_for_eligible_peer(&observer, origin.id()).await;
    for _ in 0..3 {
        assert!(observer.eligible_peer(origin.id()).await.is_some());
    }
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    observer.close_endpoint().await;
    origin.close_endpoint().await;
    Ok(())
}

async fn wait_for_eligible_peer(node: &Node, target: iroh::EndpointId) {
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while !node
            .eligible_peers()
            .await
            .iter()
            .any(|peer| peer.id == target)
        {
            tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("provider eligibility was not established within 5 s");
}

impl PluginRpcBridge for Verifier {
    fn handle_request(
        &self,
        _: String,
        _: String,
        params: String,
    ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        let require_bond = self.1;
        Box::pin(async move {
            let input: Value = serde_json::from_str(&params).unwrap();
            let arguments = &input["arguments"];
            Ok(RpcResult {result_json:json!({"accepted":(!require_bond && arguments["evidence"].is_null()) || (arguments["evidence"]["host"] == arguments["node_id"] && arguments["evidence"]["mesh"] == arguments["mesh_id"]), "valid_until":now_secs()+3600}).to_string()})
        })
    }
    fn handle_notification(&self, _: String, _: String, _: String) -> BridgeFuture<()> {
        Box::pin(async {})
    }
}

#[tokio::test]
async fn peer_evidence_policy_can_admit_unbonded_supply_without_repeated_verification() -> Result<()>
{
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    for node in [&origin, &observer] {
        node.set_mesh_id("optional-bond".into()).await;
    }
    origin.set_serving_models(vec!["free-model".into()]).await;
    origin.set_hosted_models(vec!["free-model".into()]).await;
    let calls = Arc::new(AtomicUsize::new(0));
    let manager =
        PluginManager::for_test_peer_evidence(Arc::new(Verifier(calls.clone(), false)), None);
    *observer.plugin_manager.lock().await = Some(manager);
    origin.start_accepting();
    observer.start_accepting();
    observer
        .connect_to_peer(origin.endpoint_addr_for_advertisement())
        .await?;
    wait_for_eligible_peer(&observer, origin.id()).await;
    let verified = calls.load(Ordering::SeqCst);
    assert!(verified > 0);
    for _ in 0..3 {
        assert!(
            observer
                .hosts_for_model("free-model")
                .await
                .contains(&origin.id())
        );
        assert!(
            observer
                .collect_outbound_gossip()
                .await
                .announcements
                .iter()
                .any(|ann| ann.addr.id == origin.id())
        );
    }
    assert_eq!(calls.load(Ordering::SeqCst), verified);
    observer.close_endpoint().await;
    origin.close_endpoint().await;
    Ok(())
}

#[tokio::test]
async fn legacy_missing_evidence_is_decided_by_the_verifier() -> Result<()> {
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    for node in [&origin, &observer] {
        node.set_mesh_id("legacy-proof".into()).await;
    }
    origin.set_serving_models(vec!["legacy-model".into()]).await;
    origin.set_hosted_models(vec!["legacy-model".into()]).await;
    let ann = origin
        .collect_own_outbound_gossip()
        .await
        .announcements
        .remove(0);
    for required in [false, true] {
        *observer.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
            Arc::new(Verifier(Arc::default(), required)),
            None,
        ));
        let resolved = observer
            .resolve_inbound_gossip(
                origin.id(),
                crate::protocol::InboundGossip {
                    announcements: vec![(ann.addr.clone(), ann.clone())],
                    signed_records: Vec::new(),
                    signed_cache_affinity: Vec::new(),
                },
            )
            .await;
        assert_eq!(
            resolved.len(),
            1,
            "membership preserves the direct envelope"
        );
        observer
            .apply_announced_peers(origin.id(), &resolved, None, None, true)
            .await?;
        let manager = observer.plugin_manager().await.unwrap();
        manager
            .verify_provider_evidence(
                "legacy-proof",
                &origin.id().to_string(),
                &Default::default(),
            )
            .await;
        assert_eq!(
            observer
                .evidence_announcement(origin.id(), &ann)
                .await
                .is_some(),
            !required
        );
    }
    observer.close_endpoint().await;
    origin.close_endpoint().await;
    Ok(())
}

#[tokio::test]
async fn newest_authentic_supply_is_retained_when_a_proof_is_withdrawn() -> Result<()> {
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    for node in [&origin, &observer] {
        node.set_mesh_id("latest-proof".into()).await;
    }
    origin.set_serving_models(vec!["old-model".into()]).await;
    origin.set_hosted_models(vec!["old-model".into()]).await;
    *origin.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(Arc::default(), true)),
        Some(json!({"host":origin.id().to_string(),"mesh":"latest-proof"})),
    ));
    *observer.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(Arc::default(), true)),
        None,
    ));
    origin.start_accepting();
    observer.start_accepting();
    observer
        .connect_to_peer(origin.endpoint_addr_for_advertisement())
        .await?;
    wait_for_eligible_peer(&observer, origin.id()).await;
    assert!(
        observer
            .hosts_for_model("old-model")
            .await
            .contains(&origin.id())
    );
    origin.set_serving_models(vec!["new-model".into()]).await;
    origin.set_hosted_models(vec!["new-model".into()]).await;
    *origin.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(Arc::default(), true)),
        None,
    ));
    let conn = observer
        .state
        .lock()
        .await
        .connections
        .get(&origin.id())
        .unwrap()
        .clone();
    observer
        .gossip_existing_heartbeat_connection(origin.id(), conn)
        .await;
    assert!(observer.hosts_for_model("old-model").await.is_empty());
    assert!(observer.hosts_for_model("new-model").await.is_empty());
    let latest = observer.peer_evidence_record(origin.id()).await.unwrap();
    assert!(latest.peer_evidence.is_empty());
    assert_eq!(latest.hosted_models, vec!["new-model"]);
    *observer.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(Arc::default(), false)),
        None,
    ));
    observer.refresh_peer_evidence().await;
    assert!(
        observer
            .hosts_for_model("new-model")
            .await
            .contains(&origin.id())
    );
    assert!(observer.hosts_for_model("old-model").await.is_empty());
    observer.close_endpoint().await;
    origin.close_endpoint().await;
    Ok(())
}

#[tokio::test]
async fn peer_evidence_two_nodes_admit_signed_supply_and_reject_unbonded_hosts() -> Result<()> {
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    let unfunded = make_test_node(LocalRole::Host { http_port: 9338 }).await?;
    for node in [&origin, &observer, &unfunded] {
        node.set_mesh_id("bond-test".into()).await;
    }
    origin.set_serving_models(vec!["bonded-model".into()]).await;
    origin.set_hosted_models(vec!["bonded-model".into()]).await;
    unfunded.set_serving_models(vec!["spam-model".into()]).await;
    let calls = Arc::new(AtomicUsize::new(0));
    let provider = PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(calls.clone(), true)),
        Some(json!({"host":origin.id().to_string(),"mesh":"bond-test"})),
    );
    let buyer =
        PluginManager::for_test_peer_evidence(Arc::new(Verifier(calls.clone(), true)), None);
    *origin.plugin_manager.lock().await = Some(provider);
    *observer.plugin_manager.lock().await = Some(buyer.clone());
    for node in [&origin, &observer, &unfunded] {
        node.start_accepting();
    }
    observer
        .connect_to_peer(origin.endpoint_addr_for_advertisement())
        .await?;
    wait_for_eligible_peer(&observer, origin.id()).await;
    assert!(
        observer
            .peers()
            .await
            .iter()
            .any(|peer| peer.id == origin.id())
    );
    assert!(
        observer
            .hosts_for_model("bonded-model")
            .await
            .contains(&origin.id())
    );
    assert!(calls.load(Ordering::SeqCst) > 0);
    let _ = observer
        .connect_to_peer(unfunded.endpoint_addr_for_advertisement())
        .await;
    assert!(
        !observer
            .eligible_peers()
            .await
            .iter()
            .any(|peer| peer.id == unfunded.id())
    );
    assert!(
        observer
            .peers()
            .await
            .iter()
            .any(|peer| peer.id == unfunded.id())
    );
    assert!(
        observer
            .state
            .lock()
            .await
            .node_records
            .contains(&unfunded.id())
    );
    let outbound = observer.collect_outbound_gossip().await;
    assert!(
        outbound
            .announcements
            .iter()
            .all(|ann| !ann.serving_models.contains(&"spam-model".into()))
    );
    // A stopped verifier has no concurrent background checks.
    let _refresh = buyer.peer_evidence().refresh.lock().await;
    buyer.peer_evidence().clear_plugin("evidence");
    assert!(observer.hosts_for_model("bonded-model").await.is_empty());
    assert!(
        !observer
            .collect_outbound_gossip()
            .await
            .announcements
            .iter()
            .any(|ann| ann.addr.id == origin.id())
    );
    for node in [&unfunded, &observer, &origin] {
        node.close_endpoint().await;
    }
    Ok(())
}

#[tokio::test]
async fn peer_evidence_signed_reference_cannot_authorize_unsigned_supply_or_another_key()
-> Result<()> {
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    let attacker = make_test_node(LocalRole::Worker).await?;
    for node in [&origin, &observer, &attacker] {
        node.set_mesh_id("bond-test".into()).await;
    }
    origin.set_serving_models(vec!["signed-model".into()]).await;
    let evidence = json!({"host":origin.id().to_string(),"mesh":"bond-test"});
    for node in [&origin, &attacker] {
        *node.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
            Arc::new(Verifier(Arc::default(), true)),
            Some(evidence.clone()),
        ));
    }
    *observer.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(Arc::default(), true)),
        None,
    ));
    let outbound = origin.collect_own_outbound_gossip().await;
    let mut forged = outbound.announcements[0].clone();
    forged.serving_models = vec!["unsigned-model".into()];
    let resolved = observer
        .resolve_inbound_gossip(
            origin.id(),
            crate::protocol::InboundGossip {
                announcements: vec![(forged.addr.clone(), forged)],
                signed_records: outbound.signed_records,
                signed_cache_affinity: Vec::new(),
            },
        )
        .await;
    let evidence = observer
        .peer_evidence_record(origin.id())
        .await
        .unwrap()
        .peer_evidence;
    observer
        .plugin_manager()
        .await
        .unwrap()
        .verify_provider_evidence("bond-test", &origin.id().to_string(), &evidence)
        .await;
    assert_eq!(
        observer
            .evidence_announcement(origin.id(), &resolved[0].1)
            .await
            .unwrap()
            .serving_models,
        vec!["signed-model"]
    );
    // The attacker's own valid signature does not change the proof's host binding.
    attacker
        .set_serving_models(vec!["replayed-model".into()])
        .await;
    let replay = attacker.collect_own_outbound_gossip().await;
    let resolved = observer
        .resolve_inbound_gossip(
            attacker.id(),
            crate::protocol::InboundGossip {
                announcements: replay
                    .announcements
                    .into_iter()
                    .map(|ann| (ann.addr.clone(), ann))
                    .collect(),
                signed_records: replay.signed_records,
                signed_cache_affinity: Vec::new(),
            },
        )
        .await;
    let evidence = observer
        .peer_evidence_record(attacker.id())
        .await
        .unwrap()
        .peer_evidence;
    observer
        .plugin_manager()
        .await
        .unwrap()
        .verify_provider_evidence("bond-test", &attacker.id().to_string(), &evidence)
        .await;
    assert!(
        observer
            .evidence_announcement(attacker.id(), &resolved[0].1)
            .await
            .is_none()
    );
    assert!(
        observer
            .state
            .lock()
            .await
            .node_records
            .contains(&attacker.id())
    );
    for node in [&attacker, &observer, &origin] {
        node.close_endpoint().await;
    }
    Ok(())
}

#[tokio::test]
async fn expired_evidence_blocks_every_provider_view_without_deleting_records() -> Result<()> {
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    for node in [&origin, &observer] {
        node.set_mesh_id("retained-test".into()).await;
    }
    origin
        .set_serving_models(vec!["retained-model".into()])
        .await;
    origin
        .set_hosted_models(vec!["retained-model".into()])
        .await;
    origin
        .cache_affinity_inventory
        .lock()
        .unwrap()
        .record_l1_hit("retained-model", 42, 512, 4, 0);
    let calls = Arc::new(AtomicUsize::new(0));
    *origin.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(calls.clone(), true)),
        Some(json!({"host":origin.id().to_string(),"mesh":"retained-test"})),
    ));
    let manager =
        PluginManager::for_test_peer_evidence(Arc::new(Verifier(calls.clone(), true)), None);
    *observer.plugin_manager.lock().await = Some(manager.clone());
    origin.start_accepting();
    observer.start_accepting();
    observer
        .connect_to_peer(origin.endpoint_addr_for_advertisement())
        .await?;
    wait_for_eligible_peer(&observer, origin.id()).await;
    let signed = observer
        .state
        .lock()
        .await
        .node_records
        .get(&origin.id())
        .unwrap()
        .signed_bytes()
        .to_vec();
    let affinity = observer
        .state
        .lock()
        .await
        .cache_affinity_records
        .get(&origin.id())
        .map(|record| record.signed_bytes().to_vec());
    assert!(
        affinity.is_some(),
        "fixture publishes signed cache affinity"
    );
    assert!(
        observer
            .collect_outbound_gossip()
            .await
            .announcements
            .iter()
            .any(|ann| ann.addr.id == origin.id() && ann.cache_affinity.is_some())
    );
    assert!(
        observer
            .routing_table()
            .await
            .hosts
            .iter()
            .any(|host| host.endpoint_id == origin.id())
    );
    let _refresh = manager.peer_evidence().refresh.lock().await;
    let verified = calls.load(Ordering::SeqCst);
    observer.routing_table().await;
    observer.hosts_for_model("retained-model").await;
    assert_eq!(calls.load(Ordering::SeqCst), verified);

    manager.expire_peer_evidence_for_test();
    assert!(observer.hosts_for_model("retained-model").await.is_empty());
    assert!(observer.models_being_served().await.is_empty());
    assert!(observer.mesh_catalog().await.is_empty());
    assert!(observer.routing_table().await.hosts.is_empty());
    assert!(observer.any_host().await.is_none());
    assert!(observer.connection_to_peer(origin.id()).await.is_err());
    assert!(
        observer
            .stage_connection_to_peer(origin.id())
            .await
            .is_err()
    );
    let outbound = observer.collect_outbound_gossip().await;
    assert!(
        !outbound
            .announcements
            .iter()
            .any(|ann| ann.addr.id == origin.id())
    );
    assert!(
        !outbound
            .signed_records
            .iter()
            .any(|record| record.signed == signed)
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        verified,
        "enforcement must not invoke the verifier"
    );
    {
        let state = observer.state.lock().await;
        assert!(state.peers.contains_key(&origin.id()));
        assert!(state.connections.contains_key(&origin.id()));
        assert_eq!(
            state.node_records.get(&origin.id()).unwrap().signed_bytes(),
            signed
        );
        assert_eq!(
            state
                .cache_affinity_records
                .get(&origin.id())
                .map(|record| record.signed_bytes().to_vec()),
            affinity
        );
    }
    observer.refresh_peer_evidence().await;
    assert!(
        observer
            .hosts_for_model("retained-model")
            .await
            .contains(&origin.id())
    );
    manager.expire_peer_evidence_for_test();
    // Ineligible providers remain live members while their supply is blocked.
    *origin.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(calls.clone(), true)),
        Some(json!({"host":"wrong-key","mesh":"retained-test"})),
    ));
    {
        let mut state = observer.state.lock().await;
        let peer = state.peers.get_mut(&origin.id()).unwrap();
        peer.last_seen =
            std::time::Instant::now() - std::time::Duration::from_secs(PEER_STALE_SECS * 3);
        peer.last_mentioned = peer.last_seen;
    }
    let conn = observer
        .state
        .lock()
        .await
        .connections
        .get(&origin.id())
        .unwrap()
        .clone();
    assert_eq!(
        observer
            .gossip_existing_heartbeat_connection(origin.id(), conn)
            .await,
        crate::mesh::heartbeat::GossipHealth::Alive
    );
    observer.prune_stale_heartbeat_peers().await;
    assert!(observer.state.lock().await.peers.contains_key(&origin.id()));
    assert!(observer.routing_table().await.hosts.is_empty());
    assert_ne!(
        observer
            .state
            .lock()
            .await
            .node_records
            .get(&origin.id())
            .unwrap()
            .signed_bytes(),
        signed
    );
    observer.refresh_peer_evidence().await;
    let before_refresh = calls.load(Ordering::SeqCst);
    observer.refresh_peer_evidence().await;
    assert_eq!(
        calls.load(Ordering::SeqCst),
        before_refresh,
        "rejected evidence uses the same cached decision"
    );
    assert!(observer.hosts_for_model("retained-model").await.is_empty());
    assert!(observer.routing_table().await.hosts.is_empty());
    assert!(
        observer
            .collect_outbound_gossip()
            .await
            .signed_records
            .iter()
            .all(|record| record.signed != signed)
    );
    observer.close_endpoint().await;
    origin.close_endpoint().await;
    Ok(())
}

mod connections;
mod enforcement;
mod maintenance;
mod record_budget;
