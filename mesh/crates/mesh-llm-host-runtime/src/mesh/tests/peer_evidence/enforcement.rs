use super::*;

#[tokio::test]
async fn peer_evidence_expiry_allows_stage_status_and_stop_but_blocks_new_claims() -> Result<()> {
    use crate::inference::skippy::{
        StageControlRequest, StageControlResponse, StageStatusFilter, StageStopRequest,
    };
    let coordinator = make_test_node(LocalRole::Worker).await?;
    let worker = make_test_node(LocalRole::Worker).await?;
    for node in [&coordinator, &worker] {
        node.set_mesh_id("stage-cleanup".into()).await;
    }
    let control = crate::inference::skippy::spawn_stage_control_loop(Default::default());
    *worker.stage_control_tx.lock().await = Some(control.sender());
    let manager =
        PluginManager::for_test_peer_evidence(Arc::new(Verifier(Arc::default(), false)), None);
    *coordinator.plugin_manager.lock().await = Some(manager.clone());
    coordinator.start_accepting();
    worker.start_accepting();
    coordinator
        .connect_to_peer(worker.endpoint_addr_for_advertisement())
        .await?;
    wait_for_eligible_peer(&coordinator, worker.id()).await;
    coordinator
        .state
        .lock()
        .await
        .peers
        .get_mut(&worker.id())
        .unwrap()
        .stage_protocol_generation_supported = true;
    let mut claim = identity_test_claim(coordinator.id(), 1);
    claim.lease_until_unix_ms = crate::mesh::current_time_unix_ms() + 60_000;
    assert!(matches!(
        coordinator
            .send_stage_control(worker.id(), StageControlRequest::Claim(claim.clone()))
            .await?,
        StageControlResponse::ClaimAccepted(_)
    ));
    let _refresh = manager.peer_evidence().refresh.lock().await;
    manager.expire_peer_evidence_for_test();
    let mut load = identity_test_load_request();
    load.topology_stages[0].node_id = coordinator.id();
    load.topology_stages[1].node_id = worker.id();
    worker.record_stage_load_topology(&load).await;
    assert!(matches!(
        coordinator
            .send_stage_control(
                worker.id(),
                StageControlRequest::Status(StageStatusFilter {
                    topology_id: Some(claim.topology_id.clone()),
                    run_id: Some(claim.run_id.clone()),
                    stage_id: None,
                })
            )
            .await?,
        StageControlResponse::Status(_)
    ));
    let response = coordinator
        .send_stage_control(
            worker.id(),
            StageControlRequest::Stop(StageStopRequest {
                topology_id: claim.topology_id.clone(),
                run_id: claim.run_id.clone(),
                stage_id: "stage-1".into(),
                coordinator_term: 1,
                shutdown_generation: 1,
            }),
        )
        .await?;
    assert!(matches!(response, StageControlResponse::Ready(ready) if ready.accepted));
    assert!(
        coordinator
            .send_stage_control(worker.id(), StageControlRequest::Claim(claim))
            .await
            .unwrap_err()
            .to_string()
            .contains("eligibility")
    );
    coordinator.close_endpoint().await;
    worker.close_endpoint().await;
    control.shutdown().await?;
    Ok(())
}

#[tokio::test]
async fn peer_evidence_routing_follows_relayed_signed_supply_without_reverification() -> Result<()>
{
    use crate::inference::election::InferenceTarget;
    use mesh_llm_membership::signed_record::{NODE_RECORD, VerifiedRecord};
    fn descriptor(vision: bool) -> crate::mesh::ServedModelDescriptor {
        crate::mesh::ServedModelDescriptor {
            identity: crate::mesh::ServedModelIdentity {
                model_name: "model".into(),
                ..Default::default()
            },
            capabilities_known: true,
            capabilities: crate::models::ModelCapabilities {
                vision: if vision {
                    crate::models::CapabilityLevel::Supported
                } else {
                    crate::models::CapabilityLevel::None
                },
                ..Default::default()
            },
            ..Default::default()
        }
    }
    let old_vision = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let new_vision = make_test_node(LocalRole::Host { http_port: 9338 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    let calls = Arc::new(AtomicUsize::new(0));
    *observer.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(calls.clone(), false)),
        None,
    ));
    for node in [&old_vision, &new_vision, &observer] {
        node.set_mesh_id("signed-capabilities".into()).await;
        node.start_accepting();
    }
    for (node, vision, context) in [(&old_vision, true, 4096), (&new_vision, false, 8192)] {
        node.set_serving_models(vec!["model".into()]).await;
        node.set_hosted_models(vec!["model".into()]).await;
        node.set_served_model_descriptors(vec![descriptor(vision)])
            .await;
        node.set_model_runtime_context_length("model", Some(context))
            .await;
        observer
            .connect_to_peer(node.endpoint_addr_for_advertisement())
            .await?;
        wait_for_eligible_peer(&observer, node.id()).await;
    }
    let verified = calls.load(Ordering::SeqCst);
    for (node, vision, context) in [(&old_vision, false, 8192), (&new_vision, true, 4096)] {
        node.set_served_model_descriptors(vec![descriptor(vision)])
            .await;
        node.set_model_runtime_context_length("model", Some(context))
            .await;
        let mut gossip = node.collect_own_outbound_gossip().await;
        if !vision {
            let record = &mut gossip.signed_records[0];
            let verified = VerifiedRecord::verify(NODE_RECORD, &record.signed, &record.signature)?;
            let mut body = crate::proto::node::NodeRecord::decode(verified.body())?;
            body.refuses_peer_model_downloads = Some(true);
            body.admission_state =
                crate::proto::node::InferenceAdmissionState::AcceptingDeprioritized as i32;
            let replacement = VerifiedRecord::sign(
                NODE_RECORD,
                &ed25519_dalek::SigningKey::from_bytes(&node.endpoint_secret_key.to_bytes()),
                verified.header().seq + 1,
                verified.header().issued_at_unix_ms,
                &body.encode_to_vec(),
            );
            record.signed = replacement.signed_bytes().to_vec();
            record.signature = replacement.signature().to_vec();
        }
        observer
            .resolve_inbound_gossip(
                make_test_endpoint_id(0xfb),
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
    }
    let raw = observer.peers().await;
    assert!(raw.iter().all(|peer| peer.inference_admission_state
        != Some(crate::proto::node::InferenceAdmissionState::AcceptingDeprioritized)));
    assert!(raw.iter().all(|peer| !peer.refuses_peer_model_downloads));
    assert!(
        observer
            .eligible_peer(old_vision.id())
            .await
            .unwrap()
            .refuses_peer_model_downloads
    );
    assert_eq!(
        observer.deprioritized_peer_ids().await,
        std::collections::HashSet::from([old_vision.id()])
    );
    assert_eq!(
        super::super::super::advertised_context_length(
            raw.iter().find(|peer| peer.id == old_vision.id()).unwrap(),
            "model"
        ),
        Some(4096)
    );
    assert_eq!(
        observer
            .peer_model_context_length(old_vision.id(), "model")
            .await,
        Some(8192)
    );
    assert_eq!(
        observer
            .peer_model_context_length(new_vision.id(), "model")
            .await,
        Some(4096)
    );
    let body = json!({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AA=="}}]}]});
    let targets = crate::network::openai::workload_routing::media_capable_targets(
        &observer,
        "model",
        "/v1/chat/completions",
        Some(&body),
        vec![
            InferenceTarget::Remote(old_vision.id()),
            InferenceTarget::Remote(new_vision.id()),
        ],
    )
    .await;
    assert!(matches!(targets.as_slice(), [InferenceTarget::Remote(id)] if *id == new_vision.id()));
    new_vision
        .set_served_model_descriptors(vec![descriptor(false)])
        .await;
    let gossip = new_vision.collect_own_outbound_gossip().await;
    observer
        .resolve_inbound_gossip(
            make_test_endpoint_id(0xfb),
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
    let descriptors = observer.all_served_model_descriptors().await;
    assert_eq!(descriptors.len(), 2);
    assert!(
        descriptors
            .iter()
            .all(|descriptor| descriptor.capabilities.vision
                != crate::models::CapabilityLevel::Supported)
    );
    assert_eq!(calls.load(Ordering::SeqCst), verified);
    for node in [&observer, &old_vision, &new_vision] {
        node.close_endpoint().await;
    }
    Ok(())
}

#[tokio::test]
async fn peer_evidence_publications_are_snapshotted_under_the_record_signing_lock() -> Result<()> {
    use futures_util::FutureExt as _;
    use mesh_llm_membership::signed_record::{NODE_RECORD, VerifiedRecord};
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let manager = PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(Arc::default(), false)),
        Some(json!({"reference":"old"})),
    );
    *origin.plugin_manager.lock().await = Some(manager.clone());
    let mut announcements = origin.collect_own_outbound_gossip().await.announcements;
    let state = origin.state.lock().await;
    let collecting = origin.collect_signed_records(&mut announcements, None);
    tokio::pin!(collecting);
    assert!(collecting.as_mut().now_or_never().is_none());
    manager.peer_evidence().clear_plugin("evidence");
    drop(state);
    let (records, _) = collecting.await;
    let signed = VerifiedRecord::verify(NODE_RECORD, &records[0].signed, &records[0].signature)?;
    let record = crate::proto::node::NodeRecord::decode(signed.body())?;
    assert!(record.peer_evidence.is_empty());
    origin.close_endpoint().await;
    Ok(())
}

#[tokio::test]
async fn peer_evidence_preserves_membership_and_discovery_counts_without_advertising_rejected_supply()
-> Result<()> {
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    for node in [&origin, &observer] {
        node.set_mesh_id("membership-view".into()).await;
    }
    origin.set_serving_models(vec!["withheld".into()]).await;
    origin.set_hosted_models(vec!["withheld".into()]).await;
    *observer.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(Arc::default(), true)),
        None,
    ));
    origin.start_accepting();
    observer.start_accepting();
    observer
        .connect_to_peer(origin.endpoint_addr_for_advertisement())
        .await?;
    observer.refresh_peer_evidence().await;
    assert!(
        observer
            .peers()
            .await
            .iter()
            .any(|peer| peer.id == origin.id())
    );
    assert!(observer.eligible_peers().await.is_empty());
    assert!(observer.routable_peers().await.is_empty());
    let listing =
        crate::network::discovery::build_local_mesh_listing(&observer, None, None, None).await;
    assert_eq!(listing.node_count, 2);
    assert!(!listing.serving.iter().any(|model| model == "withheld"));
    assert!(!listing.on_disk.iter().any(|model| model == "withheld"));
    observer.close_endpoint().await;
    origin.close_endpoint().await;
    Ok(())
}

#[tokio::test]
async fn peer_evidence_initial_rejection_retains_membership_and_recovers_without_gossip()
-> Result<()> {
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    for node in [&origin, &observer] {
        node.set_mesh_id("initial-outage".into()).await;
    }
    origin.set_serving_models(vec!["recoverable".into()]).await;
    origin.set_hosted_models(vec!["recoverable".into()]).await;
    *observer.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(Arc::default(), true)),
        None,
    ));
    origin.start_accepting();
    observer.start_accepting();
    observer
        .connect_to_peer(origin.endpoint_addr_for_advertisement())
        .await?;
    observer.refresh_peer_evidence().await;
    assert!(observer.hosts_for_model("recoverable").await.is_empty());
    observer.prune_stale_heartbeat_peers().await;
    assert!(
        observer
            .state
            .lock()
            .await
            .node_records
            .contains(&origin.id())
    );
    *observer.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(Arc::default(), false)),
        None,
    ));
    observer.refresh_peer_evidence().await;
    assert!(
        observer
            .hosts_for_model("recoverable")
            .await
            .contains(&origin.id())
    );
    observer.close_endpoint().await;
    origin.close_endpoint().await;
    Ok(())
}

#[tokio::test]
async fn peer_evidence_preserves_direct_proofs_when_eligibility_fails() -> Result<()> {
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    let policy = requirement_policy_without_release_attestation();
    for node in [&origin, &observer] {
        configure_requirement_node(node, &policy, None).await?;
    }
    origin.set_serving_models(vec!["proof-model".into()]).await;
    *observer.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(Arc::default(), true)),
        None,
    ));
    origin.start_accepting();
    observer.start_accepting();
    observer
        .connect_to_peer(origin.endpoint_addr_for_advertisement())
        .await?;
    observer.refresh_peer_evidence().await;
    let conn = observer.state.lock().await.connections[&origin.id()].clone();
    assert_eq!(
        observer
            .gossip_existing_heartbeat_connection(origin.id(), conn)
            .await,
        crate::mesh::heartbeat::GossipHealth::Alive
    );
    assert!(
        !observer
            .state
            .lock()
            .await
            .requirement_rejected_peers
            .contains(&origin.id())
    );
    assert!(observer.hosts_for_model("proof-model").await.is_empty());
    observer.close_endpoint().await;
    origin.close_endpoint().await;
    Ok(())
}

#[tokio::test]
async fn peer_evidence_does_not_prevent_public_mesh_id_convergence() -> Result<()> {
    let mut origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let mut observer = make_test_node(LocalRole::Client).await?;
    origin.public_mesh = true;
    observer.public_mesh = true;
    origin.set_mesh_id("1111111111111111".into()).await;
    observer.set_mesh_id("eeeeeeeeeeeeeeee".into()).await;
    origin.set_serving_models(vec!["public-model".into()]).await;
    origin.set_hosted_models(vec!["public-model".into()]).await;
    *observer.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(Arc::default(), false)),
        None,
    ));
    origin.start_accepting();
    observer.start_accepting();
    observer
        .connect_to_peer(origin.endpoint_addr_for_advertisement())
        .await?;
    wait_for_eligible_peer(&observer, origin.id()).await;
    assert_eq!(observer.mesh_id().await, origin.mesh_id().await);
    assert!(
        observer
            .hosts_for_model("public-model")
            .await
            .contains(&origin.id())
    );
    observer.close_endpoint().await;
    origin.close_endpoint().await;
    Ok(())
}

#[tokio::test]
async fn peer_evidence_cannot_project_a_new_record_that_fails_mesh_policy() -> Result<()> {
    use mesh_llm_membership::signed_record::{NODE_RECORD, VerifiedRecord};
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    let relay = make_test_node(LocalRole::Client).await?;
    let policy = requirement_policy_without_release_attestation();
    for node in [&origin, &observer, &relay] {
        configure_requirement_node(node, &policy, None).await?;
    }
    origin
        .set_serving_models(vec!["original-model".into()])
        .await;
    *observer.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(Arc::default(), false)),
        None,
    ));
    origin.start_accepting();
    observer.start_accepting();
    observer
        .connect_to_peer(origin.endpoint_addr_for_advertisement())
        .await?;
    wait_for_eligible_peer(&observer, origin.id()).await;
    let current = origin
        .collect_own_outbound_gossip()
        .await
        .signed_records
        .remove(0);
    let current = VerifiedRecord::verify(NODE_RECORD, &current.signed, &current.signature).unwrap();
    let mut body = crate::proto::node::NodeRecord::decode(current.body()).unwrap();
    body.mesh_policy_hash = "wrong-policy".into();
    body.serving_models = vec!["unauthorized-model".into()];
    let key = ed25519_dalek::SigningKey::from_bytes(&origin.endpoint_secret_key.to_bytes());
    let replacement = VerifiedRecord::sign(
        NODE_RECORD,
        &key,
        current.header().seq + 1,
        current_time_unix_ms(),
        &body.encode_to_vec(),
    );
    let direct = relay
        .collect_own_outbound_gossip()
        .await
        .announcements
        .remove(0);
    let resolved = observer
        .resolve_inbound_gossip(
            relay.id(),
            crate::protocol::InboundGossip {
                announcements: vec![(direct.addr.clone(), direct)],
                signed_records: vec![crate::proto::node::SignedNodeRecord {
                    signed: replacement.signed_bytes().to_vec(),
                    signature: replacement.signature().to_vec(),
                    hop: None,
                }],
                signed_cache_affinity: Vec::new(),
            },
        )
        .await;
    observer
        .apply_announced_peers(
            relay.id(),
            &resolved,
            None,
            Some(crate::protocol::NODE_PROTOCOL_GENERATION),
            false,
        )
        .await?;
    observer.refresh_peer_evidence().await;
    assert!(
        observer
            .hosts_for_model("unauthorized-model")
            .await
            .is_empty()
    );
    assert!(
        observer
            .collect_outbound_gossip()
            .await
            .signed_records
            .iter()
            .all(|record| record.signed != replacement.signed_bytes())
    );
    for node in [&relay, &observer, &origin] {
        node.close_endpoint().await;
    }
    Ok(())
}

#[tokio::test]
async fn peer_evidence_is_checked_again_after_a_dedicated_stage_dial() -> Result<()> {
    let server = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .alpns(vec![skippy_protocol::STAGE_ALPN_V2.to_vec()])
        .relay_mode(iroh::RelayMode::Disabled)
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))?
        .bind()
        .await?;
    let observer = make_test_node(LocalRole::Client).await?;
    observer.set_mesh_id("stage-dial".into()).await;
    let manager =
        PluginManager::for_test_peer_evidence(Arc::new(Verifier(Arc::default(), false)), None);
    manager
        .verify_provider_evidence("stage-dial", &server.id().to_string(), &Default::default())
        .await;
    *observer.plugin_manager.lock().await = Some(manager.clone());
    let mut peer = make_test_peer(server.id(), None, 0);
    peer.addr = server.addr();
    observer.state.lock().await.peers.insert(server.id(), peer);
    let client = observer.clone();
    let id = server.id();
    let dial = tokio::spawn(async move { client.stage_connection_to_peer(id).await });
    let incoming = tokio::time::timeout(std::time::Duration::from_secs(5), server.accept())
        .await?
        .unwrap();
    manager.expire_peer_evidence_for_test();
    let connection = incoming.await?;
    assert!(
        dial.await?.is_err(),
        "expired eligibility must block the new stage connection"
    );
    connection.close(0u32.into(), b"test complete");
    observer.close_endpoint().await;
    server.close().await;
    Ok(())
}

#[tokio::test]
async fn peer_evidence_pending_verifier_does_not_block_discovery_or_gossip() -> Result<()> {
    struct Slow;
    impl PluginRpcBridge for Slow {
        fn handle_request(
            &self,
            _: String,
            _: String,
            _: String,
        ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
            Box::pin(std::future::pending())
        }
        fn handle_notification(&self, _: String, _: String, _: String) -> BridgeFuture<()> {
            Box::pin(async {})
        }
    }
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    for node in [&origin, &observer] {
        node.set_mesh_id("slow-verifier".into()).await;
    }
    *observer.plugin_manager.lock().await =
        Some(PluginManager::for_test_peer_evidence(Arc::new(Slow), None));
    origin.start_accepting();
    observer.start_accepting();
    let gossip = origin.collect_own_outbound_gossip().await;
    let announcements = observer
        .resolve_inbound_gossip(
            make_test_endpoint_id(0xfb),
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
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        observer.connect_discovered_peers(&announcements, true, false),
    )
    .await?;
    let conn = observer.state.lock().await.connections[&origin.id()].clone();
    assert!(observer.eligible_peers().await.is_empty());
    assert!(
        observer
            .collect_outbound_gossip()
            .await
            .announcements
            .iter()
            .all(|ann| ann.addr.id != origin.id())
    );
    let health = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        observer.gossip_existing_heartbeat_connection(origin.id(), conn),
    )
    .await?;
    assert_eq!(health, crate::mesh::heartbeat::GossipHealth::Alive);
    observer.close_endpoint().await;
    origin.close_endpoint().await;
    Ok(())
}

#[tokio::test]
async fn peer_evidence_preserves_catalog_models_for_signed_and_passthrough_peers() -> Result<()> {
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    for node in [&origin, &observer] {
        node.set_mesh_id("catalog".into()).await;
    }
    origin.set_models(vec!["catalog-only".into()]).await;
    *observer.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(Arc::default(), false)),
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
            .mesh_catalog()
            .await
            .contains(&"catalog-only".into())
    );
    let id = make_test_endpoint_id(0x76);
    let mut peer = make_test_peer(id, None, 0);
    peer.role = LocalRole::Client;
    peer.available_models = vec!["client-catalog".into()];
    observer.state.lock().await.peers.insert(id, peer);
    assert!(
        observer
            .mesh_catalog()
            .await
            .contains(&"client-catalog".into())
    );
    observer.close_endpoint().await;
    origin.close_endpoint().await;
    Ok(())
}

#[tokio::test]
async fn peer_evidence_gossip_omits_records_changed_after_eligibility_projection() -> Result<()> {
    use std::collections::HashMap;
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    for node in [&origin, &observer] {
        node.set_mesh_id("gossip-pin".into()).await;
    }
    origin.set_serving_models(vec!["pinned-model".into()]).await;
    *observer.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(Arc::default(), false)),
        None,
    ));
    origin.start_accepting();
    observer.start_accepting();
    observer
        .connect_to_peer(origin.endpoint_addr_for_advertisement())
        .await?;
    wait_for_eligible_peer(&observer, origin.id()).await;
    let ann = origin
        .collect_own_outbound_gossip()
        .await
        .announcements
        .remove(0);
    let (canonical, pin) = observer
        .evidence_announcement_with_record(origin.id(), &ann, true)
        .await
        .unwrap();
    let pins = HashMap::from([(origin.id(), pin)]);
    origin
        .set_serving_models(vec!["different-model".into()])
        .await;
    let replacement = origin.collect_own_outbound_gossip().await;
    observer
        .resolve_inbound_gossip(
            origin.id(),
            crate::protocol::InboundGossip {
                announcements: replacement
                    .announcements
                    .into_iter()
                    .map(|ann| (ann.addr.clone(), ann))
                    .collect(),
                signed_records: replacement.signed_records,
                signed_cache_affinity: Vec::new(),
            },
        )
        .await;
    let mut announcements = vec![canonical];
    let (signed, _) = observer
        .collect_signed_records(&mut announcements, Some(&pins))
        .await;
    assert!(announcements.is_empty());
    assert!(signed.is_empty());
    observer.close_endpoint().await;
    origin.close_endpoint().await;
    Ok(())
}

#[tokio::test]
async fn peer_evidence_gossip_rechecks_cached_eligibility_for_signed_and_legacy_peers() -> Result<()>
{
    for signed in [false, true] {
        let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
        let observer = make_test_node(LocalRole::Client).await?;
        for node in [&origin, &observer] {
            node.set_mesh_id("gossip-expiry".into()).await;
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let manager =
            PluginManager::for_test_peer_evidence(Arc::new(Verifier(calls.clone(), false)), None);
        *observer.plugin_manager.lock().await = Some(manager.clone());
        origin.start_accepting();
        observer.start_accepting();
        observer
            .connect_to_peer(origin.endpoint_addr_for_advertisement())
            .await?;
        wait_for_eligible_peer(&observer, origin.id()).await;
        let _refresh = manager.peer_evidence().refresh.lock().await;
        if !signed {
            observer
                .state
                .lock()
                .await
                .node_records
                .remove(&origin.id());
        }
        let ann = origin
            .collect_own_outbound_gossip()
            .await
            .announcements
            .remove(0);
        let (canonical, pin) = observer
            .evidence_announcement_with_record(origin.id(), &ann, true)
            .await
            .unwrap();
        assert_eq!(pin.is_some(), signed);
        let pins = HashMap::from([(origin.id(), pin)]);
        let verified = calls.load(Ordering::SeqCst);
        manager.expire_peer_evidence_for_test();
        let mut announcements = vec![canonical];
        let (records, affinity) = observer
            .collect_signed_records(&mut announcements, Some(&pins))
            .await;
        assert!(announcements.is_empty());
        assert!(records.is_empty());
        assert!(affinity.is_empty());
        assert_eq!(calls.load(Ordering::SeqCst), verified);
        observer.close_endpoint().await;
        origin.close_endpoint().await;
    }
    Ok(())
}

#[cfg(unix)]
#[tokio::test]
async fn peer_evidence_is_checked_after_the_plugin_connects_to_its_stream_socket() -> Result<()> {
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    for node in [&origin, &observer] {
        node.set_mesh_id("plugin-stream".into()).await;
    }
    let manager =
        PluginManager::for_test_peer_evidence(Arc::new(Verifier(Arc::default(), false)), None);
    *observer.plugin_manager.lock().await = Some(manager.clone());
    origin.start_accepting();
    observer.start_accepting();
    observer
        .connect_to_peer(origin.endpoint_addr_for_advertisement())
        .await?;
    wait_for_eligible_peer(&observer, origin.id()).await;
    let conn = observer.state.lock().await.connections[&origin.id()].clone();
    let listener =
        crate::plugin::bind_local_listener(&uuid::Uuid::new_v4().to_string(), "evidence-test")
            .await?;
    let endpoint = listener.endpoint();
    let kind = listener.transport_kind();
    let request = proto::OpenMeshStreamRequest {
        target_peer_id: origin.id().to_string(),
        ..Default::default()
    };
    let node = observer.clone();
    let bridge = tokio::spawn(async move {
        crate::mesh::plugin_streams::bridge_outbound_plugin_mesh_stream(
            node, listener, conn, request,
        )
        .await
    });
    let _refresh = manager.peer_evidence().refresh.lock().await;
    manager.expire_peer_evidence_for_test();
    let _local = crate::plugin::connect_test_side_stream(&endpoint, kind).await?;
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(5), bridge)
            .await??
            .is_err()
    );
    observer.close_endpoint().await;
    origin.close_endpoint().await;
    Ok(())
}

#[tokio::test]
async fn peer_evidence_projection_rejects_a_record_superseded_during_policy_checks() -> Result<()> {
    use futures_util::FutureExt as _;
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    for node in [&origin, &observer] {
        node.set_mesh_id("projection-race".into()).await;
    }
    *origin.plugin_manager.lock().await = Some(PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(Arc::default(), true)),
        Some(json!({"host":origin.id().to_string(),"mesh":"projection-race"})),
    ));
    let manager =
        PluginManager::for_test_peer_evidence(Arc::new(Verifier(Arc::default(), true)), None);
    *observer.plugin_manager.lock().await = Some(manager.clone());
    origin.start_accepting();
    observer.start_accepting();
    observer
        .connect_to_peer(origin.endpoint_addr_for_advertisement())
        .await?;
    wait_for_eligible_peer(&observer, origin.id()).await;
    let _refresh = manager.peer_evidence().refresh.lock().await;
    let ann = origin
        .collect_own_outbound_gossip()
        .await
        .announcements
        .remove(0);
    let trust = observer.trust_store.lock().await;
    let projection = observer.evidence_announcement(origin.id(), &ann);
    tokio::pin!(projection);
    assert!(projection.as_mut().now_or_never().is_none());
    *origin.plugin_manager.lock().await = None;
    let replacement = origin.collect_own_outbound_gossip().await;
    observer
        .resolve_inbound_gossip(
            origin.id(),
            crate::protocol::InboundGossip {
                announcements: replacement
                    .announcements
                    .into_iter()
                    .map(|ann| (ann.addr.clone(), ann))
                    .collect(),
                signed_records: replacement.signed_records,
                signed_cache_affinity: Vec::new(),
            },
        )
        .await;
    drop(trust);
    assert!(projection.await.is_none());
    assert!(observer.connectivity_snapshot().await.admitted_peer_count > 0);
    observer.close_endpoint().await;
    origin.close_endpoint().await;
    Ok(())
}

#[tokio::test]
async fn peer_evidence_is_checked_after_stream_credit_without_declaring_peer_death() -> Result<()> {
    use futures_util::FutureExt as _;
    let key = iroh::SecretKey::generate();
    let endpoint = iroh::Endpoint::builder(iroh::endpoint::presets::Minimal)
        .secret_key(key.clone())
        .alpns(vec![crate::protocol::ALPN_V1.to_vec()])
        .transport_config(
            iroh::endpoint::QuicTransportConfig::builder()
                .max_concurrent_bidi_streams(1u32.into())
                .build(),
        )
        .bind_addr(std::net::SocketAddr::from(([127, 0, 0, 1], 0)))?
        .bind()
        .await?;
    let origin =
        Node::new_test_node_from_endpoint(LocalRole::Host { http_port: 9337 }, endpoint, key);
    let observer = make_test_node(LocalRole::Client).await?;
    for node in [&origin, &observer] {
        node.set_mesh_id("stream-credit".into()).await;
    }
    let manager =
        PluginManager::for_test_peer_evidence(Arc::new(Verifier(Arc::default(), false)), None);
    *observer.plugin_manager.lock().await = Some(manager.clone());
    let server = origin.clone();
    let accept_loop = tokio::spawn(async move { server.accept_loop().await });
    origin.start_accepting();
    observer.start_accepting();
    observer
        .connect_to_peer(origin.endpoint_addr_for_advertisement())
        .await?;
    wait_for_eligible_peer(&observer, origin.id()).await;
    let _refresh = manager.peer_evidence().refresh.lock().await;
    let conn = observer.state.lock().await.connections[&origin.id()].clone();
    let (mut occupied, reply) = conn.open_bi().await?;
    occupied
        .write_all(&[crate::protocol::STREAM_GOSSIP])
        .await?;
    let opening = observer.open_http_tunnel(origin.id());
    tokio::pin!(opening);
    assert!(opening.as_mut().now_or_never().is_none());
    manager.expire_peer_evidence_for_test();
    drop(occupied);
    drop(reply);
    let error = tokio::time::timeout(std::time::Duration::from_secs(5), opening)
        .await?
        .unwrap_err();
    assert!(error.to_string().contains("eligibility"));
    assert!(
        !observer
            .state
            .lock()
            .await
            .dead_peers
            .contains_key(&origin.id())
    );
    assert!(observer.state.lock().await.peers.contains_key(&origin.id()));
    observer.close_endpoint().await;
    origin.close_endpoint().await;
    tokio::time::timeout(std::time::Duration::from_secs(5), accept_loop).await??;
    Ok(())
}

#[tokio::test]
async fn peer_evidence_filters_work_without_blocking_member_control_transport() -> Result<()> {
    let buyer = make_test_node(LocalRole::Worker).await?;
    let seller = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    for node in [&buyer, &seller] {
        node.set_mesh_id("control-membership".into()).await;
    }
    let manager =
        PluginManager::for_test_peer_evidence(Arc::new(Verifier(Arc::default(), true)), None);
    *seller.plugin_manager.lock().await = Some(manager);
    buyer.start_accepting();
    seller.start_accepting();
    seller
        .connect_to_peer(buyer.endpoint_addr_for_advertisement())
        .await?;
    seller.refresh_peer_evidence().await;
    assert!(seller.connection_to_peer(buyer.id()).await.is_err());
    assert!(seller.admitted_connection_to_peer(buyer.id()).await.is_ok());
    assert!(seller.connectivity_snapshot().await.admitted_peer_count > 0);
    let frame = proto::MeshChannelFrame {
        plugin_id: "tempo".into(),
        message_id: "quote-to-unbonded-buyer".into(),
        message: Some(proto::ChannelMessage {
            channel: "pay.stablecoin.v1".into(),
            source_peer_id: seller.id().to_string(),
            target_peer_id: buyer.id().to_string(),
            content_type: "application/json".into(),
            body: b"{}".to_vec(),
            ..Default::default()
        }),
        ..Default::default()
    };
    seller
        .forward_targeted_plugin_frame(
            &buyer.id().to_string(),
            crate::protocol::STREAM_PLUGIN_CHANNEL,
            frame.encode_to_vec(),
        )
        .await;
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while !buyer.has_plugin_message("quote-to-unbonded-buyer").await {
            tokio::task::yield_now().await;
        }
    })
    .await?;
    seller.close_endpoint().await;
    buyer.close_endpoint().await;
    Ok(())
}
