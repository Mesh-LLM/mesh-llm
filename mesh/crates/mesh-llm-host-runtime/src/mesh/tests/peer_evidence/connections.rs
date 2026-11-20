use super::*;

#[tokio::test]
async fn peer_evidence_allows_on_demand_admission_before_the_work_gate() -> Result<()> {
    for cached_connection in [false, true] {
        let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
        let observer = make_test_node(LocalRole::Client).await?;
        for node in [&origin, &observer] {
            node.set_mesh_id("on-demand-admission".into()).await;
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
        drop(manager.peer_evidence().refresh.lock().await);
        let verified = calls.load(Ordering::SeqCst);
        {
            let mut state = observer.state.lock().await;
            state.peers.get_mut(&origin.id()).unwrap().admitted = false;
            if !cached_connection {
                state.connections.remove(&origin.id());
            }
        }
        assert!(
            observer
                .require_current_peer_evidence(origin.id())
                .await
                .is_err()
        );
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            observer.connection_to_peer(origin.id()),
        )
        .await??;
        assert!(observer.state.lock().await.peers[&origin.id()].is_admitted());
        assert_eq!(calls.load(Ordering::SeqCst), verified);
        observer.close_endpoint().await;
        origin.close_endpoint().await;
    }
    Ok(())
}

#[tokio::test]
async fn stage_dial_completes_gossip_before_requiring_provider_evidence() -> Result<()> {
    let worker = make_test_node(LocalRole::Worker).await?;
    let upstream = make_test_node(LocalRole::Worker).await?;
    let mesh = "on-demand-stage-admission";
    for node in [&worker, &upstream] {
        node.set_mesh_id(mesh.into()).await;
        node.start_accepting();
    }
    let proof = json!({"host":worker.id().to_string(),"mesh":mesh});
    let publisher = PluginManager::for_test_peer_evidence(
        Arc::new(Verifier(Arc::default(), true)),
        Some(proof),
    );
    *worker.plugin_manager.lock().await = Some(publisher);
    let calls = Arc::new(AtomicUsize::new(0));
    let verifier =
        PluginManager::for_test_peer_evidence(Arc::new(Verifier(calls.clone(), true)), None);
    *upstream.plugin_manager.lock().await = Some(verifier.clone());
    upstream
        .connect_to_peer(worker.endpoint_addr_for_advertisement())
        .await?;
    wait_for_eligible_peer(&upstream, worker.id()).await;
    drop(verifier.peer_evidence().refresh.lock().await);
    let verified = calls.load(Ordering::SeqCst);
    {
        let mut state = upstream.state.lock().await;
        state.peers.get_mut(&worker.id()).unwrap().admitted = false;
        state.connections.remove(&worker.id());
    }
    assert!(
        upstream
            .require_current_peer_evidence(worker.id())
            .await
            .is_err()
    );

    let connection = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        upstream.stage_connection_to_peer(worker.id()),
    )
    .await??;
    assert_eq!(connection.remote_id(), worker.id());
    assert!(upstream.state.lock().await.peers[&worker.id()].is_admitted());
    assert_eq!(calls.load(Ordering::SeqCst), verified);
    upstream.close_endpoint().await;
    worker.close_endpoint().await;
    Ok(())
}

#[tokio::test]
async fn peer_evidence_abandoned_plugin_attachment_stops_when_the_connection_closes() -> Result<()>
{
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    for node in [&origin, &observer] {
        node.set_mesh_id("abandoned-attachment".into()).await;
        node.start_accepting();
    }
    observer
        .connect_to_peer(origin.endpoint_addr_for_advertisement())
        .await?;
    let conn = observer.state.lock().await.connections[&origin.id()].clone();
    let listener = crate::plugin::bind_local_listener(
        &uuid::Uuid::new_v4().to_string(),
        "abandoned-attachment",
    )
    .await?;
    #[cfg(unix)]
    let path = std::path::PathBuf::from(listener.endpoint());
    let node = observer.clone();
    let target_peer_id = origin.id().to_string();
    let bridge = tokio::spawn(async move {
        crate::mesh::plugin_streams::bridge_outbound_plugin_mesh_stream(
            node,
            listener,
            conn,
            proto::OpenMeshStreamRequest {
                target_peer_id,
                ..Default::default()
            },
        )
        .await
    });
    observer.close_endpoint().await;
    let error = tokio::time::timeout(std::time::Duration::from_secs(1), bridge)
        .await??
        .unwrap_err();
    assert!(error.to_string().contains("closed before local attachment"));
    #[cfg(unix)]
    assert!(!path.exists(), "abandoned socket pathname was not removed");
    origin.close_endpoint().await;
    Ok(())
}
