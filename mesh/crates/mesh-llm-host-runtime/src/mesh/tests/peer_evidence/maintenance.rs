use super::*;
use std::{future::Future, task::Poll};

#[tokio::test]
async fn peer_evidence_verifier_restart_revalidates_without_waiting_for_a_heartbeat() -> Result<()>
{
    let origin = make_test_node(LocalRole::Host { http_port: 9337 }).await?;
    let observer = make_test_node(LocalRole::Client).await?;
    for node in [&origin, &observer] {
        node.set_mesh_id("verifier-restart".into()).await;
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
    let (delivered, received) = tokio::sync::oneshot::channel();
    observer
        .forward_plugin_event(crate::plugin::PluginMeshEvent::PluginStarted {
            plugin_id: "evidence".into(),
            generation: 0,
            delivered,
        })
        .await?;
    assert!(received.await?);
    wait_for_eligible_peer(&observer, origin.id()).await;
    assert_eq!(calls.load(Ordering::SeqCst), verified + 1);
    observer.close_endpoint().await;
    origin.close_endpoint().await;
    Ok(())
}

#[tokio::test]
async fn peer_evidence_idle_maintenance_releases_the_manager_after_endpoint_closure() -> Result<()>
{
    for with_verifier in [false, true] {
        let node = make_test_node(LocalRole::Client).await?;
        let bridge = Arc::new(Verifier(Arc::default(), false));
        let weak_bridge = Arc::downgrade(&bridge);
        let manager = if with_verifier {
            PluginManager::for_test_peer_evidence(bridge, None)
        } else {
            PluginManager::for_test_bridge(&["ordinary"], bridge)
        };
        *node.plugin_manager.lock().await = Some(manager.clone());
        let mut maintenance = Box::pin(node.maintain_peer_evidence());
        std::future::poll_fn(|cx| {
            assert!(maintenance.as_mut().poll(cx).is_pending());
            Poll::Ready(())
        })
        .await;
        drop(manager.peer_evidence().refresh.lock().await);
        node.close_endpoint().await;
        drop(manager);
        assert!(weak_bridge.upgrade().is_none());
        std::future::poll_fn(|cx| {
            assert!(maintenance.as_mut().poll(cx).is_pending());
            Poll::Ready(())
        })
        .await;
    }
    Ok(())
}

struct PendingResponse(Arc<tokio::sync::Notify>);
impl PluginRpcBridge for PendingResponse {
    fn handle_request(
        &self,
        _: String,
        _: String,
        _: String,
    ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
        let release = self.0.clone();
        Box::pin(async move {
            release.notified().await;
            Ok(RpcResult {
                result_json: json!({"accepted":true,"cache_until_restart":true}).to_string(),
            })
        })
    }
    fn handle_notification(&self, _: String, _: String, _: String) -> BridgeFuture<()> {
        Box::pin(async {})
    }
}

#[tokio::test]
async fn peer_evidence_late_results_notify_provider_views_without_another_refresh() -> Result<()> {
    let node = make_test_node(LocalRole::Client).await?;
    let release = Arc::new(tokio::sync::Notify::new());
    let manager =
        PluginManager::for_test_peer_evidence(Arc::new(PendingResponse(release.clone())), None);
    *node.plugin_manager.lock().await = Some(manager.clone());
    assert!(
        !manager
            .verify_provider_evidence_until(
                "mesh",
                "provider",
                &Default::default(),
                tokio::time::Instant::now() + std::time::Duration::from_millis(20),
            )
            .await
    );
    let mut maintenance = Box::pin(node.maintain_peer_evidence());
    std::future::poll_fn(|cx| {
        assert!(maintenance.as_mut().poll(cx).is_pending());
        Poll::Ready(())
    })
    .await;
    drop(manager.peer_evidence().refresh.lock().await);
    let mut changes = node.peer_change_rx.clone();
    changes.borrow_and_update();
    release.notify_one();
    tokio::time::timeout(std::time::Duration::from_secs(1), async {
        tokio::select! {
            _ = &mut maintenance => anyhow::bail!("evidence maintenance stopped"),
            changed = changes.changed() => changed?,
        }
        Ok::<_, anyhow::Error>(())
    })
    .await??;
    assert!(
        manager
            .peer_evidence()
            .provider_cached("mesh", "provider", &Default::default())
    );
    drop(maintenance);
    node.close_endpoint().await;
    Ok(())
}
