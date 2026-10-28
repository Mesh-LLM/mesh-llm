use super::*;
use crate::plugin::{InProcessPluginRunner, proto};
use mesh_llm_plugin::{LocalStream, read_envelope, write_envelope};
use std::sync::atomic::{AtomicUsize, Ordering};

async fn serve(mut stream: LocalStream, writes: Arc<AtomicUsize>) -> anyhow::Result<()> {
    loop {
        let incoming = read_envelope(&mut stream).await?;
        let payload = match incoming.payload {
            Some(proto::envelope::Payload::InitializeRequest(_)) => {
                proto::envelope::Payload::InitializeResponse(proto::InitializeResponse {
                    plugin_id: incoming.plugin_id.clone(),
                    plugin_protocol_version: super::super::PROTOCOL_VERSION,
                    plugin_version: "fixture".into(),
                    server_info_json: serde_json::to_string(&rmcp::model::ServerConfig::default())?,
                    manifest: Some(proto::PluginManifest::default()),
                    ..Default::default()
                })
            }
            Some(proto::envelope::Payload::InvokeServiceRequest(request)) => {
                if request.service_name == "write" {
                    writes.fetch_add(1, Ordering::SeqCst);
                }
                proto::envelope::Payload::InvokeServiceResponse(proto::InvokeServiceResponse {
                    output_json: request.input_json,
                    is_error: false,
                })
            }
            Some(proto::envelope::Payload::ShutdownRequest(_)) => return Ok(()),
            _ => continue,
        };
        write_envelope(
            &mut stream,
            &proto::Envelope {
                protocol_version: super::super::PROTOCOL_VERSION,
                plugin_id: incoming.plugin_id,
                request_id: incoming.request_id,
                payload: Some(payload),
            },
        )
        .await?;
    }
}

#[tokio::test]
async fn non_wallet_provider_uses_fixed_setup_allowlist_generation_and_revocation() {
    let writes = Arc::new(AtomicUsize::new(0));
    let runner: InProcessPluginRunner = {
        let writes = writes.clone();
        Arc::new(move |stream| Box::pin(serve(stream, writes.clone())))
    };
    let caller = super::super::runtime::tests::in_process_plugin("client", runner.clone());
    let provider = super::super::runtime::tests::in_process_plugin("state-store", runner);
    // Normal plugin initialization over the actual control connection.
    caller.call_tool("probe", "{}").await.unwrap();
    provider.call_tool("probe", "{}").await.unwrap();
    let generation = caller.bound_generation().await.unwrap();
    let grant = PluginOperationBinding {
        name: "state".into(),
        provider: "state-store".into(),
        profile: "one".into(),
        setup_operation: "open".into(),
        setup_input_json: "{\"profile\":\"one\"}".into(),
        operations: vec!["write".into()],
    };
    let mut manager = PluginManager::for_test_summaries(Vec::new());
    let inner = Arc::get_mut(&mut manager.inner).unwrap();
    inner.plugins.insert("client".into(), caller);
    inner.plugins.insert("state-store".into(), provider);
    inner.operation_bindings.0.lock().unwrap().insert(
        ("client".into(), "state".into()),
        Binding {
            grant,
            handle: None,
            capacity: Arc::new(Semaphore::new(8)),
        },
    );
    let request = |handle, operation: &str, input: &str| BoundOperationRequest {
        binding: "state".into(),
        handle,
        operation: operation.into(),
        input_json: input.into(),
    };
    assert!(matches!(
        manager
            .bound_operation("client", generation, request(None, "open", "{}"))
            .await,
        Response::NotDispatched { .. }
    ));
    let Response::Responded {
        handle,
        output_json,
        ..
    } = manager
        .bound_operation("client", generation, request(None, "", ""))
        .await
    else {
        panic!("setup must respond")
    };
    assert_eq!(output_json, "{\"profile\":\"one\"}");
    assert!(!handle.is_empty());
    assert!(matches!(
        manager
            .bound_operation(
                "client",
                generation,
                request(Some(handle.clone()), "open", "{}")
            )
            .await,
        Response::NotDispatched { .. }
    ));
    assert!(matches!(
        manager
            .bound_operation(
                "client",
                generation,
                request(Some(handle.clone()), "write", "{}")
            )
            .await,
        Response::Responded { .. }
    ));
    assert_eq!(writes.load(Ordering::SeqCst), 1);
    assert!(matches!(
        manager
            .bound_operation(
                "client",
                generation + 1,
                request(Some(handle.clone()), "write", "{}")
            )
            .await,
        Response::NotDispatched { .. }
    ));
    let Response::Responded { handle: fresh, .. } = manager
        .bound_operation("client", generation, request(None, "", ""))
        .await
    else {
        panic!("repeat setup must respond")
    };
    assert_ne!(fresh, handle);
    assert!(matches!(
        manager
            .bound_operation(
                "client",
                generation,
                request(Some(handle.clone()), "write", "{}")
            )
            .await,
        Response::NotDispatched { .. }
    ));
    // Hold the final admission lock: the call has passed initial authorization
    // and reserved queue capacity, but cannot yet reach the enqueue boundary.
    let pending = manager.inner.plugins["state-store"]
        .bound_test_pending()
        .await;
    let blocked = pending.lock().await;
    let gate = manager
        .inner
        .operation_bindings
        .0
        .lock()
        .unwrap()
        .get(&("client".into(), "state".into()))
        .unwrap()
        .capacity
        .clone();
    let queued_manager = manager.clone();
    let queued = tokio::spawn(async move {
        queued_manager
            .bound_operation(
                "client",
                generation,
                BoundOperationRequest {
                    binding: "state".into(),
                    handle: Some(fresh),
                    operation: "write".into(),
                    input_json: "{}".into(),
                },
            )
            .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while gate.available_permits() == 8 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    manager.revoke_changed_operation_bindings(&MeshConfig::default());
    drop(blocked);
    assert!(
        matches!(tokio::time::timeout(std::time::Duration::from_secs(2), queued)
        .await.unwrap().unwrap(), Response::NotDispatched { reason } if reason == "binding revoked")
    );
    assert_eq!(writes.load(Ordering::SeqCst), 1);
    manager.revoke_changed_operation_bindings(&MeshConfig::default());

    assert!(matches!(
        manager
            .bound_operation("client", generation, request(Some(handle), "write", "{}"))
            .await,
        Response::NotDispatched { .. }
    ));
    manager.shutdown().await;
}
