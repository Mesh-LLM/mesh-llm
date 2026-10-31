use super::*;
use crate::plugin::InProcessPluginRunner;
use mesh_llm_plugin::{LocalStream, read_envelope, write_envelope};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

async fn serve(
    mut stream: LocalStream,
    calls: Arc<AtomicUsize>,
    received: Option<Arc<tokio::sync::Notify>>,
) -> Result<()> {
    loop {
        let incoming = read_envelope(&mut stream).await?;
        let payload = match incoming.payload {
            Some(proto::envelope::Payload::InitializeRequest(_)) => {
                proto::envelope::Payload::InitializeResponse(proto::InitializeResponse {
                    plugin_id: "replay-fixture".into(),
                    plugin_protocol_version: crate::plugin::PROTOCOL_VERSION,
                    plugin_version: "0.0.0".into(),
                    server_info_json: serde_json::to_string(&rmcp::model::ServerConfig::default())?,
                    manifest: Some(proto::PluginManifest::default()),
                    ..Default::default()
                })
            }
            Some(proto::envelope::Payload::InvokeServiceRequest(_)) => {
                if let Some(received) = &received {
                    calls.fetch_add(1, Ordering::SeqCst);
                    received.notify_one();
                    // Keep reading so shutdown closes the stream, but never reply.
                    continue;
                }
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    // The operation was received, but its reply is lost.
                    return Ok(());
                }
                proto::envelope::Payload::InvokeServiceResponse(proto::InvokeServiceResponse {
                    output_json: "{}".into(),
                    is_error: false,
                })
            }
            Some(proto::envelope::Payload::ShutdownRequest(_)) => return Ok(()),
            _ => continue,
        };
        write_envelope(
            &mut stream,
            &proto::Envelope {
                protocol_version: crate::plugin::PROTOCOL_VERSION,
                plugin_id: incoming.plugin_id,
                request_id: incoming.request_id,
                payload: Some(payload),
            },
        )
        .await?;
    }
}

async fn exercise(operation: &str, replay: bool) {
    let calls = Arc::new(AtomicUsize::new(0));
    let starts = Arc::new(AtomicUsize::new(0));
    let runner: InProcessPluginRunner = {
        let calls = calls.clone();
        let starts = starts.clone();
        Arc::new(move |stream| {
            starts.fetch_add(1, Ordering::SeqCst);
            Box::pin(serve(stream, calls.clone(), None))
        })
    };
    let plugin = crate::plugin::runtime::tests::in_process_plugin("replay-fixture", runner);
    let result = plugin
        .call_tool_with_replay(
            operation,
            "{}",
            None,
            if replay {
                TransportReplay::Allow
            } else {
                TransportReplay::Never
            },
        )
        .await;
    assert_eq!(result.is_ok(), replay);
    assert_eq!(calls.load(Ordering::SeqCst), if replay { 2 } else { 1 });
    assert_eq!(starts.load(Ordering::SeqCst), if replay { 2 } else { 1 });
    if !replay {
        // A later caller-owned operation can restart the plugin; no permanent
        // disablement or change to ledger-owned recovery is introduced.
        plugin.call_tool("wallet_lookup", "{}").await.unwrap();
        assert_eq!(starts.load(Ordering::SeqCst), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
    plugin.shutdown().await;
}

#[tokio::test]
async fn lost_wallet_pay_reply_is_not_replayed() {
    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        exercise("wallet_pay", false),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn lost_wallet_lookup_reply_keeps_existing_retry_behavior() {
    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        exercise("wallet_lookup", true),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn shutdown_does_not_restart_an_outstanding_replayable_request() {
    let starts = Arc::new(AtomicUsize::new(0));
    let received = Arc::new(tokio::sync::Notify::new());
    let runner: InProcessPluginRunner = {
        let starts = starts.clone();
        let received = received.clone();
        Arc::new(move |stream| {
            starts.fetch_add(1, Ordering::SeqCst);
            Box::pin(serve(
                stream,
                Arc::new(AtomicUsize::new(0)),
                Some(received.clone()),
            ))
        })
    };
    let plugin = Arc::new(crate::plugin::runtime::tests::in_process_plugin(
        "replay-fixture",
        runner,
    ));
    let caller = plugin.clone();
    let request = tokio::spawn(async move {
        caller
            .call_tool_with_replay(
                "wallet_lookup",
                "{}",
                Some(std::time::Duration::from_secs(1)),
                TransportReplay::Allow,
            )
            .await
    });
    tokio::time::timeout(std::time::Duration::from_secs(10), received.notified())
        .await
        .unwrap();
    plugin.shutdown().await;
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(5), request)
            .await
            .unwrap()
            .unwrap()
            .is_err()
    );
    assert!(plugin.call_tool("wallet_lookup", "{}").await.is_err());
    let count = starts.load(Ordering::SeqCst);
    assert!(plugin.runtime.lock().await.is_none());
    plugin.shutdown().await;
    assert_eq!(
        count, 1,
        "shutdown must not resurrect the runner through replay"
    );
}

/// Serves `manifest`, counts the operations it receives and never answers one,
/// so the caller's deadline is the only thing that ends the call.
async fn never_answer(
    mut stream: LocalStream,
    manifest: proto::PluginManifest,
    calls: Arc<AtomicUsize>,
) -> Result<()> {
    loop {
        let incoming = read_envelope(&mut stream).await?;
        let payload = match incoming.payload {
            Some(proto::envelope::Payload::InitializeRequest(_)) => {
                proto::envelope::Payload::InitializeResponse(proto::InitializeResponse {
                    plugin_id: "deadline-fixture".into(),
                    plugin_protocol_version: crate::plugin::PROTOCOL_VERSION,
                    plugin_version: "0.0.0".into(),
                    server_info_json: serde_json::to_string(&rmcp::model::ServerConfig::default())?,
                    manifest: Some(manifest.clone()),
                    ..Default::default()
                })
            }
            Some(proto::envelope::Payload::InvokeServiceRequest(_)) => {
                calls.fetch_add(1, Ordering::SeqCst);
                continue;
            }
            Some(proto::envelope::Payload::ShutdownRequest(_)) => return Ok(()),
            _ => continue,
        };
        write_envelope(
            &mut stream,
            &proto::Envelope {
                protocol_version: crate::plugin::PROTOCOL_VERSION,
                plugin_id: incoming.plugin_id,
                request_id: incoming.request_id,
                payload: Some(payload),
            },
        )
        .await?;
    }
}

fn deadline_runner(
    manifest: proto::PluginManifest,
    calls: Arc<AtomicUsize>,
) -> InProcessPluginRunner {
    Arc::new(move |stream| Box::pin(never_answer(stream, manifest.clone(), calls.clone())))
}

fn manifest_with_deadline(operation: &str, timeout_secs: Option<u32>) -> proto::PluginManifest {
    proto::PluginManifest {
        operations: vec![proto::OperationManifest {
            name: operation.to_string(),
            description: operation.to_string(),
            input_schema_json: "{}".to_string(),
            timeout_secs,
            ..Default::default()
        }],
        ..Default::default()
    }
}

/// A call that runs past its deadline must be given up on once. Restarting the
/// plugin and sending the same operation again starts the work a second time
/// and kills the process that was still working on it.
#[tokio::test]
async fn timed_out_call_is_not_sent_again_on_a_fresh_process() {
    let calls = Arc::new(AtomicUsize::new(0));
    let starts = Arc::new(AtomicUsize::new(0));
    let runner_calls = calls.clone();
    let runner_starts = starts.clone();
    let runner: InProcessPluginRunner = Arc::new(move |stream| {
        runner_starts.fetch_add(1, Ordering::SeqCst);
        Box::pin(never_answer(
            stream,
            proto::PluginManifest::default(),
            runner_calls.clone(),
        ))
    });
    let plugin = crate::plugin::runtime::tests::in_process_plugin("deadline-fixture", runner);

    let error = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        plugin.call_tool_with_replay(
            "demo_slow",
            "{}",
            Some(std::time::Duration::from_millis(200)),
            TransportReplay::Allow,
        ),
    )
    .await
    .expect("the deadline must end the call")
    .expect_err("a call that never answered must fail");
    assert!(error.to_string().contains("timed out"), "error: {error}");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "a timed-out operation must not be delivered to a restarted plugin"
    );
    assert_eq!(
        starts.load(Ordering::SeqCst),
        1,
        "a timed-out operation must not start a second plugin process"
    );
    plugin.shutdown().await;
}

/// The host's request deadline has to follow the operation: a plugin that
/// declares a longer one must not be given up on at the fixed default.
#[tokio::test]
async fn a_manifest_operation_can_declare_its_own_deadline() {
    let calls = Arc::new(AtomicUsize::new(0));
    let plugin = crate::plugin::runtime::tests::in_process_plugin(
        "deadline-fixture",
        deadline_runner(manifest_with_deadline("demo_slow", Some(1)), calls.clone()),
    );

    let outcome = tokio::time::timeout(
        std::time::Duration::from_secs(4),
        plugin.call_tool("demo_slow", "{}"),
    )
    .await;
    let error = match outcome {
        Ok(result) => result.expect_err("the plugin never answers"),
        Err(_) => panic!(
            "the host ignored demo_slow's declared 1s deadline and kept waiting \
             past its fixed default"
        ),
    };
    assert!(error.to_string().contains("timed out"), "error: {error}");
    plugin.shutdown().await;
}
