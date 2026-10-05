use super::*;
use crate::plugin::InProcessPluginRunner;
use mesh_llm_plugin::{LocalStream, read_envelope, write_envelope};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

async fn serve(mut stream: LocalStream, calls: Arc<AtomicUsize>) -> Result<()> {
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
            Box::pin(serve(stream, calls.clone()))
        })
    };
    let plugin = crate::plugin::runtime::tests::in_process_plugin("replay-fixture", runner);
    let result = plugin.call_tool_with_timeout(operation, "{}", None).await;
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
