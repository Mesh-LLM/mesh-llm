use super::*;
use crate::{LocalStream, PROTOCOL_VERSION, Plugin, PluginRuntime, plugin_server_info, proto};
use tokio::time::{Duration, timeout};

#[test]
fn normalize_http_operation_name_keeps_internal_prefix() {
    assert_eq!(
        normalize_http_operation_name("GET", "/ping"),
        "__http_get_ping"
    );
    assert_eq!(normalize_http_operation_name("GET", "/"), "__http_get");
    assert_eq!(
        normalize_http_operation_name("POST", "/v1/notes/"),
        "__http_post_v1_notes"
    );
}

#[tokio::test]
async fn http_route_stays_out_of_manifest_operations_but_remains_invocable() {
    let plugin = crate::plugin! {
        metadata: PluginMetadata::new(
            "demo",
            "1.0.0",
            plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>),
        ),
        http: [
            crate::http::post("/echo")
                .description("Echo over stapled HTTP")
                .handle(|args, _ctx| Box::pin(async move { Ok(args) })),
        ],
    };

    let manifest = plugin.manifest().expect("manifest");
    assert!(
        manifest.operations.is_empty(),
        "HTTP-only route must not be published in manifest.operations: {:?}",
        manifest.operations
    );
    assert_eq!(manifest.http_bindings.len(), 1);
    assert_eq!(
        manifest.http_bindings[0].operation_name.as_deref(),
        Some("__http_post_echo")
    );

    let (plugin_io, host_io) = tokio::io::duplex(64 * 1024);
    let runtime = tokio::spawn(PluginRuntime::run_with_stream(
        plugin,
        LocalStream::Memory(plugin_io),
    ));
    let mut host = LocalStream::Memory(host_io);

    crate::write_envelope(
        &mut host,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: "demo".into(),
            request_id: 1,
            payload: Some(proto::envelope::Payload::InitializeRequest(
                proto::InitializeRequest {
                    host_protocol_version: PROTOCOL_VERSION,
                    ..Default::default()
                },
            )),
        },
    )
    .await
    .unwrap();
    let init_resp = timeout(Duration::from_secs(2), crate::read_envelope(&mut host))
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        init_resp.payload,
        Some(proto::envelope::Payload::InitializeResponse(_))
    ));

    let params = serde_json::json!({
        "name": "__http_post_echo",
        "arguments": { "ok": true }
    });
    crate::write_envelope(
        &mut host,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: "demo".into(),
            request_id: 2,
            payload: Some(proto::envelope::Payload::RpcRequest(proto::RpcRequest {
                method: "tools/call".into(),
                params_json: params.to_string(),
                invocation_id: None,
            })),
        },
    )
    .await
    .unwrap();
    let call_resp = timeout(Duration::from_secs(2), crate::read_envelope(&mut host))
        .await
        .unwrap()
        .unwrap();
    let Some(proto::envelope::Payload::RpcResponse(resp)) = call_resp.payload else {
        panic!("expected RpcResponse, got {:?}", call_resp.payload);
    };
    let decoded: serde_json::Value = serde_json::from_str(&resp.result_json).unwrap();
    assert_eq!(decoded["structuredContent"]["ok"], serde_json::json!(true));
    runtime.abort();
}
