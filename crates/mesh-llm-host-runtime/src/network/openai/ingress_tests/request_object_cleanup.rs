//! Request-object ownership ends exactly once when automatic admission fails.

use super::*;
use crate::plugin::{BridgeFuture, PluginManager, PluginRpcBridge, RpcResult, proto};
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

#[derive(Default)]
struct CompletionRecorder(Mutex<Vec<String>>);

impl PluginRpcBridge for CompletionRecorder {
    fn handle_request(
        &self,
        plugin_name: String,
        method: String,
        params_json: String,
    ) -> BridgeFuture<Result<RpcResult, proto::ErrorResponse>> {
        assert_eq!(plugin_name, "blobstore");
        assert_eq!(method, "tools/call");
        let params: serde_json::Value = serde_json::from_str(&params_json).unwrap();
        assert_eq!(
            params["name"],
            crate::plugins::blobstore::COMPLETE_REQUEST_TOOL
        );
        let request_id = params["arguments"]["request_id"]
            .as_str()
            .unwrap()
            .to_owned();
        self.0.lock().unwrap().push(request_id.clone());
        Box::pin(async move {
            let response = rmcp::model::CallToolResult::structured(serde_json::json!({
                "request_id": request_id, "removed_tokens": 1, "removed_bytes": 4,
            }));
            Ok(RpcResult {
                result_json: serde_json::to_string(&response).unwrap(),
            })
        })
    }

    fn handle_notification(&self, _: String, _: String, _: String) -> BridgeFuture<()> {
        Box::pin(async {})
    }
}

async fn rejected_request_releases_objects(model: &str, media: bool) {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    let recorder = Arc::new(CompletionRecorder::default());
    let manager = PluginManager::for_test_bridge(&["blobstore"], recorder.clone());
    manager.set_test_capability_providers(vec![crate::plugin::PluginCapabilityProvider {
        capability: crate::plugins::blobstore::OBJECT_STORE_CAPABILITY.into(),
        plugin_name: "blobstore".into(),
        plugin_status: "running".into(),
        endpoint_id: None,
        available: true,
        detail: None,
    }]);
    node.set_plugin_manager(manager.clone()).await;
    let mut targets = election::ModelTargets::default();
    targets.targets.insert(
        "text-only".into(),
        vec![election::InferenceTarget::Local(1)],
    );
    node.set_served_model_descriptors(vec![mesh::ServedModelDescriptor {
        identity: mesh::ServedModelIdentity {
            model_name: "text-only".into(),
            ..Default::default()
        },
        capabilities_known: true,
        metadata: Some(mesh::ServedModelMetadata {
            workload_class: Some(mesh::ModelWorkloadClass::CausalGeneration),
            ..Default::default()
        }),
        ..Default::default()
    }])
    .await;
    let path = if media {
        "/v1/chat/completions"
    } else {
        "/v1/embeddings"
    };
    let body = if media {
        serde_json::json!({"model": model, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        ]}]})
    } else {
        serde_json::json!({"model": model, "input": "hello"})
    }
    .to_string();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let handler = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = proxy::read_http_request(&mut stream).await.unwrap();
        // Parsing/normalization already owns these objects when this handler starts.
        request.request_object_request_ids = vec!["upload-a".into(), "upload-b".into()];
        let affinity = affinity::AffinityRouter::new();
        handle_buffered_api_request(
            stream.into(),
            request,
            ProxyConnectionContext {
                route: IngressRouteContext {
                    node: &node,
                    targets: &targets,
                    affinity: &affinity,
                    plugin_manager: Some(&manager),
                },
            },
            None,
            crate::runtime::IngressType::LocalOpenAi,
        )
        .await;
    });
    let mut client = TcpStream::connect(address).await.unwrap();
    client.write_all(format!("POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
    let mut response = String::new();
    client.read_to_string(&mut response).await.unwrap();
    handler.await.unwrap();
    assert!(response.starts_with("HTTP/1.1 422"), "{response}");
    assert_eq!(*recorder.0.lock().unwrap(), ["upload-a", "upload-b"]);
}

#[tokio::test]
async fn workload_and_media_rejections_complete_each_request_object_once() {
    for (model, media) in [("text-only", false), ("auto", false), ("auto", true)] {
        tokio::time::timeout(
            std::time::Duration::from_secs(10),
            rejected_request_releases_objects(model, media),
        )
        .await
        .expect("rejected ingress must complete");
    }
}
