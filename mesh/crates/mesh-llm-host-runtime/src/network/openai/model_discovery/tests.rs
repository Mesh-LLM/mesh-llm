use super::*;
use crate::network::affinity::AffinityRouter;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

async fn catalog_node() -> mesh::Node {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Worker)
        .await
        .unwrap();
    let manager = crate::plugin::PluginManager::for_test_summaries(Vec::new());
    manager
        .set_test_inference_endpoints(vec![
            crate::plugin::InferenceEndpointRoute {
                plugin_name: "chat".into(),
                endpoint_id: "chat".into(),
                protocol: Some("openai_compatible".into()),
                address: "http://127.0.0.1:1".into(),
                models: vec!["chat-model".into(), "shared".into()],
            },
            crate::plugin::InferenceEndpointRoute {
                plugin_name: "ocr".into(),
                endpoint_id: "ocr".into(),
                protocol: Some("document_ocr_v1".into()),
                address: "http://127.0.0.1:2".into(),
                models: vec!["ocr-model".into(), "shared".into()],
            },
        ])
        .await;
    node.set_plugin_manager(manager).await;
    node
}

async fn request(node: mesh::Node, path: &str, extra_header: &str) -> (u16, serde_json::Value) {
    request_method(node, "GET", path, extra_header).await
}

async fn request_method(
    node: mesh::Node,
    method: &str,
    path: &str,
    extra_header: &str,
) -> (u16, serde_json::Value) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        super::super::transport::handle_mesh_request(
            node,
            stream.into(),
            false,
            AffinityRouter::default(),
        )
        .await;
    });
    let mut stream = TcpStream::connect(address).await.unwrap();
    stream
        .write_all(
            format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n{extra_header}\r\n").as_bytes(),
        )
        .await
        .unwrap();
    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    server.await.unwrap();
    let response = String::from_utf8(response).unwrap();
    let status = response.split_whitespace().nth(1).unwrap().parse().unwrap();
    let (_, body) = response.split_once("\r\n\r\n").unwrap();
    (status, serde_json::from_str(body).unwrap())
}

#[tokio::test]
async fn passive_discovery_keeps_ocr_and_protocol_specific_shared_identity() {
    let node = catalog_node().await;
    let (status, body) = request(node.clone(), "/mesh/v1/models", "").await;
    assert_eq!(status, 200);
    let models = body["data"].as_array().unwrap();
    assert!(models.iter().any(|model| model["id"] == "ocr-model"));
    let shared = models.iter().find(|model| model["id"] == "shared").unwrap();
    assert!(
        shared["providers"]
            .as_array()
            .unwrap()
            .iter()
            .any(|provider| provider["workload_class"] == "document_ocr")
    );
    let (status, filtered) = request(
        node.clone(),
        "/mesh/v1/models?workload_class=document_ocr",
        "",
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(filtered["data"].as_array().unwrap().len(), 2);
    let (status, compatibility) = request(node.clone(), "/v1/models", "").await;
    assert_eq!(status, 200);
    assert_eq!(compatibility["data"].as_array().unwrap().len(), 2);
    assert!(
        compatibility["data"]
            .as_array()
            .unwrap()
            .iter()
            .all(|model| model["id"] != "ocr-model")
    );
    assert_eq!(
        request(node.clone(), "/v1/models/ocr-model", "").await.0,
        404
    );
    assert_eq!(
        request(node, "/mesh/v1/models/lookup?id=ocr-model", "")
            .await
            .0,
        200
    );
}

#[tokio::test]
async fn passive_anthropic_models_use_case_insensitive_version_header() {
    let node = catalog_node().await;
    let (status, body) = request(
        node.clone(),
        "/v1/models?limit=1",
        "AnThRoPiC-VeRsIoN: 2023-06-01\r\n",
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(body["data"][0]["type"], "model");
    assert_eq!(body["has_more"], true);
    assert!(body.get("object").is_none());
    let (status, body) = request(
        node,
        "/v1/models/chat-model",
        "anthropic-version: 2023-06-01\r\n",
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(body["id"], "chat-model");
    assert_eq!(body["display_name"], "chat-model");
    assert!(body.get("capabilities").is_none());
}

#[test]
fn only_recognized_reads_are_intercepted() {
    for path in [
        "/models",
        "/v1/models?limit=1",
        "/v1/models/org%2Fmodel",
        "/models/model",
    ] {
        assert!(is_request("GET", path), "{path}");
        assert!(!is_request("POST", path), "{path}");
    }
    for path in ["/mesh/v1/models", "/mesh/v1/models/lookup?id=one"] {
        assert!(is_request("GET", path), "{path}");
        assert!(is_request("POST", path), "{path}");
        assert!(is_request("DELETE", path), "{path}");
    }
    for path in [
        "/v1/model",
        "/v1/models-other",
        "/mesh/v1/models/other",
        "/v1/documents/ocr",
    ] {
        assert!(!is_request("GET", path), "{path}");
    }
}

#[tokio::test]
async fn mesh_catalog_rejects_mutations_with_method_not_allowed() {
    let node = catalog_node().await;
    for path in ["/mesh/v1/models", "/mesh/v1/models/lookup?id=chat-model"] {
        let (status, body) = request_method(node.clone(), "POST", path, "").await;
        assert_eq!(status, 405, "{path}");
        assert_eq!(body["error"]["message"], "method not allowed");
    }
}

#[tokio::test]
async fn mesh_lookup_rejects_invalid_utf8_instead_of_matching_a_replacement_character() {
    let node = catalog_node().await;
    let manager = node.plugin_manager().await.unwrap();
    let mut endpoints = manager.inference_endpoints().await.unwrap();
    endpoints[0].models.push("\u{fffd}".into());
    manager.set_test_inference_endpoints(endpoints).await;
    assert_eq!(
        request(node.clone(), "/mesh/v1/models/lookup?id=%FF", "")
            .await
            .0,
        400,
    );
    assert_eq!(
        request(node, "/mesh/v1/models/lookup?id=%EF%BF%BD", "")
            .await
            .0,
        200,
    );
}
