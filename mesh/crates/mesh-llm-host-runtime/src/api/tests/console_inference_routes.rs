//! The Console owns browser Chat, not a second public inference API.

use super::*;

#[tokio::test]
async fn console_rejects_native_inference_routes() {
    let state = build_test_mesh_api_with_api_port(0).await;
    for headless in [false, true] {
        state.set_headless(headless).await;
        for path in [
            "/v1/models",
            "/v1/chat/completions",
            "/v1/responses",
            "/v1/embeddings",
            "/v1/audio/transcriptions",
            "/v1/images/generations",
            "/v1/browser-poc",
            "/models",
            "/models/test-model",
            "/mesh/v1/models",
            "/mesh/v1/models/test-model",
            "/mesh/v1/documents/ocr",
            "/mesh/v1/images/similarity",
            "/mesh/v1/images/depth",
        ] {
            for method in ["GET", "POST", "OPTIONS"] {
                let (addr, server) = spawn_management_test_server(state.clone()).await;
                let response = send_management_request(
                    addr,
                    format!(
                        "{method} {path}?trace=1 HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{{}}"
                    ),
                )
                .await;
                server.await.unwrap().unwrap();
                assert!(
                    response.starts_with("HTTP/1.1 404"),
                    "headless={headless}, {method} {path}: {response}"
                );
            }
        }
    }
}

#[tokio::test]
async fn browser_chat_rejects_other_methods_and_prefix_aliases() {
    let state = build_test_mesh_api_with_api_port(0).await;
    for path in ["/api/chat", "/api/responses"] {
        for method in ["GET", "OPTIONS", "DELETE"] {
            let (addr, server) = spawn_management_test_server(state.clone()).await;
            let response = send_management_request(
                addr,
                format!("{method} {path} HTTP/1.1\r\nHost: localhost\r\n\r\n"),
            )
            .await;
            server.await.unwrap().unwrap();
            assert!(response.starts_with("HTTP/1.1 405"), "{response}");
        }
        let (addr, server) = spawn_management_test_server(state.clone()).await;
        let response = send_management_request(
            addr,
            format!("POST {path}/v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n"),
        )
        .await;
        server.await.unwrap().unwrap();
        assert!(response.starts_with("HTTP/1.1 404"), "{response}");
    }
}
