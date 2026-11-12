//! A private virtual-model turn keeps the plugin's nested inference private.

use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Answer one nested request and return the raw request it sent.
async fn nested_request(
    bridge: &InferenceRpcBridge,
    plugin: &str,
    port_listener: tokio::net::TcpListener,
) -> String {
    let server = tokio::spawn(async move {
        let (mut stream, _) = port_listener.accept().await.unwrap();
        let mut buf = vec![0u8; 8192];
        let read = stream.read(&mut buf).await.unwrap();
        let body = r#"{"choices":[]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).await.unwrap();
        String::from_utf8_lossy(&buf[..read]).to_ascii_lowercase()
    });
    let params = r#"{"model_id":"qwen","request":{"messages":[]}}"#;
    bridge
        .handle_request(
            plugin.to_string(),
            "inference/chat_completions".to_string(),
            params.to_string(),
        )
        .await
        .unwrap();
    server.await.unwrap()
}

#[tokio::test]
async fn nested_calls_are_private_only_while_a_private_turn_runs() {
    let plugins = PluginManager::for_test_summaries(Vec::new());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let bridge = InferenceRpcBridge::new(port, plugins.clone());

    let turn = plugins.begin_private_invocation("moa");
    let sent = nested_request(&bridge, "moa", listener).await;
    assert!(sent.contains("x-mesh-scope: private"), "{sent}");

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let bridge = InferenceRpcBridge::new(listener.local_addr().unwrap().port(), plugins.clone());
    let sent = nested_request(&bridge, "other", listener).await;
    assert!(!sent.contains("x-mesh-scope"), "{sent}");

    drop(turn);
    let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
        .await
        .unwrap();
    let bridge = InferenceRpcBridge::new(listener.local_addr().unwrap().port(), plugins);
    let sent = nested_request(&bridge, "moa", listener).await;
    assert!(!sent.contains("x-mesh-scope"), "{sent}");
}
