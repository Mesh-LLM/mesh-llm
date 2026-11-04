use super::*;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

fn provider(class: Value, context: Value) -> Value {
    json!({"available":true, "workload_class": class, "runtime":{"context_length":context}})
}

#[test]
fn context_is_minimum_only_when_every_generation_provider_reports_a_window() {
    let causal = provider(json!("causal_generation"), json!(32768));
    let smaller = provider(json!("causal_generation"), json!(16384));
    let ocr = provider(json!("document_ocr"), json!(512));
    assert_eq!(
        provider_context_limit(&json!({"providers":[causal,smaller,ocr]})),
        Some(16384)
    );
    for missing in [
        provider(Value::Null, Value::Null),
        provider(json!("causal_generation"), Value::Null),
        provider(json!("causal_generation"), json!(0)),
    ] {
        assert_eq!(
            provider_context_limit(&json!({"providers":[causal,missing]})),
            None
        );
    }
    let mut unavailable = provider(json!("causal_generation"), Value::Null);
    unavailable["available"] = json!(false);
    assert_eq!(
        provider_context_limit(&json!({"providers":[causal,unavailable]})),
        Some(32768)
    );
}

#[tokio::test]
async fn compatibility_ids_are_enriched_from_bounded_paginated_mesh_discovery() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        for index in 0..3 {
            let (stream, _) = listener.accept().await.unwrap();
            let mut reader = BufReader::new(stream);
            let mut request = String::new();
            reader.read_line(&mut request).await.unwrap();
            let mut header = String::new();
            loop {
                header.clear();
                reader.read_line(&mut header).await.unwrap();
                if header == "\r\n" {
                    break;
                }
            }
            let body = match index {
                0 => {
                    assert!(request.starts_with("GET /v1/models "));
                    json!({"object":"list","data":[{"id":"org/llm:Q4"},{"id":"mesh"}]})
                }
                1 => {
                    assert!(request.starts_with("GET /mesh/v1/models?limit=500 "));
                    json!({"data":[{"id":"ocr-only","providers":[provider(json!("document_ocr"),json!(256))]}],"next_cursor":"page/2+next"})
                }
                _ => {
                    assert!(request.starts_with("GET /mesh/v1/models?limit=500&cursor=page%2F2%2Bnext "));
                    json!({"data":[{"id":"org/llm:Q4","providers":[provider(json!("causal_generation"),json!(16384)), provider(json!("document_ocr"),json!(256))]}],"next_cursor":null})
                }
            }.to_string();
            reader.into_inner().write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).as_bytes()).await.unwrap();
        }
    });
    let (inventory, chosen) = super::super::fetch_mesh_models(
        &reqwest::Client::new(),
        &format!("http://{address}/v1/models"),
        &Some("org/llm:Q4".into()),
    )
    .await
    .unwrap();
    assert_eq!(chosen, "org/llm:Q4");
    assert_eq!(inventory.names, ["org/llm:Q4", "mesh"]);
    assert_eq!(inventory.context_limit("org/llm:Q4"), 16384);
    assert_eq!(inventory.context_limit("mesh"), 16384);
    server.await.unwrap();
}

#[tokio::test]
async fn legacy_server_without_mesh_discovery_preserves_explicit_fallback() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let body = "{}";
        stream.write_all(format!("HTTP/1.1 404 Not Found\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).as_bytes()).await.unwrap();
    });
    let mut inventory = ModelInventory::from_response(&json!({"data":[{"id":"legacy"}]}));
    enrich_from_mesh(
        &reqwest::Client::new(),
        &format!("http://{address}/v1/models"),
        &mut inventory,
    )
    .await;
    assert_eq!(inventory.context_limit("legacy"), 8192);
    assert_eq!(inventory.fallbacks, [("legacy".into(), 8192)]);
    server.await.unwrap();
}
