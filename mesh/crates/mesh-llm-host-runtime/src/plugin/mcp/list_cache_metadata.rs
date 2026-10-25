//! Regression coverage for MCP 2026-07-28 list-result cache metadata.
//!
//! The 2026-07-28 revision makes `ttlMs` and `cacheScope` required on every
//! `resultType: "complete"` result. `rmcp`'s `with_all_items` constructor leaves
//! both fields `None`, and they are serialized with `skip_serializing_if`, so a
//! handler that does not set them emits a list result that a conforming client
//! must reject. These tests drive the shipped `PluginMcpHttpEndpoint` (the real
//! `rmcp` `StreamableHttpService`) over loopback and assert the four list results
//! carry the fields.

use super::PluginMcpHttpEndpoint;
use crate::plugin::{PluginManager, PluginRpcBridge, RpcResult};
use http_body_util::{BodyExt, Full};
use serde_json::{Value, json};
use std::sync::Arc;

/// Wire value for the MCP revision that made list results cacheable.
const PROTOCOL_2026_07_28: &str = "2026-07-28";
const PROTOCOL_2025_11_25: &str = "2025-11-25";
const NOTIFY_INITIALIZED: &str = "notifications/initialized";
const CACHE_METADATA_FIELDS: [&str; 2] = ["ttlMs", "cacheScope"];

struct NoopBridge;

impl PluginRpcBridge for NoopBridge {
    fn handle_request(
        &self,
        _plugin_name: String,
        _method: String,
        _params_json: String,
    ) -> crate::plugin::BridgeFuture<Result<RpcResult, crate::plugin::proto::ErrorResponse>> {
        Box::pin(async move {
            Err(crate::plugin::proto::ErrorResponse {
                code: -32603,
                message: "unexpected test bridge request".into(),
                data_json: String::new(),
            })
        })
    }

    fn handle_notification(
        &self,
        _plugin_name: String,
        _method: String,
        _params_json: String,
    ) -> crate::plugin::BridgeFuture<()> {
        Box::pin(async {})
    }
}

async fn serve_loopback_endpoint() -> String {
    let endpoint =
        PluginMcpHttpEndpoint::new(PluginManager::for_test_bridge(&[], Arc::new(NoopBridge)));
    let router = axum::Router::new()
        .route("/mcp", axum::routing::any(forward_to_mcp_endpoint))
        .with_state(endpoint);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    format!("http://{address}/mcp")
}

async fn forward_to_mcp_endpoint(
    axum::extract::State(endpoint): axum::extract::State<PluginMcpHttpEndpoint>,
    request: axum::extract::Request,
) -> axum::response::Response {
    let (parts, body) = request.into_parts();
    let body = match BodyExt::collect(body).await {
        Ok(collected) => collected.to_bytes(),
        Err(_) => bytes::Bytes::new(),
    };
    let request = http::Request::from_parts(parts, Full::new(body));
    let response = endpoint.handle(request).await;
    let (parts, body) = response.into_parts();
    axum::response::Response::from_parts(
        parts,
        axum::body::Body::from_stream(body.into_data_stream()),
    )
}

/// 2026-07-28 requires the negotiated protocol version and client capabilities on
/// every request `_meta`; without them the transport rejects the call outright.
fn request_meta() -> Value {
    json!({
        "io.modelcontextprotocol/protocolVersion": PROTOCOL_2026_07_28,
        "io.modelcontextprotocol/clientCapabilities": {}
    })
}

struct McpSession {
    client: reqwest::Client,
    url: String,
    session_id: String,
    version: &'static str,
    /// 2026-07-28 carries the negotiated version and client capabilities in every
    /// request `_meta`; older revisions negotiate once in `initialize`.
    inline_meta: bool,
}

impl McpSession {
    async fn open(url: &str, version: &'static str, inline_meta: bool) -> Self {
        let client = reqwest::Client::new();
        let mut params = json!({
            "protocolVersion": version,
            "capabilities": {},
            "clientInfo": { "name": "mesh-llm-list-cache-test", "version": "1" },
        });
        if inline_meta {
            params["_meta"] = request_meta();
        }
        let response = client
            .post(url)
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .header("mcp-protocol-version", version)
            .json(&json!({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": params,
            }))
            .send()
            .await
            .unwrap();
        let session_id = response
            .headers()
            .get("mcp-session-id")
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default()
            .to_string();
        assert!(
            !session_id.is_empty(),
            "initialize must return a session id"
        );

        let session = Self {
            client,
            url: url.to_string(),
            session_id,
            version,
            inline_meta,
        };
        let mut notification = json!({
            "jsonrpc": "2.0",
            "method": NOTIFY_INITIALIZED,
        });
        if inline_meta {
            notification["params"] = json!({ "_meta": request_meta() });
        }
        session.post(NOTIFY_INITIALIZED, notification).await;
        session
    }

    /// 2026-07-28 (SEP-2243) requires `Mcp-Method` on every non-initialize POST,
    /// mirroring the body method; the transport rejects the call without it.
    async fn post(&self, method: &str, payload: Value) -> String {
        self.client
            .post(&self.url)
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream")
            .header("mcp-protocol-version", self.version)
            .header("mcp-method", method)
            .header("mcp-session-id", &self.session_id)
            .json(&payload)
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap()
    }

    /// Send one request and return the decoded JSON-RPC `result` object.
    async fn result(&self, id: i64, method: &str) -> Value {
        let mut request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
        });
        if self.inline_meta {
            request["params"] = json!({ "_meta": request_meta() });
        }
        let body = self.post(method, request).await;
        assert!(
            !body.contains("\"error\""),
            "{method} returned an error response: {body}"
        );
        for line in body.lines() {
            let Some(payload) = line.strip_prefix("data:") else {
                continue;
            };
            let Ok(value) = serde_json::from_str::<Value>(payload.trim()) else {
                continue;
            };
            if value.get("id").and_then(Value::as_i64) == Some(id) {
                return value
                    .get("result")
                    .cloned()
                    .unwrap_or_else(|| panic!("{method} response has no result: {body}"));
            }
        }
        panic!("{method} response contained no matching JSON-RPC result: {body}");
    }
}

#[tokio::test]
async fn mcp_list_results_carry_2026_07_28_cache_metadata() {
    let url = serve_loopback_endpoint().await;
    let session = McpSession::open(&url, PROTOCOL_2026_07_28, true).await;

    for (id, method) in [
        (10_i64, "tools/list"),
        (11, "resources/list"),
        (12, "prompts/list"),
        (13, "resources/templates/list"),
    ] {
        let result = session.result(id, method).await;
        assert_eq!(
            result.get("resultType").and_then(Value::as_str),
            Some("complete"),
            "{method} must be served as a complete 2026-07-28 result"
        );
        for field in CACHE_METADATA_FIELDS {
            assert!(
                result.get(field).is_some(),
                "{method} result is missing required 2026-07-28 cache metadata field \
                 '{field}'; got keys {:?} in {result}",
                result
                    .as_object()
                    .map(|map| map.keys().cloned().collect::<Vec<_>>()),
            );
        }
    }
}

/// The gate must stay one-directional: a peer that negotiated an older revision
/// keeps the wire shape it understands.
#[tokio::test]
async fn mcp_list_results_omit_cache_metadata_for_legacy_peer() {
    let url = serve_loopback_endpoint().await;
    let session = McpSession::open(&url, PROTOCOL_2025_11_25, false).await;

    for (id, method) in [
        (20_i64, "tools/list"),
        (21, "resources/list"),
        (22, "prompts/list"),
        (23, "resources/templates/list"),
    ] {
        let result = session.result(id, method).await;
        for field in CACHE_METADATA_FIELDS {
            assert!(
                result.get(field).is_none(),
                "{method} result must not advertise 2026-07-28 cache metadata to a \
                 {PROTOCOL_2025_11_25} peer; got {field} in {result}",
            );
        }
    }
}
