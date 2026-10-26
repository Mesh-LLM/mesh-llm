//! Session-isolation regression coverage for the plugin MCP HTTP endpoint.
//!
//! Every HTTP MCP session shares the endpoint's [`ActiveBridge`] peer slot, so a
//! server-to-client callback issued on behalf of one client can be delivered to a
//! different client that merely wrote the slot last. These tests drive the real
//! `POST /mcp` endpoint over a real TCP listener with two independent MCP sessions
//! and assert a callback resolves only to the session that caused it.

use super::*;
use axum::{
    Router,
    extract::State,
    http::{Request, Response},
    routing::any,
};
use http_body_util::Full;
use rmcp::{
    ErrorData as McpError, RoleClient, ServiceExt,
    handler::client::ClientHandler,
    model::{CallToolRequestParams, ListRootsResult},
    service::{Peer, RequestContext, RunningService},
    transport::StreamableHttpClientTransport,
};
use tokio::sync::mpsc;

/// RPC bridge standing in for a plugin that never issues host requests of its own.
struct NoopRpcBridge;

impl PluginRpcBridge for NoopRpcBridge {
    fn handle_request(
        &self,
        _plugin_name: String,
        method: String,
        _params_json: String,
    ) -> crate::plugin::BridgeFuture<Result<RpcResult, plugin::proto::ErrorResponse>> {
        Box::pin(async move { Err(proto_error::internal(format!("unexpected call '{method}'"))) })
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

/// MCP client session that records its own name whenever the host delivers a
/// server-to-client `roots/list` request to it.
#[derive(Clone)]
struct RecordingClient {
    name: &'static str,
    deliveries: mpsc::UnboundedSender<&'static str>,
}

impl ClientHandler for RecordingClient {
    async fn list_roots(
        &self,
        _context: RequestContext<RoleClient>,
    ) -> Result<ListRootsResult, McpError> {
        let _ = self.deliveries.send(self.name);
        Ok(ListRootsResult::default())
    }
}

async fn spawn_mcp_http_endpoint(endpoint: PluginMcpHttpEndpoint) -> String {
    async fn forward(
        State(endpoint): State<PluginMcpHttpEndpoint>,
        request: Request<axum::body::Body>,
    ) -> Response<axum::body::Body> {
        let (parts, body) = request.into_parts();
        let bytes = axum::body::to_bytes(body, usize::MAX)
            .await
            .expect("test request body should be readable");
        let request = Request::from_parts(parts, Full::new(bytes));
        let (response_parts, response_body) = endpoint.handle(request).await.into_parts();
        Response::from_parts(response_parts, axum::body::Body::new(response_body))
    }

    let router = Router::new()
        .route("/mcp", any(forward))
        .with_state(endpoint);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, router).await;
    });
    format!("http://{address}/mcp")
}

/// One connected MCP session and the identity it reports when delivered to.
struct Session {
    running: RunningService<RoleClient, RecordingClient>,
    name: &'static str,
}

impl Session {
    fn peer(&self) -> &Peer<RoleClient> {
        self.running.peer()
    }
}

async fn connect_session(
    url: &str,
    name: &'static str,
    deliveries: &mpsc::UnboundedSender<&'static str>,
) -> Session {
    let client = RecordingClient {
        name,
        deliveries: deliveries.clone(),
    };
    let running = client
        .serve(StreamableHttpClientTransport::from_uri(url.to_owned()))
        .await
        .unwrap_or_else(|err| panic!("{name} session should initialize: {err}"));
    Session { running, name }
}

/// Issue the server-to-client callback the way a plugin invocation does: through
/// the endpoint's bridge, the only routing state between a plugin-initiated
/// callback and the MCP client meant to receive it.
async fn issue_callback(bridge: &ActiveBridge) -> Result<String, plugin::proto::ErrorResponse> {
    bridge
        .handle_request(
            "adapter".to_owned(),
            "roots/list".to_owned(),
            serde_json::json!({}).to_string(),
        )
        .await
        .map(|result| result.result_json)
}

fn drain(deliveries: &mut mpsc::UnboundedReceiver<&'static str>) -> Vec<&'static str> {
    let mut seen = Vec::new();
    while let Ok(name) = deliveries.try_recv() {
        seen.push(name);
    }
    seen
}

/// A callback caused by one session's tool invocation must not be captured by a
/// different session that wrote the shared peer slot in the meantime.
#[tokio::test]
async fn mcp_callback_is_not_delivered_to_a_session_that_did_not_invoke_it() {
    let plugin_manager = PluginManager::for_test_bridge(&[], Arc::new(NoopRpcBridge));
    let endpoint = PluginMcpHttpEndpoint::new(plugin_manager);
    let bridge = endpoint.bridge.clone();
    let url = spawn_mcp_http_endpoint(endpoint).await;

    let (sender, mut deliveries) = mpsc::unbounded_channel();

    // The victim is the session that invokes a tool, so it owns any callback that
    // invocation causes.
    let victim = connect_session(&url, "victim", &sender).await;
    let _ = victim
        .peer()
        .call_tool(CallToolRequestParams::new("missing"))
        .await;

    // The attacker never invokes a tool; it only completes its own handshake.
    let attacker = connect_session(&url, "attacker", &sender).await;
    assert_eq!(attacker.name, "attacker");

    let outcome = issue_callback(&bridge).await;
    let delivered_to = drain(&mut deliveries);

    assert!(
        outcome.is_ok(),
        "callback should resolve to its originating session, got {:?}",
        outcome.err().map(|err| err.message)
    );
    assert_eq!(
        delivered_to,
        vec!["victim"],
        "callback must be delivered only to the session whose tool invocation caused it"
    );
}

/// Once the originating session is gone the callback has no owner left and must be
/// rejected rather than quietly handed to whichever session happens to be live.
#[tokio::test]
async fn mcp_callback_is_rejected_after_originating_session_expires() {
    let plugin_manager = PluginManager::for_test_bridge(&[], Arc::new(NoopRpcBridge));
    let endpoint = PluginMcpHttpEndpoint::new(plugin_manager);
    let bridge = endpoint.bridge.clone();
    let url = spawn_mcp_http_endpoint(endpoint).await;

    let (sender, mut deliveries) = mpsc::unbounded_channel();

    let victim = connect_session(&url, "victim", &sender).await;
    let _ = victim
        .peer()
        .call_tool(CallToolRequestParams::new("missing"))
        .await;
    drop(victim);

    let _unrelated = connect_session(&url, "attacker", &sender).await;

    let outcome = issue_callback(&bridge).await;
    let delivered_to = drain(&mut deliveries);

    assert!(
        outcome.is_err(),
        "callback whose originating session expired must be rejected, got {outcome:?}"
    );
    assert!(
        delivered_to.is_empty(),
        "an expired session's callback must not reach another session, got {delivered_to:?}"
    );
}
