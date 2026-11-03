//! Session-isolation regression coverage for the plugin MCP HTTP endpoint.
//!
//! Each plugin invocation owns its callback destination. These tests drive the real
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
        Ok(ListRootsResult::new(vec![rmcp::model::Root::new(format!(
            "file:///{}",
            self.name
        ))]))
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

/// An invalid tool call must not become the owner of subsequent callbacks.
#[tokio::test]
async fn missing_tool_does_not_bind_a_callback_session() {
    let plugin_manager = PluginManager::for_test_bridge(&[], Arc::new(NoopRpcBridge));
    let endpoint = PluginMcpHttpEndpoint::new(plugin_manager);
    let bridge = endpoint.bridge.clone();
    let url = spawn_mcp_http_endpoint(endpoint).await;

    let (sender, mut deliveries) = mpsc::unbounded_channel();

    // No plugin is invoked by this missing tool request.
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
        outcome.is_err(),
        "missing tool must not bind a callback owner"
    );
    assert!(delivered_to.is_empty());
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

#[tokio::test]
async fn pending_invocation_keeps_callbacks_when_another_session_calls() {
    callback_invocations(false).await;
}

#[tokio::test]
async fn capable_plugin_callbacks_route_two_concurrent_invocations() {
    callback_invocations(true).await;
}

fn callback_plugin_runner(
    entered: mpsc::UnboundedSender<()>,
    proceed: Arc<tokio::sync::Semaphore>,
    concurrent: bool,
) -> plugin::InProcessPluginRunner {
    use mesh_llm_plugin::{
        InternalRpcPluginBuilder, OperationRouter, PluginMetadata, PluginRuntime,
        operation_with_schema, plugin_server_info,
    };
    Arc::new(move |stream| {
        let mut router = OperationRouter::new();
        let entered = entered.clone();
        let proceed = proceed.clone();
        router.add_json(
            operation_with_schema("callback", "Callback", serde_json::Map::new()),
            move |_args: serde_json::Value, context| {
                let entered = entered.clone();
                let proceed = proceed.clone();
                Box::pin(async move {
                    entered.send(()).unwrap();
                    proceed.acquire().await.unwrap().forget();
                    let roots: serde_json::Value = context
                        .request_host("roots/list", serde_json::json!({}))
                        .await?;
                    Ok(roots)
                })
            },
        );
        let plugin = InternalRpcPluginBuilder::new(PluginMetadata::new(
            "adapter",
            "0.0.0",
            plugin_server_info(
                "adapter",
                "0.0.0",
                "Adapter",
                "Callback fixture",
                None::<String>,
            ),
        ))
        .with_manifest(plugin::proto::PluginManifest {
            operations: vec![plugin::proto::OperationManifest {
                name: "callback".into(),
                description: "Callback".into(),
                input_schema_json: "{}".into(),
                ..Default::default()
            }],
            ..Default::default()
        })
        .with_operation_router(router)
        .with_capabilities(if concurrent {
            vec!["invocation_callbacks.v1".into()]
        } else {
            vec![]
        })
        .build();
        Box::pin(PluginRuntime::run_with_stream(plugin, stream))
    })
}

async fn callback_invocations(concurrent: bool) {
    let (entered, mut entries) = mpsc::unbounded_channel();
    let proceed = Arc::new(tokio::sync::Semaphore::new(0));
    let runner = callback_plugin_runner(entered, proceed.clone(), concurrent);
    let manager = PluginManager::start_with_in_process(
        &plugin::ResolvedPlugins {
            externals: vec![plugin::in_process_builtin_spec("adapter")],
            inactive: vec![],
        },
        plugin::PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        mpsc::channel(1).0,
        plugin::InProcessPlugins::default().with("adapter", runner),
    )
    .await
    .unwrap();
    let url = spawn_mcp_http_endpoint(PluginMcpHttpEndpoint::new(manager.clone())).await;
    let (sender, mut deliveries) = mpsc::unbounded_channel();
    let victim = connect_session(&url, "victim", &sender).await;
    let attacker = connect_session(&url, "attacker", &sender).await;
    let victim_peer = victim.peer().clone();
    let mut first = tokio::spawn(async move {
        victim_peer
            .call_tool(CallToolRequestParams::new("adapter.callback"))
            .await
    });
    await_callback_entry(&mut entries, &mut first).await;

    let second =
        if concurrent {
            let peer = attacker.peer().clone();
            let second = tokio::spawn(async move {
                peer.call_tool(CallToolRequestParams::new("adapter.callback"))
                    .await
            });
            let entered =
                tokio::time::timeout(std::time::Duration::from_secs(2), entries.recv()).await;
            entered.unwrap_or_else(|error| {
            proceed.add_permits(2);
            first.abort();
            second.abort();
            panic!("capable plugin must admit both invocations before either completes: {error}");
        }).unwrap();
            Some(second)
        } else {
            assert!(
                attacker
                    .peer()
                    .call_tool(CallToolRequestParams::new("missing"))
                    .await
                    .is_err()
            );
            None
        };
    proceed.add_permits(if concurrent { 2 } else { 1 });

    let first_result = tokio::time::timeout(std::time::Duration::from_secs(10), first)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(
        serde_json::to_string(&first_result)
            .unwrap()
            .contains("file:///victim")
    );
    if let Some(second) = second {
        let second_result = tokio::time::timeout(std::time::Duration::from_secs(10), second)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(
            serde_json::to_string(&second_result)
                .unwrap()
                .contains("file:///attacker")
        );
        let mut delivered = drain(&mut deliveries);
        delivered.sort_unstable();
        assert_eq!(delivered, vec!["attacker", "victim"]);
    } else {
        assert_eq!(drain(&mut deliveries), vec!["victim"]);
    }
    manager.shutdown().await;
}

async fn await_callback_entry<T: std::fmt::Debug>(
    entries: &mut mpsc::UnboundedReceiver<()>,
    call: &mut tokio::task::JoinHandle<T>,
) {
    tokio::select! {
        entered = tokio::time::timeout(std::time::Duration::from_secs(10), entries.recv()) => {
            entered.unwrap().unwrap();
        }
        result = call => panic!("invocation ended before entering the plugin: {result:?}"),
    }
}
