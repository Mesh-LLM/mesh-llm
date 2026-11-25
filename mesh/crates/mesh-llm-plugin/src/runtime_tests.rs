use super::*;
use crate::{mcp, plugin, plugin_server_info, read_envelope, write_envelope};
use rmcp::model::{ArgumentInfo, ContentBlock, PromptMessage, Reference, Role};
use serde_json::json;
#[cfg(unix)]
use tokio::sync::{Barrier, Notify};
use tokio::time::{Duration, timeout};

#[derive(Clone, Debug, Default, Deserialize, Serialize, schemars::JsonSchema)]
struct DemoArgs {
    #[serde(default)]
    message: String,
}

fn test_context() -> PluginContext<'static> {
    let (outbound_tx, _outbound_rx) = mpsc::channel(8);
    PluginContext::new(
        "demo".into(),
        outbound_tx,
        Arc::new(Mutex::new(HashMap::new())),
    )
}

#[cfg(unix)]
fn test_channel_message(message_kind: &str) -> proto::ChannelMessage {
    proto::ChannelMessage {
        channel: "events".into(),
        source_peer_id: "peer-a".into(),
        target_peer_id: "peer-b".into(),
        content_type: "text/plain".into(),
        body: Vec::new(),
        message_kind: message_kind.into(),
        correlation_id: String::new(),
        metadata_json: String::new(),
    }
}

fn acknowledgement_plugin() -> SimplePlugin {
    SimplePlugin::new(PluginMetadata::new(
        "demo",
        "1.0.0",
        plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>),
    ))
    .with_subscribe_resource(|_request, _context| Box::pin(async { Ok(()) }))
    .with_unsubscribe_resource(|_request, _context| Box::pin(async { Ok(()) }))
    .with_task_router(
        crate::TaskRouter::new()
            .with_update(|_request, _context| Box::pin(async { Ok(()) }))
            .with_cancel(|_request, _context| Box::pin(async { Ok(()) })),
    )
}

async fn rpc_result_json(
    plugin: &mut impl Plugin,
    method: &str,
    params: serde_json::Value,
) -> serde_json::Value {
    let response = plugin
        .handle_rpc(
            proto::RpcRequest {
                invocation_id: None,
                method: method.into(),
                params_json: params.to_string(),
            },
            &mut test_context(),
        )
        .await
        .unwrap();
    let proto::envelope::Payload::RpcResponse(response) = response else {
        panic!("expected an RPC response");
    };
    serde_json::from_str(&response.result_json).unwrap()
}

#[tokio::test]
async fn resource_subscription_acknowledgements_are_empty_objects() {
    let mut plugin = acknowledgement_plugin();

    for method in ["resources/subscribe", "resources/unsubscribe"] {
        assert_eq!(
            rpc_result_json(&mut plugin, method, json!({ "uri": "demo://state" })).await,
            json!({})
        );
    }
}

#[tokio::test]
async fn task_acknowledgements_include_the_complete_result_discriminator() {
    let mut plugin = acknowledgement_plugin();

    assert_eq!(
        rpc_result_json(
            &mut plugin,
            "tasks/update",
            json!({ "taskId": "task-1", "inputResponses": {} }),
        )
        .await,
        json!({ "resultType": "complete" })
    );
    assert_eq!(
        rpc_result_json(&mut plugin, "tasks/cancel", json!({ "taskId": "task-1" }),).await,
        json!({ "resultType": "complete" })
    );
}

#[tokio::test]
async fn invoke_service_dispatches_operation_prompt_resource_and_completion() {
    let mut plugin = plugin! {
        metadata: PluginMetadata::new(
            "demo",
            "1.0.0",
            plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>),
        ),
        mcp: [
            mcp::tool("echo")
                .description("Echo input")
                .input::<DemoArgs>()
                .handle(|args, _context| Box::pin(async move {
                    Ok(json!({ "echo": args.message }))
                })),
            mcp::resource("demo://state")
                .name("State")
                .handle(|request, _context| Box::pin(async move {
                    Ok(crate::read_resource_result(vec![
                        rmcp::model::ResourceContents::text("state", request.uri),
                    ]))
                })),
            mcp::prompt("brief")
                .description("Brief")
                .handle(|request, _context| Box::pin(async move {
                    Ok(crate::get_prompt_result(vec![PromptMessage::new(
                        Role::User,
                        ContentBlock::text(format!("brief:{}", request.name)),
                    )]))
                })),
            mcp::completion("prompt.brief.topic")
                .handle(|_request, _context| Box::pin(async move {
                    crate::complete_result(vec!["alpha".into()])
                })),
        ],
    };

    let mut context = test_context();

    let op = plugin
        .invoke_service(
            proto::InvokeServiceRequest {
                invocation_id: None,
                kind: proto::ServiceKind::Operation as i32,
                service_name: "echo".into(),
                input_json: json!({ "message": "hello" }).to_string(),
            },
            &mut context,
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&op.output_json).unwrap(),
        json!({"echo": "hello"})
    );

    let prompt = plugin
        .invoke_service(
            proto::InvokeServiceRequest {
                invocation_id: None,
                kind: proto::ServiceKind::Prompt as i32,
                service_name: "brief".into(),
                input_json: serde_json::to_string(&GetPromptRequestParams::new("brief")).unwrap(),
            },
            &mut context,
        )
        .await
        .unwrap()
        .unwrap();
    let prompt_result: GetPromptResult = serde_json::from_str(&prompt.output_json).unwrap();
    assert_eq!(prompt_result.messages.len(), 1);

    let resource = plugin
        .invoke_service(
            proto::InvokeServiceRequest {
                invocation_id: None,
                kind: proto::ServiceKind::Resource as i32,
                service_name: "demo://state".into(),
                input_json: serde_json::to_string(&ReadResourceRequestParams::new("demo://state"))
                    .unwrap(),
            },
            &mut context,
        )
        .await
        .unwrap()
        .unwrap();
    let resource_result: ReadResourceResult = serde_json::from_str(&resource.output_json).unwrap();
    assert_eq!(resource_result.contents.len(), 1);

    let completion = plugin
        .invoke_service(
            proto::InvokeServiceRequest {
                invocation_id: None,
                kind: proto::ServiceKind::Completion as i32,
                service_name: "brief".into(),
                input_json: serde_json::to_string(&CompleteRequestParams::new(
                    Reference::for_prompt("brief"),
                    ArgumentInfo::new("topic", "a"),
                ))
                .unwrap(),
            },
            &mut context,
        )
        .await
        .unwrap()
        .unwrap();
    let completion_result: CompleteResult = serde_json::from_str(&completion.output_json).unwrap();
    assert_eq!(
        completion_result.completion.values,
        vec![String::from("alpha")]
    );
}

#[tokio::test]
async fn invoke_service_dispatches_virtual_model_handler() {
    let mut router = crate::VirtualModelRouter::new();
    router.add_json(
        crate::operation_with_schema("chat", "Virtual chat", serde_json::Map::new()),
        |request: serde_json::Value, _context| {
            Box::pin(async move {
                Ok(crate::VirtualModelResponse {
                    status_code: 200,
                    body: json!({"echo": request["request"]}),
                    headers: vec![("x-test".into(), "ok".into())],
                    event_stream: false,
                })
            })
        },
    );
    let mut plugin = crate::SimplePlugin::new(PluginMetadata::new(
        "virtual-demo",
        "1.0.0",
        plugin_server_info(
            "virtual-demo",
            "1.0.0",
            "Virtual Demo",
            "Virtual model test",
            None::<String>,
        ),
    ))
    .with_virtual_model_router(router);
    let mut context = test_context();

    let response = plugin
        .invoke_service(
            proto::InvokeServiceRequest {
                invocation_id: None,
                kind: proto::ServiceKind::VirtualModel as i32,
                service_name: "chat".into(),
                input_json: json!({"request": {"model": "virtual-demo"}}).to_string(),
            },
            &mut context,
        )
        .await
        .unwrap()
        .unwrap();
    let response: crate::VirtualModelResponse =
        serde_json::from_str(&response.output_json).unwrap();
    assert_eq!(response.status_code, 200);
    assert_eq!(response.body["echo"]["model"], "virtual-demo");
    assert_eq!(response.headers, [("x-test".into(), "ok".into())]);
}

// Unix only: this drives a real `UnixStream` pair, and `LocalStream` has no
// Unix variant on Windows, where the transport is a named pipe.
#[cfg(unix)]
#[tokio::test]
async fn health_request_returns_while_operation_is_running() {
    let started = Arc::new(Notify::new());
    let started_for_tool = started.clone();
    let plugin = plugin! {
        metadata: PluginMetadata::new(
            "demo",
            "1.0.0",
            plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>),
        ),
        mcp: [
            mcp::tool("slow")
                .description("Slow operation")
                .input::<DemoArgs>()
                .handle(move |_args, _context| {
                    let started = started_for_tool.clone();
                    Box::pin(async move {
                        started.notify_one();
                        tokio::time::sleep(Duration::from_millis(300)).await;
                        Ok(json!({ "done": true }))
                    })
                }),
        ],
    };

    let (plugin_stream, host_stream) = tokio::net::UnixStream::pair().unwrap();

    let runtime = tokio::spawn(PluginRuntime::run_with_stream(
        plugin,
        LocalStream::Unix(plugin_stream),
    ));
    let mut host_stream = LocalStream::Unix(host_stream);
    write_envelope(
        &mut host_stream,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: "demo".into(),
            request_id: 1,
            payload: Some(proto::envelope::Payload::InvokeServiceRequest(
                proto::InvokeServiceRequest {
                    invocation_id: None,
                    kind: proto::ServiceKind::Operation as i32,
                    service_name: "slow".into(),
                    input_json: "{}".into(),
                },
            )),
        },
    )
    .await
    .unwrap();

    started.notified().await;
    write_envelope(
        &mut host_stream,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: "demo".into(),
            request_id: 2,
            payload: Some(proto::envelope::Payload::HealthRequest(
                proto::HealthRequest {},
            )),
        },
    )
    .await
    .unwrap();

    let health = timeout(Duration::from_millis(150), read_envelope(&mut host_stream))
        .await
        .expect("health response should not wait for slow operation")
        .unwrap();
    assert_eq!(health.request_id, 2);
    assert!(matches!(
        health.payload,
        Some(proto::envelope::Payload::HealthResponse(_))
    ));

    let invoke = timeout(Duration::from_secs(1), read_envelope(&mut host_stream))
        .await
        .expect("slow operation should still complete")
        .unwrap();
    assert_eq!(invoke.request_id, 1);

    runtime.abort();
}

// Unix only: this drives a real `UnixStream` pair, and `LocalStream` has no
// Unix variant on Windows, where the transport is a named pipe.
#[cfg(unix)]
#[tokio::test]
async fn invokes_service_requests_concurrently() {
    let barrier = Arc::new(Barrier::new(2));
    let barrier_for_tool = barrier.clone();
    let plugin = plugin! {
        metadata: PluginMetadata::new(
            "demo",
            "1.0.0",
            plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>),
        ),
        mcp: [
            mcp::tool("barrier")
                .description("Waits for another request")
                .input::<DemoArgs>()
                .handle(move |_args, _context| {
                    let barrier = barrier_for_tool.clone();
                    Box::pin(async move {
                        barrier.wait().await;
                        Ok(json!({ "done": true }))
                    })
                }),
        ],
    };

    let (plugin_stream, host_stream) = tokio::net::UnixStream::pair().unwrap();

    let runtime = tokio::spawn(PluginRuntime::run_with_stream(
        plugin,
        LocalStream::Unix(plugin_stream),
    ));
    let mut host_stream = LocalStream::Unix(host_stream);

    for request_id in [1, 2] {
        write_envelope(
            &mut host_stream,
            &proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: "demo".into(),
                request_id,
                payload: Some(proto::envelope::Payload::InvokeServiceRequest(
                    proto::InvokeServiceRequest {
                        invocation_id: None,
                        kind: proto::ServiceKind::Operation as i32,
                        service_name: "barrier".into(),
                        input_json: "{}".into(),
                    },
                )),
            },
        )
        .await
        .unwrap();
    }

    let first = timeout(Duration::from_secs(1), read_envelope(&mut host_stream))
        .await
        .expect("first invoke response should not block behind another request")
        .unwrap();
    let second = timeout(Duration::from_secs(1), read_envelope(&mut host_stream))
        .await
        .expect("second invoke response should not block behind another request")
        .unwrap();
    let mut request_ids = vec![first.request_id, second.request_id];
    request_ids.sort_unstable();
    assert_eq!(request_ids, vec![1, 2]);

    runtime.abort();
}

#[tokio::test]
async fn dropped_open_mesh_stream_request_removes_pending_response() {
    let (outbound_tx, mut outbound_rx) = mpsc::channel(8);
    let pending_host_responses = Arc::new(Mutex::new(HashMap::new()));
    let mut context =
        PluginContext::new("demo".into(), outbound_tx, pending_host_responses.clone());

    let request = tokio::spawn(async move {
        let _ = context
            .open_mesh_stream(proto::OpenMeshStreamRequest::default())
            .await;
    });

    let outbound = outbound_rx
        .recv()
        .await
        .expect("request should be sent before awaiting host response");
    assert_ne!(outbound.request_id, 0);
    assert_eq!(
        pending_host_responses
            .lock()
            .expect("pending map should not be poisoned")
            .len(),
        1
    );

    request.abort();
    let _ = request.await;

    assert!(
        pending_host_responses
            .lock()
            .expect("pending map should not be poisoned")
            .is_empty()
    );
}

// Unix only: this drives a real `UnixStream` pair, and `LocalStream` has no
// Unix variant on Windows, where the transport is a named pipe.
#[cfg(unix)]
#[tokio::test]
async fn ordered_notifications_do_not_overtake_each_other() {
    let first_started = Arc::new(Notify::new());
    let release_first = Arc::new(Notify::new());
    let handled = Arc::new(Mutex::new(Vec::<String>::new()));
    let first_started_for_handler = first_started.clone();
    let release_first_for_handler = release_first.clone();
    let handled_for_handler = handled.clone();
    let plugin = SimplePlugin::new(PluginMetadata::new(
        "demo",
        "1.0.0",
        plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>),
    ))
    .on_channel_message(move |message, _context| {
        let first_started = first_started_for_handler.clone();
        let release_first = release_first_for_handler.clone();
        let handled = handled_for_handler.clone();
        Box::pin(async move {
            if message.message_kind == "first" {
                first_started.notify_one();
                release_first.notified().await;
            }
            handled
                .lock()
                .expect("handled list should not be poisoned")
                .push(message.message_kind);
            Ok(())
        })
    });

    let (plugin_stream, host_stream) = tokio::net::UnixStream::pair().unwrap();

    let runtime = tokio::spawn(PluginRuntime::run_with_stream(
        plugin,
        LocalStream::Unix(plugin_stream),
    ));
    let mut host_stream = LocalStream::Unix(host_stream);

    for (request_id, message_kind) in [(1, "first"), (2, "second")] {
        write_envelope(
            &mut host_stream,
            &proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: "demo".into(),
                request_id,
                payload: Some(proto::envelope::Payload::ChannelMessage(
                    test_channel_message(message_kind),
                )),
            },
        )
        .await
        .unwrap();
    }

    first_started.notified().await;
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        handled
            .lock()
            .expect("handled list should not be poisoned")
            .is_empty(),
        "second message should wait behind the first ordered handler"
    );

    release_first.notify_one();
    timeout(Duration::from_secs(1), async {
        loop {
            if handled
                .lock()
                .expect("handled list should not be poisoned")
                .len()
                == 2
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("ordered handlers should finish");

    assert_eq!(
        *handled.lock().expect("handled list should not be poisoned"),
        vec![String::from("first"), String::from("second")]
    );

    runtime.abort();
}

#[cfg(unix)]
async fn initialize(host_stream: &mut LocalStream, host_capabilities: Vec<String>) {
    write_envelope(
        host_stream,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: "demo".into(),
            request_id: 1,
            payload: Some(proto::envelope::Payload::InitializeRequest(
                proto::InitializeRequest {
                    host_protocol_version: PROTOCOL_VERSION,
                    host_capabilities,
                    ..Default::default()
                },
            )),
        },
    )
    .await
    .unwrap();
    let response = timeout(Duration::from_secs(1), read_envelope(host_stream))
        .await
        .expect("initialize response")
        .unwrap();
    assert!(matches!(
        response.payload,
        Some(proto::envelope::Payload::InitializeResponse(_))
    ));
}

// A plugin announces its key from `on_initialized`, and the host's reply
// comes in on the same read loop that delivered `InitializeRequest`. The
// loop must keep reading while `on_initialized` waits, or the reply is
// never read and the announce times out.
#[cfg(unix)]
#[tokio::test]
async fn a_key_announced_from_on_initialized_gets_the_hosts_reply() {
    let (announced_tx, announced_rx) = tokio::sync::oneshot::channel();
    let announced_tx = Arc::new(Mutex::new(Some(announced_tx)));
    let plugin = SimplePlugin::new(PluginMetadata::new(
        "demo",
        "1.0.0",
        plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>),
    ))
    .on_initialized(move |context| {
        let announced_tx = announced_tx.clone();
        Box::pin(async move {
            let response = context.announce_plugin_key(vec![7; 32]).await;
            if let Some(tx) = announced_tx.lock().unwrap().take() {
                let _ = tx.send(response.map_err(|err| err.to_string()));
            }
            Ok(())
        })
    });

    let (plugin_stream, host_stream) = tokio::net::UnixStream::pair().unwrap();
    let runtime = tokio::spawn(PluginRuntime::run_with_stream(
        plugin,
        LocalStream::Unix(plugin_stream),
    ));
    let mut host_stream = LocalStream::Unix(host_stream);
    initialize(
        &mut host_stream,
        vec![crate::host_capabilities::PLUGIN_KEYS.into()],
    )
    .await;

    let request = timeout(Duration::from_secs(1), read_envelope(&mut host_stream))
        .await
        .expect("the plugin should ask the host to announce its key")
        .unwrap();
    let Some(proto::envelope::Payload::PluginKeyRequest(key_request)) = request.payload else {
        panic!("expected a plugin key request, got {:?}", request.payload);
    };
    assert_eq!(key_request.public_key, vec![7; 32]);
    write_envelope(
        &mut host_stream,
        &proto::Envelope {
            protocol_version: PROTOCOL_VERSION,
            plugin_id: "demo".into(),
            request_id: request.request_id,
            payload: Some(proto::envelope::Payload::PluginKeyResponse(
                proto::PluginKeyResponse {
                    node_id: "a".repeat(64),
                    binding_signature: vec![9; 64],
                },
            )),
        },
    )
    .await
    .unwrap();

    let response = timeout(Duration::from_secs(1), announced_rx)
        .await
        .expect("the announce should get the reply, not time out")
        .unwrap()
        .expect("the announce should succeed");
    assert_eq!(response.node_id, "a".repeat(64));
    assert_eq!(response.binding_signature, vec![9; 64]);

    runtime.abort();
}

/// A plugin whose manifest gains a local channel only while it initializes.
#[derive(Clone)]
struct LateLocalChannel {
    declared: bool,
}

#[crate::async_trait]
impl Plugin for LateLocalChannel {
    fn plugin_id(&self) -> &str {
        "demo"
    }
    fn plugin_version(&self) -> String {
        "1.0.0".into()
    }
    fn server_info(&self) -> ServerConfig {
        plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>)
    }
    fn manifest(&self) -> Option<proto::PluginManifest> {
        self.declared.then(|| {
            crate::plugin_manifest()
                .item(crate::local_channel("payment.lifecycle.v1"))
                .build()
        })
    }
    async fn initialize(
        &mut self,
        _request: PluginInitializeRequest,
        _context: &mut PluginContext<'_>,
    ) -> crate::PluginResult<()> {
        self.declared = true;
        Ok(())
    }
}

#[tokio::test]
async fn a_local_channel_declared_during_initialize_is_refused_without_host_support() {
    for initially_declared in [false, true] {
        let (plugin_stream, host_stream) = tokio::io::duplex(64 * 1024);
        let runtime = tokio::spawn(PluginRuntime::run_with_stream(
            LateLocalChannel {
                declared: initially_declared,
            },
            LocalStream::Memory(plugin_stream),
        ));
        let mut host_stream = LocalStream::Memory(host_stream);
        write_envelope(
            &mut host_stream,
            &proto::Envelope {
                protocol_version: PROTOCOL_VERSION,
                plugin_id: "demo".into(),
                request_id: 1,
                payload: Some(proto::envelope::Payload::InitializeRequest(
                    proto::InitializeRequest {
                        host_protocol_version: PROTOCOL_VERSION,
                        host_capabilities: Vec::new(),
                        ..Default::default()
                    },
                )),
            },
        )
        .await
        .unwrap();
        let response = timeout(Duration::from_secs(1), read_envelope(&mut host_stream))
            .await
            .expect("initialize response")
            .unwrap();
        let Some(proto::envelope::Payload::ErrorResponse(error)) = response.payload else {
            panic!("expected a refusal, got {:?}", response.payload);
        };
        assert_eq!(
            error.code,
            crate::STARTUP_DISABLED_ERROR_CODE,
            "initially_declared={initially_declared}: {error:?}"
        );
        assert!(error.message.contains("local_channels.v1"), "{error:?}");
        runtime.abort();
    }
}

#[cfg(unix)]
#[tokio::test]
async fn a_panic_in_on_initialized_ends_the_runtime_with_an_error() {
    let plugin = SimplePlugin::new(PluginMetadata::new(
        "demo",
        "1.0.0",
        plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>),
    ))
    .on_initialized(|_context| Box::pin(async { panic!("startup panicked") }));

    let (plugin_stream, host_stream) = tokio::net::UnixStream::pair().unwrap();
    let runtime = tokio::spawn(PluginRuntime::run_with_stream(
        plugin,
        LocalStream::Unix(plugin_stream),
    ));
    let mut host_stream = LocalStream::Unix(host_stream);
    initialize(&mut host_stream, Vec::new()).await;

    let result = timeout(Duration::from_secs(1), runtime)
        .await
        .expect("the runtime should end")
        .unwrap();
    let err = result.expect_err("a panic in on_initialized should end the runtime");
    assert!(err.to_string().contains("on_initialized panicked"), "{err}");
    assert!(err.to_string().contains("startup panicked"), "{err}");
}

#[cfg(unix)]
#[tokio::test]
async fn an_error_from_on_initialized_still_ends_the_runtime() {
    let plugin = SimplePlugin::new(PluginMetadata::new(
        "demo",
        "1.0.0",
        plugin_server_info("demo", "1.0.0", "Demo", "Demo plugin", None::<String>),
    ))
    .on_initialized(|_context| Box::pin(async { bail!("startup failed") }));

    let (plugin_stream, host_stream) = tokio::net::UnixStream::pair().unwrap();
    let runtime = tokio::spawn(PluginRuntime::run_with_stream(
        plugin,
        LocalStream::Unix(plugin_stream),
    ));
    let mut host_stream = LocalStream::Unix(host_stream);
    initialize(&mut host_stream, Vec::new()).await;

    let result = timeout(Duration::from_secs(1), runtime)
        .await
        .expect("the runtime should end")
        .unwrap();
    let err = result.expect_err("the error from on_initialized should end the runtime");
    assert!(err.to_string().contains("startup failed"), "{err}");
}
