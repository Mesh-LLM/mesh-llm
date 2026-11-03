// Issue #2383: what a caller sees when the host calls into a plugin over HTTP.
//
// Every case here drives the real management API over TCP against an
// in-process plugin that keeps JSON key order, so the assertions are the bytes
// a `curl` client reads.

use crate::plugin::{InProcessPluginRunner, proto};
use mesh_llm_plugin::{LocalStream, read_envelope, write_envelope};

const ORDERED: &str = r#"{"zeta":1,"alpha":{"b":1,"a":2}}"#;

fn demo_operation(name: &str) -> proto::OperationManifest {
    proto::OperationManifest {
        name: name.to_string(),
        description: name.to_string(),
        input_schema_json: "{}".to_string(),
        ..Default::default()
    }
}

fn demo_http_binding(path: &str) -> proto::HttpBindingManifest {
    proto::HttpBindingManifest {
        binding_id: path.trim_start_matches('/').to_string(),
        method: proto::HttpMethod::Post as i32,
        path: path.to_string(),
        operation_name: Some(path.trim_start_matches('/').to_string()),
        request_body_mode: proto::HttpBodyMode::Buffered as i32,
        response_body_mode: proto::HttpBodyMode::Buffered as i32,
        ..Default::default()
    }
}

fn demo_manifest() -> proto::PluginManifest {
    proto::PluginManifest {
        operations: ["echo", "ordered", "add", "missing", "boom"]
            .map(demo_operation)
            .to_vec(),
        http_bindings: vec![demo_http_binding("/echo"), demo_http_binding("/ordered")],
        ..Default::default()
    }
}

fn demo_plugin_error(code: i32, message: &str) -> proto::envelope::Payload {
    proto::envelope::Payload::ErrorResponse(proto::ErrorResponse {
        code,
        message: message.to_string(),
        data_json: String::new(),
    })
}

/// Serves the operations the demo manifest publishes. `echo` returns the exact
/// text it was given, so its reply shows what the plugin actually received.
async fn serve_demo(mut stream: LocalStream) -> anyhow::Result<()> {
    loop {
        let incoming = read_envelope(&mut stream).await?;
        let payload = match incoming.payload {
            Some(proto::envelope::Payload::InitializeRequest(_)) => {
                proto::envelope::Payload::InitializeResponse(proto::InitializeResponse {
                    plugin_id: "demo".into(),
                    plugin_protocol_version: crate::plugin::PROTOCOL_VERSION,
                    plugin_version: "0.0.0".into(),
                    server_info_json: serde_json::to_string(
                        &rmcp::model::ServerConfig::default(),
                    )?,
                    manifest: Some(demo_manifest()),
                    ..Default::default()
                })
            }
            Some(proto::envelope::Payload::InvokeServiceRequest(request)) => {
                match request.service_name.as_str() {
                    "echo" => {
                        proto::envelope::Payload::InvokeServiceResponse(proto::InvokeServiceResponse
                        { output_json: request.input_json, is_error: false })
                    }
                    "ordered" => {
                        proto::envelope::Payload::InvokeServiceResponse(proto::InvokeServiceResponse
                        { output_json: ORDERED.to_string(), is_error: false })
                    }
                    "add" => demo_plugin_error(-32602, "Invalid arguments for tool 'add'"),
                    "missing" => demo_plugin_error(-32601, "Unknown operation 'missing'"),
                    "boom" => {
                        proto::envelope::Payload::InvokeServiceResponse(proto::InvokeServiceResponse
                        { output_json: r#"{"code":"BAD_INPUT","message":"nope"}"#.into(), is_error: true })
                    }
                    other => demo_plugin_error(-32601, &format!("Unknown operation '{other}'")),
                }
            }
            Some(proto::envelope::Payload::HealthRequest(_)) => {
                proto::envelope::Payload::HealthResponse(proto::HealthResponse {
                    status: proto::health_response::Status::Ok as i32,
                    detail: String::new(),
                })
            }
            Some(proto::envelope::Payload::ShutdownRequest(_)) => return Ok(()),
            _ => continue,
        };
        write_envelope(
            &mut stream,
            &proto::Envelope {
                protocol_version: crate::plugin::PROTOCOL_VERSION,
                plugin_id: incoming.plugin_id,
                request_id: incoming.request_id,
                payload: Some(payload),
            },
        )
        .await?;
    }
}

async fn build_demo_plugin_mesh_api() -> MeshApi {
    let node = crate::mesh::Node::new_for_tests(crate::mesh::NodeRole::Worker)
        .await
        .unwrap();
    let runner: InProcessPluginRunner =
        std::sync::Arc::new(|stream| Box::pin(serve_demo(stream)));
    let plugin_manager = crate::plugin::PluginManager::start_with_in_process(
        &crate::plugin::ResolvedPlugins {
            externals: vec![crate::plugin::in_process_builtin_spec("demo")],
            inactive: vec![],
        },
        crate::plugin::PluginHostMode {
            mesh_visibility: MeshVisibility::Private,
        },
        mpsc::channel(1).0,
        crate::plugin::InProcessPlugins::default().with("demo", runner),
    )
    .await
    .unwrap();
    let summaries = plugin_manager.list().await;
    assert_eq!(
        summaries.first().map(|summary| summary.status.as_str()),
        Some("running"),
        "the demo plugin must be up for these route tests"
    );
    let runtime_data_collector = node.runtime_data_collector();
    let runtime_data_producer =
        runtime_data_collector.producer(crate::runtime_data::RuntimeDataSource {
            scope: "runtime",
            plugin_data_key: None,
            plugin_endpoint_key: None,
        });
    MeshApi::new(MeshApiConfig {
        node,
        model_name: "test-model".to_string(),
        api_port: 3131,
        model_size_bytes: 0,
        owner_key_path: None,
        plugin_manager,
        affinity_router: crate::network::affinity::AffinityRouter::default(),
        runtime_data_collector,
        runtime_data_producer,
    })
}

fn post_plugin_request(path: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
        body.len(),
        body
    )
}

async fn call_plugin_api(path: &str, body: &str) -> String {
    let state = build_demo_plugin_mesh_api().await;
    let (address, server) = spawn_management_test_server(state).await;
    let response = send_management_request(address, post_plugin_request(path, body)).await;
    server.await.unwrap().unwrap();
    response
}

fn response_body(response: &str) -> &str {
    response
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .unwrap_or_default()
}

/// The buffered HTTP binding must not re-sort the body on the way in, nor the
/// plugin's reply on the way out.
#[tokio::test]
async fn buffered_http_binding_keeps_plugin_json_key_order() {
    let response = call_plugin_api("/api/plugins/demo/http/echo", ORDERED).await;
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "response: {response}"
    );
    assert_eq!(
        response_body(&response),
        ORDERED,
        "buffered HTTP binding re-sorted the plugin JSON"
    );
}

#[tokio::test]
async fn buffered_http_binding_keeps_the_order_the_plugin_replied_with() {
    let response = call_plugin_api("/api/plugins/demo/http/ordered", "{}").await;
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "response: {response}"
    );
    assert_eq!(response_body(&response), ORDERED);
}

/// The tools route already passes text through; it is the reference behaviour
/// the buffered binding has to match.
#[tokio::test]
async fn tools_route_keeps_plugin_json_key_order() {
    let response = call_plugin_api("/api/plugins/demo/tools/echo", ORDERED).await;
    assert!(
        response.starts_with("HTTP/1.1 200"),
        "response: {response}"
    );
    assert_eq!(response_body(&response), ORDERED);
}

#[tokio::test]
async fn plugin_invalid_params_keeps_a_400_status() {
    let response = call_plugin_api("/api/plugins/demo/tools/add", r#"{"a":"x","b":1}"#).await;
    assert!(
        response.starts_with("HTTP/1.1 400"),
        "a caller's invalid params must be a client error, got: {response}"
    );
    assert!(response.contains("-32602"), "response: {response}");
}

#[tokio::test]
async fn plugin_method_not_found_keeps_a_404_status() {
    let response = call_plugin_api("/api/plugins/demo/tools/missing", "{}").await;
    assert!(
        response.starts_with("HTTP/1.1 404"),
        "a plugin method-not-found must be a client error, got: {response}"
    );
}

/// An error result's JSON must stay a JSON object in the response body.
#[tokio::test]
async fn plugin_error_result_is_returned_as_a_json_object() {
    let response = call_plugin_api("/api/plugins/demo/tools/boom", "{}").await;
    assert!(
        response.starts_with("HTTP/1.1 502"),
        "response: {response}"
    );
    let body = json_body(&response);
    assert!(
        body["error"].is_object(),
        "the error detail must be an object, got: {body}"
    );
    assert_eq!(body["error"]["code"], "BAD_INPUT");
}
