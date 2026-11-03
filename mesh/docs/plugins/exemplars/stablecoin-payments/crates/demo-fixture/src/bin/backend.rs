//! Canned OpenAI backend owned by a real Mesh plugin process; never a model benchmark.
use anyhow::{Context, Result, ensure};
use axum::{
    Json, Router,
    body::Body,
    http::{Response, StatusCode},
    routing::{get, post},
};
use mesh_llm_plugin::{
    PluginMetadata, PluginRuntime, SimplePlugin, openai_http_inference_endpoint, plugin_manifest,
    plugin_server_info,
};
use serde_json::{Value, json};

const MODEL: &str = "mesh-payment-fixture";

fn manifest(url: &str) -> mesh_llm_plugin::proto::PluginManifest {
    plugin_manifest()
        .item(openai_http_inference_endpoint("fixture", url).managed_by_plugin(true))
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    let url =
        std::env::var("MESH_LLM_PLUGIN_URL").unwrap_or_else(|_| "http://127.0.0.1:18080/v1".into());
    if std::env::args().any(|a| a == "--manifest") {
        println!(
            "{}",
            mesh_llm_plugin::package_manifest_json(&manifest(&url))?
        );
        return Ok(());
    }
    let address = url
        .strip_prefix("http://127.0.0.1:")
        .and_then(|s| s.strip_suffix("/v1"))
        .context("fixture URL must be loopback HTTP with /v1")?;
    let port: u16 = address.parse()?;
    ensure!(port != 0, "fixture requires an explicit reserved port");
    let listener = tokio::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, port)).await?;
    let router = Router::new()
        .route("/health", get(|| async { "offline fixture" }))
        .route("/v1/models", get(|| async { Json(json!({"object":"list","data":[{"id":MODEL,"object":"model","owned_by":"offline-fixture"}]})) }))
        .route("/v1/chat/completions", post(completion));
    let server = tokio::spawn(async move { axum::serve(listener, router).await });
    let plugin = SimplePlugin::new(
        PluginMetadata::new(
            "demo-backend-fixture",
            env!("CARGO_PKG_VERSION"),
            plugin_server_info(
                "demo-backend-fixture",
                env!("CARGO_PKG_VERSION"),
                "Offline canned backend",
                "Returns declared fixture usage; no model weights",
                None::<String>,
            ),
        )
        .with_manifest(manifest(&url)),
    );
    let result = PluginRuntime::run(plugin).await;
    server.abort();
    result
}

async fn completion(Json(request): Json<Value>) -> Result<Response<Body>, StatusCode> {
    if request["model"] != MODEL
        || request["max_tokens"]
            .as_u64()
            .is_none_or(|n| n == 0 || n > 16)
    {
        return Err(StatusCode::BAD_REQUEST);
    }
    let output = request["max_tokens"].as_u64().unwrap().min(4);
    let usage = json!({"prompt_tokens":1,"completion_tokens":output,"total_tokens":output+1});
    let content = "OFFLINE FIXTURE RESPONSE";
    if request["stream"] == true {
        let chunk = json!({"id":"fixture","object":"chat.completion.chunk","created":1,"model":MODEL,
            "choices":[{"index":0,"delta":{"role":"assistant","content":content},"finish_reason":null}]});
        let finish = json!({"id":"fixture","object":"chat.completion.chunk","created":1,"model":MODEL,
            "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]});
        let usage_chunk = json!({"id":"fixture","object":"chat.completion.chunk","created":1,"model":MODEL,
            "choices":[],"usage":usage});
        return Response::builder()
            .header("content-type", "text/event-stream")
            .body(Body::from(format!(
                "data: {chunk}\n\ndata: {finish}\n\ndata: {usage_chunk}\n\ndata: [DONE]\n\n"
            )))
            .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR);
    }
    let response = json!({"id":"fixture","object":"chat.completion","created":1,"model":MODEL,
        "choices":[{"index":0,"message":{"role":"assistant","content":content},"finish_reason":"stop"}],"usage":usage});
    Response::builder()
        .header("content-type", "application/json")
        .body(Body::from(response.to_string()))
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)
}
