//! An operator's endpoint model allowlist over the real plugin protocol: SDK
//! plugins declare an OpenAI-compatible endpoint whose upstream lists several
//! models, and only the allowed ones are advertised or routed.

use super::*;
use crate::plugin::{InProcessPluginRunner, InProcessPlugins, config};
use mesh_llm_plugin::{PluginMetadata, PluginRuntime, SimplePlugin};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Answers every `GET /v1/models` with the same three models.
async fn spawn_upstream() -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = format!("http://{}/v1", listener.local_addr().unwrap());
    let handle = tokio::spawn(async move {
        let body = r#"{"data":[{"id":"sold"},{"id":"unpriced"},{"id":"also-unpriced"}]}"#;
        loop {
            let Ok((mut stream, _)) = listener.accept().await else {
                return;
            };
            let mut buf = vec![0u8; 4096];
            let _ = stream.read(&mut buf).await;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len(),
            );
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.shutdown().await;
        }
    });
    (address, handle)
}

fn endpoint_plugin(name: &'static str, address: String) -> InProcessPluginRunner {
    Arc::new(move |stream| {
        let manifest = mesh_llm_plugin::plugin_manifest()
            .item(mesh_llm_plugin::openai_http_inference_endpoint(
                "upstream",
                address.clone(),
            ))
            .build();
        let plugin = SimplePlugin::new(
            PluginMetadata::new(
                name,
                "1.0.0",
                mesh_llm_plugin::plugin_server_info(
                    name,
                    "1.0.0",
                    "Endpoint",
                    "Serves an upstream",
                    None::<String>,
                ),
            )
            .with_manifest(manifest),
        );
        Box::pin(PluginRuntime::run_with_stream(plugin, stream))
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn only_allowed_endpoint_models_are_advertised_and_routed() {
    let (address, upstream) = spawn_upstream().await;
    let (mesh_tx, _mesh_rx) = mpsc::channel(16);
    let mut in_process = InProcessPlugins::default();
    let mut externals = Vec::new();
    // limited: the operator allows one model. open: no allowlist.
    for (name, allowed) in [
        (
            "limited",
            Some(vec!["sold".to_string(), "not-listed".to_string()]),
        ),
        ("open", None),
    ] {
        in_process = in_process.with(name, endpoint_plugin(name, address.clone()));
        let mut spec = config::in_process_builtin_spec(name);
        spec.startup.optional = false;
        spec.endpoint_models = allowed;
        externals.push(spec);
    }
    let manager = PluginManager::start_with_in_process(
        &config::ResolvedPlugins {
            externals,
            inactive: Vec::new(),
        },
        config::PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        mesh_tx,
        in_process,
    )
    .await
    .unwrap();

    let endpoints = manager.inference_endpoints().await.unwrap();
    let models_of = |plugin: &str| {
        endpoints
            .iter()
            .find(|endpoint| endpoint.plugin_name == plugin)
            .map(|endpoint| endpoint.models.clone())
            .unwrap_or_else(|| panic!("{plugin} endpoint is routable: {endpoints:?}"))
    };
    assert_eq!(models_of("limited"), ["sold"]);
    assert_eq!(models_of("open"), ["sold", "unpriced", "also-unpriced"]);

    // Asking for an excluded model by name reaches only the open plugin.
    let route = manager
        .inference_endpoint_for_model("unpriced")
        .await
        .unwrap()
        .expect("the open plugin still serves it");
    assert_eq!(route.plugin_name, "open");

    manager.shutdown().await;

    // With only the limited plugin, excluded models are not served at all.
    let (mesh_tx, _mesh_rx) = mpsc::channel(16);
    let mut spec = config::in_process_builtin_spec("limited");
    spec.startup.optional = false;
    spec.endpoint_models = Some(vec!["sold".to_string()]);
    let manager = PluginManager::start_with_in_process(
        &config::ResolvedPlugins {
            externals: vec![spec],
            inactive: Vec::new(),
        },
        config::PluginHostMode {
            mesh_visibility: mesh_llm_plugin::MeshVisibility::Private,
        },
        mesh_tx,
        InProcessPlugins::default().with("limited", endpoint_plugin("limited", address)),
    )
    .await
    .unwrap();
    assert_eq!(manager.inference_models().await.unwrap(), ["sold"]);
    for excluded in ["unpriced", "also-unpriced", "not-listed"] {
        assert!(
            manager
                .inference_endpoint_for_model(excluded)
                .await
                .unwrap()
                .is_none(),
            "{excluded} must not be routed to the endpoint"
        );
    }
    assert!(
        manager
            .inference_endpoint_for_model("sold")
            .await
            .unwrap()
            .is_some()
    );
    manager.shutdown().await;
    upstream.abort();
}
