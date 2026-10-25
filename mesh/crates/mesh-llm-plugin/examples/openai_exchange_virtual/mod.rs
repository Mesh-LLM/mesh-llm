//! Optional deterministic virtual backend for installed dispatch conformance.
pub fn router() -> mesh_llm_plugin::VirtualModelRouter {
    let mut router = mesh_llm_plugin::VirtualModelRouter::new();
    router.add_json(mesh_llm_plugin::operation_with_schema("virtual_echo", "Lifecycle conformance echo", serde_json::Map::new()), |request: mesh_llm_plugin::VirtualModelInvocation, _context| {
        Box::pin(async move {
            if let Ok(path) = std::env::var("MESH_LLM_EXEMPLAR_VIRTUAL_LOG") {
                use tokio::io::AsyncWriteExt;
                let mut options = tokio::fs::OpenOptions::new();
                options.create(true).write(true).truncate(true);
                #[cfg(unix)]
                options.mode(0o600);
                let mut file = options.open(path).await.map_err(anyhow::Error::from)?;
                file.write_all(&serde_json::to_vec(&request).map_err(anyhow::Error::from)?).await.map_err(anyhow::Error::from)?;
            }
            Ok(mesh_llm_plugin::VirtualModelResponse {
                status_code: 200,
                body: serde_json::json!({"id":"echo","object":"chat.completion","created":0,"model":"exchange-echo","choices":[{"index":0,"message":{"role":"assistant","content":"echo"},"finish_reason":"stop"}]}),
                headers: Vec::new(), event_stream: false,
            })
        })
    });
    router
}
