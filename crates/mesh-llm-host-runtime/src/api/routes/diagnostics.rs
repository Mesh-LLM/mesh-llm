use super::super::{
    MeshApi,
    http::{respond_error, respond_json},
};
use tokio::net::TcpStream;
use url::form_urlencoded;

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &MeshApi,
    path: &str,
) -> anyhow::Result<()> {
    let model_ref = match split_readiness_model_ref(path) {
        Some(model_ref) => model_ref,
        None => {
            return respond_error(stream, 400, "Missing required 'model_ref' query parameter")
                .await;
        }
    };
    let report = state.split_readiness_report(&model_ref).await;
    respond_json(stream, 200, &report).await
}

fn split_readiness_model_ref(path: &str) -> Option<String> {
    let (_, raw_query) = path.split_once('?')?;
    for (key, value) in form_urlencoded::parse(raw_query.as_bytes()) {
        if matches!(key.as_ref(), "model_ref" | "model") {
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::split_readiness_model_ref;

    #[test]
    fn split_readiness_query_accepts_percent_encoded_model_ref() {
        assert_eq!(
            split_readiness_model_ref(
                "/api/diagnostics/split-readiness?model_ref=meshllm%2FQwen3-8B-Q4_K_M-layers"
            ),
            Some("meshllm/Qwen3-8B-Q4_K_M-layers".to_string())
        );
    }

    #[test]
    fn split_readiness_query_rejects_blank_model_ref() {
        assert_eq!(
            split_readiness_model_ref("/api/diagnostics/split-readiness?model_ref=%20"),
            None
        );
    }
}
