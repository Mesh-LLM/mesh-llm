use std::{
    error::Error,
    fmt,
    path::Path,
    sync::{Arc, Mutex},
};

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use mesh_native_serving_plugin_api as native_plugin_api;
use model_artifact::gguf::scan_gguf_tokenizer_inventory;
use serde::{Deserialize, Serialize};
use skippy_protocol::{
    StageConfig,
    tokenizer::{MAX_TOKENIZE_INPUT_BYTES, TokenizeRequest, TokenizeResponse, TokenizerIdentity},
};

use crate::runtime_state::RuntimeState;

pub const MAX_TOKENIZE_TOKENS: usize = 262_144;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerCapabilityError {
    StageZeroRequired,
    IdentityUnavailable,
    IdentityMismatch,
    InputTooLarge,
    TooManyTokens,
    BackendFailure,
}

impl TokenizerCapabilityError {
    pub const fn code(self) -> &'static str {
        match self {
            Self::StageZeroRequired => "stage_zero_required",
            Self::IdentityUnavailable => "identity_unavailable",
            Self::IdentityMismatch => "identity_mismatch",
            Self::InputTooLarge => "input_too_large",
            Self::TooManyTokens => "too_many_tokens",
            Self::BackendFailure => "backend_failure",
        }
    }
}

impl fmt::Display for TokenizerCapabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

impl Error for TokenizerCapabilityError {}

pub(crate) fn tokenizer_identity_from_stage(
    stage_index: u32,
    model_id: &str,
    source_model_sha256: Option<&str>,
) -> Result<TokenizerIdentity, TokenizerCapabilityError> {
    if stage_index != 0 {
        return Err(TokenizerCapabilityError::StageZeroRequired);
    }
    if model_id.trim().is_empty() {
        return Err(TokenizerCapabilityError::IdentityUnavailable);
    }
    let source_model_sha256 = source_model_sha256
        .filter(|value| is_sha256(value))
        .ok_or(TokenizerCapabilityError::IdentityUnavailable)?
        .to_ascii_lowercase();
    Ok(TokenizerIdentity {
        model_id: model_id.to_owned(),
        tokenizer_id: format!("gguf-source-sha256:{source_model_sha256}"),
        source_model_sha256,
    })
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

trait TokenizerSource: Send + Sync {
    fn tokenize(&self, text: &str, add_special: bool)
    -> Result<Vec<i32>, TokenizerCapabilityError>;
    fn token_pieces(&self, token_ids: &[i32]) -> Result<Vec<Vec<u8>>, TokenizerCapabilityError>;
}

struct LoadedStageZeroTokenizer {
    runtime: Arc<Mutex<RuntimeState>>,
}

impl TokenizerSource for LoadedStageZeroTokenizer {
    fn tokenize(
        &self,
        text: &str,
        add_special: bool,
    ) -> Result<Vec<i32>, TokenizerCapabilityError> {
        self.runtime
            .lock()
            .map_err(|_| TokenizerCapabilityError::BackendFailure)?
            .model
            .tokenize(text, add_special)
            .map_err(|_| TokenizerCapabilityError::BackendFailure)
    }

    fn token_pieces(&self, token_ids: &[i32]) -> Result<Vec<Vec<u8>>, TokenizerCapabilityError> {
        let runtime = self
            .runtime
            .lock()
            .map_err(|_| TokenizerCapabilityError::BackendFailure)?;
        token_ids
            .iter()
            .map(|token_id| {
                runtime
                    .model
                    .detokenize_bytes(&[*token_id])
                    .map_err(|_| TokenizerCapabilityError::BackendFailure)
            })
            .collect()
    }
}

#[derive(Clone)]
pub struct TokenizerCapability {
    identity: TokenizerIdentity,
    source: Arc<dyn TokenizerSource>,
    inventory: Option<Arc<native_plugin_api::TokenizerInventory>>,
}

impl TokenizerCapability {
    pub(crate) fn from_stage_zero(
        config: &StageConfig,
        runtime: Arc<Mutex<RuntimeState>>,
    ) -> Result<Self, TokenizerCapabilityError> {
        let identity = tokenizer_identity_from_stage(
            config.stage_index,
            &config.model_id,
            config.source_model_sha256.as_deref(),
        )?;
        let source: Arc<dyn TokenizerSource> = Arc::new(LoadedStageZeroTokenizer { runtime });
        let inventory = inventory_from_stage(config, &identity, source.as_ref()).map(Arc::new);
        Ok(Self {
            identity,
            source,
            inventory,
        })
    }

    pub fn identity(&self) -> &TokenizerIdentity {
        &self.identity
    }

    /// A fully materialized, immutable native vocabulary, when the stage was
    /// bound from a readable source GGUF. Inventory construction occurs while
    /// binding the model, never from the decode proposal path.
    pub fn inventory(&self) -> Option<&native_plugin_api::TokenizerInventory> {
        self.inventory.as_deref()
    }

    pub fn tokenize(
        &self,
        request: TokenizeRequest,
    ) -> Result<TokenizeResponse, TokenizerCapabilityError> {
        if request.expected_identity != self.identity {
            return Err(TokenizerCapabilityError::IdentityMismatch);
        }
        if request.text.len() > MAX_TOKENIZE_INPUT_BYTES {
            return Err(TokenizerCapabilityError::InputTooLarge);
        }
        let token_ids = self.source.tokenize(&request.text, request.add_special)?;
        if token_ids.len() > MAX_TOKENIZE_TOKENS {
            return Err(TokenizerCapabilityError::TooManyTokens);
        }
        let token_pieces = request
            .include_token_pieces
            .then(|| self.source.token_pieces(&token_ids))
            .transpose()?;
        Ok(TokenizeResponse {
            identity: self.identity.clone(),
            token_ids,
            token_pieces,
        })
    }
}

fn source_gguf_path(config: &StageConfig) -> Option<&Path> {
    [
        config.source_model_path.as_deref(),
        config.model_path.as_deref(),
    ]
    .into_iter()
    .flatten()
    .map(Path::new)
    .find(|path| path.is_file())
}

fn inventory_from_stage(
    config: &StageConfig,
    identity: &TokenizerIdentity,
    source: &dyn TokenizerSource,
) -> Option<native_plugin_api::TokenizerInventory> {
    let source_path = source_gguf_path(config)?;
    let vocabulary = scan_gguf_tokenizer_inventory(source_path)?;
    let token_ids = (0..vocabulary.tokens.len())
        .map(|id| i32::try_from(id).ok())
        .collect::<Option<Vec<_>>>()?;
    let token_pieces = source.token_pieces(&token_ids).ok()?;
    if token_pieces.len() != vocabulary.tokens.len() {
        return None;
    }
    let mut tokens = Vec::with_capacity(vocabulary.tokens.len());
    for (id, (token, bytes)) in vocabulary.tokens.into_iter().zip(token_pieces).enumerate() {
        let id = u32::try_from(id).ok()?;
        let piece = if token.is_control {
            native_plugin_api::TokenizerInventoryPiece::Control {
                identity: String::from_utf8(token.raw).ok()?,
            }
        } else {
            native_plugin_api::TokenizerInventoryPiece::Bytes { bytes }
        };
        tokens.push(native_plugin_api::TokenizerInventoryToken { id, piece });
    }
    Some(native_plugin_api::TokenizerInventory {
        schema_version: native_plugin_api::TOKENIZER_INVENTORY_SCHEMA,
        model_id: identity.model_id.clone(),
        source_model_sha256: identity.source_model_sha256.clone(),
        tokenizer_id: identity.tokenizer_id.clone(),
        tokens,
    })
}

#[derive(Debug, Serialize)]
struct TokenizerErrorBody {
    error: &'static str,
}

struct TokenizerHttpError(TokenizerCapabilityError);

impl IntoResponse for TokenizerHttpError {
    fn into_response(self) -> Response {
        let status = match self.0 {
            TokenizerCapabilityError::InputTooLarge => StatusCode::PAYLOAD_TOO_LARGE,
            TokenizerCapabilityError::TooManyTokens => StatusCode::UNPROCESSABLE_ENTITY,
            TokenizerCapabilityError::BackendFailure => StatusCode::INTERNAL_SERVER_ERROR,
            TokenizerCapabilityError::StageZeroRequired
            | TokenizerCapabilityError::IdentityUnavailable
            | TokenizerCapabilityError::IdentityMismatch => StatusCode::CONFLICT,
        };
        (
            status,
            Json(TokenizerErrorBody {
                error: self.0.code(),
            }),
        )
            .into_response()
    }
}

/// Skippy's tokenizer extension for the product OpenAI endpoint.
///
/// The capability must come from the same already-loaded stage-0 runtime used
/// for generation. This router owns the only HTTP tokenizer route; the stage
/// transport server deliberately does not expose it.
pub(crate) fn tokenizer_http_router(capability: TokenizerCapability) -> Router {
    Router::new()
        .route("/v1/tokenize", post(tokenize_entrypoint))
        .with_state(capability)
}

async fn tokenize_entrypoint(
    State(capability): State<TokenizerCapability>,
    Json(request): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, TokenizerHttpError> {
    tokio::task::spawn_blocking(move || capability.tokenize(request))
        .await
        .map_err(|_| TokenizerHttpError(TokenizerCapabilityError::BackendFailure))?
        .map(Json)
        .map_err(TokenizerHttpError)
}

#[cfg(test)]
mod tests {
    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode, header::CONTENT_TYPE},
    };
    use skippy_protocol::LoadMode;
    use tower::ServiceExt;

    use super::*;

    const SHA256: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn stage_config(source_model_path: Option<String>, model_path: Option<String>) -> StageConfig {
        StageConfig {
            run_id: "run".to_owned(),
            topology_id: "topology".to_owned(),
            model_id: "model".to_owned(),
            package_ref: None,
            manifest_sha256: None,
            source_model_path,
            source_model_sha256: None,
            source_model_bytes: None,
            materialized_path: None,
            materialized_pinned: false,
            model_path,
            projector_path: None,
            stage_id: "stage-0".to_owned(),
            stage_index: 0,
            layer_start: 0,
            layer_end: 1,
            ctx_size: 1_024,
            lane_count: 1,
            n_batch: None,
            n_ubatch: None,
            n_gpu_layers: 0,
            mmap: None,
            mlock: false,
            cache_type_k: "f16".to_owned(),
            cache_type_v: "f16".to_owned(),
            flash_attn_type: Default::default(),
            filter_tensors_on_load: false,
            selected_device: None,
            kv_cache: None,
            native_mtp_enabled: true,
            load_mode: LoadMode::RuntimeSlice,
            bind_addr: "127.0.0.1:0".to_owned(),
            upstream: None,
            downstream: None,
        }
    }

    struct RecordingTokenizer {
        tokens: Vec<i32>,
    }

    struct UnreachableOpenAiBackend;

    #[async_trait::async_trait]
    impl openai_frontend::OpenAiBackend for UnreachableOpenAiBackend {
        async fn models(&self) -> openai_frontend::OpenAiResult<Vec<openai_frontend::ModelObject>> {
            unreachable!("tokenizer requests must not enter the generic OpenAI backend")
        }

        async fn chat_completion(
            &self,
            _request: openai_frontend::ChatCompletionRequest,
        ) -> openai_frontend::OpenAiResult<openai_frontend::ChatCompletionResponse> {
            unreachable!("tokenizer requests must not enter the generic OpenAI backend")
        }

        async fn chat_completion_stream(
            &self,
            _request: openai_frontend::ChatCompletionRequest,
            _context: openai_frontend::OpenAiRequestContext,
        ) -> openai_frontend::OpenAiResult<openai_frontend::ChatCompletionStream> {
            unreachable!("tokenizer requests must not enter the generic OpenAI backend")
        }
    }

    impl TokenizerSource for RecordingTokenizer {
        fn tokenize(
            &self,
            _text: &str,
            add_special: bool,
        ) -> Result<Vec<i32>, TokenizerCapabilityError> {
            let mut tokens = self.tokens.clone();
            if add_special {
                tokens.insert(0, 1);
            }
            Ok(tokens)
        }

        fn token_pieces(
            &self,
            token_ids: &[i32],
        ) -> Result<Vec<Vec<u8>>, TokenizerCapabilityError> {
            Ok(token_ids
                .iter()
                .map(|token_id| token_id.to_string().into_bytes())
                .collect())
        }
    }

    fn identity() -> TokenizerIdentity {
        tokenizer_identity_from_stage(0, "model", Some(SHA256)).unwrap()
    }

    fn capability(tokens: Vec<i32>) -> (TokenizerCapability, Arc<RecordingTokenizer>) {
        let source = Arc::new(RecordingTokenizer { tokens });
        (
            TokenizerCapability {
                identity: identity(),
                source: source.clone(),
                inventory: None,
            },
            source,
        )
    }

    fn request(text: String) -> TokenizeRequest {
        TokenizeRequest {
            expected_identity: identity(),
            text,
            add_special: false,
            include_token_pieces: false,
        }
    }

    async fn post_tokenize(capability: TokenizerCapability, request: &TokenizeRequest) -> Response {
        tokenizer_http_router(capability)
            .oneshot(
                Request::post("/v1/tokenize")
                    .header(CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_vec(request).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap()
    }

    async fn response_json(response: Response) -> serde_json::Value {
        serde_json::from_slice(
            &to_bytes(response.into_body(), 64 * 1024 * 1024)
                .await
                .unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn tokenization_returns_source_tokens() {
        let (capability, _) = capability(vec![4, 5]);
        let response = capability.tokenize(request("hello".to_string())).unwrap();
        assert_eq!(response.token_ids, vec![4, 5]);
    }

    #[test]
    fn bounds_input_and_output() {
        let (input_bounded, _) = capability(Vec::new());
        let error = input_bounded
            .tokenize(request("x".repeat(MAX_TOKENIZE_INPUT_BYTES + 1)))
            .unwrap_err();
        assert_eq!(error, TokenizerCapabilityError::InputTooLarge);

        let (output_bounded, _) = capability(vec![7; MAX_TOKENIZE_TOKENS + 1]);
        let error = output_bounded
            .tokenize(request("x".to_string()))
            .unwrap_err();
        assert_eq!(error, TokenizerCapabilityError::TooManyTokens);
    }

    #[test]
    fn identity_is_authoritative_and_fail_closed() {
        assert_eq!(
            tokenizer_identity_from_stage(1, "model", Some(SHA256)).unwrap_err(),
            TokenizerCapabilityError::StageZeroRequired
        );
        assert_eq!(
            tokenizer_identity_from_stage(0, "model", None).unwrap_err(),
            TokenizerCapabilityError::IdentityUnavailable
        );
        let (capability, _) = capability(Vec::new());
        let mut request = request("x".to_string());
        request.expected_identity.model_id = "another-model".to_string();
        assert_eq!(
            capability.tokenize(request).unwrap_err(),
            TokenizerCapabilityError::IdentityMismatch
        );
    }

    #[test]
    fn source_gguf_path_falls_back_when_source_model_path_is_unavailable() {
        let temp_dir = tempfile::tempdir().expect("create temporary GGUF directory");
        let model_path = temp_dir.path().join("model.gguf");
        std::fs::write(&model_path, []).expect("create temporary GGUF");
        let config = stage_config(
            Some(temp_dir.path().join("missing.gguf").display().to_string()),
            Some(model_path.display().to_string()),
        );

        assert_eq!(source_gguf_path(&config), Some(model_path.as_path()));
    }

    #[test]
    fn optional_pieces_align_one_to_one_with_exact_token_ids() {
        let (capability, _) = capability(vec![4, 29, 8]);
        let mut request = request("hello".to_string());
        request.include_token_pieces = true;
        let response = capability.tokenize(request).unwrap();
        assert_eq!(response.token_ids, vec![4, 29, 8]);
        assert_eq!(
            response.token_pieces.unwrap(),
            vec![b"4".to_vec(), b"29".to_vec(), b"8".to_vec()]
        );
    }

    #[tokio::test]
    async fn openai_tokenizer_route_preserves_the_exact_wire_contract() {
        let (capability, _) = capability(vec![4, 29, 8]);
        let mut request = request("hello".to_owned());
        request.include_token_pieces = true;

        let response =
            crate::embedded::openai_backend_router(Arc::new(UnreachableOpenAiBackend), capability)
                .oneshot(
                    Request::post("/v1/tokenize")
                        .header(CONTENT_TYPE, "application/json")
                        .body(Body::from(serde_json::to_vec(&request).unwrap()))
                        .unwrap(),
                )
                .await
                .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response_json(response).await,
            serde_json::json!({
                "identity": identity(),
                "token_ids": [4, 29, 8],
                "token_pieces": [[52], [50, 57], [56]],
            })
        );
    }

    #[tokio::test]
    async fn openai_tokenizer_route_rejects_identity_mismatch() {
        let (capability, _) = capability(vec![4, 5]);
        let mut request = request("hello".to_owned());
        request.expected_identity.model_id = "wrong-model".to_owned();

        let response = post_tokenize(capability, &request).await;

        assert_eq!(response.status(), StatusCode::CONFLICT);
        assert_eq!(
            response_json(response).await,
            serde_json::json!({"error": "identity_mismatch"})
        );
    }

    #[tokio::test]
    async fn openai_tokenizer_route_enforces_input_and_output_bounds() {
        let (input_bounded, _) = capability(Vec::new());
        let input_response = post_tokenize(
            input_bounded,
            &request("x".repeat(MAX_TOKENIZE_INPUT_BYTES + 1)),
        )
        .await;
        assert_eq!(input_response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert_eq!(
            response_json(input_response).await,
            serde_json::json!({"error": "input_too_large"})
        );

        let (output_bounded, _) = capability(vec![7; MAX_TOKENIZE_TOKENS + 1]);
        let output_response = post_tokenize(output_bounded, &request("x".to_owned())).await;
        assert_eq!(output_response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(
            response_json(output_response).await,
            serde_json::json!({"error": "too_many_tokens"})
        );
    }
}
