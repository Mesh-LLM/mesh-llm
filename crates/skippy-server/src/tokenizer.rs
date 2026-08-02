use std::{
    error::Error,
    fmt,
    sync::{Arc, Mutex},
};

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use serde::{Deserialize, Serialize};
use skippy_protocol::{
    MAX_TOKENIZE_INPUT_BYTES, StageConfig, TokenizeRequest, TokenizeResponse, TokenizerIdentity,
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
    fn token_piece(&self, token_id: i32) -> Result<Vec<u8>, TokenizerCapabilityError>;
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

    fn token_piece(&self, token_id: i32) -> Result<Vec<u8>, TokenizerCapabilityError> {
        self.runtime
            .lock()
            .map_err(|_| TokenizerCapabilityError::BackendFailure)?
            .model
            .detokenize_bytes(&[token_id])
            .map_err(|_| TokenizerCapabilityError::BackendFailure)
    }
}

#[derive(Clone)]
pub struct TokenizerCapability {
    identity: TokenizerIdentity,
    source: Arc<dyn TokenizerSource>,
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
        Ok(Self::from_loaded_stage_zero(identity, runtime))
    }

    pub(crate) fn from_loaded_stage_zero(
        identity: TokenizerIdentity,
        runtime: Arc<Mutex<RuntimeState>>,
    ) -> Self {
        Self {
            identity,
            source: Arc::new(LoadedStageZeroTokenizer { runtime }),
        }
    }

    pub fn identity(&self) -> &TokenizerIdentity {
        &self.identity
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
            .then(|| {
                token_ids
                    .iter()
                    .map(|token_id| self.source.token_piece(*token_id))
                    .collect()
            })
            .transpose()?;
        Ok(TokenizeResponse {
            identity: self.identity.clone(),
            token_ids,
            token_pieces,
        })
    }
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
    capability
        .tokenize(request)
        .map(Json)
        .map_err(TokenizerHttpError)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use axum::{
        body::{Body, to_bytes},
        http::{Request, StatusCode, header::CONTENT_TYPE},
    };
    use tower::ServiceExt;

    use super::*;

    const SHA256: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    struct RecordingTokenizer {
        tokens: Vec<i32>,
        generation_mutations: AtomicUsize,
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

        fn token_piece(&self, token_id: i32) -> Result<Vec<u8>, TokenizerCapabilityError> {
            Ok(token_id.to_string().into_bytes())
        }
    }

    fn identity() -> TokenizerIdentity {
        tokenizer_identity_from_stage(0, "model", Some(SHA256)).unwrap()
    }

    fn capability(tokens: Vec<i32>) -> (TokenizerCapability, Arc<RecordingTokenizer>) {
        let source = Arc::new(RecordingTokenizer {
            tokens,
            generation_mutations: AtomicUsize::new(0),
        });
        (
            TokenizerCapability {
                identity: identity(),
                source: source.clone(),
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
    fn tokenization_does_not_mutate_generation_state() {
        let (capability, source) = capability(vec![4, 5]);
        let response = capability.tokenize(request("hello".to_string())).unwrap();
        assert_eq!(response.token_ids, vec![4, 5]);
        assert_eq!(source.generation_mutations.load(Ordering::SeqCst), 0);
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
