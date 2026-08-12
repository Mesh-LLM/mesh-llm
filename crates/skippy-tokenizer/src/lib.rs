//! A model-bound tokenizer capability for in-process Skippy consumers.
//!
//! This crate intentionally contains only the stable contract. Runtime
//! implementations bind it to an already-loaded tokenizer; they must not open
//! another model on behalf of a caller.

use std::{error::Error, fmt};

use serde::{Deserialize, Deserializer, Serialize, Serializer, ser::SerializeStruct};

pub const MAX_TOKENIZE_INPUT_BYTES: usize = 1_048_576;
pub const MAX_TOKENIZE_BATCH_SIZE: usize = 64;
pub const MAX_TOKENIZE_BATCH_INPUT_BYTES: usize = 8 * MAX_TOKENIZE_INPUT_BYTES;
pub const MAX_TOKENIZE_TOKENS: usize = 262_144;
pub const MAX_TOKENIZE_PIECES: usize = 4_096;
pub const TOKENIZER_VERSION: &str = "gguf-native-v1";

/// One piece of a structured, lossless encode request. Control descriptors are
/// opaque to Mesh; a runtime may accept them only when it can preserve their
/// native meaning without decoding them as ordinary text.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum InputPiece {
    Bytes(Vec<u8>),
    Control { descriptor: Vec<u8> },
}

impl InputPiece {
    fn len(&self) -> usize {
        match self {
            Self::Bytes(bytes) | Self::Control { descriptor: bytes } => bytes.len(),
        }
    }
}

/// A bounded request to encode structured input. The byte run is not decoded
/// with replacement semantics: runtimes that cannot preserve it must return
/// [`TokenizerError::UnsupportedInput`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodeRequest {
    pub expected_identity: TokenizerIdentity,
    pub pieces: Vec<InputPiece>,
}

impl EncodeRequest {
    pub fn new(expected_identity: TokenizerIdentity, pieces: Vec<InputPiece>) -> Self {
        Self {
            expected_identity,
            pieces,
        }
    }

    pub fn bytes(expected_identity: TokenizerIdentity, bytes: Vec<u8>) -> Self {
        Self::new(expected_identity, vec![InputPiece::Bytes(bytes)])
    }

    pub fn total_input_bytes(&self) -> Option<usize> {
        self.pieces
            .iter()
            .map(InputPiece::len)
            .try_fold(0, usize::checked_add)
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct EncodeResponse {
    pub identity: TokenizerIdentity,
    pub token_ids: Vec<i32>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TokenizerIdentity {
    pub model_id: String,
    pub source_model_sha256: String,
    pub tokenizer_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer_version: Option<String>,
    #[serde(default)]
    pub stage_index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serving_profile: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecialTokenPolicy {
    #[default]
    Omit,
    Add,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizeRequest {
    pub expected_identity: TokenizerIdentity,
    pub text: String,
    pub special_tokens: SpecialTokenPolicy,
    pub include_token_pieces: bool,
}

#[derive(Deserialize)]
struct TokenizeRequestWire {
    expected_identity: TokenizerIdentity,
    text: String,
    #[serde(default)]
    special_tokens: Option<SpecialTokenPolicy>,
    #[serde(default)]
    add_special: Option<bool>,
    #[serde(default)]
    include_token_pieces: bool,
}

impl<'de> Deserialize<'de> for TokenizeRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = TokenizeRequestWire::deserialize(deserializer)?;
        let special_tokens = wire
            .add_special
            .map(|add_special| {
                if add_special {
                    SpecialTokenPolicy::Add
                } else {
                    SpecialTokenPolicy::Omit
                }
            })
            .or(wire.special_tokens)
            .unwrap_or_default();
        Ok(Self {
            expected_identity: wire.expected_identity,
            text: wire.text,
            special_tokens,
            include_token_pieces: wire.include_token_pieces,
        })
    }
}

impl Serialize for TokenizeRequest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("TokenizeRequest", 5)?;
        state.serialize_field("expected_identity", &self.expected_identity)?;
        state.serialize_field("text", &self.text)?;
        state.serialize_field("special_tokens", &self.special_tokens)?;
        // Keep the legacy REST spelling in serialized requests so a new
        // consumer can still request special tokens from an older server.
        state.serialize_field(
            "add_special",
            &(self.special_tokens == SpecialTokenPolicy::Add),
        )?;
        state.serialize_field("include_token_pieces", &self.include_token_pieces)?;
        state.end()
    }
}

impl TokenizeRequest {
    pub fn with_special_tokens(
        expected_identity: TokenizerIdentity,
        text: String,
        special_tokens: SpecialTokenPolicy,
    ) -> Self {
        Self {
            expected_identity,
            text,
            special_tokens,
            include_token_pieces: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TokenizeResponse {
    pub identity: TokenizerIdentity,
    pub token_ids: Vec<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_pieces: Option<Vec<Vec<u8>>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenizerLimits {
    pub max_batch_size: usize,
    pub max_batch_input_bytes: usize,
    pub max_input_bytes: usize,
    pub max_output_tokens: usize,
}

impl Default for TokenizerLimits {
    fn default() -> Self {
        Self {
            max_batch_size: MAX_TOKENIZE_BATCH_SIZE,
            max_batch_input_bytes: MAX_TOKENIZE_BATCH_INPUT_BYTES,
            max_input_bytes: MAX_TOKENIZE_INPUT_BYTES,
            max_output_tokens: MAX_TOKENIZE_TOKENS,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "code")]
pub enum TokenizerError {
    StageZeroRequired,
    UnsupportedStage {
        stage_index: u32,
    },
    IdentityUnavailable,
    RuntimeUnavailable,
    IdentityMismatch {
        expected: Box<TokenizerIdentity>,
        actual: Box<TokenizerIdentity>,
    },
    BatchTooLarge {
        limit: usize,
    },
    BatchInputTooLarge {
        limit: usize,
    },
    InputTooLarge {
        limit: usize,
    },
    TooManyPieces {
        limit: usize,
    },
    TooManyTokens {
        limit: usize,
    },
    UnsupportedInput {
        reason: String,
    },
    BackendFailure {
        message: String,
    },
}

impl TokenizerError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::StageZeroRequired => "stage_zero_required",
            Self::UnsupportedStage { .. } => "unsupported_stage",
            Self::IdentityUnavailable => "identity_unavailable",
            Self::RuntimeUnavailable => "runtime_unavailable",
            Self::IdentityMismatch { .. } => "identity_mismatch",
            Self::BatchTooLarge { .. } => "batch_too_large",
            Self::BatchInputTooLarge { .. } => "batch_input_too_large",
            Self::InputTooLarge { .. } => "input_too_large",
            Self::TooManyPieces { .. } => "too_many_pieces",
            Self::TooManyTokens { .. } => "too_many_tokens",
            Self::UnsupportedInput { .. } => "unsupported_input",
            Self::BackendFailure { .. } => "backend_failure",
        }
    }
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StageZeroRequired => formatter.write_str("stage-zero tokenizer required"),
            Self::UnsupportedStage { stage_index } => {
                write!(formatter, "tokenizer is unavailable on stage {stage_index}")
            }
            Self::IdentityUnavailable => formatter.write_str("tokenizer identity unavailable"),
            Self::RuntimeUnavailable => formatter.write_str("tokenizer runtime unavailable"),
            Self::IdentityMismatch { .. } => formatter.write_str("tokenizer identity mismatch"),
            Self::BatchTooLarge { limit } => {
                write!(formatter, "tokenizer batch exceeds {limit} requests")
            }
            Self::BatchInputTooLarge { limit } => {
                write!(formatter, "tokenizer batch input exceeds {limit} bytes")
            }
            Self::InputTooLarge { limit } => {
                write!(formatter, "tokenizer input exceeds {limit} bytes")
            }
            Self::TooManyPieces { limit } => {
                write!(formatter, "tokenizer input exceeds {limit} pieces")
            }
            Self::TooManyTokens { limit } => {
                write!(formatter, "tokenizer output exceeds {limit} tokens")
            }
            Self::UnsupportedInput { reason } => {
                write!(formatter, "tokenizer does not support this input: {reason}")
            }
            Self::BackendFailure { message } => {
                write!(formatter, "tokenizer backend failure: {message}")
            }
        }
    }
}

impl Error for TokenizerError {}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TokenizeBatchItem {
    pub request_index: usize,
    pub result: Result<TokenizeResponse, TokenizerError>,
}

pub trait Tokenizer: Send + Sync {
    fn identity(&self) -> &TokenizerIdentity;

    fn limits(&self) -> TokenizerLimits {
        TokenizerLimits::default()
    }

    fn tokenize_batch(
        &self,
        requests: &[TokenizeRequest],
    ) -> Result<Vec<TokenizeBatchItem>, TokenizerError>;

    /// Encode one structured byte/control sequence with automatic special-token
    /// insertion disabled. Implementations must reject input they cannot
    /// preserve exactly instead of applying lossy decoding. Mesh does not
    /// interpret Rosetta or control identities; control descriptors remain
    /// opaque to this contract.
    fn encode(&self, request: EncodeRequest) -> Result<EncodeResponse, TokenizerError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> TokenizerIdentity {
        TokenizerIdentity {
            model_id: "model".to_owned(),
            source_model_sha256: "a".repeat(64),
            tokenizer_id: "gguf-source-sha256:".to_owned() + &"a".repeat(64),
            tokenizer_version: Some("gguf-v1".to_owned()),
            stage_index: 0,
            serving_profile: Some("stage-zero".to_owned()),
        }
    }

    #[test]
    fn request_defaults_are_safe_and_identity_is_serialized() {
        let request: TokenizeRequest = serde_json::from_value(serde_json::json!({
            "expected_identity": identity(),
            "text": "hello"
        }))
        .unwrap();
        assert_eq!(request.special_tokens, SpecialTokenPolicy::Omit);
        assert!(!request.include_token_pieces);
        assert_eq!(
            serde_json::from_value::<TokenizerIdentity>(serde_json::to_value(identity()).unwrap())
                .unwrap(),
            identity()
        );
        let serialized = serde_json::to_value(&request).unwrap();
        assert_eq!(serialized["add_special"], false);
        let legacy: TokenizeRequest = serde_json::from_value(serde_json::json!({
            "expected_identity": identity(),
            "text": "hello",
            "add_special": true
        }))
        .unwrap();
        assert_eq!(legacy.special_tokens, SpecialTokenPolicy::Add);
    }

    #[test]
    fn error_codes_are_machine_readable() {
        assert_eq!(
            TokenizerError::BatchTooLarge { limit: 1 }.code(),
            "batch_too_large"
        );
        assert_eq!(
            TokenizerError::UnsupportedStage { stage_index: 1 }.code(),
            "unsupported_stage"
        );
        assert_eq!(
            TokenizerError::UnsupportedInput {
                reason: "invalid UTF-8".to_owned()
            }
            .code(),
            "unsupported_input"
        );
    }
}
