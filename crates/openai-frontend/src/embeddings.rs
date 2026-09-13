use base64::{Engine, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};

use crate::{OpenAiError, OpenAiResult, Usage};

const DEFAULT_ENCODING_FORMAT: &str = "float";

/// OpenAI-compatible embedding input. Token arrays are preserved so clients
/// can avoid a second tokenizer pass when they already own tokenization.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Text(String),
    Texts(Vec<String>),
    Tokens(Vec<i32>),
    TokenArrays(Vec<Vec<i32>>),
}

impl EmbeddingInput {
    /// Count independent inputs; a single token array is one input, not a batch.
    pub fn len(&self) -> usize {
        match self {
            Self::Text(_) | Self::Tokens(_) => 1,
            Self::Texts(values) => values.len(),
            Self::TokenArrays(values) => values.len(),
        }
    }

    /// Reject empty batches and empty members in text or pre-tokenized input.
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Text(value) => value.is_empty(),
            Self::Texts(values) => values.is_empty() || values.iter().any(String::is_empty),
            Self::Tokens(values) => values.is_empty(),
            Self::TokenArrays(values) => values.is_empty() || values.iter().any(Vec::is_empty),
        }
    }

    /// Detect negative token IDs without applying tokenizer rules to text input.
    fn contains_invalid_token(&self) -> bool {
        match self {
            Self::Text(_) | Self::Texts(_) => false,
            Self::Tokens(tokens) => tokens.iter().any(|token| *token < 0),
            Self::TokenArrays(inputs) => inputs.iter().flatten().any(|token| *token < 0),
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct EmbeddingsRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default = "default_encoding_format")]
    pub encoding_format: String,
    #[serde(default)]
    pub dimensions: Option<usize>,
    #[serde(default)]
    pub user: Option<String>,
}

/// Preserve the endpoint's default numeric-vector response representation.
fn default_encoding_format() -> String {
    DEFAULT_ENCODING_FORMAT.to_string()
}

impl EmbeddingsRequest {
    /// Validate batch content, output encoding, and optional positive dimensions.
    pub fn validate(&self) -> OpenAiResult<()> {
        if self.model.trim().is_empty() {
            return Err(OpenAiError::invalid_request("model must not be empty"));
        }
        if self.input.is_empty() {
            return Err(OpenAiError::invalid_request(
                "embedding input must contain at least one non-empty item",
            ));
        }
        if self.input.contains_invalid_token() {
            return Err(OpenAiError::invalid_request(
                "embedding token IDs must be non-negative",
            ));
        }
        if !matches!(self.encoding_format.as_str(), "float" | "base64") {
            return Err(OpenAiError::invalid_request(
                "encoding_format must be 'float' or 'base64'",
            ));
        }
        if self.dimensions == Some(0) {
            return Err(OpenAiError::invalid_request(
                "dimensions must be greater than zero",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    pub values: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(untagged)]
pub enum EmbeddingOutput {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct EmbeddingObject {
    pub object: &'static str,
    pub embedding: EmbeddingOutput,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingObject>,
    pub model: String,
    pub usage: Usage,
}

impl EmbeddingResponse {
    /// Preserve input indexes and encode vectors with the already-validated format.
    /// Embedding usage contains prompt tokens only; this endpoint generates none.
    pub fn from_embeddings(
        model: String,
        embeddings: Vec<Embedding>,
        prompt_tokens: u32,
        encoding_format: &str,
    ) -> Self {
        let data = embeddings
            .into_iter()
            .map(|embedding| EmbeddingObject {
                object: "embedding",
                embedding: if encoding_format == "base64" {
                    EmbeddingOutput::Base64(encode_f32_base64(&embedding.values))
                } else {
                    EmbeddingOutput::Float(embedding.values)
                },
                index: embedding.index,
            })
            .collect();
        Self {
            object: "list",
            data,
            model,
            usage: Usage {
                prompt_tokens,
                completion_tokens: 0,
                total_tokens: prompt_tokens,
                ..Usage::default()
            },
        }
    }
}

/// Encode little-endian IEEE-754 samples, independent of the host byte order.
fn encode_f32_base64(values: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    STANDARD.encode(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_encoding_is_little_endian_f32() {
        assert_eq!(encode_f32_base64(&[1.0, -2.0]), "AACAPwAAAMA=");
    }

    #[test]
    fn request_rejects_empty_batches_and_unknown_formats() {
        let mut request = EmbeddingsRequest {
            model: "embed".into(),
            input: EmbeddingInput::Texts(Vec::new()),
            encoding_format: "float".into(),
            dimensions: None,
            user: None,
        };
        assert!(request.validate().is_err());
        request.input = EmbeddingInput::Text("hello".into());
        request.encoding_format = "hex".into();
        assert!(request.validate().is_err());
    }

    #[test]
    fn request_rejects_negative_token_ids() {
        let request = EmbeddingsRequest {
            model: "embed".into(),
            input: EmbeddingInput::TokenArrays(vec![vec![1, 2], vec![3, -1]]),
            encoding_format: "float".into(),
            dimensions: None,
            user: None,
        };

        let error = request.validate().expect_err("negative IDs must fail");
        assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST);
        assert!(error.body().error.message.contains("non-negative"));
    }
}
