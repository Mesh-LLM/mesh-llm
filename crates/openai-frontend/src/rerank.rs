use serde::{Deserialize, Serialize};

use crate::{OpenAiError, OpenAiResult, Usage};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(untagged)]
pub enum RerankDocument {
    Text(String),
    Object(serde_json::Value),
}

impl RerankDocument {
    /// Borrow plain text or a document object's required string `text` field.
    pub fn text(&self) -> OpenAiResult<&str> {
        match self {
            Self::Text(text) => Ok(text),
            Self::Object(value) => value
                .get("text")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    OpenAiError::invalid_request(
                        "rerank document objects must contain a string 'text' field",
                    )
                }),
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<RerankDocument>,
    #[serde(default)]
    pub top_n: Option<usize>,
    #[serde(default)]
    pub return_documents: bool,
}

impl RerankRequest {
    /// Validate every document before admission, including optional top-N bounds.
    pub fn validate(&self) -> OpenAiResult<()> {
        if self.model.trim().is_empty() {
            return Err(OpenAiError::invalid_request("model must not be empty"));
        }
        if self.query.is_empty() {
            return Err(OpenAiError::invalid_request("query must not be empty"));
        }
        if self.documents.is_empty() {
            return Err(OpenAiError::invalid_request(
                "documents must contain at least one item",
            ));
        }
        for document in &self.documents {
            if document.text()?.is_empty() {
                return Err(OpenAiError::invalid_request(
                    "rerank documents must not be empty",
                ));
            }
        }
        if self.top_n == Some(0) {
            return Err(OpenAiError::invalid_request(
                "top_n must be greater than zero",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<RerankDocument>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct RerankResponse {
    pub id: String,
    pub results: Vec<RerankResult>,
    pub usage: Usage,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_objects_require_text() {
        let document = RerankDocument::Object(serde_json::json!({"title": "missing"}));
        assert!(document.text().is_err());
    }
}
