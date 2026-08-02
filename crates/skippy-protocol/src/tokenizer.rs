use serde::{Deserialize, Serialize};

pub const MAX_TOKENIZE_INPUT_BYTES: usize = 1_048_576;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TokenizerIdentity {
    pub model_id: String,
    pub source_model_sha256: String,
    pub tokenizer_id: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TokenizeRequest {
    pub expected_identity: TokenizerIdentity,
    pub text: String,
    #[serde(default)]
    pub add_special: bool,
    #[serde(default)]
    pub include_token_pieces: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TokenizeResponse {
    pub identity: TokenizerIdentity,
    pub token_ids: Vec<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_pieces: Option<Vec<Vec<u8>>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_wire_contract_is_exact() {
        let identity = TokenizerIdentity {
            model_id: "model".to_owned(),
            source_model_sha256: "a".repeat(64),
            tokenizer_id: format!("gguf-source-sha256:{}", "a".repeat(64)),
        };
        let request: TokenizeRequest = serde_json::from_value(serde_json::json!({
            "expected_identity": identity,
            "text": "hello",
        }))
        .unwrap();
        assert!(!request.add_special);
        assert!(!request.include_token_pieces);

        let response = TokenizeResponse {
            identity: request.expected_identity,
            token_ids: vec![1, 2],
            token_pieces: None,
        };
        assert_eq!(
            serde_json::to_value(response).unwrap(),
            serde_json::json!({
                "identity": {
                    "model_id": "model",
                    "source_model_sha256": "a".repeat(64),
                    "tokenizer_id": format!("gguf-source-sha256:{}", "a".repeat(64)),
                },
                "token_ids": [1, 2],
            })
        );
        assert_eq!(MAX_TOKENIZE_INPUT_BYTES, 1_048_576);
    }
}
