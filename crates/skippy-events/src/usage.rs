//! Authoritative inference token accounting.

use serde::{Deserialize, Serialize};

/// Upstream token counts exactly as reported by a compatible API. Each field
/// remains absent when upstream omitted it; consumers must never estimate it.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct TokenUsage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_prompt_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_tokens: Option<u64>,
}

impl TokenUsage {
    /// Construct an authoritative usage record only when the provider supplied
    /// every count and the total is arithmetically consistent. Missing,
    /// overflowing, or internally inconsistent usage must not be estimated.
    pub fn from_counts(
        prompt_tokens: Option<u64>,
        completion_tokens: Option<u64>,
        total_tokens: Option<u64>,
    ) -> Option<Self> {
        let (Some(prompt_tokens), Some(completion_tokens), Some(total_tokens)) =
            (prompt_tokens, completion_tokens, total_tokens)
        else {
            return None;
        };
        if prompt_tokens.checked_add(completion_tokens) != Some(total_tokens) {
            return None;
        }
        Some(Self {
            prompt_tokens: Some(prompt_tokens),
            cached_prompt_tokens: None,
            completion_tokens: Some(completion_tokens),
            total_tokens: Some(total_tokens),
        })
    }

    pub fn with_cached_prompt_tokens(mut self, cached_prompt_tokens: Option<u64>) -> Self {
        self.cached_prompt_tokens = match (self.prompt_tokens, cached_prompt_tokens) {
            (Some(prompt_tokens), Some(cached_tokens)) if cached_tokens > prompt_tokens => None,
            (_, cached_tokens) => cached_tokens,
        };
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_prompt_tokens_cannot_exceed_prompt_tokens() {
        let usage = TokenUsage::from_counts(Some(8), Some(3), Some(11)).unwrap();

        assert_eq!(
            usage
                .with_cached_prompt_tokens(Some(8))
                .cached_prompt_tokens,
            Some(8)
        );
        assert_eq!(
            usage
                .with_cached_prompt_tokens(Some(9))
                .cached_prompt_tokens,
            None
        );
    }
}
