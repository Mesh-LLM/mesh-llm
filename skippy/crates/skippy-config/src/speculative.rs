//! Validation of speculative generation configuration.

/// Validate that draft_min_tokens <= draft_max_tokens for speculative decoding.
pub fn validate_draft_min_max(draft_min_tokens: u32, draft_max_tokens: u32) -> Result<(), String> {
    if draft_min_tokens > draft_max_tokens {
        Err(
            "skippy speculative draft_min_tokens must be less than or equal to draft_max_tokens"
                .to_string(),
        )
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_draft_min_max_accepts_equal_and_ordered_values() {
        assert!(validate_draft_min_max(0, 0).is_ok());
        assert!(validate_draft_min_max(0, 3).is_ok());
        assert!(validate_draft_min_max(3, 3).is_ok());
    }

    #[test]
    fn validate_draft_min_max_rejects_min_greater_than_max() {
        let error = validate_draft_min_max(4, 3).expect_err("min greater than max should fail");
        assert!(error.contains("draft_min_tokens must be less than or equal to draft_max_tokens"));
    }
}
