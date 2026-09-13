use anyhow::{Context, Result};

/// Reject syntax errors before routing can mistake them for a missing model.
/// Uploads and plugin routes keep their own body contracts; tokenize already
/// validates its metadata separately. Empty bodies retain existing behavior.
pub(super) fn validate_inference_json(method: &str, path: &str, body: &[u8]) -> Result<()> {
    if method == "POST"
        && matches!(
            path.split('?').next().unwrap_or(path),
            "/v1/chat/completions" | "/v1/completions" | "/v1/responses" | "/v1/embeddings"
        )
        && !body.is_empty()
    {
        serde_json::from_slice::<serde_json::Value>(body)
            .context("request body is not valid JSON")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_routes_reject_malformed_json_including_query_strings() {
        for path in [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/responses",
            "/v1/embeddings",
        ] {
            for body in [
                b"{broken".as_slice(),
                b"  ",
                b"{} trailing",
                b"{\"x\":\xff}",
                b"{\"x\":\"\xff\"}",
                br#"{"x":"\uD800"}"#,
            ] {
                assert!(validate_inference_json("POST", path, body).is_err());
                assert!(validate_inference_json("POST", &format!("{path}?trace=1"), body).is_err());
            }
        }
    }

    #[test]
    fn validation_does_not_impose_a_model_or_metadata_schema() {
        for body in [b"".as_slice(), b"{}", b"null", b"[]", br#"{"model":42}"#] {
            assert!(validate_inference_json("POST", "/v1/chat/completions", body).is_ok());
        }
    }

    #[test]
    fn non_json_routes_and_other_methods_are_untouched() {
        for path in [
            "/api/objects",
            "/v1/audio/transcriptions",
            "/api/plugins/demo/http",
            "/v1/tokenize",
        ] {
            assert!(validate_inference_json("POST", path, b"not json").is_ok());
        }
        assert!(validate_inference_json("GET", "/v1/chat/completions", b"not json").is_ok());
    }
}
