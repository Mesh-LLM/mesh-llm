//! Admit immutable Hub commit identity before any cache-path construction.
use crate::error::{HFError, HFResult};
pub(in super::super) fn admit_commit(commit: &str, requested: &str) -> HFResult<String> {
    let valid =
        |value: &str| value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit());
    if !valid(commit) || (valid(requested) && !commit.eq_ignore_ascii_case(requested)) {
        return Err(HFError::InvalidParameter(
            "Hub commit identity is malformed or differs from requested immutable commit".into(),
        ));
    }
    Ok(commit.to_owned())
}
pub(in super::super) fn admit_etag(etag: &str) -> HFResult<String> {
    if etag.is_empty()
        || matches!(etag, "." | "..")
        || etag
            .chars()
            .any(|ch| ch.is_control() || matches!(ch, '/' | '\\' | ':'))
    {
        return Err(HFError::InvalidParameter(
            "Hub ETag is not a cache filename".into(),
        ));
    }
    Ok(etag.to_owned())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn malformed_and_substituted_commit_cannot_enter_cache_paths() {
        let commit = "a".repeat(40);
        assert_eq!(admit_commit(&commit, "release/étape#v2").unwrap(), commit);
        assert!(admit_commit(&"b".repeat(40), &commit).is_err());
        for invalid in ["../outside", "/tmp/foreign", "", "abc\ndef"] {
            assert!(admit_commit(invalid, "main").is_err());
        }
        assert!(admit_commit(&"g".repeat(40), "main").is_err());
        assert_eq!(admit_etag("etag-http").unwrap(), "etag-http");
        for invalid in ["../outside", "a/b", "a\\b", "", ".", "..", "a:b"] {
            assert!(admit_etag(invalid).is_err());
        }
    }
}
