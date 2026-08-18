//! Prefix-diffing for text that is re-parsed in full on every decoded token.
//!
//! The chat parser hands back the whole message on every token, so content,
//! reasoning, and tool-call arguments all arrive as growing strings. Streaming
//! them means sending only the part that is new since the last chunk.

/// Append the part of `current` that extends `emitted`, or `None` when there is
/// nothing new or `current` is not an extension of what was already emitted.
///
/// Returning `None` for a non-extension is deliberate: a client concatenates the
/// fragments it receives, so a value that has been revised rather than extended
/// cannot be corrected by sending more bytes.
pub(in crate::frontend) fn suffix_delta(
    current: Option<&str>,
    emitted: &mut String,
) -> Option<String> {
    let current = current?;
    let delta = current.strip_prefix(emitted.as_str())?;
    if delta.is_empty() {
        return None;
    }
    emitted.push_str(delta);
    Some(delta.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn suffix_delta_reports_only_the_new_suffix() {
        let mut emitted = String::from("abc");

        assert_eq!(
            suffix_delta(Some("abcdef"), &mut emitted).as_deref(),
            Some("def")
        );
        assert_eq!(emitted, "abcdef");
        assert!(suffix_delta(Some("abcdef"), &mut emitted).is_none());
        assert!(suffix_delta(Some("xyz"), &mut emitted).is_none());
        assert!(suffix_delta(None, &mut emitted).is_none());
    }
}
