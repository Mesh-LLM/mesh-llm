//! Limits on request stop sequences and the per-token stop search.
//!
//! Generation checks the stop sequences after every token. Searching the
//! whole accumulated text for every stop string made each token cost
//! `stops × text length`, so a request with many long stop strings could keep
//! a generation lane busy almost indefinitely. Requests are bounded here, and
//! each check only searches the text a new token could have completed.

use skippy_inference_api::{InferenceError, InferenceResult, StopSequence};

/// Most stop sequences one request may set.
pub(super) const MAX_STOP_SEQUENCES: usize = 64;
/// Longest stop sequence a request may set, in bytes.
pub(super) const MAX_STOP_SEQUENCE_BYTES: usize = 1024;

/// Rejects a request whose stop sequences exceed the count or length limits.
pub(super) fn validate_stop_sequences(stop: Option<&StopSequence>) -> InferenceResult<()> {
    let Some(stop) = stop else {
        return Ok(());
    };
    let values = stop.values();
    if values.len() > MAX_STOP_SEQUENCES {
        return Err(InferenceError::invalid_request(format!(
            "stop supports at most {MAX_STOP_SEQUENCES} sequences"
        )));
    }
    if values
        .iter()
        .any(|value| value.len() > MAX_STOP_SEQUENCE_BYTES)
    {
        return Err(InferenceError::invalid_request(format!(
            "each stop sequence must be at most {MAX_STOP_SEQUENCE_BYTES} bytes"
        )));
    }
    Ok(())
}

/// Returns whether a stop sequence ends inside `text[searched_len..]`.
///
/// `searched_len` is how much of `text` earlier checks already searched. A
/// stop that ends in the new text starts at most `max_stop_bytes - 1` bytes
/// before it, so only that tail and the new text are searched.
pub(super) fn stop_in_new_text(
    text: &str,
    searched_len: usize,
    stop_values: &[&str],
    max_stop_bytes: usize,
) -> bool {
    let mut start = searched_len
        .min(text.len())
        .saturating_sub(max_stop_bytes.saturating_sub(1));
    while !text.is_char_boundary(start) {
        start -= 1;
    }
    let window = &text[start..];
    stop_values
        .iter()
        .any(|stop| !stop.is_empty() && window.contains(stop))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_too_many_or_too_long_stop_sequences() {
        let at_limit = StopSequence::Many(vec!["x".to_string(); MAX_STOP_SEQUENCES]);
        validate_stop_sequences(Some(&at_limit)).unwrap();
        let too_many = StopSequence::Many(vec!["x".to_string(); MAX_STOP_SEQUENCES + 1]);
        assert!(validate_stop_sequences(Some(&too_many)).is_err());

        let longest = StopSequence::One("x".repeat(MAX_STOP_SEQUENCE_BYTES));
        validate_stop_sequences(Some(&longest)).unwrap();
        let too_long = StopSequence::One("x".repeat(MAX_STOP_SEQUENCE_BYTES + 1));
        assert!(validate_stop_sequences(Some(&too_long)).is_err());

        validate_stop_sequences(None).unwrap();
    }

    #[test]
    fn finds_stops_that_span_the_previous_check() {
        // "STOP" straddles the boundary between earlier text and the new token.
        assert!(stop_in_new_text("hello STOP", 8, &["STOP"], 4));
        assert!(stop_in_new_text("STOP", 0, &["STOP"], 4));
        assert!(!stop_in_new_text("hello STO", 6, &["STOP"], 4));
    }

    #[test]
    fn does_not_rescan_text_earlier_checks_covered() {
        // A stop wholly inside already-searched text was reported then; the
        // next check only looks at the tail a new token could complete.
        let text = format!("STOP{}", "a".repeat(100));
        assert!(!stop_in_new_text(&text, text.len() - 1, &["STOP"], 4));
    }

    #[test]
    fn tail_start_respects_char_boundaries() {
        let text = "ééSTOP";
        for searched_len in 0..=text.len() {
            stop_in_new_text(text, searched_len, &["éSTOP"], "éSTOP".len());
        }
        assert!(stop_in_new_text(text, 3, &["éSTOP"], "éSTOP".len()));
    }
}
