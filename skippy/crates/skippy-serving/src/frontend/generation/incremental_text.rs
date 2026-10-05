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

/// One named fixture's partial-parse snapshots and terminal parse, read from a
/// `RECORDED` fixture file (see `tests/fixtures/README.md`).
///
/// Kept next to `suffix_delta` so `tool_call_stream` and `chat_stream_deltas`
/// read the fixture's line format exactly once; a format change (comment/
/// blank-line skipping, the `splitn` layout, the `final` marker) only needs
/// updating here.
#[cfg(test)]
pub(in crate::frontend) struct RecordedFixture {
    pub(in crate::frontend) snapshots: Vec<Vec<serde_json::Value>>,
    pub(in crate::frontend) final_call: Vec<serde_json::Value>,
}

#[cfg(test)]
pub(in crate::frontend) fn recorded_fixture(recorded: &str, fixture: &str) -> RecordedFixture {
    let mut snapshots = Vec::new();
    let mut final_call = None;
    for line in recorded.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(3, ' ');
        let (name, marker, payload) = (
            parts.next().expect("fixture name"),
            parts.next().expect("prefix marker"),
            parts.next().expect("tool_calls json"),
        );
        if name != fixture {
            continue;
        }
        let calls = serde_json::from_str::<Vec<serde_json::Value>>(payload).expect("fixture json");
        if marker == "final" {
            final_call = Some(calls);
        } else {
            snapshots.push(calls);
        }
    }
    assert!(!snapshots.is_empty(), "no snapshots for fixture {fixture}");
    RecordedFixture {
        snapshots,
        final_call: final_call.expect("fixture final parse"),
    }
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
