//! When a busy node is asked again.
//!
//! A stage with too many openings waiting answers 503 with a `Retry-After`
//! and admits nothing, so the same request can be sent again. The node
//! keeps a turn open for [`WINDOW`] after its receipt, so a request is sent
//! again only while the pause still ends inside that time.

use std::time::Duration;

use reqwest::StatusCode;

/// How long after its first try a request is still sent again.
pub(super) const WINDOW: Duration = Duration::from_secs(10);
/// The shortest pause, whatever the node's `Retry-After` says.
const MIN_PAUSE: Duration = Duration::from_secs(1);

/// The pause before a request is sent again: `status` and `retry_after`
/// are the node's answer, and `asked_for` is the time since the first try.
/// `None` when the answer is not a busy one, or when the pause would end
/// past the window.
pub(super) fn pause(
    status: StatusCode,
    retry_after: Option<&str>,
    asked_for: Duration,
) -> Option<Duration> {
    if status != StatusCode::SERVICE_UNAVAILABLE {
        return None;
    }
    let seconds = retry_after?.trim().parse().ok()?;
    let pause = Duration::from_secs(seconds).max(MIN_PAUSE);
    (asked_for + pause <= WINDOW).then_some(pause)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUSY: StatusCode = StatusCode::SERVICE_UNAVAILABLE;

    #[test]
    fn a_busy_node_is_asked_again_after_the_pause_it_names() {
        let two = Some(Duration::from_secs(2));
        assert_eq!(pause(BUSY, Some("2"), Duration::ZERO), two);
        assert_eq!(pause(BUSY, Some(" 2 "), Duration::from_secs(8)), two);
        // no pause asked for is still a second
        assert_eq!(
            pause(BUSY, Some("0"), Duration::ZERO),
            Some(Duration::from_secs(1))
        );
    }

    #[test]
    fn a_pause_past_the_window_is_not_waited_for() {
        assert_eq!(pause(BUSY, Some("2"), Duration::from_millis(8_001)), None);
        assert_eq!(pause(BUSY, Some("60"), Duration::ZERO), None);
    }

    #[test]
    fn only_a_busy_answer_is_asked_again() {
        // an unloaded stage answers 503 without a retry time
        assert_eq!(pause(BUSY, None, Duration::ZERO), None);
        assert_eq!(pause(BUSY, Some("soon"), Duration::ZERO), None);
        for status in [StatusCode::OK, StatusCode::CONFLICT, StatusCode::GONE] {
            assert_eq!(pause(status, Some("2"), Duration::ZERO), None);
        }
    }
}
