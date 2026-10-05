use std::time::Duration;

const STAGE_REPLY_TIMEOUT_ENV: &str = "SKIPPY_STAGE_REPLY_TIMEOUT_SECS";
const DEFAULT_STAGE_REPLY_TIMEOUT_SECS: u64 = 30;
const MIN_STAGE_REPLY_TIMEOUT_SECS: u64 = 1;
const MAX_STAGE_REPLY_TIMEOUT_SECS: u64 = 60 * 60;

pub(in crate::frontend) fn stage_reply_timeout() -> Duration {
    stage_reply_timeout_from(std::env::var(STAGE_REPLY_TIMEOUT_ENV).ok().as_deref())
}

fn stage_reply_timeout_from(value: Option<&str>) -> Duration {
    let seconds = value
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_STAGE_REPLY_TIMEOUT_SECS)
        .clamp(MIN_STAGE_REPLY_TIMEOUT_SECS, MAX_STAGE_REPLY_TIMEOUT_SECS);
    Duration::from_secs(seconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reply_timeout_defaults_to_thirty_seconds() {
        assert_eq!(stage_reply_timeout_from(None), Duration::from_secs(30));
        assert_eq!(
            stage_reply_timeout_from(Some("invalid")),
            Duration::from_secs(30)
        );
    }

    #[test]
    fn reply_timeout_accepts_and_bounds_override() {
        assert_eq!(
            stage_reply_timeout_from(Some(" 180 ")),
            Duration::from_secs(180)
        );
        assert_eq!(stage_reply_timeout_from(Some("0")), Duration::from_secs(1));
        assert_eq!(
            stage_reply_timeout_from(Some("99999")),
            Duration::from_secs(3600)
        );
    }
}
