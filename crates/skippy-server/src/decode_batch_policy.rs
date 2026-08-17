use std::{env, time::Duration};

const COLLECTION_US_ENV: &str = "SKIPPY_DECODE_BATCH_COLLECTION_US";
const MAX_COLLECTION_US: u64 = 10_000;

pub(crate) fn collection_window(max_batch_size: usize) -> Duration {
    collection_window_from_value(max_batch_size, env::var(COLLECTION_US_ENV).ok().as_deref())
}

fn collection_window_from_value(max_batch_size: usize, value: Option<&str>) -> Duration {
    if max_batch_size <= 1 {
        return Duration::ZERO;
    }
    let micros = value
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(0)
        .min(MAX_COLLECTION_US);
    Duration::from_micros(micros)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_lane_never_waits() {
        assert_eq!(
            collection_window_from_value(1, Some("1000")),
            Duration::ZERO
        );
    }

    #[test]
    fn invalid_or_missing_values_disable_collection() {
        assert_eq!(collection_window_from_value(8, None), Duration::ZERO);
        assert_eq!(
            collection_window_from_value(8, Some("invalid")),
            Duration::ZERO
        );
    }

    #[test]
    fn collection_window_is_bounded() {
        assert_eq!(
            collection_window_from_value(8, Some("750")),
            Duration::from_micros(750)
        );
        assert_eq!(
            collection_window_from_value(8, Some("50000")),
            Duration::from_micros(MAX_COLLECTION_US)
        );
    }
}
