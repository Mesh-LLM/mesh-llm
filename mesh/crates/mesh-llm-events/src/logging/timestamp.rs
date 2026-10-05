//! Canonical timestamp representation for ordered logging data.

use std::fmt;

use chrono::{DateTime, SecondsFormat, Utc};

/// Error returned when a logging timestamp is not valid RFC 3339.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InvalidLoggingTimestamp;

impl fmt::Display for InvalidLoggingTimestamp {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("logging timestamp must be RFC 3339")
    }
}

impl std::error::Error for InvalidLoggingTimestamp {}

/// Normalize an RFC 3339 instant to a fixed-width, lexically sortable UTC key.
///
/// Nanosecond precision preserves every instant accepted by `chrono` while the
/// fixed width makes ordinary string ordering equivalent to chronological
/// ordering. Offset, whole-second, and fractional inputs therefore share one
/// representation before they reach SQLite indexes or in-memory ledgers.
pub fn canonical_logging_timestamp(value: &str) -> Result<String, InvalidLoggingTimestamp> {
    DateTime::parse_from_rfc3339(value)
        .map(|timestamp| {
            timestamp
                .with_timezone(&Utc)
                .to_rfc3339_opts(SecondsFormat::Nanos, true)
        })
        .map_err(|_| InvalidLoggingTimestamp)
}

#[cfg(test)]
mod tests {
    use super::canonical_logging_timestamp;

    #[test]
    fn canonical_timestamp_is_fixed_width_utc_and_preserves_precision() {
        assert_eq!(
            canonical_logging_timestamp("2026-08-03T00:00:00Z").unwrap(),
            "2026-08-03T00:00:00.000000000Z"
        );
        assert_eq!(
            canonical_logging_timestamp("2026-08-03T00:00:00.123Z").unwrap(),
            "2026-08-03T00:00:00.123000000Z"
        );
        assert_eq!(
            canonical_logging_timestamp("2026-08-03T01:00:00.123456789+01:00").unwrap(),
            "2026-08-03T00:00:00.123456789Z"
        );
    }
}
