//! Per-client-IP reconnect rate limiting, from the frozen bounds table:
//! 10 connects per client key per 60 seconds; client key is peer IP.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

use crate::runtime_events::config::{RECONNECT_LIMIT_PER_WINDOW, RECONNECT_WINDOW};

static ATTEMPTS: LazyLock<Mutex<HashMap<Option<IpAddr>, Vec<Instant>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Record a connect attempt for `key` (the caller's peer IP, or `None` when
/// unavailable) and report whether it is within the frozen rate limit.
/// `None` is bucketed together and rate-limited the same as any other key
/// rather than exempted, since an unavailable peer address is not proof of
/// a distinct caller.
pub(super) fn record_attempt(key: Option<IpAddr>) -> bool {
    record_attempt_at(key, Instant::now())
}

fn record_attempt_at(key: Option<IpAddr>, now: Instant) -> bool {
    let mut attempts = ATTEMPTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let entry = attempts.entry(key).or_default();
    entry.retain(|instant| now.duration_since(*instant) < RECONNECT_WINDOW);
    if entry.len() >= RECONNECT_LIMIT_PER_WINDOW {
        return false;
    }
    entry.push(now);
    true
}

#[cfg(test)]
pub(super) fn clear() {
    ATTEMPTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial(runtime_events_reconnect)]
    fn allows_up_to_the_frozen_limit_then_rejects() {
        clear();
        let key = Some(IpAddr::from([127, 0, 0, 1]));
        let start = Instant::now();
        for _ in 0..RECONNECT_LIMIT_PER_WINDOW {
            assert!(record_attempt_at(key, start));
        }
        assert!(!record_attempt_at(key, start));
        clear();
    }

    #[test]
    #[serial_test::serial(runtime_events_reconnect)]
    fn resets_after_the_window_elapses() {
        clear();
        let key = Some(IpAddr::from([127, 0, 0, 2]));
        let start = Instant::now();
        for _ in 0..RECONNECT_LIMIT_PER_WINDOW {
            assert!(record_attempt_at(key, start));
        }
        assert!(record_attempt_at(key, start + RECONNECT_WINDOW));
        clear();
    }

    #[test]
    #[serial_test::serial(runtime_events_reconnect)]
    fn distinct_keys_have_independent_budgets() {
        clear();
        let a = Some(IpAddr::from([127, 0, 0, 3]));
        let b = Some(IpAddr::from([127, 0, 0, 4]));
        let start = Instant::now();
        for _ in 0..RECONNECT_LIMIT_PER_WINDOW {
            assert!(record_attempt_at(a, start));
        }
        assert!(!record_attempt_at(a, start));
        assert!(record_attempt_at(b, start));
        clear();
    }
}
