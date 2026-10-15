//! The bar window: a node with a referee-signed contradiction for a model is
//! not asked to referee that model for D days (the operator's
//! `referee_bar_days`, 30 unless set).
//!
//! The window runs from the time THIS node recorded the verdict (its
//! `received_at`, or `issued_at` for a verdict it issued), never from a time
//! the referee wrote, so a referee cannot backdate a bar away. A later
//! corroboration does not end it early. It is per model: a bar for one model
//! leaves the node eligible for another. The history stays in the counts.
//!
//! The rules here are decided (2026-09-30), each pinned by a constant and a
//! test of its own name, so a later change flips one constant and one test:
//!
//! - [`BAR_WINDOW_INCLUDES_ITS_END`]: the window is `[t, t + D)`: barred one
//!   second before `t + D`, eligible at exactly `t + D`
//!   (`the_bar_window_is_t_to_t_plus_d_half_open`).
//! - [`LAPSED_BAR_STARTS_WITHOUT_HISTORY`]: once a bar lapses the node is in
//!   tier 2 until it has a corroboration for that model dated after the
//!   lapse (`a_lapsed_bar_is_tier_2_until_a_fresh_corroboration`).

use chrono::{DateTime, Duration, Utc};

use crate::verdict_counts::{CONTRADICTED, CORROBORATED};

/// `referee_bar_days` when the operator has not set it.
pub const DEFAULT_BAR_DAYS: u32 = 30;
/// The setting: a positive whole number of days.
pub const ENV_REFEREE_BAR_DAYS: &str = "CAPSULES_REFEREE_BAR_DAYS";

/// Decided: the bar window is half-open, `[t, t + D)`. `true` would make it
/// `[t, t + D]`.
pub const BAR_WINDOW_INCLUDES_ITS_END: bool = false;
/// Decided: a node whose bar has lapsed starts again with no history for that
/// model; a corroboration counts toward tier 1 only when it is dated after
/// the lapse. `false` would let a corroboration from before the lapse count.
pub const LAPSED_BAR_STARTS_WITHOUT_HISTORY: bool = true;

/// Where the time comes from, so tests inject it.
pub trait Clock {
    fn now(&self) -> DateTime<Utc>;
}

/// The wall clock.
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

/// The operator's `referee_bar_days`: a positive whole number, else 30.
pub fn bar_days_from(raw: Option<&str>) -> u32 {
    raw.map(str::trim)
        .filter(|raw| !raw.is_empty() && raw.bytes().all(|b| b.is_ascii_digit()))
        .and_then(|raw| raw.parse::<u32>().ok())
        .filter(|days| *days > 0)
        .unwrap_or(DEFAULT_BAR_DAYS)
}

pub fn bar_days() -> u32 {
    bar_days_from(crate::settings::var(ENV_REFEREE_BAR_DAYS).ok().as_deref())
}

/// One counted verdict about a node, for one model, as this node recorded it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recorded {
    pub node_id: String,
    pub model_hash: String,
    /// One of the four buckets (`crate::verdict_counts::BUCKETS`).
    pub bucket: String,
    /// When THIS node recorded it.
    pub recorded_at: DateTime<Utc>,
}

/// The bar of one contradiction recorded at `at`, as `(from, until)`.
fn window(at: DateTime<Utc>, days: u32) -> (DateTime<Utc>, DateTime<Utc>) {
    (at, at + Duration::days(i64::from(days)))
}

fn inside(now: DateTime<Utc>, (from, until): (DateTime<Utc>, DateTime<Utc>)) -> bool {
    now >= from && (now < until || (BAR_WINDOW_INCLUDES_ITS_END && now == until))
}

/// `node` has a contradiction for `model` whose bar covers `now`.
pub fn barred(
    node: &str,
    model: &str,
    verdicts: &[Recorded],
    now: DateTime<Utc>,
    days: u32,
) -> bool {
    if cfg!(feature = "mutant-referee-select-skips-bar-window") {
        return false;
    }
    verdicts
        .iter()
        .filter(|v| v.node_id == node && v.model_hash == model && v.bucket == CONTRADICTED)
        .any(|v| inside(now, window(v.recorded_at, days)))
}

/// `node` has a corroboration for `model` that counts toward tier 1: after
/// its last bar lapsed, when it had one.
pub fn corroborated(node: &str, model: &str, verdicts: &[Recorded], days: u32) -> bool {
    let about = || {
        verdicts
            .iter()
            .filter(|v| v.node_id == node && v.model_hash == model)
    };
    let lapsed = about()
        .filter(|v| v.bucket == CONTRADICTED)
        .map(|v| window(v.recorded_at, days).1)
        .max();
    about()
        .filter(|v| v.bucket == CORROBORATED)
        .any(|v| match lapsed {
            Some(lapse) if LAPSED_BAR_STARTS_WITHOUT_HISTORY => v.recorded_at > lapse,
            _ => true,
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(text: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(text).unwrap().into()
    }

    fn recorded(bucket: &str, when: &str) -> Recorded {
        Recorded {
            node_id: "node-c".into(),
            model_hash: "m".into(),
            bucket: bucket.into(),
            recorded_at: at(when),
        }
    }

    /// Decided (b): the bar window is [t, t+D).
    #[test]
    fn the_bar_window_is_t_to_t_plus_d_half_open() {
        const { assert!(!BAR_WINDOW_INCLUDES_ITS_END) };
        let v = [recorded(CONTRADICTED, "2026-08-30T00:00:00Z")];
        assert!(
            barred("node-c", "m", &v, at("2026-08-30T00:00:00Z"), 30),
            "barred from t"
        );
        assert!(
            barred("node-c", "m", &v, at("2026-09-28T23:59:59Z"), 30),
            "one second before t+D"
        );
        assert!(
            !barred("node-c", "m", &v, at("2026-09-29T00:00:00Z"), 30),
            "eligible at t+D"
        );
        assert!(
            !barred("node-c", "m", &v, at("2026-08-29T23:59:59Z"), 30),
            "a verdict recorded later bars nothing now"
        );
    }

    /// Decided (a): a lapsed bar is tier 2 until a fresh corroboration.
    #[test]
    fn a_lapsed_bar_is_tier_2_until_a_fresh_corroboration() {
        const { assert!(LAPSED_BAR_STARTS_WITHOUT_HISTORY) };
        let contradiction = recorded(CONTRADICTED, "2026-07-01T00:00:00Z");
        let before = recorded(CORROBORATED, "2026-06-01T00:00:00Z");
        let inside_window = recorded(CORROBORATED, "2026-07-15T00:00:00Z");
        let after = recorded(CORROBORATED, "2026-08-01T00:00:00Z");
        assert!(!corroborated(
            "node-c",
            "m",
            &[contradiction.clone(), before, inside_window],
            30
        ));
        assert!(corroborated("node-c", "m", &[contradiction, after], 30));
        assert!(
            corroborated(
                "node-c",
                "m",
                &[recorded(CORROBORATED, "2026-06-01T00:00:00Z")],
                30
            ),
            "never barred"
        );
    }

    #[test]
    fn a_bar_is_per_model() {
        let v = [recorded(CONTRADICTED, "2026-09-28T00:00:00Z")];
        assert!(barred("node-c", "m", &v, at("2026-09-29T00:00:00Z"), 30));
        assert!(!barred(
            "node-c",
            "other",
            &v,
            at("2026-09-29T00:00:00Z"),
            30
        ));
    }

    #[test]
    fn the_setting_is_a_positive_whole_number_else_30() {
        assert_eq!(bar_days_from(None), 30);
        assert_eq!(bar_days_from(Some(" 10 ")), 10);
        for raw in ["0", "-3", "ten", "", "+5", "1.5"] {
            assert_eq!(bar_days_from(Some(raw)), 30, "{raw:?}");
        }
    }
}
